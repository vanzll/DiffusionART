from __future__ import annotations

import math
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass

import torch


@dataclass
class Progress:
    attempts: int = 0
    completed_rollouts: int = 0
    optimizer_steps: int = 0
    elapsed_seconds: float = 0.0
    training_seconds: float = 0.0

    def state_dict(self):
        return asdict(self)

    @classmethod
    def from_state_dict(cls, state):
        if set(state) != set(cls.__dataclass_fields__):
            raise ValueError("checkpoint progress fields differ")
        for key in ("attempts", "completed_rollouts", "optimizer_steps"):
            if type(state[key]) is not int or state[key] < 0:
                raise ValueError(f"invalid progress {key}")
        for key in ("elapsed_seconds", "training_seconds"):
            if not math.isfinite(state[key]) or state[key] < 0:
                raise ValueError(f"invalid progress {key}")
        if state["completed_rollouts"] > state["attempts"]:
            raise ValueError("completed rollouts exceed attempts")
        if state["training_seconds"] > state["elapsed_seconds"]:
            raise ValueError("training time exceeds elapsed time")
        return cls(**state)


class CostClock:
    """Actual elapsed and training-only costs; retries consume real time."""

    def __init__(self, progress: Progress, gpu_count: int, clock=time.monotonic):
        if type(gpu_count) is not int or gpu_count < 0:
            raise ValueError("gpu_count must be a nonnegative integer")
        self.progress, self.gpu_count, self.clock = progress, gpu_count, clock
        self.last = self.clock()
        self.training = False

    def tick(self):
        now = self.clock()
        elapsed = now - self.last
        if elapsed < 0:
            raise RuntimeError("monotonic clock moved backwards")
        self.progress.elapsed_seconds += elapsed
        if self.training:
            self.progress.training_seconds += elapsed
        self.last = now

    @contextmanager
    def training_phase(self):
        if self.training:
            raise RuntimeError("nested cost phases are not supported")
        self.tick()
        self.training = True
        try:
            yield
        finally:
            self.tick()
            self.training = False

    def metrics(self):
        self.tick()
        return {
            "train/optimizer_step": self.progress.optimizer_steps,
            "system/rollout_attempts": self.progress.attempts,
            "system/completed_rollouts": self.progress.completed_rollouts,
            "cost/elapsed_seconds": self.progress.elapsed_seconds,
            "cost/training_seconds": self.progress.training_seconds,
            "cost/elapsed_gpu_hours": self.progress.elapsed_seconds * self.gpu_count / 3600,
            "cost/training_gpu_hours": self.progress.training_seconds * self.gpu_count / 3600,
            "cost/gpu_count": self.gpu_count,
        }


class PolicyShadow:
    """Named, local unsharded policy state. FSDP requires its own adapter.

    EMA applies to floating parameters. Buffers are copied, not averaged.
    Swaps restore the current policy even if inference raises an exception.
    """

    def __init__(self, model: torch.nn.Module, *, trainable_only: bool = False):
        from torch.distributed.fsdp import FullyShardedDataParallel
        from torch.nn.parallel import DistributedDataParallel

        if isinstance(model, (DistributedDataParallel, FullyShardedDataParallel)):
            raise ValueError("pass the unsharded module; FSDP shadows need a shard-aware adapter")
        self.model = model
        self.active = False
        self.trainable_only = trainable_only
        parameters = [
            (name, value)
            for name, value in model.named_parameters(remove_duplicate=False)
            if not trainable_only or value.requires_grad
        ]
        self.parameter_names = {name for name, _ in parameters}
        if trainable_only and not parameters:
            raise ValueError("trainable policy shadow requires trainable parameters")
        self.shadow = self._snapshot()
        aliases = {}
        for name, parameter in parameters:
            aliases.setdefault(id(parameter), []).append(name)
        self.aliases = tuple(tuple(names) for names in aliases.values() if len(names) > 1)

    def _live(self):
        if self.trainable_only:
            return {
                k: v
                for k, v in self.model.named_parameters(remove_duplicate=False)
                if v.requires_grad
            }
        return self.model.state_dict()

    def _snapshot(self):
        return {k: v.detach().to(device="cpu", copy=True) for k, v in self._live().items()}

    @torch.no_grad()
    def _copy_to_model(self, state):
        if self.trainable_only:
            for name, value in self._live().items():
                value.copy_(state[name])
        else:
            self.model.load_state_dict(state, strict=True)

    def _validate(self, state):
        live = self._live()
        if set(state) != set(live):
            raise ValueError("policy shadow state names changed")
        for key, value in state.items():
            if value.shape != live[key].shape or value.dtype != live[key].dtype:
                raise ValueError(f"policy shadow metadata changed: {key}")
            if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(
                value
            ).all():
                raise ValueError(f"nonfinite policy shadow: {key}")
        for names in self.aliases:
            if any(not torch.equal(state[names[0]], state[name]) for name in names[1:]):
                raise ValueError(
                    f"policy shadow has inconsistent shared parameter aliases: {names}"
                )

    @torch.no_grad()
    def update(self, decay: float):
        if self.active or not math.isfinite(decay) or not 0 <= decay < 1:
            raise ValueError("invalid decay or active shadow swap")
        current = self._snapshot()
        self._validate(current)
        for key, target in self.shadow.items():
            if key in self.parameter_names and target.is_floating_point() and decay:
                target.mul_(decay).add_(current[key], alpha=1 - decay)
            else:
                target.copy_(current[key])

    @contextmanager
    def use(self):
        if self.active or any(p.grad is not None for p in self.model.parameters()):
            raise RuntimeError("shadow swap requires no active swap or accumulated gradients")
        self._validate(self.shadow)
        current = self._snapshot()
        self.active = True
        try:
            self._copy_to_model(self.shadow)
            yield
        finally:
            self._copy_to_model(current)
            self.active = False

    def state_dict(self):
        if self.active:
            raise RuntimeError("cannot checkpoint an active policy swap")
        return {key: value.clone() for key, value in self.shadow.items()}

    def load_state_dict(self, state):
        if self.active:
            raise RuntimeError("cannot load during policy swap")
        self._validate(state)
        self.shadow = {key: value.detach().cpu().clone() for key, value in state.items()}
