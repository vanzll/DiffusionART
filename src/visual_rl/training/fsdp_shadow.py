"""Rank-local, parameters-only CPU shadows for FSDP1 FULL_SHARD.

Buffers must be frozen; they are neither copied nor averaged. All operations
are local, including cache invalidation. Callers coordinate enter/exit failures
(e.g. with session.local); forwards inside use() still run FSDP collectives.
The checkpoint format is deliberately same-version, same-topology only.
"""

from __future__ import annotations

import copy
import math
from contextlib import contextmanager
from types import SimpleNamespace
from weakref import WeakSet

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._runtime_utils import _reshard

from .fsdp_checkpoint import _FSDPGroup, _layout

_ACTIVE_MODELS = WeakSet()


class FSDPPolicyShadow:
    """One CPU local-shard shadow, plus temporary CPU current shards in use().

    Requires cleared gradients and no outstanding forward/backward graph.
    use() is inference-only and disables autograd. A failed restore is terminal:
    the active guard remains set rather than advertising a usable current policy.
    """

    def __init__(self, model):
        self.model = model
        self.active = False
        self._group = _FSDPGroup(model)
        self._check_idle()
        self._metadata()  # Validate the supported FSDP configuration before mutation.
        self._reshard()
        self._parameters = dict(model.named_parameters())
        if not self._parameters:
            raise ValueError("policy shadow requires parameters")
        self._layout = self._metadata()
        self.shadow = self._snapshot()

    def _metadata(self):
        # Reuse checkpoint's exact flat layout/alias/precision/topology checks,
        # without constructing an optimizer or invoking any state_dict hooks.
        layout = _layout(self.model, SimpleNamespace(param_groups=[]), self._group)
        del layout["optimizer"], layout["optimizer_groups"]
        return layout

    def _check_idle(self):
        if self.model in _ACTIVE_MODELS:
            raise RuntimeError("policy shadow requires no active swap")
        self._check_gradients()

    def _check_gradients(self):
        if any(p.grad is not None for p in self.model.parameters()):
            raise RuntimeError("policy shadow requires cleared gradients")

    def _reshard(self):
        # Root no-grad forwards retain gathered weights. Merely copying local
        # shards leaves the next forward using that stale policy unless freed.
        for module in FSDP.fsdp_modules(self.model):
            handle = module._handle
            if handle is not None and hasattr(handle.flat_param, "_local_shard"):
                _reshard(module, handle, True)

    def _prepare(self):
        self._check_gradients()
        self._reshard()
        live = dict(self.model.named_parameters())
        if (live.keys() != self._parameters.keys()
                or any(live[name] is not p for name, p in self._parameters.items())
                or self._metadata() != self._layout):
            raise ValueError("policy shadow model topology/layout changed")

    def _snapshot(self):
        return {name: p.detach().to(device="cpu", copy=True)
                for name, p in self._parameters.items()}

    def _validate(self, values):
        if not isinstance(values, dict) or values.keys() != self._parameters.keys():
            raise ValueError("policy shadow parameter names differ")
        for name, p in self._parameters.items():
            value = values[name]
            if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                    or value.shape != p.shape or value.dtype != p.dtype
                    or value.stride() != p.stride()):
                raise ValueError(f"policy shadow shard shape/dtype/layout differs: {name}")
            if not torch.isfinite(value).all():
                raise ValueError(f"nonfinite policy shadow: {name}")

    @torch.no_grad()
    def _copy(self, values):
        for name, p in self._parameters.items():
            p.copy_(values[name])

    @contextmanager
    def use(self):
        self._check_idle()
        self._prepare()
        self._validate(self.shadow)
        current = self._snapshot()
        self.active = True
        _ACTIVE_MODELS.add(self.model)
        try:
            self._copy(self.shadow)
            with torch.no_grad():
                yield
        finally:
            self._prepare()
            self._copy(current)
            _ACTIVE_MODELS.remove(self.model)
            self.active = False

    @torch.no_grad()
    def update(self, decay: float):
        if not math.isfinite(decay) or not 0 <= decay < 1:
            raise ValueError("decay must be finite and in [0, 1)")
        self._check_idle()
        self._prepare()
        for name, p in self._parameters.items():
            current = p.detach().to(device="cpu", copy=True)
            if decay == 0:
                self.shadow[name].copy_(current)
            else:
                self.shadow[name].mul_(decay).add_(current, alpha=1 - decay)

    def state_dict(self):
        self._check_idle()
        self._prepare()
        return {"schema_version": 1, "rank": self._group.rank,
                "layout": copy.deepcopy(self._layout),
                "parameters": {name: p.clone() for name, p in self.shadow.items()}}

    def load_state_dict(self, state):
        self._check_idle()
        self._prepare()
        if (not isinstance(state, dict)
                or state.keys() != {"schema_version", "rank", "layout", "parameters"}
                or state["schema_version"] != 1 or state["rank"] != self._group.rank
                or state["layout"] != self._layout):
            raise ValueError("policy shadow checkpoint schema/rank/topology/layout differs")
        self._validate(state["parameters"])
        self.shadow = {name: p.detach().to(device="cpu", copy=True)
                       for name, p in state["parameters"].items()}
