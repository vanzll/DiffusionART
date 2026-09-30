"""Explicit torchrun ownership for replicated or sharded data-parallel execution.

This is an execution primitive, not a declaration that a native backend has
distributed evaluation/checkpoint support. Backends must opt in separately.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from typing import Mapping

import torch
import torch.distributed as dist

from ..config import ConfigError


@dataclass(frozen=True)
class RankInfo:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    local_world_size: int = 1

    def __post_init__(self):
        values = (self.rank, self.world_size, self.local_rank, self.local_world_size)
        if not all(type(value) is int for value in values) or not (
            self.world_size > 0 and self.local_world_size > 0
            and self.world_size % self.local_world_size == 0
            and 0 <= self.rank < self.world_size
            and 0 <= self.local_rank < self.local_world_size
            and self.rank % self.local_world_size == self.local_rank
        ):
            raise ConfigError("invalid homogeneous torchrun rank topology")

    def local_prompt_count(self, global_prompt_count: int) -> int:
        if (
            type(global_prompt_count) is not int
            or global_prompt_count < self.world_size
            or global_prompt_count % self.world_size
        ):
            raise ConfigError("global prompt groups must divide evenly over data-parallel ranks")
        return global_prompt_count // self.world_size


def launch_identity(strategy: str, env: Mapping[str, str] | None = None) -> RankInfo:
    """Validate launcher identity without initializing CUDA or rendezvous."""
    if strategy not in {"none", "ddp", "fsdp"}:
        raise ConfigError("this execution adapter supports none, ddp or fsdp")
    env = os.environ if env is None else env
    keys = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE")
    if not any(key in env for key in keys):
        if strategy != "none":
            raise ConfigError(f"{strategy} requires torchrun RANK/WORLD_SIZE/LOCAL_RANK/LOCAL_WORLD_SIZE")
        return RankInfo()
    if not all(key in env for key in keys):
        raise ConfigError("incomplete torchrun rank environment")
    try:
        rank, world, local, local_world = (int(env[key]) for key in keys)
    except (TypeError, ValueError) as exc:
        raise ConfigError("torchrun rank fields must be integers") from exc
    identity = RankInfo(rank, world, local, local_world)
    if strategy == "none" and world != 1:
        raise ConfigError("multiple torchrun workers require a distributed backend, not distributed=none")
    return identity


@dataclass(frozen=True)
class DistributedSession:
    identity: RankInfo
    device: torch.device
    distributed: bool

    def local(self, label, function):
        """Coordinate a local-only phase, never a callback containing collectives."""
        if not self.distributed:
            return function()
        error, result = None, None
        try:
            result = function()
        except Exception as exc:
            error = exc
        flag = torch.tensor(int(error is None), device=self.device, dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        if not flag.item():
            raise RuntimeError(f"distributed {label} failed on at least one rank") from error
        return result

    def primary(self, label, function):
        return self.local(label, lambda: function() if self.identity.rank == 0 else None)

    def agree(self, value):
        encoded = self.local("identity encoding", lambda: json.dumps(value, sort_keys=True, allow_nan=False).encode())
        self.require_same_digest(hashlib.sha256(encoded).hexdigest())

    def require_same_digest(self, digest: str) -> None:
        """All ranks agree on a SHA256 identity before entering model work.

        Use for the resolved science config/data manifest, not rank-local paths.
        Fixed-size tensor exchange avoids pickle/object collectives on NCCL.
        Every rank must call this in the same order.
        """
        valid = (
            isinstance(digest, str) and len(digest) == 64
            and all(char in "0123456789abcdef" for char in digest)
        )
        payload = bytes.fromhex(digest) if valid else bytes(32)
        value = torch.tensor([int(valid), *payload], dtype=torch.int32, device=self.device)
        if self.distributed:
            lower, upper = value.clone(), value.clone()
            dist.all_reduce(lower, op=dist.ReduceOp.MIN)
            dist.all_reduce(upper, op=dist.ReduceOp.MAX)
            valid = bool(lower[0].item()) and torch.equal(lower, upper)
        if not valid:
            raise ConfigError("rank configuration/data identity differs or is not a SHA256 digest")


@contextmanager
def distributed_session(
    strategy: str, *, device: str, timeout_seconds: float, init_method: str = "env://"
):
    """Own exactly one default process group; never adopt another caller's group.

    The supervisor must terminate peer workers on failure. Cleanup deliberately
    has no barrier: a failed peer cannot participate. A finite collective timeout
    is not a wall-clock guarantee for a hung driver or filesystem call.
    """
    identity = launch_identity(strategy)
    if (
        type(timeout_seconds) not in (int, float)
        or not math.isfinite(timeout_seconds) or timeout_seconds <= 0
    ):
        raise ConfigError("process-group timeout must be positive and finite")
    if device not in {"cpu", "cuda"}:
        raise ConfigError("device must be cpu or cuda")
    if dist.is_initialized():
        raise ConfigError("refusing to adopt or replace an existing process group")
    if device == "cuda":
        if not torch.cuda.is_available() or identity.local_world_size > torch.cuda.device_count():
            raise ConfigError("LOCAL_RANK has no visible CUDA device")
        if strategy != "none" and not dist.is_nccl_available():
            raise ConfigError("CUDA distributed training requires NCCL support")
        torch.cuda.set_device(identity.local_rank)
        target = torch.device("cuda", identity.local_rank)
    else:
        target = torch.device("cpu")
    initialized = False
    try:
        if strategy != "none":
            dist.init_process_group(
                backend="nccl" if device == "cuda" else "gloo",
                init_method=init_method,
                rank=identity.rank,
                world_size=identity.world_size,
                timeout=timedelta(seconds=timeout_seconds),
            )
            initialized = True
        yield DistributedSession(identity, target, initialized)
    finally:
        if initialized:
            dist.destroy_process_group()
