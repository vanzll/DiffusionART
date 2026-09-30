"""Checksummed, immutable checkpoints for local unsharded training state.

This synchronous local-filesystem primitive is not an FSDP checkpoint writer
or a deadline guarantee for unresponsive network mounts.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
import uuid
from pathlib import Path

import numpy as np
import torch

from .state import Progress


def _atomic_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".publish-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _digest(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rng_state(*, cuda_device=None):
    """Capture legacy all-device state or an explicitly rank-local device.

    Distributed callers must bind their device, avoiding CUDA contexts on peer
    GPUs visible in the same process environment.
    """
    numpy = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": [numpy[0], numpy[1].tolist(), *numpy[2:]],
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if cuda_device is None and torch.cuda.is_initialized() else [],
    }
    if cuda_device is not None:
        device = torch.device(cuda_device)
        state["local_cuda_rng"] = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    return state


def restore_rng(state, *, cuda_device=None):
    """Restore only the declared ownership mode; never widen rank-local state."""
    if cuda_device is not None and ("local_cuda_rng" not in state or state["torch_cuda"]):
        raise ValueError("rank-local RNG restore rejects global/mixed CUDA state")
    if "local_cuda_rng" in state:
        if cuda_device is None:
            raise ValueError("rank-local RNG restore requires an explicit device")
        device = torch.device(cuda_device)
        if (state["local_cuda_rng"] is not None) != (device.type == "cuda"):
            raise ValueError("rank-local RNG device type differs")
        if state["local_cuda_rng"] is not None:
            torch.cuda.set_rng_state(state["local_cuda_rng"], device)
    if state["torch_cuda"]:
        if not torch.cuda.is_available() or len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG topology differs from checkpoint")
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), *n[2:]))
    torch.set_rng_state(state["torch_cpu"])


def save_checkpoint(
    root,
    *,
    model,
    optimizer,
    progress: Progress,
    config_sha256: str,
    stateful: dict | None = None,
    generators: dict | None = None,
) -> Path:
    """Capture a completed optimizer boundary, including caller-owned data state.

    ``stateful`` must include scheduler, old policy, sampler/dataset cursor etc.
    when present. Resuming requires exactly the same set, never silent omission.
    Multi-rank/FSDP checkpointing is deliberately rejected by this writer.
    """
    if torch.distributed.is_initialized():
        raise ValueError("use a coordinated distributed checkpoint adapter")
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError("checkpoint only after gradient buffers are cleared")
    Progress.from_state_dict(progress.state_dict())
    if len(config_sha256) != 64 or any(c not in "0123456789abcdef" for c in config_sha256):
        raise ValueError("config_sha256 must be a SHA256 digest")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    stem = f"step-{progress.optimizer_steps:09d}-{uuid.uuid4().hex}"
    target, receipt = root / (stem + ".pt"), root / (stem + ".json")
    state = {
        "schema_version": 1,
        "config_sha256": config_sha256,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "progress": progress.state_dict(),
        "rng": rng_state(),
        "stateful": {k: v.state_dict() for k, v in (stateful or {}).items()},
        "generators": {k: v.get_state() for k, v in (generators or {}).items()},
    }
    fd, temporary = tempfile.mkstemp(prefix=".checkpoint-", dir=root)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        checksum = _digest(Path(temporary))
        os.replace(temporary, target)
        record = {
            "schema_version": 1,
            "file": target.name,
            "sha256": checksum,
            "bytes": target.stat().st_size,
            "optimizer_step": progress.optimizer_steps,
            "config_sha256": config_sha256,
        }
        _atomic_json(receipt, record)
        _atomic_json(root / "latest.json", {"receipt": receipt.name, **record})
    finally:
        Path(temporary).unlink(missing_ok=True)
    return receipt


def load_checkpoint(
    receipt,
    *,
    model,
    optimizer,
    config_sha256: str,
    stateful: dict | None = None,
    generators: dict | None = None,
) -> Progress:
    if torch.distributed.is_initialized():
        raise ValueError("use a coordinated distributed checkpoint adapter")
    receipt = Path(receipt)
    metadata = json.loads(receipt.read_text(encoding="utf-8"))
    name = metadata.get("file")
    if not isinstance(name, str) or Path(name).name != name or not name.endswith(".pt"):
        raise ValueError("invalid checkpoint filename")
    if metadata.get("schema_version") != 1 or metadata.get("config_sha256") != config_sha256:
        raise ValueError("checkpoint schema/config differs")
    path = receipt.parent / name
    if path.stat().st_size != metadata["bytes"] or _digest(path) != metadata["sha256"]:
        raise ValueError("checkpoint integrity check failed")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("schema_version") != 1 or state.get("config_sha256") != config_sha256:
        raise ValueError("checkpoint payload schema/config differs")
    progress = Progress.from_state_dict(state["progress"])
    if progress.optimizer_steps != metadata["optimizer_step"]:
        raise ValueError("checkpoint step identity differs")
    if set(state["stateful"]) != set(stateful or {}):
        raise ValueError("checkpoint stateful component names differ")
    if set(state["generators"]) != set(generators or {}):
        raise ValueError("checkpoint generator names differ")
    # A failed load is terminal. Callers must not continue with partially loaded
    # state if a downstream component rejects its own schema.
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    for key, component in (stateful or {}).items():
        component.load_state_dict(state["stateful"][key])
    for key, generator in (generators or {}).items():
        generator.set_state(state["generators"][key])
    restore_rng(state["rng"])
    return progress
