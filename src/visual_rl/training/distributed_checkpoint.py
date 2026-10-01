"""Coordinated DDP checkpoint publication on a shared, responsive filesystem.

Shared parameters/optimizer and explicitly replicated components are written
once. Rank-local RNG, buffers and caller-owned components are stored separately.
This synchronous primitive is not a storage deadline or an FSDP adapter.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from collections import OrderedDict
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .checkpoint import _atomic_json, _digest, restore_rng, rng_state
from .checkpoint_retention import enqueue_retention, validate_keep_last
from .state import Progress


class DistributedCheckpointError(RuntimeError):
    """All participating ranks must leave the failed checkpoint operation."""


class _Group:
    def __init__(self, model):
        if not dist.is_initialized() or not isinstance(model, DistributedDataParallel):
            raise ValueError("distributed checkpoint requires initialized DDP")
        self.group = model.process_group
        self.rank = dist.get_rank(self.group)
        self.world = dist.get_world_size(self.group)
        self.device = next(model.parameters()).device
        self.module = model.module

    def local(self, label, function):
        error, result = None, None
        try:
            result = function()
        except Exception as exc:
            error = exc
        flag = torch.tensor(int(error is None), device=self.device, dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=self.group)
        if not flag.item():
            raise DistributedCheckpointError(f"checkpoint {label} failed on at least one rank") from error
        return result

    def agree(self, value):
        encoded = self.local("identity encoding", lambda: json.dumps(value, sort_keys=True, allow_nan=False).encode())
        digest = torch.tensor(list(hashlib.sha256(encoded).digest()), device=self.device, dtype=torch.int32)
        lower, upper = digest.clone(), digest.clone()
        dist.all_reduce(lower, op=dist.ReduceOp.MIN, group=self.group)
        dist.all_reduce(upper, op=dist.ReduceOp.MAX, group=self.group)
        if not torch.equal(lower, upper):
            raise DistributedCheckpointError("checkpoint identities differ across ranks")

    def token(self):
        value = torch.tensor(list(uuid.uuid4().bytes) if self.rank == 0 else [0] * 16,
                             device=self.device, dtype=torch.uint8)
        dist.broadcast(value, src=dist.get_global_rank(self.group, 0), group=self.group)
        return bytes(value.cpu().tolist()).hex()


def _validate(model, progress, config_sha256, stateful, replicated_stateful, generators):
    from torch.distributed.fsdp import FullyShardedDataParallel

    if any(isinstance(module, FullyShardedDataParallel) for module in model.modules()):
        raise ValueError("FSDP requires a shard-aware checkpoint adapter")
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError("checkpoint requires cleared gradient buffers")
    if not isinstance(config_sha256, str) or len(config_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in config_sha256
    ):
        raise ValueError("config_sha256 must be a SHA256 digest")
    if progress is not None:
        Progress.from_state_dict(progress.state_dict())
    for components in (stateful, replicated_stateful, generators):
        if not isinstance(components, dict) or any(not isinstance(key, str) or not key for key in components):
            raise ValueError("checkpoint component names must be nonempty strings")
    if set(stateful) & set(replicated_stateful):
        raise ValueError("local and replicated stateful component names overlap")


def _write(path, state):
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        checksum = _digest(Path(temporary))
        size = Path(temporary).stat().st_size
        os.replace(temporary, path)
        return {"file": path.name, "sha256": checksum, "bytes": size}
    finally:
        Path(temporary).unlink(missing_ok=True)


def _read(root, record, filename):
    if not isinstance(record, dict) or record.get("file") != filename:
        raise ValueError("checkpoint file identity differs")
    path = root / filename
    if path.is_symlink() or not path.is_file():
        raise ValueError("checkpoint requires regular, non-symlink files")
    if path.stat().st_size != record.get("bytes") or _digest(path) != record.get("sha256"):
        raise ValueError("checkpoint integrity check failed")
    return torch.load(path, map_location="cpu", weights_only=True)


def save_ddp_checkpoint(
    root, *, model, optimizer, progress: Progress, config_sha256: str,
    stateful: dict | None = None, replicated_stateful: dict | None = None,
    generators: dict | None = None,
    keep_last: int | None = None,
) -> Path:
    """All DDP-group ranks call at the same completed optimizer boundary.

    Parameters and optimizer state must already be synchronized by the normal
    DDP update. Callers explicitly classify components: data cursors are local;
    a synchronized old-policy/eval shadow can be replicated. No sharded optimizer
    is supported. Rank-local buffers are preserved even when they differ.
    ``keep_last`` retains registered publications only, in publication order;
    None disables cleanup. All ranks must supply the same retention setting.
    Cleanup is asynchronous/advisory; a busy worker may skip publications and
    cannot guarantee N snapshots at save return.
    """
    group = _Group(model)
    group.local("retention preflight", lambda: validate_keep_last(keep_last))
    stateful, replicated_stateful, generators = stateful or {}, replicated_stateful or {}, generators or {}
    group.local("preflight", lambda: _validate(model, progress, config_sha256, stateful, replicated_stateful, generators))
    root = group.local("root path", lambda: Path(root).absolute())
    counts = {key: progress.state_dict()[key] for key in ("attempts", "completed_rollouts", "optimizer_steps")}
    group.agree({"root": str(root), "config": config_sha256, "counts": counts,
                 "world": group.world, "stateful": sorted(stateful),
                 "replicated": sorted(replicated_stateful), "generators": sorted(generators),
                 "keep_last": keep_last})
    directory = root / f"step-{progress.optimizer_steps:09d}-{group.token()}"
    group.local("directory creation", lambda: directory.mkdir(parents=True, exist_ok=False) if group.rank == 0 else None)

    def publish_local():
        common = {"schema_version": 1, "config_sha256": config_sha256, "counts": counts, "world_size": group.world}
        model_state = group.module.state_dict()
        parameters = dict(group.module.named_parameters(remove_duplicate=False))
        local = {
            **common, "rank": group.rank, "progress": progress.state_dict(),
            "model_local": {key: value for key, value in model_state.items() if key not in parameters},
            "model_metadata": getattr(model_state, "_metadata", None),
            "rng": rng_state(cuda_device=group.device),
            "stateful": {key: value.state_dict() for key, value in stateful.items()},
            "generators": {key: value.get_state() for key, value in generators.items()},
        }
        if group.rank == 0:
            shared = _write(directory / "shared.pt", {
                **common, "model": model_state, "optimizer": optimizer.state_dict(),
                "replicated_stateful": {key: value.state_dict() for key, value in replicated_stateful.items()},
            })
            _atomic_json(directory / "shared.record.json", shared)
        record = _write(directory / f"rank-{group.rank:05d}.pt", local)
        _atomic_json(directory / f"rank-{group.rank:05d}.record.json", record)

    group.local("state serialization", publish_local)
    receipt = directory / "manifest.json"

    def publish_manifest():
        if group.rank != 0:
            return
        records = [json.loads((directory / f"rank-{rank:05d}.record.json").read_text()) for rank in range(group.world)]
        shared = json.loads((directory / "shared.record.json").read_text())
        # Records are read only after every writer acknowledged fsync/rename.
        record = {"schema_version": 1, "kind": "ddp", "config_sha256": config_sha256,
                  "world_size": group.world, "counts": counts, "shared": shared, "ranks": records}
        _atomic_json(receipt, record)
        _atomic_json(root / "latest.json", {"manifest": f"{directory.name}/manifest.json"})

    group.local("manifest publication", publish_manifest)
    if group.rank == 0:
        enqueue_retention(
            root, receipt, identity={"kind": "ddp", "config_sha256": config_sha256,
                                    "topology": {"world_size": group.world,
                                                 "global_world_size": dist.get_world_size(),
                                                 "members": dist.get_process_group_ranks(group.group),
                                                 "device_type": group.device.type}}, keep_last=keep_last)
    return receipt


def load_ddp_checkpoint(
    receipt, *, model, optimizer, config_sha256: str, stateful: dict | None = None,
    replicated_stateful: dict | None = None, generators: dict | None = None,
) -> Progress:
    """Stage and verify all payloads before any rank changes live training state.

    World-size changes are rejected. A load/application error is terminal; never
    continue training with partially restored state.
    """
    group = _Group(model)
    stateful, replicated_stateful, generators = stateful or {}, replicated_stateful or {}, generators or {}
    group.local("preflight", lambda: _validate(model, None, config_sha256, stateful, replicated_stateful, generators))
    receipt = group.local("receipt path", lambda: Path(receipt).absolute())
    group.agree({"receipt": str(receipt), "config": config_sha256, "world": group.world,
                 "stateful": sorted(stateful), "replicated": sorted(replicated_stateful), "generators": sorted(generators)})

    def stage():
        manifest = json.loads(receipt.read_text())
        if (manifest.get("schema_version") != 1 or manifest.get("kind") != "ddp"
                or manifest.get("config_sha256") != config_sha256 or manifest.get("world_size") != group.world
                or len(manifest.get("ranks", [])) != group.world):
            raise ValueError("checkpoint manifest config/topology differs")
        shared = _read(receipt.parent, manifest["shared"], "shared.pt")
        local = _read(receipt.parent, manifest["ranks"][group.rank], f"rank-{group.rank:05d}.pt")
        for state in (shared, local):
            if (state.get("schema_version") != 1 or state.get("config_sha256") != config_sha256
                    or state.get("world_size") != group.world or state.get("counts") != manifest["counts"]):
                raise ValueError("checkpoint payload identity differs")
        progress = Progress.from_state_dict(local["progress"])
        if local.get("rank") != group.rank or any(progress.state_dict()[key] != manifest["counts"][key]
                                                  for key in ("attempts", "completed_rollouts", "optimizer_steps")):
            raise ValueError("checkpoint rank/step differs")
        if (set(local["stateful"]) != set(stateful) or set(shared["replicated_stateful"]) != set(replicated_stateful)
                or set(local["generators"]) != set(generators)):
            raise ValueError("checkpoint component names differ")
        parameters = dict(group.module.named_parameters(remove_duplicate=False))
        live = group.module.state_dict()
        expected_local = set(live) - set(parameters)
        if set(local["model_local"]) != expected_local:
            raise ValueError("checkpoint rank-local model state differs")
        merged = {**shared["model"], **local["model_local"]}
        if set(merged) != set(live):
            raise ValueError("checkpoint model keys differ")
        for key, tensor in live.items():
            if isinstance(tensor, torch.Tensor) and (
                not isinstance(merged[key], torch.Tensor) or tensor.shape != merged[key].shape
                or tensor.dtype != merged[key].dtype
            ):
                raise ValueError(f"checkpoint model shape/dtype differs: {key}")
        return manifest, shared, local, progress

    manifest, shared, local, progress = group.local("integrity validation", stage)
    group.agree(manifest)

    def apply():
        merged = OrderedDict(shared["model"])
        merged.update(local["model_local"])
        # Versioned modules consult state_dict metadata during loading. Plain
        # dict expansion would silently turn current state into legacy state.
        if local["model_metadata"] is not None:
            merged._metadata = local["model_metadata"]
        group.module.load_state_dict(merged, strict=True)
        optimizer.load_state_dict(shared["optimizer"])
        for key, value in replicated_stateful.items():
            value.load_state_dict(shared["replicated_stateful"][key])
        for key, value in stateful.items():
            value.load_state_dict(local["stateful"][key])
        for key, value in generators.items():
            value.set_state(local["generators"][key])
        restore_rng(local["rng"], cuda_device=group.device)

    group.local("state restoration (terminal on error)", apply)
    return progress
