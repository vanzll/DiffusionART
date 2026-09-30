"""Same-topology FSDP1 checkpoints on a responsive shared filesystem.

PyTorch's LOCAL_STATE_DICT owns flat-parameter serialization, including padding
and shard metadata. The ordinary optimizer state_dict is already rank-local
with use_orig_params=False. Neither path gathers model or optimizer weights.
Caller-owned stateful components must likewise be local/shard-aware; this API
does not turn a full-model EMA into a sharded EMA.

Native FSDP phases are collective and failures are terminal: the caller must
abort its process group, not continue training or attempt another checkpoint.
Only rank-local IO/validation/component callbacks use coordinated error voting.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed._shard.api import load_with_process_group
from torch.distributed._shard.metadata import ShardMetadata
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._shard.sharded_tensor.metadata import (
    MEM_FORMAT_ENCODING,
    ShardedTensorMetadata,
    TensorProperties,
)
from torch.distributed._shard.sharded_tensor.shard import Shard
from torch.distributed._shard.sharding_spec import ChunkShardingSpec, EnumerableShardingSpec
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import LocalStateDictConfig, ShardingStrategy, StateDictType
from torch.distributed.remote_device import _remote_device

from .checkpoint import _atomic_json, _digest, restore_rng, rng_state
from .distributed_checkpoint import DistributedCheckpointError as DistributedCheckpointError
from .distributed_checkpoint import _Group, _write
from .state import Progress

_COUNTS = ("attempts", "completed_rollouts", "optimizer_steps")


def _shard_getattr(owner, name):
    # Protocol 2 encodes PyTorch's nested ProcessGroupState class via getattr.
    if owner is ShardedTensor and name == "ProcessGroupState":
        return ShardedTensor.ProcessGroupState
    raise ValueError("unsupported checkpoint class lookup")


_SHARD_TYPES = [
    ShardedTensor, Shard, ShardMetadata, ShardedTensorMetadata, TensorProperties,
    ShardedTensor.ProcessGroupState, ChunkShardingSpec, EnumerableShardingSpec, _remote_device,
    MEM_FORMAT_ENCODING,
    (_shard_getattr, "builtins.getattr"),
]


class _FSDPGroup(_Group):
    def __init__(self, model):
        if not dist.is_initialized() or not isinstance(model, FSDP):
            raise ValueError("FSDP checkpoint requires an initialized FSDP1 root")
        self.group = model.process_group
        self.rank = dist.get_rank(self.group)
        self.world = dist.get_world_size(self.group)
        self.device = model.compute_device


def _type(value):
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _layout(model, optimizer, group):
    """Only inspect FSDP layout metadata; never reconstruct its flat tensors.

    LOCAL_STATE_DICT cannot detect a same-numel reordering on its own. FSDP1
    exposes no public original-parameter layout query with use_orig_params=False,
    so this deliberately version-locked format also checks its layout metadata.
    """
    units = []
    for name, module in model.named_modules():
        if not isinstance(module, FSDP):
            continue
        if (module.sharding_strategy != ShardingStrategy.FULL_SHARD
                or module._use_orig_params or module.process_group is not group.group
                or module._device_mesh is not None):
            raise ValueError("requires FULL_SHARD, use_orig_params=False, one process group, no device mesh")
        flat = module._handle.flat_param if module._handle is not None else None
        if flat is not None and flat.dtype != torch.float32:
            raise ValueError("FSDP master parameters must be FP32")
        mp = module.mixed_precision
        units.append({
            "name": name,
            "parameters": [] if flat is None else [
                [fqn, list(shape), list(stride)]
                for fqn, shape, stride in zip(flat._fqns, flat._shapes, flat._strides)
            ],
            "aliases": [] if flat is None else [
                [p.module_name, p.param_name, p.prim_module_name, p.prim_param_name]
                for p in flat._shared_param_infos
            ],
            "trainable": None if flat is None else flat.requires_grad,
            "precision": {key: str(getattr(mp, key)) for key in (
                "param_dtype", "reduce_dtype", "buffer_dtype", "keep_low_precision_grads",
                "cast_forward_inputs", "cast_root_forward_inputs",
            )},
            "cpu_offload": module.cpu_offload.offload_params,
        })
    parameters = {id(p): name for name, p in model.named_parameters()}
    groups = [[parameters[id(p)] for p in entry["params"]] for entry in optimizer.param_groups]
    if len({name for entry in groups for name in entry}) != sum(map(len, groups)):
        raise ValueError("optimizer has duplicate parameters")
    return {
        "torch": str(torch.__version__), "world_size": group.world,
        "global_world_size": dist.get_world_size(),
        "members": dist.get_process_group_ranks(group.group),
        "device_type": group.device.type, "units": units,
        "modules": [[name, _type(module)] for name, module in model.named_modules()],
        "optimizer": _type(optimizer), "optimizer_groups": groups,
        "parameters": [[name, list(p.shape), str(p.dtype), p.requires_grad]
                       for name, p in model.named_parameters()],
        "buffers": [[name, list(b.shape)] for name, b in model.named_buffers()],
    }


def _preflight(model, optimizer, progress, config, stateful, generators, group):
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError("checkpoint requires cleared gradient buffers")
    if not isinstance(config, str) or len(config) != 64 or any(c not in "0123456789abcdef" for c in config):
        raise ValueError("config_sha256 must be a SHA256 digest")
    if progress is not None:
        Progress.from_state_dict(progress.state_dict())
    for components in (stateful, generators):
        if not isinstance(components, dict) or any(not isinstance(k, str) or not k for k in components):
            raise ValueError("checkpoint component names must be nonempty strings")
    return _layout(model, optimizer, group)


def _local_state(model):
    # Despite the name, LOCAL_STATE_DICT calls init_from_local_shards(), which
    # all-gathers shard *metadata*. Do not wrap this phase in group.local(): its
    # error all-reduce could mismatch another rank's pending metadata collective.
    with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT,
                              LocalStateDictConfig(offload_to_cpu=True)):
        return model.state_dict()


def _check_model(saved, live):
    if saved.keys() != live.keys() or getattr(saved, "_metadata", None) is None:
        raise ValueError("checkpoint model keys/metadata differ")
    for key, target in live.items():
        source = saved[key]
        if isinstance(target, ShardedTensor):
            if not isinstance(source, ShardedTensor) or source.metadata() != target.metadata():
                raise ValueError(f"checkpoint shard metadata differs: {key}")
            a, b = source.local_shards(), target.local_shards()
            if len(a) != len(b) or any(x.metadata != y.metadata or x.tensor.shape != y.tensor.shape
                                      or x.tensor.dtype != y.tensor.dtype for x, y in zip(a, b)):
                raise ValueError(f"checkpoint local shard differs: {key}")
        elif isinstance(target, torch.Tensor):
            if not isinstance(source, torch.Tensor) or source.shape != target.shape:
                raise ValueError(f"checkpoint tensor shape differs: {key}")


def save_fsdp_checkpoint(
    root, *, model, optimizer, progress: Progress, config_sha256: str,
    stateful: dict | None = None, generators: dict | None = None,
) -> Path:
    """All model-group ranks call after an update and zero_grad(set_to_none=True).

    The caller owns publication at root. Failed writes leave unpublished orphan
    directories; the previous latest pointer remains valid. No storage timeout
    or process-crash recovery is provided by this synchronous primitive.
    Stateful callbacks must be rank-local and must not perform collectives.
    """
    group = _FSDPGroup(model)
    stateful = {} if stateful is None else stateful
    generators = {} if generators is None else generators
    layout = group.local("preflight", lambda: _preflight(
        model, optimizer, progress, config_sha256, stateful, generators, group))
    root = group.local("root path", lambda: Path(root).absolute())
    counts = {key: progress.state_dict()[key] for key in _COUNTS}
    identity = {"schema_version": 1, "kind": "fsdp-local", "config_sha256": config_sha256,
                "layout": layout, "counts": counts, "stateful": sorted(stateful),
                "generators": sorted(generators)}
    group.agree({"root": str(root), **identity})
    directory = root / f"step-{progress.optimizer_steps:09d}-{group.token()}"
    group.local("directory creation", lambda: directory.mkdir(parents=True) if group.rank == 0 else None)
    model_state = _local_state(model)

    def write_rank():
        payload = {
            **identity, "rank": group.rank, "model": model_state,
            "optimizer": optimizer.state_dict(), "progress": progress.state_dict(),
            # Includes nonpersistent buffers; mixed precision may have changed
            # their dtype since construction. Preserve the actual runtime state.
            "buffers": {name: b.detach().cpu() for name, b in model.named_buffers()},
            "rng": rng_state(cuda_device=group.device),
            "stateful": {key: value.state_dict() for key, value in stateful.items()},
            "generators": {key: value.get_state() for key, value in generators.items()},
        }
        record = _write(directory / f"rank-{group.rank:05d}.pt", payload)
        _atomic_json(directory / f"rank-{group.rank:05d}.record.json", record)

    group.local("state serialization", write_rank)
    receipt = directory / "manifest.json"

    def publish():
        if group.rank != 0:
            return
        records = [json.loads((directory / f"rank-{rank:05d}.record.json").read_text())
                   for rank in range(group.world)]
        _atomic_json(receipt, {**identity, "ranks": records})
        _atomic_json(root / "latest.json", {"manifest": f"{directory.name}/manifest.json"})

    group.local("manifest publication", publish)
    return receipt


def load_fsdp_checkpoint(
    receipt, *, model, optimizer, config_sha256: str,
    stateful: dict | None = None, generators: dict | None = None,
) -> Progress:
    """Validate every rank before mutation; require identical FSDP/optimizer layout.

    Pass the manifest, not latest.json. Only load trusted checkpoint directories;
    hashes detect corruption, not malicious replacement. Application errors are
    terminal, not transactional rollback of arbitrary stateful.load_state_dict.
    Stateful callbacks must be rank-local and must not perform collectives.
    """
    group = _FSDPGroup(model)
    stateful = {} if stateful is None else stateful
    generators = {} if generators is None else generators
    layout = group.local("preflight", lambda: _preflight(
        model, optimizer, None, config_sha256, stateful, generators, group))
    receipt = group.local("receipt path", lambda: Path(receipt).absolute())
    group.agree({"receipt": str(receipt), "layout": layout, "config": config_sha256,
                 "stateful": sorted(stateful), "generators": sorted(generators)})

    def read():
        manifest = json.loads(receipt.read_text())
        if (manifest.get("schema_version") != 1 or manifest.get("kind") != "fsdp-local"
                or manifest.get("config_sha256") != config_sha256 or manifest.get("layout") != layout
                or manifest.get("stateful") != sorted(stateful)
                or manifest.get("generators") != sorted(generators)
                or len(manifest.get("ranks", [])) != group.world):
            raise ValueError("checkpoint manifest config/topology/components differ")
        record = manifest["ranks"][group.rank]
        filename = f"rank-{group.rank:05d}.pt"
        if not isinstance(record, dict) or record.get("file") != filename:
            raise ValueError("checkpoint file identity differs")
        path = receipt.parent / filename
        if (path.is_symlink() or not path.is_file() or path.stat().st_size != record.get("bytes")
                or _digest(path) != record.get("sha256")):
            raise ValueError("checkpoint integrity check failed")
        with load_with_process_group(group.group), torch.serialization.safe_globals(_SHARD_TYPES):
            # PyTorch __setstate__ only validates ranks and calls _post_init.
            # LOCAL_STATE_DICT has init_rrefs=False, so it performs no collective.
            payload = torch.load(path, map_location="cpu", weights_only=True)
        for key in ("schema_version", "kind", "config_sha256", "layout", "counts"):
            if payload.get(key) != manifest[key]:
                raise ValueError("checkpoint payload identity differs")
        progress = Progress.from_state_dict(payload["progress"])
        if (payload.get("rank") != group.rank
                or {key: progress.state_dict()[key] for key in _COUNTS} != manifest["counts"]
                or set(payload["stateful"]) != set(stateful)
                or set(payload["generators"]) != set(generators)):
            raise ValueError("checkpoint rank/progress/components differ")
        buffers = dict(model.named_buffers())
        if set(payload["buffers"]) != set(buffers) or any(
            not isinstance(payload["buffers"][name], torch.Tensor)
            or payload["buffers"][name].shape != b.shape for name, b in buffers.items()
        ):
            raise ValueError("checkpoint buffers differ")
        saved_groups = payload["optimizer"]["param_groups"]
        if [g["params"] for g in saved_groups] != [g["params"] for g in optimizer.state_dict()["param_groups"]]:
            raise ValueError("checkpoint optimizer parameter mapping differs")
        return manifest, payload, progress

    manifest, payload, progress = group.local("integrity validation", read)
    group.agree(manifest)
    live = _local_state(model)
    group.local("model validation", lambda live=live: _check_model(payload["model"], live))
    del live

    # Keep native model hooks outside local error voting as well. The stock
    # LOCAL load hook only selects/pads its local shard, but user model hooks
    # may communicate. Any model-application failure is terminal.
    with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT,
                              LocalStateDictConfig(offload_to_cpu=True)):
        model.load_state_dict(payload["model"], strict=True)

    def apply():
        optimizer.load_state_dict(payload["optimizer"])
        with torch.no_grad():
            for name, buffer in model.named_buffers():
                buffer.data = payload["buffers"][name].to(device=buffer.device, copy=True)
        for key, component in stateful.items():
            component.load_state_dict(payload["stateful"][key])
        for key, generator in generators.items():
            generator.set_state(payload["generators"][key])
        restore_rng(payload["rng"], cuda_device=group.device)

    group.local("state restoration (terminal on error)", apply)
    return progress
