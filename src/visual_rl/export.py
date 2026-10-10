"""Offline CPU export of trusted, complete SD3/FLUX policy checkpoints."""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path

import torch

from .config import Experiment, load_config
from .models.finetune import FineTuneConfig, prepare_policy
from .training.checkpoint import _digest
from .training.state import Progress


def _safe_path(value) -> Path:
    path = Path(value).absolute()
    if ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("paths must not contain parent traversal or symlinks")
    return path


def _json(path):
    if not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise ValueError("missing or invalid checkpoint publication JSON")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("checkpoint publication must be a mapping")
    return value


def _payload(root, record, filename=None):
    name = record.get("file") if isinstance(record, dict) else None
    if (not isinstance(name, str) or Path(name).name != name or not name.endswith(".pt")
            or (filename is not None and name != filename)):
        raise ValueError("invalid checkpoint file identity")
    path = _safe_path(root / name)
    if (not path.is_file() or path.stat().st_size != record.get("bytes")
            or _digest(path) != record.get("sha256")):
        raise ValueError("checkpoint integrity check failed")
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, RuntimeError, EOFError) as exc:
        raise ValueError("unsupported or invalid checkpoint serialization; no unsafe pickle fallback") from exc
    if not isinstance(value, dict):
        raise ValueError("checkpoint payload must be a mapping")
    return value


def _require(state, keys):
    if any(key not in state for key in keys):
        raise ValueError("incomplete checkpoint payload")
    if any(not isinstance(state[key], dict) for key in keys):
        raise ValueError("invalid checkpoint component state")


def _policy_state(receipt, config):
    metadata = _json(receipt)
    if metadata.get("kind") not in (None, "ddp"):
        raise ValueError("unsupported checkpoint backend (including fsdp-local); no resharding")
    if (metadata.get("schema_version") != 1
            or metadata.get("config_sha256") != config.fingerprint()):
        raise ValueError("checkpoint schema/config differs; use the original resolved config")
    if metadata.get("kind") is None:
        if config.runtime.distributed != "none" or receipt.name == "latest.json":
            raise ValueError("use a published local receipt, not latest.json or a distributed config")
        state = _payload(receipt.parent, metadata)
        _require(state, ("model", "optimizer", "progress", "rng", "stateful", "generators"))
        if (state.get("schema_version") != 1
                or state.get("config_sha256") != metadata["config_sha256"]):
            raise ValueError("checkpoint payload identity differs")
        progress = Progress.from_state_dict(state["progress"])
        if progress.optimizer_steps != metadata.get("optimizer_step"):
            raise ValueError("checkpoint step identity differs")
        return state["model"], metadata, progress.optimizer_steps

    if receipt.name != "manifest.json" or config.runtime.distributed != "ddp":
        raise ValueError("DDP export requires its published manifest and original DDP config")
    world, records = metadata.get("world_size"), metadata.get("ranks")
    if (type(world) is not int or world < 1 or not isinstance(records, list)
            or len(records) != world):
        raise ValueError("incomplete DDP manifest")
    shared = _payload(receipt.parent, metadata.get("shared"), "shared.pt")
    _require(shared, ("model", "optimizer", "replicated_stateful"))
    identity = ("schema_version", "config_sha256", "world_size", "counts")
    if any(shared.get(key) != metadata.get(key) for key in identity):
        raise ValueError("DDP shared identity differs")
    # Verify every published rank, not just the inference-relevant shared file.
    buffer_names = None
    for rank, record in enumerate(records):
        local = _payload(receipt.parent, record, f"rank-{rank:05d}.pt")
        _require(local, ("model_local", "progress", "rng", "stateful", "generators"))
        if (local.get("rank") != rank
                or any(local.get(key) != metadata.get(key) for key in identity)):
            raise ValueError("DDP rank identity differs")
        progress = Progress.from_state_dict(local["progress"])
        counts = {key: progress.state_dict()[key]
                  for key in ("attempts", "completed_rollouts", "optimizer_steps")}
        if counts != metadata.get("counts"):
            raise ValueError("DDP progress identity differs")
        names = set(local["model_local"])
        if buffer_names is not None and names != buffer_names:
            raise ValueError("DDP rank-local buffer names differ")
        buffer_names = names
        for key, value in local["model_local"].items():
            target = shared["model"].get(key)
            if (not isinstance(value, torch.Tensor) or not isinstance(target, torch.Tensor)
                    or value.dtype != target.dtype or not torch.equal(value, target)):
                raise ValueError("DDP rank-local buffers differ; no canonical inference policy")
    return shared["model"], metadata, progress.optimizer_steps


def export_policy(config: Experiment, checkpoint, output) -> Path:
    """Export the current policy only; never resume, merge LoRA, or load CUDA state.

    Checkpoint hashes detect corruption, not hostile replacement. Inputs must be
    trusted and immutable during export. The output parent must already exist.
    """
    if torch.distributed.is_initialized():
        raise ValueError("export runs in a standalone CPU process, not a process group")
    if config.model.name not in ("sd3", "flux"):
        raise ValueError("unsupported model backend; export supports only sd3 and flux")
    if not config.model.checkpoint or not config.model.revision:
        raise ValueError("base model checkpoint and revision must be explicit")
    options = config.model.options.get("finetune")
    if not isinstance(options, dict):
        raise ValueError("configured finetune settings must be explicit")
    try:
        fine = FineTuneConfig(**options)
    except TypeError as exc:
        raise ValueError("unsupported finetune settings") from exc
    receipt, destination = _safe_path(checkpoint), _safe_path(output)
    if destination.exists():
        raise FileExistsError(f"export destination already exists: {destination}")
    if not destination.parent.is_dir():
        raise ValueError("export destination parent must already exist")
    base = Path(config.model.checkpoint).absolute()
    if destination.is_relative_to(receipt.parent) or destination.is_relative_to(base):
        raise ValueError("export destination must be outside checkpoint and base-model directories")
    state, metadata, step = _policy_state(receipt, config)
    from diffusers import FluxTransformer2DModel, SD3Transformer2DModel

    cls = SD3Transformer2DModel if config.model.name == "sd3" else FluxTransformer2DModel
    model = cls.from_pretrained(
        config.model.checkpoint, subfolder="transformer", revision=config.model.revision,
        local_files_only=True, torch_dtype=torch.float32,
    ).cpu()
    model = prepare_policy(model, fine)
    live = model.state_dict()
    if state.keys() != live.keys():
        raise ValueError("checkpoint policy keys differ from configured transformer/LoRA")
    frozen = {name for name, p in model.named_parameters() if not p.requires_grad}
    for key, value in state.items():
        if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                or value.shape != live[key].shape or value.is_meta
                or value.dtype != live[key].dtype):
            raise ValueError(f"checkpoint policy tensor metadata differs: {key}")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"nonfinite policy tensor: {key}")
        if fine.mode == "lora" and key in frozen and not torch.equal(value, live[key]):
            raise ValueError(f"LoRA frozen base differs; adapter-only export would lose state: {key}")
    model.load_state_dict(state, strict=True)
    model.eval()
    if fine.mode == "lora":
        model.peft_config["default"].base_model_name_or_path = config.model.checkpoint
        model.peft_config["default"].revision = config.model.revision
    # Exclusive creation prevents clobbering even if another exporter races us.
    destination.mkdir(exist_ok=False)
    try:
        kwargs = {"safe_serialization": True}
        if fine.mode == "lora":
            kwargs["save_embedding_layers"] = False
        model.save_pretrained(destination, **kwargs)
        record = {
            "schema_version": 1, "kind": "policy-export", "format": fine.mode,
            "model": config.model.name, "base_model": config.model.checkpoint,
            "base_revision": config.model.revision, "base_subfolder": "transformer",
            "checkpoint": str(receipt), "checkpoint_receipt_sha256": _digest(receipt),
            "config_sha256": metadata["config_sha256"], "optimizer_step": step,
            "output": str(destination),
        }
        (destination / "export.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except BaseException:
        shutil.rmtree(destination)
        raise
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="original experiment YAML")
    parser.add_argument("--checkpoint", required=True, help="published receipt/manifest JSON")
    parser.add_argument("--output", required=True, help="new directory in an existing parent")
    args = parser.parse_args(argv)
    try:
        destination = export_policy(load_config(args.config), args.checkpoint, args.output)
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"export failed: {exc}\n")
    print(destination)


if __name__ == "__main__":
    main()
