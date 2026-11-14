"""Verify and load a named local policy export into a fresh native adapter."""

import json
from pathlib import Path

import torch

from .data import file_sha256


def verify_final(directory):
    directory = Path(directory).resolve(strict=True)
    marker = json.loads((directory / "COMPLETE.json").read_text())
    if marker.get("actual_optimizer_steps") != 10 or not marker.get("files"):
        raise ValueError("named final requires a complete ten-step marker")
    names = set()
    for item in marker["files"]:
        path = directory / item["path"]
        if (
            Path(item["path"]).is_absolute()
            or ".." in Path(item["path"]).parts
            or item["path"] in names
            or not path.resolve(strict=True).is_relative_to(directory)
            or file_sha256(path) != item["sha256"]
        ):
            raise ValueError("export inventory/path/content identity differs")
        names.add(item["path"])
    identity = json.loads((directory / "identity.json").read_text())
    from .training.small_video_checkpoint import SmallVideoIdentity

    frozen = SmallVideoIdentity(**identity["identity"])
    if marker.get("identity_sha256") != frozen.sha256:
        raise ValueError("export identity marker differs")
    return identity


def load_final_into_adapter(c, directory, adapter):
    """Only policy weights; text/VAE/assets stay verified and no Adam/RNG is restored.

    The caller constructs a fresh native adapter with matching prepare_policy
    settings and owns any GPU lease. Checkpoint resume is a separate interface.
    """
    from .backends.small_video import resolved_protocol

    identity = verify_final(directory)
    frozen, protocol = identity["identity"], identity["protocol"]
    mode = c.model.options["finetune"]["mode"]
    if (
        frozen["model_name"] != c.model.name
        or frozen["model_revision"] != c.model.revision
        or frozen["finetune_mode"] != mode
        or protocol != resolved_protocol(c)
    ):
        raise ValueError("export model/base/mode/scientific protocol differs")
    model = adapter.trainable_module
    directory = Path(directory)
    if mode == "full":
        if identity["format"] != "native_transformer_state_dict":
            raise ValueError("full export format differs")
        weights = torch.load(directory / "policy.pt", map_location="cpu", weights_only=True)
        if not isinstance(weights, dict) or any(
            not isinstance(v, torch.Tensor) or not torch.isfinite(v).all() for v in weights.values()
        ):
            raise ValueError("full export requires finite native policy tensors")
        model.load_state_dict(weights, strict=True)
    else:
        if identity["format"] != "peft_adapter" or not getattr(model, "peft_config", None):
            raise ValueError("LoRA export requires the matching prepared PEFT adapter")
        from peft import set_peft_model_state_dict
        from peft.utils.save_and_load import load_peft_weights

        config = json.loads((directory / "adapter/adapter_config.json").read_text())
        expected = c.model.options["finetune"]
        if (
            config["r"] != expected["rank"]
            or config["lora_alpha"] != expected["alpha"]
            or set(config["target_modules"]) != set(expected["target_modules"])
            or config["lora_dropout"] != 0
        ):
            raise ValueError("LoRA adapter targets/rank/scaling differ")
        weights = load_peft_weights(str(directory / "adapter"), device="cpu")
        if not weights or any(not torch.isfinite(v).all() for v in weights.values()):
            raise ValueError("LoRA export contains missing/nonfinite weights")
        result = set_peft_model_state_dict(model, weights, adapter_name="default")
        if result.unexpected_keys or any("lora_" in key for key in result.missing_keys):
            raise ValueError("LoRA export did not populate every expected trainable tensor")
    return adapter
