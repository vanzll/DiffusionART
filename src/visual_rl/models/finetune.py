"""Explicit full-weight/LoRA setup, separate from forward autocast precision."""

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FineTuneConfig:
    mode: str
    rank: int | None = None
    alpha: float | None = None
    target_modules: tuple[str, ...] = ()
    initialization: str = "gaussian"

    def __post_init__(self):
        if self.mode not in ("full", "lora"):
            raise ValueError("fine-tuning mode must be full or lora")
        if self.initialization != "gaussian":
            raise ValueError("only the recorded Gaussian LoRA initialization is supported")
        if self.mode == "full":
            if self.rank is not None or self.alpha is not None or self.target_modules:
                raise ValueError("full-weight mode must not silently ignore LoRA options")
            return
        if type(self.rank) is not int or self.rank < 1:
            raise ValueError("LoRA rank must be a positive integer")
        if (
            isinstance(self.alpha, bool) or not isinstance(self.alpha, (float, int))
            or not math.isfinite(self.alpha) or self.alpha <= 0
        ):
            raise ValueError("LoRA alpha must be finite and positive")
        if (
            not isinstance(self.target_modules, (tuple, list)) or not self.target_modules
            or any(not isinstance(x, str) or not x for x in self.target_modules)
            or len(set(self.target_modules)) != len(self.target_modules)
        ):
            raise ValueError("LoRA target_modules must be unique nonempty module suffixes")
        object.__setattr__(self, "target_modules", tuple(self.target_modules))


def prepare_policy(model: torch.nn.Module, config: FineTuneConfig):
    """Return an unwrapped trainable policy with FP32 optimizer parameters.

    Call before constructing an optimizer, shadow or DDP wrapper. Does not
    alter the frozen base dtype, enable EMA, change RNG seeds or turn on AMP.
    The caller owns eval/train mode, autocast and activation checkpointing.
    Loading a resumed adapter is a checkpoint concern, not another init here.
    """
    if getattr(model, "peft_config", None) or isinstance(
        model, torch.nn.parallel.DistributedDataParallel
    ):
        raise ValueError("prepare a fresh unwrapped policy, not an existing adapter or DDP model")
    parameters = list(model.parameters())
    if not parameters or any(p.is_meta or not p.is_floating_point() for p in parameters):
        raise ValueError("policy must have materialized floating parameters")
    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        raise ValueError("quantized policy training needs a separate verified recipe")
    if config.mode == "full":
        model.requires_grad_(True)
        model.float()
    else:
        names = dict(model.named_modules())
        for suffix in config.target_modules:
            matches = [module for name, module in names.items()
                       if name == suffix or name.endswith("." + suffix)]
            if not matches or any(not isinstance(m, torch.nn.Linear) for m in matches):
                raise ValueError(f"LoRA target must match linear modules: {suffix}")
        from peft import LoraConfig, get_peft_model

        model.requires_grad_(False)
        model = get_peft_model(model, LoraConfig(
            r=config.rank, lora_alpha=config.alpha, target_modules=list(config.target_modules),
            init_lora_weights=config.initialization, lora_dropout=0.0, bias="none",
            use_rslora=False, use_dora=False,
        ))
    for parameter in model.parameters():
        parameter.grad = None
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    if not any(p.requires_grad for p in model.parameters()):
        raise ValueError("fine-tuning selected no trainable parameters")
    return model


def parameter_inventory(model: torch.nn.Module):
    """JSON-ready precision/count record; shared parameters are counted once."""
    result = {"trainable": {}, "frozen": {}}
    for parameter in model.parameters():
        bucket = result["trainable" if parameter.requires_grad else "frozen"]
        dtype = str(parameter.dtype).removeprefix("torch.")
        bucket[dtype] = bucket.get(dtype, 0) + parameter.numel()
    return result
