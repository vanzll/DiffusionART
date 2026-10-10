"""Optional, frozen SD3 text preparation; no sampling or conditioning cache.

Conventions inspected at research revision
3ca2a36f6b20c53a06813013234666d9c033a660: train_sd3.py::compute_text_embeddings
and diffusers_patch/train_dreambooth_lora_sd3.py::encode_prompt (two CLIP
penultimate states plus T5, concatenated CLIP pools, encoded empty negatives).
This delegates to public Diffusers, not the historical implementation. In
particular, every pooled row is repeated alongside its token row; historical
mixed-precision/checkpoint parity is not claimed. Sequence length is caller
owned (the inspected training recipe used 128).
"""

import os

import torch
from torch import Tensor, nn

from .base import guidance_scale
from .sd3 import SD3Conditioning


def _placement(device, torch_dtype):
    if not isinstance(device, (str, torch.device)):
        raise ValueError("device must be an explicit string or torch.device")
    if torch_dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError("torch_dtype must be an explicit floating torch dtype")
    try:
        device = torch.device(device)
        if device.type == "meta":
            raise ValueError("meta is not an execution device")
        # Resolve the device and reject unavailable hardware before any loader I/O.
        return torch.empty(0, device=device).device
    except (RuntimeError, TypeError, AssertionError) as exc:
        raise ValueError(f"unavailable or invalid device: {device}") from exc


def _prompts(value, name, *, allow_empty=False):
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a string or a nonempty list/tuple of strings")
    if any(not isinstance(item, str) or (not allow_empty and not item.strip()) for item in value):
        raise ValueError(f"{name} contains an invalid or empty prompt")
    return list(value)


class SD3TextEncoder:
    """Own placement/freeze state of an injected pipeline's three text encoders.

    All CLIP/T5 encoders and tokenizers must be present. Other pipeline assets
    are untouched. Do not share encoders with text training or install offload
    hooks: this helper uses explicit, resident device/dtype placement. No LoRA,
    clip-skip, precomputed embeddings, or per-encoder prompt overrides are exposed.
    Import directly from ``visual_rl.models.sd3_text``; no registry is involved.
    """

    def __init__(self, pipeline, *, device, torch_dtype):
        self.device = _placement(device, torch_dtype)
        self.torch_dtype = torch_dtype
        if not callable(getattr(pipeline, "encode_prompt", None)):
            raise ValueError("pipeline must provide encode_prompt")
        encoders = []
        for suffix in ("", "_2", "_3"):
            encoder = getattr(pipeline, "text_encoder" + suffix, None)
            tokenizer = getattr(pipeline, "tokenizer" + suffix, None)
            if not isinstance(encoder, nn.Module) or not callable(tokenizer):
                raise ValueError("all three CLIP/T5 encoders and tokenizers are required")
            encoders.append(encoder)
        self.pipeline = pipeline
        self._encoders = tuple(encoders)
        for encoder in self._encoders:
            encoder.to(device=self.device, dtype=self.torch_dtype)
        self._freeze()

    def _freeze(self):
        for encoder in self._encoders:
            encoder.requires_grad_(False).eval()
            for parameter in encoder.parameters():
                parameter.grad = None

    @classmethod
    def from_pretrained(cls, model_id_or_path, *, revision, device, torch_dtype,
                        local_files_only=True):
        """Load text assets only; downloads require explicit local_files_only=False.

        Revision is required and nonempty; use a commit hash for reproducibility.
        Hugging Face's normal environment/credential/cache configuration applies.
        No transformer or VAE weights are loaded, even if present in the repo.
        """
        if not isinstance(model_id_or_path, (str, os.PathLike)):
            raise ValueError("model_id_or_path must be a nonempty string or path")
        model_id_or_path = os.fspath(model_id_or_path)
        if not isinstance(model_id_or_path, str) or not model_id_or_path.strip():
            raise ValueError("model_id_or_path must be a nonempty string or path")
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError("revision must be explicit and nonempty")
        if not isinstance(local_files_only, bool):
            raise ValueError("local_files_only must be a bool")
        device = _placement(device, torch_dtype)

        from diffusers import StableDiffusion3Pipeline

        pipeline = StableDiffusion3Pipeline.from_pretrained(
            model_id_or_path, revision=revision, local_files_only=local_files_only,
            torch_dtype=torch_dtype, transformer=None, vae=None,
        )
        return cls(pipeline, device=device, torch_dtype=torch_dtype)

    def encode(self, prompt, *, max_sequence_length, cfg_scale=None,
               negative_prompt=None, num_images_per_prompt=1) -> SD3Conditioning:
        """Return fresh detached tensors in prompt-major image-repeat order.

        For prompts [p0, p1] and two images each, every token/pooled tensor
        has rows [p0, p0, p1, p1]; negative rows use the same ordering.
        ``cfg_scale=None`` omits negatives; any finite nonnegative scale, even
        0 or 1, requests them. None/empty-string negatives encode actual empty
        text, not zeros. A scalar negative broadcasts; a list must match the
        positive batch. SD3Adapter owns negative-first CFG concatenation.
        ``max_sequence_length`` bounds T5 to 1..512 tokens; CLIP retains its
        tokenizer limit, so the joint token length also includes CLIP tokens.
        """
        prompts = _prompts(prompt, "prompt")
        if (type(max_sequence_length) is not int or not 1 <= max_sequence_length <= 512):
            raise ValueError("max_sequence_length must be an integer in [1, 512]")
        if type(num_images_per_prompt) is not int or num_images_per_prompt < 1:
            raise ValueError("num_images_per_prompt must be a positive integer")
        if cfg_scale is not None and (isinstance(cfg_scale, bool)
                                      or not isinstance(cfg_scale, (int, float))):
            raise ValueError("cfg_scale must be a finite nonnegative number or None")
        guidance_scale(cfg_scale)
        if negative_prompt is None:
            negatives = [""] * len(prompts)
        elif isinstance(negative_prompt, str):
            negatives = [negative_prompt] * len(prompts)
        else:
            negatives = _prompts(negative_prompt, "negative_prompt", allow_empty=True)
        if len(negatives) != len(prompts):
            raise ValueError("negative_prompt batch must match prompt batch")
        if cfg_scale is None and negative_prompt is not None:
            raise ValueError("negative_prompt requires cfg_scale (None disables CFG)")

        self._freeze()
        # Ordinary no-grad tensors remain usable by a trainable transformer,
        # even when the caller happened to enter inference_mode or autocast.
        with torch.inference_mode(False), torch.no_grad(), torch.autocast(
                device_type=self.device.type, enabled=False):
            pos, neg, pooled_pos, pooled_neg = self.pipeline.encode_prompt(
                prompt=prompts, prompt_2=None, prompt_3=None,
                device=self.device, num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=cfg_scale is not None,
                negative_prompt=negatives if cfg_scale is not None else None,
                max_sequence_length=max_sequence_length,
            )
            batch = len(prompts) * num_images_per_prompt

            def checked(value, ndim, name):
                if (not isinstance(value, Tensor) or value.ndim != ndim
                        or value.shape[0] != batch or any(size == 0 for size in value.shape)
                        or not value.is_floating_point()):
                    raise ValueError(f"{name} has invalid rank, dtype, or repeated batch size")
                return value.detach().to(device=self.device, dtype=self.torch_dtype).clone()

            pos = checked(pos, 3, "prompt_embeds")
            pooled_pos = checked(pooled_pos, 2, "pooled_prompt_embeds")
            if cfg_scale is not None:
                neg = checked(neg, 3, "negative_prompt_embeds")
                pooled_neg = checked(pooled_neg, 2, "negative_pooled_prompt_embeds")
                if neg.shape != pos.shape or pooled_neg.shape != pooled_pos.shape:
                    raise ValueError("positive and negative conditioning shapes must match")
            else:
                neg, pooled_neg = None, None
        return SD3Conditioning(
            prompt_embeds=pos, pooled_prompt_embeds=pooled_pos,
            negative_prompt_embeds=neg, negative_pooled_prompt_embeds=pooled_neg,
        )
