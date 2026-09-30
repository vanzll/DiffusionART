"""FLUX.1 adapter with an unpacked BCHW public latent representation."""

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .base import (
    VelocityAdapter,
    affine_decode,
    batch_tensor,
    channel_count,
    check_component_config,
    checked_velocity,
    guidance_scale,
    positive,
    sigma_batch,
    validate_latents,
)


def pack_latents(latents: Tensor) -> Tensor:
    if latents.ndim != 4 or any(size == 0 for size in latents.shape):
        raise ValueError("FLUX packing requires nonempty BCHW latents")
    b, c, h, w = latents.shape
    if h % 2 or w % 2:
        raise ValueError("FLUX latent height and width must be even")
    return (
        latents.reshape(b, c, h // 2, 2, w // 2, 2)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(b, h * w // 4, c * 4)
    )


def unpack_latents(packed: Tensor, height: int, width: int) -> Tensor:
    if (
        packed.ndim != 3
        or height <= 0
        or width <= 0
        or height % 2
        or width % 2
        or packed.shape[1] != height * width // 4
        or packed.shape[2] % 4
    ):
        raise ValueError("packed FLUX geometry does not match the explicit latent height/width")
    b, _, c4 = packed.shape
    return (
        packed.reshape(b, height // 2, width // 2, c4 // 4, 2, 2)
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(b, c4 // 4, height, width)
    )


def image_ids(latents: Tensor) -> Tensor:
    h, w = latents.shape[-2:]
    rows, columns = torch.meshgrid(
        torch.arange(h // 2, device=latents.device),
        torch.arange(w // 2, device=latents.device),
        indexing="ij",
    )
    return (
        torch.stack([torch.zeros_like(rows), rows, columns], dim=-1)
        .reshape(-1, 3)
        .to(latents.dtype)
    )


@dataclass(frozen=True)
class FluxConfig:
    scaling_factor: float
    shift_factor: float
    latent_channels: int = 16
    embedded_guidance_scale: float | None = 3.5
    scheduler_timestep_scale: float = 1000.0
    vae_affine_dtype: torch.dtype | None = None

    def __post_init__(self):
        positive(self.scaling_factor, "scaling_factor")
        channel_count(self.latent_channels)
        positive(self.scheduler_timestep_scale, "scheduler_timestep_scale")
        if not math.isfinite(self.shift_factor):
            raise ValueError("shift_factor must be finite")
        guidance_scale(self.embedded_guidance_scale)
        if self.vae_affine_dtype not in (None, torch.float32, torch.bfloat16):
            raise ValueError("FLUX VAE affine precision must be fp32 or bf16")


@dataclass(frozen=True)
class FluxConditioning:
    prompt_embeds: Tensor
    pooled_prompt_embeds: Tensor
    text_ids: Tensor


class FluxAdapter(VelocityAdapter):
    def __init__(self, transformer, vae, config: FluxConfig):
        model_config = getattr(transformer, "config", None)
        expected = getattr(model_config, "guidance_embeds", None)
        if expected is not None and bool(expected) != (config.embedded_guidance_scale is not None):
            raise ValueError(
                "embedded_guidance_scale must agree with transformer.config.guidance_embeds"
            )
        check_component_config(transformer, in_channels=4 * config.latent_channels)
        # Diffusers may leave out_channels=None, meaning use in_channels.
        if getattr(model_config, "out_channels", None) is not None:
            check_component_config(transformer, out_channels=4 * config.latent_channels)
        check_component_config(
            vae,
            scaling_factor=config.scaling_factor,
            shift_factor=config.shift_factor,
            latent_channels=config.latent_channels,
        )
        super().__init__(transformer, vae)
        self.config = config

    def forward(self, latents, sigma, conditioning, *, timestep_dtype=None):
        return self.predict_velocity(latents, sigma, conditioning, timestep_dtype=timestep_dtype)

    def predict_velocity(self, latents, sigma, conditioning: FluxConditioning, *, timestep_dtype=None):
        validate_latents(latents, 4, self.config.latent_channels)
        c = conditioning
        batch_tensor(c.prompt_embeds, latents, "prompt_embeds", 3)
        batch_tensor(c.pooled_prompt_embeds, latents, "pooled_prompt_embeds", 2)
        if c.text_ids.shape != (c.prompt_embeds.shape[1], 3) or c.text_ids.device != latents.device:
            raise ValueError("text_ids must be unbatched [text_sequence, 3] on the latent device")
        packed = pack_latents(latents)
        # Rollout casts before division; historical training replays FP32 timesteps.
        if timestep_dtype not in (None, torch.float32, torch.bfloat16):
            raise ValueError("FLUX timestep precision must be fp32 or bf16")
        t = (sigma_batch(sigma, latents) * self.config.scheduler_timestep_scale).to(
            latents.dtype if timestep_dtype is None else timestep_dtype
        ) / 1000.0
        guidance = None
        if self.config.embedded_guidance_scale is not None:
            guidance = torch.full(
                (latents.shape[0],),
                self.config.embedded_guidance_scale,
                device=latents.device,
                dtype=torch.float32,
            )
        output = self.transformer(
            hidden_states=packed,
            timestep=t,
            guidance=guidance,
            pooled_projections=c.pooled_prompt_embeds,
            encoder_hidden_states=c.prompt_embeds,
            txt_ids=c.text_ids,
            img_ids=image_ids(latents),
            return_dict=False,
        )[0]
        checked_velocity(output, packed)
        return unpack_latents(output, *latents.shape[-2:])

    def to_vae_latents(self, latents):
        validate_latents(latents, 4, self.config.latent_channels)
        pack_latents(latents)  # Reject geometries that cannot be predicted.
        if self.config.vae_affine_dtype is not None:
            # Historical pipelines perform this affine before casting to VAE dtype.
            value = latents.to(self.config.vae_affine_dtype)
            return value / self.config.scaling_factor + self.config.shift_factor
        return affine_decode(latents, self.config.scaling_factor, self.config.shift_factor)

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path,
        *,
        config: FluxConfig,
        revision=None,
        local_files_only=True,
        torch_dtype=None,
    ):
        from diffusers import AutoencoderKL, FluxTransformer2DModel

        kwargs = dict(revision=revision, local_files_only=local_files_only)
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        transformer = FluxTransformer2DModel.from_pretrained(
            model_id_or_path, subfolder="transformer", **kwargs
        )
        vae = AutoencoderKL.from_pretrained(model_id_or_path, subfolder="vae", **kwargs)
        return cls(transformer, vae, config)
