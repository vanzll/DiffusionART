"""SD3/SD3.5 joint-attention transformer adapter."""

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


@dataclass(frozen=True)
class SD3Config:
    scaling_factor: float
    shift_factor: float
    latent_channels: int = 16
    timestep_scale: float = 1000.0
    cfg_scale: float | None = None

    def __post_init__(self):
        positive(self.scaling_factor, "scaling_factor")
        positive(self.timestep_scale, "timestep_scale")
        channel_count(self.latent_channels)
        if not math.isfinite(self.shift_factor):
            raise ValueError("shift_factor must be finite")
        guidance_scale(self.cfg_scale)


@dataclass(frozen=True)
class SD3Conditioning:
    prompt_embeds: Tensor
    pooled_prompt_embeds: Tensor
    negative_prompt_embeds: Tensor | None = None
    negative_pooled_prompt_embeds: Tensor | None = None


class SD3Adapter(VelocityAdapter):
    def __init__(self, transformer, vae, config: SD3Config):
        check_component_config(
            transformer, in_channels=config.latent_channels, out_channels=config.latent_channels
        )
        check_component_config(
            vae,
            scaling_factor=config.scaling_factor,
            shift_factor=config.shift_factor,
            latent_channels=config.latent_channels,
        )
        super().__init__(transformer, vae)
        self.config = config

    def forward(self, latents, sigma, conditioning, *, upcast_output=False):
        return self.predict_velocity(latents, sigma, conditioning, upcast_output=upcast_output)

    def predict_velocity(self, latents, sigma, conditioning: SD3Conditioning, *, upcast_output=False):
        validate_latents(latents, 4, self.config.latent_channels)
        c = conditioning
        batch_tensor(c.prompt_embeds, latents, "prompt_embeds", 3)
        batch_tensor(c.pooled_prompt_embeds, latents, "pooled_prompt_embeds", 2)
        t = sigma_batch(sigma, latents) * self.config.timestep_scale
        hidden, prompt, pooled = latents, c.prompt_embeds, c.pooled_prompt_embeds
        if self.config.cfg_scale is not None:
            batch_tensor(c.negative_prompt_embeds, latents, "negative_prompt_embeds", 3)
            batch_tensor(
                c.negative_pooled_prompt_embeds, latents, "negative_pooled_prompt_embeds", 2
            )
            if (
                c.negative_prompt_embeds.shape != prompt.shape
                or c.negative_pooled_prompt_embeds.shape != pooled.shape
            ):
                raise ValueError("positive and negative conditioning shapes must match")
            hidden, t = torch.cat([latents, latents]), torch.cat([t, t])
            prompt = torch.cat([c.negative_prompt_embeds, prompt])
            pooled = torch.cat([c.negative_pooled_prompt_embeds, pooled])
        output = self.transformer(
            hidden_states=hidden,
            timestep=t,
            encoder_hidden_states=prompt,
            pooled_projections=pooled,
            return_dict=False,
        )[0]
        checked_velocity(output, hidden)
        if upcast_output:
            output = output.float()
        if self.config.cfg_scale is not None:
            unconditional, conditional = output.chunk(2)
            output = unconditional + self.config.cfg_scale * (conditional - unconditional)
        return checked_velocity(output, latents)

    def to_vae_latents(self, latents):
        validate_latents(latents, 4, self.config.latent_channels)
        return affine_decode(latents, self.config.scaling_factor, self.config.shift_factor)

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path,
        *,
        config: SD3Config,
        revision=None,
        local_files_only=True,
        torch_dtype=None,
    ):
        """Explicit, lazy component loading. Hugging Face reads HF_TOKEN itself."""
        from diffusers import AutoencoderKL, SD3Transformer2DModel

        kwargs = dict(revision=revision, local_files_only=local_files_only)
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        transformer = SD3Transformer2DModel.from_pretrained(
            model_id_or_path, subfolder="transformer", **kwargs
        )
        vae = AutoencoderKL.from_pretrained(model_id_or_path, subfolder="vae", **kwargs)
        return cls(transformer, vae, config)
