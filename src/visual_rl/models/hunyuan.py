"""Public HY13B FastVideo ABI, deliberately not a Diffusers HY pipeline."""

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
class HunyuanVideoConfig:
    scaling_factor: float = 0.476986
    latent_channels: int = 16
    timestep_scale: float = 1000.0
    embedded_guidance_raw: float = 6018.0
    guidance_dtype: torch.dtype = torch.bfloat16

    def __post_init__(self):
        positive(self.scaling_factor, "scaling_factor")
        channel_count(self.latent_channels)
        positive(self.timestep_scale, "timestep_scale")
        guidance_scale(self.embedded_guidance_raw)
        if self.embedded_guidance_raw is None or not self.guidance_dtype.is_floating_point:
            raise ValueError("native embedded guidance requires a value and floating dtype")


@dataclass(frozen=True)
class HunyuanVideoConditioning:
    # FastVideo packs pooled CLIP features in token 0, followed by text tokens.
    prompt_embeds: Tensor
    attention_mask: Tensor


class HunyuanVideoAdapter(VelocityAdapter):
    def __init__(self, transformer, vae, config: HunyuanVideoConfig = HunyuanVideoConfig()):
        check_component_config(
            transformer, in_channels=config.latent_channels, out_channels=config.latent_channels
        )
        check_component_config(
            vae,
            scaling_factor=config.scaling_factor,
            shift_factor=0.0,
            latent_channels=config.latent_channels,
        )
        super().__init__(transformer, vae)
        self.config = config

    def predict_velocity(self, latents, sigma, conditioning: HunyuanVideoConditioning):
        validate_latents(latents, 5, self.config.latent_channels)
        c = conditioning
        batch_tensor(c.prompt_embeds, latents, "prompt_embeds", 3)
        batch_tensor(c.attention_mask, latents, "attention_mask", 2)
        if c.prompt_embeds.shape[1] != c.attention_mask.shape[1] + 1:
            raise ValueError(
                "native HY prompt must include one pooled token before the masked text tokens"
            )
        t = (sigma_batch(sigma, latents) * self.config.timestep_scale).long()
        guidance = torch.tensor(
            [self.config.embedded_guidance_raw],
            device=latents.device,
            dtype=self.config.guidance_dtype,
        )
        output = self.transformer(
            hidden_states=latents,
            encoder_hidden_states=c.prompt_embeds,
            timestep=t,
            guidance=guidance,
            encoder_attention_mask=c.attention_mask,
            return_dict=False,
        )[0]
        return checked_velocity(output, latents)

    def to_vae_latents(self, latents):
        validate_latents(latents, 5, self.config.latent_channels)
        return affine_decode(latents, self.config.scaling_factor, 0.0)
