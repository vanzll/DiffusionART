"""Single-transformer Wan2.1 CLIP-conditioned first-frame I2V adapter."""

import inspect
import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .base import (
    VelocityAdapter,
    batch_tensor,
    check_component_config,
    checked_velocity,
    guidance_scale,
    positive,
    sigma_batch,
    validate_latents,
)


@dataclass(frozen=True)
class WanI2VConfig:
    latents_mean: tuple[float, ...]
    latents_std: tuple[float, ...]
    temporal_compression: int = 4
    timestep_scale: float = 1000.0
    cfg_scale: float | None = 5.0

    def __post_init__(self):
        object.__setattr__(self, "latents_mean", tuple(self.latents_mean))
        object.__setattr__(self, "latents_std", tuple(self.latents_std))
        if not self.latents_mean or len(self.latents_mean) != len(self.latents_std):
            raise ValueError("Wan requires one mean and std per latent channel")
        if any(not math.isfinite(v) for v in self.latents_mean):
            raise ValueError("latents_mean must be finite")
        for value in self.latents_std:
            positive(value, "latents_std")
        if not isinstance(self.temporal_compression, int) or self.temporal_compression < 1:
            raise ValueError("temporal_compression must be a positive integer")
        positive(self.timestep_scale, "timestep_scale")
        guidance_scale(self.cfg_scale)

    @property
    def latent_channels(self):
        return len(self.latents_mean)


@dataclass(frozen=True)
class WanI2VConditioning:
    prompt_embeds: Tensor
    image_embeds: Tensor
    image_condition: Tensor
    negative_prompt_embeds: Tensor | None = None


def first_frame_condition(
    normalized_video_latents: Tensor, temporal_compression: int = 4
) -> Tensor:
    """Prepend the first-frame mask to an ALREADY encoded/normalized video.

    The caller encodes [reference image, zero pixel frames] using the VAE mode,
    then normalizes every latent frame. Zero-padding an encoded first frame is
    NOT equivalent. This helper never encodes or overwrites the noise state.
    """
    if (
        normalized_video_latents.ndim != 5
        or any(n == 0 for n in normalized_video_latents.shape)
        or not normalized_video_latents.is_floating_point()
    ):
        raise ValueError("conditioning latents must be nonempty floating BCTHW")
    if not isinstance(temporal_compression, int) or temporal_compression < 1:
        raise ValueError("temporal_compression must be a positive integer")
    b, _, t, h, w = normalized_video_latents.shape
    mask = normalized_video_latents.new_zeros(b, temporal_compression, t, h, w)
    mask[:, :, 0] = 1
    return torch.cat([mask, normalized_video_latents], dim=1)


def decode_wan_differentiable(vae, raw_latents: Tensor) -> Tensor:
    """Untiled native Wan2.1 primitives with per-call differentiable caches.

    Input is already denormalized. No VAE affine transform occurs here.
    Native tiling and patched Wan2.2 decoders are intentionally unsupported.
    """
    if getattr(vae.config, "patch_size", None) is not None:
        raise ValueError("only the unpatched Wan2.1 VAE is supported")
    if getattr(vae, "use_tiling", False):
        raise ValueError("differentiable Wan decode is untiled; disable VAE tiling explicitly")
    count = sum(type(module).__name__ == "WanCausalConv3d" for module in vae.decoder.modules())
    if not count:
        raise ValueError("native Wan causal decoder blocks are required")
    has_first_chunk = "first_chunk" in inspect.signature(vae.decoder.forward).parameters
    x = vae.post_quant_conv(raw_latents)
    cache = [None] * count
    frames = []
    for index in range(x.shape[2]):
        kwargs = {"feat_cache": cache, "feat_idx": [0]}
        if has_first_chunk:
            kwargs["first_chunk"] = index == 0
        frames.append(vae.decoder(x[:, :, index : index + 1], **kwargs))
    return torch.cat(frames, dim=2).clamp(-1.0, 1.0)


class WanI2VAdapter(VelocityAdapter):
    def __init__(self, transformer, vae, config: WanI2VConfig):
        check_component_config(
            transformer,
            in_channels=2 * config.latent_channels + config.temporal_compression,
            out_channels=config.latent_channels,
        )
        model_config = getattr(transformer, "config", None)
        if (
            model_config is not None
            and hasattr(model_config, "image_dim")
            and model_config.image_dim is None
        ):
            raise ValueError("Wan2.1 I2V requires a CLIP image-conditioned transformer")
        check_component_config(
            vae,
            z_dim=config.latent_channels,
            latents_mean=config.latents_mean,
            latents_std=config.latents_std,
        )
        super().__init__(transformer, vae)
        self.config = config

    def predict_velocity(self, latents, sigma, conditioning: WanI2VConditioning):
        config, c = self.config, conditioning
        validate_latents(latents, 5, config.latent_channels)
        batch_tensor(c.prompt_embeds, latents, "prompt_embeds", 3)
        batch_tensor(c.image_embeds, latents, "image_embeds", 3)
        batch_tensor(c.image_condition, latents, "image_condition", 5)
        expected = (
            latents.shape[0],
            config.temporal_compression + config.latent_channels,
            *latents.shape[2:],
        )
        if c.image_condition.shape != expected:
            raise ValueError(
                "image_condition must contain mask channels followed by normalized VAE channels"
            )
        mask = c.image_condition[:, : config.temporal_compression]
        if not torch.all(mask[:, :, 0] == 1) or not torch.all(mask[:, :, 1:] == 0):
            raise ValueError(
                "only first-frame I2V masks (first latent frame one, rest zero) are supported"
            )
        hidden = torch.cat([latents, c.image_condition], dim=1)
        t = sigma_batch(sigma, latents) * config.timestep_scale
        prompt, image = c.prompt_embeds, c.image_embeds
        if config.cfg_scale is not None:
            batch_tensor(c.negative_prompt_embeds, latents, "negative_prompt_embeds", 3)
            if c.negative_prompt_embeds.shape != prompt.shape:
                raise ValueError("positive and negative prompt shapes must match")
            hidden, t = torch.cat([hidden, hidden]), torch.cat([t, t])
            prompt = torch.cat([prompt, c.negative_prompt_embeds])
            image = torch.cat([image, image])
        output = self.transformer(
            hidden_states=hidden,
            timestep=t,
            encoder_hidden_states=prompt,
            encoder_hidden_states_image=image,
            return_dict=False,
        )[0]
        expected_batch = latents.shape[0] * (2 if config.cfg_scale is not None else 1)
        if output.shape != (expected_batch, *latents.shape[1:]):
            raise ValueError(
                "Wan velocity must exclude conditioning channels and match the full CFG batch"
            )
        if config.cfg_scale is not None:
            conditional, unconditional = output.float().chunk(2)
            output = unconditional + config.cfg_scale * (conditional - unconditional)
        return checked_velocity(output, latents)

    def to_vae_latents(self, latents):
        validate_latents(latents, 5, self.config.latent_channels)
        dtype = torch.float64 if latents.dtype == torch.float64 else torch.float32
        mean = torch.tensor(self.config.latents_mean, device=latents.device, dtype=dtype).view(
            1, -1, 1, 1, 1
        )
        std = torch.tensor(self.config.latents_std, device=latents.device, dtype=dtype).view(
            1, -1, 1, 1, 1
        )
        return latents.to(dtype) * std + mean

    def _decode_raw(self, latents):
        return decode_wan_differentiable(self.vae, latents)
