"""Differentiable T2AV adapter for the pinned native Diffusers MiniMax-H3 ABI.

Native helpers are imported lazily; importing this module needs only PyTorch.
Video is [B,C,T,H,W], audio is [B,stereo,C,T]. Public velocity is dz/dsigma,
whereas H3 predicts dz/d(1-sigma), so sign conversion happens exactly here.
"""

import inspect
from dataclasses import dataclass

import torch

from ..training.audio_video_types import AVMedia, AVState

DIFFUSERS_REVISION = "73faca368c0da77645ea58836aa8780d2d8c5353"
MODEL_ID = "MiniMaxAI/MiniMax-H3"
MODEL_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"


@dataclass(frozen=True)
class MiniMaxH3Config:
    fps: float = 24.0
    audio_channels: int = 2
    pixel_mean: tuple = (0.485, 0.456, 0.406)
    pixel_std: tuple = (0.229, 0.224, 0.225)

    def __post_init__(self):
        if self.fps != 24 or self.audio_channels != 2:
            raise ValueError("Released H3 supports native 24fps stereo only")


@dataclass(frozen=True)
class MiniMaxH3Conditioning:
    prompt_embeds: torch.Tensor
    attention_mask: torch.Tensor

    def select(self, rows):
        if isinstance(rows, torch.Tensor):
            rows = rows.to(self.prompt_embeds.device)
        return MiniMaxH3Conditioning(self.prompt_embeds[rows], self.attention_mask[rows])

    def to(self, device):
        return MiniMaxH3Conditioning(self.prompt_embeds.to(device), self.attention_mask.to(device))


def native_h3_helpers():
    """Load upstream implementations rather than vendoring layout/patch code."""
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        MiniMaxH3PrepareLayoutStep,
        patchify_video_latents,
    )
    return MiniMaxH3PrepareLayoutStep.build_packed_sequence, patchify_video_latents


def validate_native_abi(transformer):
    expected = (
        "hidden_states", "audio_hidden_states", "encoder_hidden_states", "timestep",
        "timestep_indices", "token_tags", "position_ids", "video_indices", "audio_indices",
        "text_indices", "attention_kwargs", "return_dict",
    )
    if tuple(inspect.signature(transformer.forward).parameters) != expected:
        raise RuntimeError(f"Unsupported H3 forward ABI; install Diffusers@{DIFFUSERS_REVISION}")
    config = transformer.config
    if tuple(config.patch_size)[0] != 1:
        raise ValueError("Pinned H3 layout requires temporal patch size 1")


def _sigma_batch(sigma, state):
    value = torch.as_tensor(sigma, device=state.device, dtype=torch.float32)
    if value.ndim == 0:
        value = value.expand(len(state))
    if value.shape != (len(state),) or not torch.isfinite(value).all():
        raise ValueError("sigma must be a finite scalar or [batch]")
    if ((value < 0) | (value > 1)).any():
        raise ValueError("sigma must be in [0,1]")
    return value


def unpack_video_rows(rows, shape, patch_size):
    b, c, t, h, w = shape
    pt, ph, pw = patch_size
    return rows.reshape(b, t // pt, h // ph, w // pw, c, pt, ph, pw).permute(
        0, 4, 1, 5, 2, 6, 3, 7
    ).reshape(shape)


class MiniMaxH3Adapter:
    def __init__(self, transformer, vae=None, audio_vae=None, config=MiniMaxH3Config()):
        validate_native_abi(transformer)
        from .minimax_h3_batch import enable_batched_h3
        self.transformer = enable_batched_h3(transformer)
        self.vae, self.audio_vae, self.config = vae, audio_vae, config
        # Save immutable architecture metadata before distributed wrappers replace the module.
        self.transformer_config = transformer.config
        for component in (vae, audio_vae):
            if component is not None:
                component.requires_grad_(False).eval()

    @property
    def trainable_module(self):
        return self.transformer

    def set_trainable_module(self, module):
        self.transformer = module

    def staged_decoders(self, device):
        if self.vae is None or self.audio_vae is None:
            raise ValueError("decoder staging requires both native VAEs")
        from .minimax_h3_assets import stage_native_components
        return stage_native_components((self.vae, self.audio_vae), device)

    def latent_shapes(self, batch, height, width, frames, fps=24):
        from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
            audio_latent_num_frames,
            video_latent_num_frames,
        )
        if self.vae is None or self.audio_vae is None:
            raise ValueError("latent geometry requires loaded video/audio VAEs")
        if fps != self.config.fps:
            raise ValueError("H3 has a fixed native frame rate")
        ratio = self.vae.spatial_compression_ratio
        if min(batch, height, width, frames) <= 0 or height % ratio or width % ratio:
            raise ValueError("positive dimensions and VAE-divisible canvas required")
        pt, ph, pw = self.transformer_config.patch_size
        if (height // ratio) % ph or (width // ratio) % pw:
            raise ValueError("latent canvas must be divisible by transformer patch size")
        vt = video_latent_num_frames(frames, self.vae.config.clip_length, self.vae.tokens_chunk_size)
        # Pinned native H3 audio VAE has hop800 at32k; derive from component architecture.
        hop = self.audio_vae.hop_length
        rate = self.audio_vae.config.sampling_rate / hop
        at = audio_latent_num_frames(frames, fps, rate)
        return ((batch, self.vae.config.latent_channels, vt, height // ratio, width // ratio),
                (batch, self.config.audio_channels, self.audio_vae.config.latent_channels, at))

    def _validate_state(self, state, conditioning):
        v, a = state.video, state.audio
        cfg = self.transformer_config
        if v.ndim != 5 or a.ndim != 4 or len(v) != len(a) or not len(v):
            raise ValueError("H3 state requires video[B,C,T,H,W], audio[B,2,C,T]")
        if v.shape[1] != cfg.in_channels or a.shape[1:3] != (2, cfg.audio_in_channels):
            raise ValueError("H3 latent channel count mismatch")
        if v.device != a.device or not v.is_floating_point() or not a.is_floating_point():
            raise ValueError("H3 floating latent modalities must share a device")
        if state.video_mask is not None or state.audio_mask is not None:
            raise ValueError("H3 v1 accepts homogeneous unpadded native latent batches only")
        c = conditioning
        if (c.prompt_embeds.ndim != 3 or c.prompt_embeds.shape[0] != len(v)
                or c.prompt_embeds.shape[2] != cfg.text_dim
                or c.attention_mask.shape != c.prompt_embeds.shape[:2]
                or c.prompt_embeds.device != v.device or c.attention_mask.device != v.device):
            raise ValueError("H3 conditioning batch/embedding/mask/device mismatch")
        if c.attention_mask.dtype != torch.bool:
            raise ValueError("conditioning mask must be boolean")
        lengths = c.attention_mask.sum(1)
        prefix = torch.arange(c.attention_mask.shape[1], device=v.device)[None] < lengths[:, None]
        if (lengths == 0).any() or not torch.equal(prefix, c.attention_mask):
            raise ValueError("H3 text masks must be nonempty right-padded prefixes")

    def predict_velocity(self, state, video_sigma, audio_sigma, conditioning):
        self._validate_state(state, conditioning)
        build_layout, patchify = native_h3_helpers()
        v, a = state.video, state.audio
        sv, sa = _sigma_batch(video_sigma, v), _sigma_batch(audio_sigma, a)
        patch = tuple(self.transformer_config.patch_size)
        batch = len(v)
        # One common padded layout; each request keeps its original media clock.
        lengths = conditioning.attention_mask.sum(1)
        text_length = int(lengths.max())
        positions, tags, vi, ai, ti, _, _ = build_layout(
            torch.ones(text_length, dtype=torch.long), v.shape[2], v.shape[3],
            v.shape[4], a.shape[-1], patch, 2, 2, 0)
        positions = positions.to(v.device, dtype=torch.float32 if v.device.type == "mps" else torch.float64)
        positions = positions[None].expand(batch, -1, -1).clone()
        positions[:, text_length:, 0] += (lengths - text_length)[:, None]
        tags, vi, ai, ti = (x.to(v.device) for x in (tags, vi, ai, ti))
        row_times = (1 - sv)[:, None].expand(batch, len(tags)).clone()
        row_times[:, ai] = (1 - sa)[:, None]
        times, indices = torch.unique(row_times, sorted=True, return_inverse=True)
        video, audio = self.transformer(
            hidden_states=patchify(v, patch).reshape(batch, len(vi), -1),
            audio_hidden_states=a.permute(0, 1, 3, 2).flatten(1, 2),
            encoder_hidden_states=conditioning.prompt_embeds[:, :text_length],
            timestep=times, timestep_indices=indices, token_tags=tags,
            position_ids=positions, video_indices=vi, audio_indices=ai, text_indices=ti,
            attention_kwargs={"h3_text_mask": conditioning.attention_mask[:, :text_length]},
            return_dict=False,
        )
        return AVState(-unpack_video_rows(video, v.shape, patch),
                       -audio.reshape(batch, 2, a.shape[-1], a.shape[2]).permute(0, 1, 3, 2))

    @torch.no_grad()
    def decode(self, state):
        if self.vae is None or self.audio_vae is None:
            raise ValueError("decode requires both native VAEs")
        v, a = state.video, state.audio
        def denormalize(x, config, ndim):
            mean = x.new_tensor(config.latents_mean).reshape(1, -1, *([1] * (ndim - 2)))
            std = x.new_tensor(config.latents_std).reshape(1, -1, *([1] * (ndim - 2)))
            return x * std + mean
        vp = next(self.vae.parameters())
        ap = next(self.audio_vae.parameters())
        video = self.vae.decode(denormalize(v.to(vp), self.vae.config, 5), return_dict=False)[0]
        mean = video.new_tensor(self.config.pixel_mean).reshape(1, 3, 1, 1, 1)
        std = video.new_tensor(self.config.pixel_std).reshape(1, 3, 1, 1, 1)
        video = (video.float() * std + mean).clamp(0, 1)
        batch, stereo, channels, frames = a.shape
        audio = denormalize(a.reshape(batch * stereo, channels, frames).to(ap), self.audio_vae.config, 3)
        audio = self.audio_vae.decode(audio, return_dict=False)[0].float()
        if audio.ndim != 3 or audio.shape[1] != 1:
            raise RuntimeError("native mono audio decoder returned unexpected shape")
        return AVMedia(video, audio.reshape(batch, stereo, -1), self.config.fps,
                       int(self.audio_vae.config.sampling_rate))


def native_h3_block_types():
    from diffusers.models.transformers.transformer_minimax_h3 import (
        MiniMaxH3TokenRefinerBlock,
        MiniMaxH3TransformerBlock,
    )
    return (MiniMaxH3TransformerBlock, MiniMaxH3TokenRefinerBlock)
