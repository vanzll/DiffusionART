"""Injected velocity adapters. Importing this package never imports Diffusers."""

from .base import VelocityAdapter
from .flux import FluxAdapter, FluxConditioning, FluxConfig, pack_latents, unpack_latents
from .hunyuan import HunyuanVideoAdapter, HunyuanVideoConditioning, HunyuanVideoConfig
from .sd3 import SD3Adapter, SD3Conditioning, SD3Config
from .wan import WanI2VAdapter, WanI2VConditioning, WanI2VConfig, first_frame_condition

__all__ = [
    "VelocityAdapter",
    "SD3Adapter",
    "SD3Conditioning",
    "SD3Config",
    "FluxAdapter",
    "FluxConditioning",
    "FluxConfig",
    "pack_latents",
    "unpack_latents",
    "HunyuanVideoAdapter",
    "HunyuanVideoConditioning",
    "HunyuanVideoConfig",
    "WanI2VAdapter",
    "WanI2VConditioning",
    "WanI2VConfig",
    "first_frame_condition",
]
