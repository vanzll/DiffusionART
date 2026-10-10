"""Differentiable component adapters; no pipeline inference or asset loading."""

import math
from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn


def positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def channel_count(value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("latent_channels must be a positive integer")


def check_component_config(module: nn.Module, **expected) -> None:
    """Check declared metadata when available, without unwrapping injected wrappers."""
    config = getattr(module, "config", None)
    for name, wanted in expected.items():
        if config is None or not hasattr(config, name):
            continue
        actual = getattr(config, name)
        if actual is None and name == "shift_factor":
            actual = 0.0
        if isinstance(wanted, (tuple, list)):
            matches = (
                actual is not None
                and len(actual) == len(wanted)
                and all(
                    math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-7) for a, b in zip(actual, wanted)
                )
            )
        else:
            matches = actual is not None and math.isclose(
                actual, wanted, rel_tol=1e-6, abs_tol=1e-7
            )
        if not matches:
            raise ValueError(
                f"component config {name}={actual!r} disagrees with adapter {wanted!r}"
            )


def guidance_scale(value: float | None) -> None:
    if value is not None and (not math.isfinite(value) or value < 0):
        raise ValueError("guidance scale must be finite and nonnegative, or None")


def validate_latents(latents: Tensor, ndim: int, channels: int) -> None:
    if (
        latents.ndim != ndim
        or latents.shape[1] != channels
        or any(size == 0 for size in latents.shape)
        or not latents.is_floating_point()
    ):
        raise ValueError(f"latents must be nonempty floating {ndim}D with {channels} channels")


def batch_tensor(value: Tensor, latents: Tensor, name: str, ndim: int) -> None:
    if not isinstance(value, Tensor) or value.ndim != ndim or value.shape[0] != latents.shape[0]:
        raise ValueError(
            f"{name} must have rank {ndim} and the latent batch size (no implicit repeat)"
        )
    if value.device != latents.device:
        raise ValueError(f"{name} and latents must be on the same device")


def sigma_batch(sigma: Tensor | float, latents: Tensor) -> Tensor:
    """Already schedule-shifted noise level, never a scheduler index."""
    value = torch.as_tensor(sigma, device=latents.device, dtype=torch.float32)
    if value.ndim == 0:
        value = value.expand(latents.shape[0])
    elif value.ndim == 1 and value.numel() == 1:
        value = value.expand(latents.shape[0])
    elif value.shape != (latents.shape[0],):
        raise ValueError("sigma must be scalar, [1], or [batch]")
    if not torch.isfinite(value).all() or ((value < 0) | (value > 1)).any():
        raise ValueError("sigma must be finite and in [0, 1]")
    return value


def checked_velocity(output: Tensor, latents: Tensor) -> Tensor:
    if not isinstance(output, Tensor) or output.shape != latents.shape:
        raise ValueError("velocity must have exactly the input latent shape")
    return output


def module_dtype(module: nn.Module, fallback: torch.dtype) -> torch.dtype:
    parameter = next(module.parameters(), None)
    return fallback if parameter is None else parameter.dtype


class VelocityAdapter(nn.Module, ABC):
    """A policy module plus a frozen, input-differentiable VAE.

    Placement, autocast, text/image encoding, schedules and optimizers belong
    to the caller. Constructing an adapter freezes the injected VAE in place.
    """

    def __init__(self, transformer: nn.Module, vae: nn.Module):
        super().__init__()
        if transformer is vae:
            raise ValueError("transformer and VAE must be separate modules")
        self.transformer = transformer
        self.vae = vae.requires_grad_(False).eval()
        for parameter in self.vae.parameters():
            parameter.grad = None

    @property
    def trainable_module(self) -> nn.Module:
        return self.transformer

    def train(self, mode: bool = True):
        super().train(mode)
        self.vae.eval()
        return self

    @abstractmethod
    def predict_velocity(self, latents: Tensor, sigma: Tensor | float, conditioning) -> Tensor:
        """Return dz/dsigma in the same coordinates and shape as latents."""

    def forward(self, latents: Tensor, sigma: Tensor | float, conditioning) -> Tensor:
        return self.predict_velocity(latents, sigma, conditioning)

    @abstractmethod
    def to_vae_latents(self, latents: Tensor) -> Tensor:
        """Apply the model-to-VAE affine transform exactly once."""

    def _decode_raw(self, latents: Tensor) -> Tensor:
        return self.vae.decode(latents, return_dict=False)[0]

    def decode(self, latents: Tensor) -> Tensor:
        raw = self.to_vae_latents(latents)
        raw = raw.to(dtype=module_dtype(self.vae, raw.dtype))
        with torch.autocast(device_type=raw.device.type, enabled=False):
            decoded = self._decode_raw(raw)
        if not isinstance(decoded, Tensor) or decoded.shape[0] != latents.shape[0]:
            raise ValueError("VAE must return a tensor with the input batch size")
        if torch.is_grad_enabled() and latents.requires_grad and not decoded.requires_grad:
            raise RuntimeError("VAE decode detached the latent graph; use a differentiable decoder")
        return decoded


def affine_decode(latents: Tensor, scale: float, shift: float) -> Tensor:
    # Normalize low-precision states in float32; retain float64 for numerical tests.
    dtype = torch.float64 if latents.dtype == torch.float64 else torch.float32
    return latents.to(dtype) / scale + shift
