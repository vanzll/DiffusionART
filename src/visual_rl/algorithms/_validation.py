"""Shape checks shared by the pure tensor kernels."""

import math

import torch


def scalar(value: float, name: str, *, positive: bool = False) -> float:
    value = float(value)
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


def batch(value: torch.Tensor, name: str) -> None:
    if value.ndim < 2 or value.shape[0] == 0 or value.numel() == 0:
        raise ValueError(f"{name} must have nonempty batch and feature dimensions")


def same_shape(reference: torch.Tensor, **values: torch.Tensor) -> None:
    for name, value in values.items():
        if value.shape != reference.shape:
            raise ValueError(f"{name} must have shape {tuple(reference.shape)}")


def vector(value: torch.Tensor, reference: torch.Tensor, name: str) -> torch.Tensor:
    """Accept one value per item, including explicit singleton feature axes."""
    if (
        value.ndim == 0
        or value.numel() != reference.shape[0]
        or value.shape[0] != reference.shape[0]
    ):
        raise ValueError(f"{name} must contain one value per item")
    return value.detach().to(device=reference.device).reshape(-1)


def expand(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return value.reshape((-1,) + (1,) * (reference.ndim - 1))
