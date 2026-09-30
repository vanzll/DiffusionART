"""Differentiable negative-reward objective; reward evaluation is external."""

import math
from collections.abc import Mapping

import torch

from ._validation import same_shape, scalar


def refl_loss(
    rewards: Mapping[str, torch.Tensor],
    component_weights: Mapping[str, float],
    *,
    reward_loss_scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Negative weighted fixed-normalized reward, preserving input gradients.

    Weights are a convex mixture as in the reference. No group centering,
    reward detachment, gradient normalization, or implicit pixel reduction.
    """
    if not rewards or set(rewards) != set(component_weights):
        raise ValueError("rewards and component_weights must have identical nonempty keys")
    weights = {key: scalar(value, f"weight {key}") for key, value in component_weights.items()}
    if not math.isclose(sum(weights.values()), 1.0):
        raise ValueError("component_weights must sum to one")
    scale = scalar(reward_loss_scale, "reward_loss_scale")
    first = next(iter(rewards.values()))
    if first.ndim != 1 or not first.numel():
        raise ValueError("rewards must be nonempty per-item vectors")
    same_shape(first, **rewards)
    if not all(torch.isfinite(value).all() for value in rewards.values()):
        raise ValueError("rewards must be finite")
    weighted = sum(weights[key] * value for key, value in rewards.items())
    return {"per_item_loss": -scale * weighted, "weighted_reward": weighted}
