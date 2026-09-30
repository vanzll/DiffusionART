"""Prompt-group normalization before combining fixed reward objectives."""

import math
from collections.abc import Sequence
from numbers import Real

import numpy as np
import torch
from torch import Tensor


def _positive_scalar(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{label} must be finite and positive")
    try:
        value = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite and positive") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return value


def combined_group_advantages(
    rewards_by_name: dict[str, Tensor],
    prompts: Sequence[str],
    weights: dict[str, float],
    *,
    eps=1e-4,
    renormalize_combined=False,
) -> tuple[Tensor, dict[str, Tensor], dict[str, float]]:
    """Return detached CPU FP64 combined/component advantages and diagnostics.

    Each [B] or [B, T] reward is centered and population-std normalized within
    identical prompt texts, independently per column, with eps added AFTER std.
    Positive weights are normalized by their sum, unlike scalar reward reporting.
    Optional combined normalization uses global, per-column mean and std.

    Formulas and diagnostic names follow train_sd3.py at canonical 3ca2a36.
    Sorted keys replace historical insertion-order accumulation (including the
    weight sum), so last-bit rounding can differ. Mathematically tied columns
    retain any FP64 mean/std rounding residual, matching canonical arithmetic.
    Historical group_std diagnostics flatten columns; advantages never do.
    Nonfinite inputs or calculated advantages/diagnostics are rejected.
    """
    eps = _positive_scalar(eps, "eps")
    if type(renormalize_combined) is not bool:
        raise ValueError("renormalize_combined must be a bool")
    if not isinstance(rewards_by_name, dict) or not rewards_by_name:
        raise ValueError("rewards_by_name must be a nonempty dict")
    if any(not isinstance(name, str) or not name.strip() for name in rewards_by_name):
        raise ValueError("reward names must be nonempty strings")
    if not isinstance(weights, dict) or set(weights) != set(rewards_by_name):
        raise ValueError("weight keys must match reward keys")
    if (
        isinstance(prompts, (str, bytes))
        or not isinstance(prompts, Sequence)
        or not prompts
        or any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts)
    ):
        raise ValueError("prompts must be a nonempty sequence of nonempty strings")

    names = sorted(rewards_by_name)
    numeric_weights = {name: _positive_scalar(weights[name], "weights") for name in names}
    weight_sum = sum(numeric_weights.values())
    if not math.isfinite(weight_sum):
        raise ValueError("weight sum must be finite")
    normalized_weights = {name: value / weight_sum for name, value in numeric_weights.items()}
    rewards = {}
    shape = None
    for name in names:
        value = rewards_by_name[name]
        if (
            not isinstance(value, Tensor)
            or value.layout != torch.strided
            or value.is_complex()
            or value.dtype == torch.bool
        ):
            raise ValueError("rewards must be real, dense tensors")
        if value.ndim not in (1, 2) or value.numel() == 0 or value.shape[0] != len(prompts):
            raise ValueError("rewards must have nonempty shape [B] or [B, T], one row per prompt")
        if shape is not None and value.shape != shape:
            raise ValueError("all rewards must have matching shapes")
        shape = value.shape
        raw = value.detach().to(device="cpu", dtype=torch.float64).resolve_neg().numpy()
        if not np.isfinite(raw).all():
            raise ValueError("rewards must be finite")
        rewards[name] = raw

    groups = np.asarray(prompts)
    masks = [groups == group for group in np.unique(groups)]
    per_component = {}
    combined = np.zeros(shape, dtype=np.float64)
    for name in names:
        advantages = np.zeros(shape, dtype=np.float64)
        for mask in masks:
            group_rewards = rewards[name][mask]
            mean = np.mean(group_rewards, axis=0, keepdims=True)
            std = np.std(group_rewards, axis=0, keepdims=True)
            advantages[mask] = (group_rewards - mean) / (std + eps)
        per_component[name] = advantages
        combined += normalized_weights[name] * advantages
    if renormalize_combined:
        combined = combined - combined.mean(axis=0, keepdims=True)
        combined = combined / (combined.std(axis=0, keepdims=True) + eps)
    if not np.isfinite(combined).all() or any(
        not np.isfinite(value).all() for value in per_component.values()
    ):
        raise ValueError("calculated advantages must be finite")

    stacked = np.stack([per_component[name] for name in names], axis=0)
    conflict = (stacked > 0).any(axis=0) & (stacked < 0).any(axis=0)
    metrics = {
        "combined_mean": float(np.mean(combined)),
        "combined_std": float(np.std(combined)),
        "combined_abs_mean": float(np.mean(np.abs(combined))),
        "conflict_fraction": float(np.mean(conflict)),
        "all_zero_fraction": float(np.mean(np.all(stacked == 0, axis=0))),
        "renormalized": float(renormalize_combined),
    }
    weighted_component_abs_mean = 0.0
    for name, advantages in per_component.items():
        raw = rewards[name]
        group_stds = np.asarray([np.std(raw[mask]) for mask in masks])
        metrics[f"{name}/mean"] = float(np.mean(advantages))
        metrics[f"{name}/std"] = float(np.std(advantages))
        metrics[f"{name}/abs_mean"] = float(np.mean(np.abs(advantages)))
        metrics[f"{name}/zero_fraction"] = float(np.mean(advantages == 0))
        metrics[f"{name}/weight"] = normalized_weights[name]
        metrics[f"{name}/raw_mean"] = float(np.mean(raw))
        metrics[f"{name}/raw_std"] = float(np.std(raw))
        metrics[f"{name}/group_std_mean"] = float(np.mean(group_stds))
        metrics[f"{name}/group_zero_std_fraction"] = float(np.mean(group_stds <= eps))
        weighted_component_abs_mean += normalized_weights[name] * metrics[f"{name}/abs_mean"]
    metrics["scalarization_retention"] = float(
        np.mean(np.abs(combined)) / max(weighted_component_abs_mean, eps)
    )
    correlations = []
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            left = per_component[left_name].reshape(-1)
            right = per_component[right_name].reshape(-1)
            correlation = (
                float(np.corrcoef(left, right)[0, 1])
                if np.std(left) > eps and np.std(right) > eps
                else 0.0
            )
            metrics[f"correlation/{left_name}__{right_name}"] = correlation
            correlations.append(correlation)
    if correlations:
        metrics["correlation_mean"] = float(np.mean(correlations))
        metrics["correlation_min"] = float(np.min(correlations))
    if any(not math.isfinite(value) for value in metrics.values()):
        raise ValueError("calculated metrics must be finite")
    return (
        torch.from_numpy(combined),
        {name: torch.from_numpy(value) for name, value in per_component.items()},
        metrics,
    )
