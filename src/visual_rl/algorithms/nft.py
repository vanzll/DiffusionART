"""DiffusionNFT mirror objective, without model or optimizer ownership."""

import math

import torch

from ._validation import batch, expand, same_shape, scalar, vector


def old_policy_decay(step: int, mode: str, rate: float = 0.00125, maximum: float = 0.6) -> float:
    """Zero-based completed-update index; refresh 0 copies the current policy."""
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step must be a nonnegative integer")
    if mode not in ("no-ema", "slow-ema"):
        raise ValueError("unknown old-policy mode")
    if not math.isfinite(rate) or rate < 0 or not math.isfinite(maximum) or not 0 <= maximum < 1:
        raise ValueError("invalid old-policy EMA schedule")
    return 0.0 if mode == "no-ema" else min(step * rate, maximum)


def nft_loss(
    prediction: torch.Tensor,
    old_prediction: torch.Tensor,
    x0: torch.Tensor,
    xt: torch.Tensor,
    sigma: torch.Tensor,
    advantages: torch.Tensor,
    *,
    nft_beta: float = 1.0,
    adv_clip_max: float = 5.0,
    adv_mode: str = "all",
    reference_prediction: torch.Tensor | None = None,
    reference_mse_beta: float = 0.0,
    eps: float = 1e-5,
) -> dict[str, torch.Tensor]:
    """Mirror/standard/normalized_x0 only; no batch loss reduction.

    Each branch divides its squared endpoint residual by its own detached
    float64 mean absolute residual. The advantage scale factor is retained.
    """
    batch(prediction, "prediction")
    same_shape(prediction, old_prediction=old_prediction, x0=x0, xt=xt)
    if not math.isfinite(nft_beta) or nft_beta == 0:
        raise ValueError("nft_beta must be finite and nonzero")
    clip = scalar(adv_clip_max, "adv_clip_max", positive=True)
    eps = scalar(eps, "eps", positive=True)
    ref_beta = scalar(reference_mse_beta, "reference_mse_beta")
    if adv_mode not in ("all", "positive_only", "negative_only", "one_only", "binary"):
        raise ValueError("unsupported adv_mode")
    advantage = vector(advantages, prediction, "advantages").clamp(-clip, clip)
    if adv_mode == "positive_only":
        advantage = advantage.clamp_min(0)
    elif adv_mode == "negative_only":
        advantage = advantage.clamp_max(0)
    elif adv_mode == "one_only":
        advantage = (advantage > 0).to(advantage.dtype)
    elif adv_mode == "binary":
        advantage = advantage.sign()
    probability = ((advantage / clip) / 2 + 0.5).clamp(0, 1)
    old = old_prediction.detach().to(prediction)
    t = expand(vector(sigma, prediction, "sigma"), prediction)
    endpoint = x0.detach().to(device=prediction.device)
    state = xt.detach().to(device=prediction.device)
    positive_delta = state - t * (nft_beta * prediction + (1 - nft_beta) * old) - endpoint
    negative_delta = state - t * ((1 + nft_beta) * old - nft_beta * prediction) - endpoint
    dims = tuple(range(1, prediction.ndim))
    positive_weight = positive_delta.detach().double().abs().mean(dims, keepdim=True).clamp_min(eps)
    negative_weight = negative_delta.detach().double().abs().mean(dims, keepdim=True).clamp_min(eps)
    positive = (positive_delta.square() / positive_weight).mean(dims)
    negative = (negative_delta.square() / negative_weight).mean(dims)
    policy = (probability * positive / nft_beta + (1 - probability) * negative / nft_beta) * clip
    reference = torch.zeros_like(policy)
    if reference_prediction is not None:
        same_shape(prediction, reference_prediction=reference_prediction)
        reference = (
            (prediction - reference_prediction.detach().to(device=prediction.device))
            .square()
            .mean(dims)
        )
    elif ref_beta:
        raise ValueError("nonzero reference_mse_beta requires reference_prediction")
    return {
        "per_item_loss": policy + ref_beta * reference,
        "per_item_policy_loss": policy,
        "per_item_reference_mse": reference,
        "positive_loss": positive,
        "negative_loss": negative,
        "positive_weight": positive_weight.flatten(),
        "negative_weight": negative_weight.flatten(),
        "positive_probability": probability,
    }
