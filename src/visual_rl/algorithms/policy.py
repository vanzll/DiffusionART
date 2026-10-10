"""Clipped GRPO and FlowDPPO transition objectives."""

from collections.abc import Mapping

import torch

from ._validation import batch, expand, same_shape, scalar, vector


def _log_ratio(log_prob: torch.Tensor, old_log_prob: torch.Tensor) -> torch.Tensor:
    if log_prob.ndim != 1 or not log_prob.numel():
        raise ValueError(
            "log_prob must be a nonempty per-item vector, not per-element log densities"
        )
    same_shape(log_prob, old_log_prob=old_log_prob)
    return log_prob - old_log_prob.detach().to(log_prob)


def clipped_grpo_loss(
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor | Mapping[str, torch.Tensor],
    *,
    ratio_clip: float,
    advantage_clip: float | None = 5.0,
    component_weights: Mapping[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    """max(-A*r, -A*clip(r)); component losses are clipped BEFORE mixing.

    ratio_clip is intentionally required: pinned DanceGRPO uses 1e-4 and
    pinned Fast-SDE uses 1e-5. Advantages must already be globally prepared.
    """
    delta = _log_ratio(log_prob, old_log_prob)
    clip = scalar(ratio_clip, "ratio_clip")
    if clip >= 1:
        raise ValueError("ratio_clip must be less than one")
    if advantage_clip is not None:
        scalar(advantage_clip, "advantage_clip", positive=True)
    ratio = delta.exp()
    clipped = ratio.clamp(1 - clip, 1 + clip)

    def term(values):
        advantage = vector(values, ratio, "advantages").to(ratio)
        if advantage_clip is not None:
            advantage = advantage.clamp(-advantage_clip, advantage_clip)
        return torch.maximum(-advantage * ratio, -advantage * clipped)

    components = {}
    if isinstance(advantages, Mapping):
        if not advantages or component_weights is None or set(advantages) != set(component_weights):
            raise ValueError("component advantages and weights must have identical nonempty keys")
        components = {key: term(value) for key, value in advantages.items()}
        policy = sum(
            scalar(component_weights[key], f"weight {key}") * value
            for key, value in components.items()
        )
    else:
        if component_weights is not None:
            raise ValueError("component_weights require component advantages")
        policy = term(advantages)
    return {
        "per_item_loss": policy,
        "ratio": ratio,
        "clip_fraction": (torch.abs(ratio - 1) > clip).float().detach(),
        **{f"component_loss/{key}": value for key, value in components.items()},
    }


def transition_kl_per_sample(
    current_mean: torch.Tensor,
    old_mean: torch.Tensor,
    transition_std: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Feature-mean KL for equal-variance Gaussians, not a feature sum.

    transition_std is the ACTUAL transition std, including sqrt(-dt) when
    required by the sampler. It may be scalar, per-item, or broadcast-shaped.
    """
    batch(current_mean, "current_mean")
    same_shape(current_mean, old_mean=old_mean)
    eps = scalar(eps, "eps", positive=True)
    std = transition_std.detach().to(current_mean).clamp_min(eps)
    if std.ndim == 1 and std.shape == (len(current_mean),):
        std = expand(std, current_mean)
    if torch.broadcast_shapes(current_mean.shape, std.shape) != current_mean.shape:
        raise ValueError("transition_std must broadcast to current_mean without adding axes")
    kl = (
        current_mean.float() - old_mean.detach().to(device=current_mean.device).float()
    ).square() / (2 * std.float().square())
    return kl.flatten(1).mean(1)


def flowdppo_loss(
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    current_mean: torch.Tensor,
    old_mean: torch.Tensor,
    transition_std: torch.Tensor,
    advantages: torch.Tensor,
    *,
    kl_mask_threshold: float = 1e-5,
) -> dict[str, torch.Tensor]:
    """Reference KL-ADV mask; no PPO ratio clipping or advantage clipping."""
    delta = _log_ratio(log_prob, old_log_prob)
    if len(current_mean) != len(log_prob):
        raise ValueError("transition means and log probabilities must share item count")
    threshold = scalar(kl_mask_threshold, "kl_mask_threshold")
    ratio = delta.exp()
    advantage = vector(advantages, ratio, "advantages").to(ratio)
    kl = transition_kl_per_sample(current_mean, old_mean, transition_std)
    high = kl >= threshold
    removed = high & (((ratio > 1) & (advantage > 0)) | ((ratio < 1) & (advantage < 0)))
    keep = (~removed).detach()
    policy = torch.where(keep, -advantage * ratio, torch.zeros_like(ratio))
    return {"per_item_loss": policy, "ratio": ratio, "keep_mask": keep, "kl": kl.detach()}
