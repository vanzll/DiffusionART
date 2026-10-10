"""Frozen-target ART MSE and directional corrective masking."""

import torch

from ._validation import batch, expand, same_shape, scalar, vector


def art_loss(
    current_velocity: torch.Tensor,
    old_velocity: torch.Tensor,
    sigma: torch.Tensor,
    guidance_direction: torch.Tensor,
    guidance_magnitude: torch.Tensor,
    *,
    state: torch.Tensor | None = None,
    ode_velocity_coefficient: torch.Tensor | None = None,
    current_ode_mean: torch.Tensor | None = None,
    old_ode_mean: torch.Tensor | None = None,
    loss_space: str = "velocity_linear_sigma",
    direction_normalization: str = "rms",
    mask_space: str = "velocity",
    velocity_mask_metric: str = "scheduler_reference_metric",
    corrective_mask_enabled: bool = True,
    movement_mask_threshold: float = 1e-5,
    noise_normalized_v_radius: float = 0.02,
    q2_reference_sde_noise_level: float = 0.8,
    q2_reference_kl_threshold: float = 1e-5,
    restore_anchor_mask: torch.Tensor | None = None,
    target_projection_radius: float = 0.0,
    reference_prediction: torch.Tensor | None = None,
    reference_mse_beta: float = 0.0,
    eps: float = 1e-5,
) -> dict[str, torch.Tensor]:
    """Return per-item losses; all data except current predictions are frozen.

    Supports x0, uniform velocity, and sigma-weighted velocity MSE. The mask
    removes large outward moves but permits target-correcting moves. Inactive
    rows train only if explicitly requested as restoring anchors.
    """
    batch(current_velocity, "current_velocity")
    same_shape(current_velocity, old_velocity=old_velocity, guidance_direction=guidance_direction)
    if loss_space not in ("x0", "velocity_uniform", "velocity_linear_sigma"):
        raise ValueError("unsupported ART loss_space")
    if direction_normalization not in ("rms", "none"):
        raise ValueError("direction_normalization must be rms or none")
    if mask_space not in ("velocity", "ode_mean"):
        raise ValueError("mask_space must be velocity or ode_mean")
    metrics = ("fixed_mse", "noise_normalized_v", "q2_sde_kl", "scheduler_reference_metric")
    if velocity_mask_metric not in metrics:
        raise ValueError("unsupported velocity_mask_metric")
    if mask_space == "ode_mean" and velocity_mask_metric != "fixed_mse":
        raise ValueError("ode_mean masking requires fixed_mse")
    eps = scalar(eps, "eps", positive=True)
    threshold = scalar(movement_mask_threshold, "movement_mask_threshold")
    radius = scalar(noise_normalized_v_radius, "noise_normalized_v_radius", positive=True)
    noise = scalar(q2_reference_sde_noise_level, "q2_reference_sde_noise_level", positive=True)
    kl_threshold = scalar(q2_reference_kl_threshold, "q2_reference_kl_threshold")
    projection = scalar(target_projection_radius, "target_projection_radius")
    reference_beta = scalar(reference_mse_beta, "reference_mse_beta")
    if projection and loss_space != "x0":
        raise ValueError("target projection is supported only in x0 space")
    v = current_velocity.float()
    old = old_velocity.detach().to(v)
    s = vector(sigma, v, "sigma").float()
    magnitude = vector(guidance_magnitude, v, "guidance_magnitude").float()
    if not torch.isfinite(s).all() or ((s < 0) | (s > 1)).any():
        raise ValueError("sigma must be finite in [0, 1]")
    if not torch.isfinite(magnitude).all() or (magnitude < 0).any():
        raise ValueError("guidance_magnitude must be finite and nonnegative")
    raw = guidance_direction.detach().to(v)
    raw_rms = raw.square().flatten(1).mean(1).sqrt()
    direction = raw / expand(raw_rms.clamp_min(eps), v) if direction_normalization == "rms" else raw
    displacement = expand(magnitude, v) * direction
    weight = s if loss_space == "velocity_linear_sigma" else torch.ones_like(s)
    if loss_space == "x0":
        if state is None:
            raise ValueError("x0 loss requires state")
        same_shape(v, state=state)
        x = state.detach().to(v)
        if projection:
            displacement = displacement * expand(
                (
                    projection / displacement.square().flatten(1).mean(1).sqrt().clamp_min(eps)
                ).clamp_max(1),
                v,
            )
        target = (x - expand(s, v) * old + displacement).detach()
        residual = target - (x - expand(s, v) * v)
        desired_step = -expand(s, v) * residual.detach()
    else:
        target = (old - displacement).detach()
        residual = target - v
        desired_step = residual.detach()
    unmasked = weight * residual.square().flatten(1).mean(1)
    movement = v - old
    energy = movement.square().flatten(1).mean(1)
    alignment = (movement * desired_step).flatten(1).mean(1)
    if mask_space == "ode_mean" or velocity_mask_metric in (
        "q2_sde_kl",
        "scheduler_reference_metric",
    ):
        if ode_velocity_coefficient is None:
            raise ValueError("this mask requires ode_velocity_coefficient")
        dt = vector(ode_velocity_coefficient, v, "ode_velocity_coefficient").float()
        if not torch.isfinite(dt).all() or (dt >= 0).any():
            raise ValueError("ODE coefficients must be finite and negative")
        if mask_space == "ode_mean":
            if current_ode_mean is None or old_ode_mean is None:
                raise ValueError("ode_mean masking requires current and old ODE means")
            same_shape(v, current_ode_mean=current_ode_mean, old_ode_mean=old_ode_mean)
            mean_movement = current_ode_mean.float() - old_ode_mean.detach().to(v)
            energy = mean_movement.square().flatten(1).mean(1)
            alignment = (mean_movement * expand(dt, v) * desired_step).flatten(1).mean(1)
        else:
            initial = torch.isclose(s, torch.ones_like(s), atol=eps, rtol=0)
            std = torch.where(
                initial,
                torch.full_like(s, noise),
                (noise**2 * s / (1 - s).clamp_min(eps) * (-dt)).clamp_min(0).sqrt(),
            )
            coefficient = torch.where(initial, dt, dt * (1 + 0.5 * noise**2))
            energy = coefficient.square() * energy / (2 * std.square().clamp_min(eps**2))
            threshold = kl_threshold
    elif velocity_mask_metric == "noise_normalized_v":
        energy = torch.where(s > 0, energy / torch.where(s > 0, s, 1.0), 0.0)
        threshold = radius**2
    active = (magnitude > 0) & (raw_rms > 0) & (s > 0)
    requested_anchor = (
        torch.zeros_like(active)
        if restore_anchor_mask is None
        else vector(restore_anchor_mask, v, "restore_anchor_mask").bool()
    )
    anchor = requested_anchor & ~active & (s > 0)
    masked = (active & corrective_mask_enabled & (energy >= threshold) & (alignment > 0)).detach()
    keep = (anchor | (active & ~masked)).detach()
    policy = torch.where(keep, unmasked, torch.zeros_like(unmasked))
    reference = torch.zeros_like(policy)
    if reference_prediction is not None:
        same_shape(v, reference_prediction=reference_prediction)
        reference = (v - reference_prediction.detach().to(v)).square().flatten(1).mean(1)
    elif reference_beta:
        raise ValueError("nonzero reference_mse_beta requires reference_prediction")
    return {
        "per_item_loss": policy + reference_beta * reference,
        "per_item_policy_loss": policy,
        "per_item_reference_mse": reference,
        "per_item_unmasked_loss": unmasked,
        "target": target,
        "keep_mask": keep,
        "masked": masked,
        "movement_score": energy.detach(),
        "outward_alignment": alignment.detach(),
    }
