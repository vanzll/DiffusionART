"""Native T2V trajectories and explicit conditional affine ART replay.

No sampler is replaced by Euler. The optional ART contract is named explicitly:
velocity MSE is the cancellation-free evaluation of native mean MSE / a**2,
with canonical sigma weights/restoration and the canonical Q2 REFERENCE mask.
The Q2 reference metric is not a KL for either native transition kernel.
"""

import json
import math
from dataclasses import dataclass, field

import torch

from ..algorithms import art_loss
from ..models.ltx_video_sampling import (
    draw_ltx_transition_innovation,
    ltx_video_schedule,
    ltx_video_stochastic_transition,
)
from ..models.wan_t2v_schedule import (
    capture_native_unipc,
    native_unipc_velocity_coefficient,
    native_wan_schedule,
    replay_native_unipc,
)

NATIVE_AFFINE_ART_CONTRACT = "native_affine_normalized_mean_canonical_q2_reference_v1"


class NativeAffineAuditError(ValueError):
    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        super().__init__(
            "native frozen-history affine replay audit failed: "
            + json.dumps(diagnostics, sort_keys=True, allow_nan=False)
        )


def audit_native_unipc_affine(
    snapshot, velocity, mean, coefficient, *, index, sigma, scheduler_factory
):
    """Distinguish exact frozen-history reconstruction from finite-precision affinity.

    Retain the caller's native autocast scope and the existing3e-5 gate.
    Replay the unchanged velocity only after failure, to classify the error
    without adding work to successful production audits.
    """
    delta = torch.full_like(velocity, 0.125)
    perturbed = replay_native_unipc(snapshot, velocity + delta, scheduler_factory=scheduler_factory)
    expected = mean.float() + coefficient.to(velocity.device) * delta.float()
    error = float((perturbed.float() - expected).abs().max())
    scale = float(expected.abs().max().clamp_min(1))
    enabled = torch.is_autocast_enabled(velocity.device.type)
    diagnostics = {
        "index": index,
        "sigma": float(sigma),
        "coefficient": float(coefficient),
        "max_abs_error": error,
        "scale": scale,
        "relative_max_error": error / scale,
        "threshold": 3e-5,
        "perturbation": 0.125,
        "velocity_dtype": str(velocity.dtype),
        "mean_dtype": str(mean.dtype),
        "snapshot_state_dtype": str(snapshot.state.dtype),
        "device": str(velocity.device),
        "snapshot_step_index": snapshot.history.get("_step_index"),
        "previous_order": snapshot.history.get("this_order"),
        "history_dtypes": [
            None if x is None else str(x.dtype) for x in snapshot.history["model_outputs"]
        ],
        "autocast_enabled": enabled,
        "autocast_dtype": str(torch.get_autocast_dtype(velocity.device.type)) if enabled else None,
        "restore_policy": "sigmas_on_cpu; tensor_history/state_to_velocity_device; preserve_dtype",
    }
    if not torch.isfinite(perturbed).all() or error > 3e-5 * scale:
        restored = replay_native_unipc(snapshot, velocity, scheduler_factory=scheduler_factory)
        identity_error = float((restored.float() - mean.float()).abs().max())
        identity_scale = float(mean.float().abs().max().clamp_min(1))
        diagnostics.update(identity_max_abs_error=identity_error, identity_scale=identity_scale)
        diagnostics["failure_kind"] = (
            "snapshot_identity"
            if not torch.isfinite(restored).all() or identity_error > 3e-5 * identity_scale
            else "finite_precision_affinity"
        )
        diagnostics = {
            key: str(value) if isinstance(value, float) and not math.isfinite(value) else value
            for key, value in diagnostics.items()
        }
        raise NativeAffineAuditError(diagnostics)
    return diagnostics


@dataclass(frozen=True)
class NativeVideoRollout:
    """Complete detached CPU trajectory, even when Flash trains two positions.

    For UniPC, old_mean and a encode the frozen history's affine offset exactly
    at the cached old prediction; native history is never recomputed from the
    updated policy. This avoids retaining redundant multi-vector histories at
    every slot. Sample/mean/innovation remain distinct for stochastic LTX.
    """

    endpoints: torch.Tensor
    sigmas: torch.Tensor
    timesteps: torch.Tensor
    states: torch.Tensor
    old_velocities: torch.Tensor
    old_means: torch.Tensor
    next_states: torch.Tensor
    velocity_coefficients: torch.Tensor
    transition_stds: torch.Tensor
    innovations: torch.Tensor | None
    selected: tuple[int, ...]
    kind: str
    history_audit_max_error: float = 0.0

    @property
    def eligible(self):
        return tuple(
            torch.where((self.sigmas[:-1] > 0) & (self.velocity_coefficients != 0))[0].tolist()
        )

    def __post_init__(self):
        n, slots = self.states.shape[:2]
        expected = (n, slots, *self.endpoints.shape[1:])
        if (
            self.states.shape != expected
            or self.old_velocities.shape != expected
            or self.old_means.shape != expected
            or self.next_states.shape != expected
            or self.sigmas.shape != (slots + 1,)
            or self.timesteps.shape != (slots,)
            or self.velocity_coefficients.shape != (slots,)
            or self.transition_stds.shape != (slots,)
            or self.innovations is not None
            and self.innovations.shape != expected
        ):
            raise ValueError("native trajectory geometry is inconsistent")
        if self.kind not in ("wan_unipc_order2", "ltx_rf_stochastic"):
            raise ValueError("unrecognized native transition/cache branch")
        if (
            not self.selected
            or tuple(sorted(set(self.selected))) != self.selected
            or not set(self.selected) <= set(self.eligible)
        ):
            raise ValueError("selected positions must be sorted unique eligible native transitions")
        for name in (
            "endpoints",
            "sigmas",
            "states",
            "old_velocities",
            "old_means",
            "next_states",
            "velocity_coefficients",
            "transition_stds",
        ):
            value = getattr(self, name)
            if value.device.type != "cpu" or value.requires_grad or not torch.isfinite(value).all():
                raise ValueError(f"native cache {name} must be detached finite CPU data")
        if (
            self.sigmas[-1] != 0
            or not (self.sigmas[:-1] > self.sigmas[1:]).all()
            or (self.transition_stds < 0).any()
        ):
            raise ValueError("invalid native physical schedule")
        if self.kind == "ltx_rf_stochastic" and self.innovations is None:
            raise ValueError("stochastic native cache requires every actual innovation")
        if self.innovations is not None and not torch.isfinite(self.innovations).all():
            raise ValueError("nonfinite native innovations")

    def conditional_mean(self, velocity, rows, steps):
        """Actual native conditional mean with history and noise-independent cache frozen."""
        old = self.old_velocities[rows, steps].to(velocity)
        mean = self.old_means[rows, steps].to(device=velocity.device, dtype=torch.float32)
        a = (
            self.velocity_coefficients[steps]
            .to(velocity.device)
            .reshape((-1,) + (1,) * (velocity.ndim - 1))
        )
        return mean + a * (velocity.float() - old.float())


@torch.no_grad()
def collect_native_video(
    predict,
    noise,
    *,
    model,
    selected,
    microbatch=8,
    scheduler_factory=None,
    audit_history=True,
    progress=None,
):
    """Collect full native trajectory; predict(state, physical_sigma, rows, index).

    Noise is supplied, never redrawn. LTX draws innovations in native packed
    layout using the current global RNG, including its zero-weight last draw.
    Fixed-input parity must use the same physical microbatch and RNG stream.
    """
    if (
        noise.ndim != 5
        or not noise.is_floating_point()
        or not len(noise)
        or not torch.isfinite(noise).all()
        or type(microbatch) is not int
        or microbatch < 1
    ):
        raise ValueError("require finite nonempty video noise and positive physical microbatch")
    if model == "wan_t2v_1_3b":
        schedule = native_wan_schedule()
        sigmas, timesteps = schedule.sigmas, schedule.timesteps
        if scheduler_factory is None:
            raise ValueError("Wan native UniPC factory is required")
        coefficients = torch.stack(
            [
                native_unipc_velocity_coefficient(sigmas, i, previous_order=1 if i == 1 else 2)
                for i in range(len(timesteps))
            ]
        )
        stds = torch.zeros_like(coefficients)
        kind = "wan_unipc_order2"
    elif model == "ltx_video_2b_distilled":
        sigmas = ltx_video_schedule()
        timesteps = sigmas[:-1].clone()
        coefficients, stds = -sigmas[:-1] * (1 - sigmas[1:]), sigmas[1:].clone()
        kind = "ltx_rf_stochastic"
    else:
        raise ValueError("unsupported small T2V model")
    slots = len(timesteps)
    selected = tuple(selected)
    if (
        not selected
        or tuple(sorted(set(selected))) != selected
        or any(type(i) is not int or not 0 <= i < slots for i in selected)
    ):
        raise ValueError("invalid native selected positions")
    shape = (len(noise), slots, *noise.shape[1:])
    states, old, means, following = [torch.empty(shape, dtype=torch.float32) for _ in range(4)]
    innovations = torch.empty(shape, dtype=torch.float32) if kind == "ltx_rf_stochastic" else None
    endpoints = torch.empty_like(noise, device="cpu", dtype=torch.float32)
    max_error = 0.0
    for start in range(0, len(noise), microbatch):
        stop = min(start + microbatch, len(noise))
        state = noise[start:stop].clone()
        rows = torch.arange(start, stop, device=state.device)
        if kind == "wan_unipc_order2":
            scheduler = scheduler_factory(
                num_train_timesteps=1000, shift=1, use_dynamic_shifting=False
            )
            scheduler.set_timesteps(50, device=state.device, shift=8)
            if not torch.equal(scheduler.timesteps.cpu(), timesteps) or not torch.equal(
                scheduler.sigmas.cpu(), sigmas
            ):
                raise ValueError("native scheduler differs from the frozen physical/time schedule")
        for index in range(slots):
            velocity = predict(
                state, sigmas[index].to(state.device).expand(len(state)), rows, index
            )
            if velocity.shape != state.shape or not torch.isfinite(velocity).all():
                raise ValueError("nonfinite or wrong-shape native policy velocity")
            snapshot = (
                capture_native_unipc(scheduler, state, timesteps[index])
                if kind == "wan_unipc_order2" and audit_history and start == 0
                else None
            )
            if kind == "wan_unipc_order2":
                mean = scheduler.step(
                    velocity, timesteps[index].to(state.device), state, return_dict=False
                )[0]
                next_state = mean
                if snapshot is not None:
                    try:
                        diagnostics = audit_native_unipc_affine(
                            snapshot,
                            velocity,
                            mean,
                            coefficients[index],
                            index=index,
                            sigma=sigmas[index],
                            scheduler_factory=scheduler_factory,
                        )
                    except NativeAffineAuditError as exc:
                        if progress is not None:
                            progress(
                                native_position=index,
                                microbatch_start=start,
                                microbatch_stop=stop,
                                affine_audit_failure=exc.diagnostics,
                            )
                        raise
                    max_error = max(max_error, diagnostics["relative_max_error"])
            else:
                innovation = draw_ltx_transition_innovation(state)
                transition = ltx_video_stochastic_transition(
                    state, velocity, sigmas[index], sigmas[index + 1], innovation=innovation
                )
                mean, next_state = transition.mean, transition.sample
                innovations[start:stop, index].copy_(innovation.cpu())
            if not torch.isfinite(next_state).all():
                raise ValueError("nonfinite native trajectory; no clipping or skipped transition")
            for target, source in (
                (states, state),
                (old, velocity),
                (means, mean),
                (following, next_state),
            ):
                target[start:stop, index].copy_(source.detach().float().cpu())
            state = next_state
            if progress is not None:
                progress(
                    native_position=index, microbatch_start=start, microbatch_stop=stop, model=model
                )
        endpoints[start:stop].copy_(state.float().cpu())
    return NativeVideoRollout(
        endpoints,
        sigmas,
        timesteps,
        states,
        old,
        means,
        following,
        coefficients,
        stds,
        innovations,
        selected,
        kind,
        max_error,
    )


@dataclass
class NativeARTObjective:
    """Explicit opt-in algebraic contract; no native-kernel KL claim.

    This implementation evaluates velocity residuals directly instead of
    subtracting nearly equal native means and dividing by small native a.
    This equals native normalized mean regression only when the native
    frozen-history affine audit passes. Native CUDA AMP UniPC can fail this
    condition; no equivalence follows from CPU algebra alone. Training launch authorization
    is separate from this pure objective's implementation and CPU proofs.
    """

    cache: NativeVideoRollout
    predict: object
    row_directions: torch.Tensor
    row_magnitudes: torch.Tensor
    row_anchors: torch.Tensor
    device: torch.device
    contract: str
    options: dict = field(default_factory=dict)
    observations: list = field(default_factory=list)

    def __post_init__(self):
        if self.contract != NATIVE_AFFINE_ART_CONTRACT:
            raise ValueError(
                "native replay requires an explicit normalized-mean/Q2-reference contract"
            )
        if (
            self.options.get("mask_space", "velocity") != "velocity"
            or self.options.get("velocity_mask_metric", "scheduler_reference_metric")
            != "scheduler_reference_metric"
            or self.options.get("loss_space", "velocity_linear_sigma") != "velocity_linear_sigma"
            or self.options.get("reference_mse_beta", 0) != 0
        ):
            raise ValueError(
                "native contract freezes canonical sigma-weighted velocity/Q2 mask and KL0"
            )
        n = len(self.cache.endpoints)
        if (
            self.row_directions.shape != self.cache.endpoints.shape
            or self.row_magnitudes.shape != (n,)
            or self.row_anchors.shape != (n,)
        ):
            raise ValueError("native ART guidance/membership shape differs")

    @property
    def has_guidance(self):
        return bool(
            (
                (self.row_magnitudes > 0)
                & (self.row_directions.flatten(1).float().square().mean(1) > 0)
            ).any()
        )

    @property
    def has_training_signal(self):
        return self.has_guidance or bool(self.row_anchors.any())

    def __call__(self, items):
        indices = torch.tensor(items, dtype=torch.int64)
        if indices.ndim != 2 or indices.shape[1] != 2 or not len(indices):
            raise ValueError("require native (row,selected-slot) items")
        rows, slots = indices.unbind(1)
        if (
            (rows < 0).any()
            or (rows >= len(self.cache.endpoints)).any()
            or (slots < 0).any()
            or (slots >= len(self.cache.selected)).any()
        ):
            raise ValueError("native replay item outside the frozen trajectory")
        steps = torch.tensor(self.cache.selected)[slots]
        if len(steps.unique()) != 1:
            raise ValueError("batch native replay by one position; no per-row transformer loop")
        state = self.cache.states[rows, steps].to(self.device)
        sigma = self.cache.sigmas[steps].to(self.device)
        velocity = self.predict(state, sigma, rows.to(self.device), int(steps[0]))
        old = self.cache.old_velocities[rows, steps].to(self.device)
        terms = art_loss(
            velocity,
            old,
            sigma,
            self.row_directions[rows].to(self.device),
            self.row_magnitudes[rows].to(self.device),
            state=state,
            ode_velocity_coefficient=(self.cache.sigmas[steps + 1] - self.cache.sigmas[steps]).to(
                self.device
            ),
            restore_anchor_mask=self.row_anchors[rows].to(self.device),
            **self.options,
        )
        self.observations.append(
            {
                "items": len(items),
                "position": int(steps[0]),
                "masked": int(terms["masked"].sum()),
                "kept": int(terms["keep_mask"].sum()),
                "loss_sum": float(terms["per_item_loss"].detach().double().sum()),
                "movement_score_sum": float(terms["movement_score"].double().sum()),
            }
        )
        return terms["per_item_loss"]


def native_window_items(rows, selected_count):
    """Position-major batches preserve a real multi-sample transformer forward."""
    return tuple((int(row), slot) for slot in range(selected_count) for row in rows)
