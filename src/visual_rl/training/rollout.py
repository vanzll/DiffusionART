"""Model-independent rollout capture with explicit stochasticity and replay data."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch

from ..sampling import flow_transition, validate_schedule


@dataclass(frozen=True)
class Rollout:
    endpoints: torch.Tensor
    selected: tuple[int, ...]
    sigmas: torch.Tensor
    states: torch.Tensor
    old_velocities: torch.Tensor
    next_states: torch.Tensor
    old_log_probs: torch.Tensor | None
    old_means: torch.Tensor
    transition_stds: torch.Tensor
    stochastic: tuple[int, ...]
    transition_kind: str
    likelihood: str
    noise_level: float
    step_indices: torch.Tensor | None = None

    def slot(self, timestep: int) -> int:
        if self.step_indices is not None:
            raise ValueError("row-specific selections have no global timestep slot")
        return self.selected.index(timestep)

    def indices(self, rows, slots):
        if self.step_indices is not None:
            return self.step_indices[rows.cpu(), slots.cpu()]
        return torch.tensor(self.selected)[slots.cpu()]


@torch.no_grad()
def collect_rollout(
    predict: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    noise: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    selected: Sequence[int],
    stochastic: Sequence[int] = (),
    innovations: Mapping[int, torch.Tensor] | None = None,
    transition_kind: str = "sde",
    likelihood: str = "gaussian",
    noise_level: float = 0.7,
    microbatch: int = 8,
    preserve_state_dtype: bool = False,
    unselected_kind: str = "ode",
    row_selected: torch.Tensor | None = None,
    transition_fn: Callable = flow_transition,
    endpoint_kind: str = "sample",
) -> Rollout:
    """``predict(state, sigma_batch, row_indices)`` owns model conditioning.

    Inputs and innovations are fixed before microbatching. Cache values are
    detached CPU copies. No extra sampling call is used to calculate old-policy
    predictions on trajectory states; NFT re-noised states are a different API.

    ``preserve_state_dtype`` reproduces FLUX's post-transition latent cast.
    Old log-probs are computed before that cast, while replay observations and
    next model inputs are rounded. This historical convention need not yield
    exactly unit initial ratios; it is not an exact quantized likelihood.
    """
    validate_schedule(sigmas)
    if noise.ndim < 2 or not noise.is_floating_point() or not torch.isfinite(noise).all():
        raise ValueError("noise must be a finite [batch, ...] tensor")
    if type(microbatch) is not int or microbatch < 1 or len(noise) < 1:
        raise ValueError("microbatch and batch size must be positive")
    count = len(sigmas) - 1
    selected, stochastic = tuple(selected), tuple(stochastic)
    if row_selected is not None:
        if (row_selected.dtype != torch.int64 or row_selected.shape != (len(noise), len(selected))
                or row_selected.numel() == 0 or (row_selected < 0).any() or (row_selected >= count).any()
                or any(len(row.unique()) != len(selected) for row in row_selected)):
            raise ValueError("row_selected must contain distinct valid steps for each trajectory")
        row_selected = row_selected.detach().cpu().clone()
        if stochastic and not set(row_selected.flatten().tolist()) <= set(stochastic):
            raise ValueError("row selections must be stochastic for likelihood replay")
    if type(preserve_state_dtype) is not bool:
        raise ValueError("preserve_state_dtype must be boolean")
    if endpoint_kind not in ("sample", "last_x0") or (
        endpoint_kind == "last_x0" and transition_kind != "dance"
    ):
        raise ValueError("last_x0 endpoint selection is Dance-only")
    if unselected_kind not in ("ode", "cps_zero_noise"):
        raise ValueError("invalid deterministic unselected transition")
    for values in (selected, stochastic):
        if len(set(values)) != len(values) or any(
            type(i) is not int or not 0 <= i < count for i in values
        ):
            raise ValueError("timesteps must be unique valid integer indices")
    if not selected:
        raise ValueError("select at least one state to cache")
    if stochastic and not set(selected) <= set(stochastic):
        raise ValueError("likelihood replay can select only stochastic transitions")
    innovations = {} if innovations is None else innovations
    if set(innovations) != set(stochastic):
        raise ValueError("provide exactly one fixed innovation tensor per stochastic step")
    for value in innovations.values():
        if value.shape != noise.shape or not torch.isfinite(value).all():
            raise ValueError("innovation shape/finiteness mismatch")
    batch, slots = len(noise), len(selected)
    shape = (batch, slots, *noise.shape[1:])
    states, velocities, following, means = [
        torch.empty(shape, dtype=torch.float32) for _ in range(4)
    ]
    stds = torch.empty((batch, slots), dtype=torch.float32)
    log_probs = torch.empty((batch, slots), dtype=torch.float32) if stochastic else None
    endpoints = torch.empty_like(noise, device="cpu", dtype=torch.float32)
    lookup = {step: slot for slot, step in enumerate(selected)}
    for start in range(0, batch, microbatch):
        end = min(start + microbatch, batch)
        rows = torch.arange(start, end, device=noise.device)
        state = noise[start:end].detach().clone()
        for index in range(count):
            t = sigmas[index].to(device=state.device).expand(end - start)
            velocity = predict(state, t, rows)
            if endpoint_kind == "last_x0" and index == count - 1:
                final_endpoint = state.float() - t.reshape((-1,) + (1,) * (state.ndim - 1)) * velocity.float()
            kind = transition_kind if index in stochastic else unselected_kind
            result = transition_fn(
                state,
                velocity,
                t,
                sigmas[index + 1],
                kind=kind,
                noise_level=noise_level,
                sigma_max=float(sigmas[1]),
                likelihood=likelihood,
                innovation=innovations[index][start:end].to(state.device)
                if index in stochastic
                else None,
            )
            next_state = result.sample.to(noise.dtype) if preserve_state_dtype else result.sample
            if index in lookup or row_selected is not None:
                if row_selected is None:
                    local_rows = torch.arange(end - start)
                    slots_at_step = torch.full_like(local_rows, lookup[index])
                else:
                    local_rows, slots_at_step = (row_selected[start:end] == index).nonzero(as_tuple=True)
                if len(local_rows):
                    rows_at_step = local_rows + start
                    source_rows = local_rows.to(state.device)
                    states[rows_at_step, slots_at_step] = state[source_rows].float().cpu()
                    velocities[rows_at_step, slots_at_step] = velocity[source_rows].float().cpu()
                    following[rows_at_step, slots_at_step] = next_state[source_rows].float().cpu()
                    means[rows_at_step, slots_at_step] = result.mean[source_rows].float().cpu()
                    stds[rows_at_step, slots_at_step] = result.std.flatten().expand(end - start).cpu()[local_rows]
                    if log_probs is not None:
                        log_probs[rows_at_step, slots_at_step] = result.log_prob.cpu()[local_rows]
            state = next_state
        endpoints[start:end].copy_((final_endpoint if endpoint_kind == "last_x0" else state).cpu())
    return Rollout(
        endpoints,
        selected,
        sigmas.detach().float().cpu().clone(),
        states,
        velocities,
        following,
        log_probs,
        means,
        stds,
        stochastic,
        transition_kind,
        likelihood,
        noise_level,
        step_indices=row_selected,
    )


def replay_transition(
    cache: Rollout, velocity: torch.Tensor, rows: torch.Tensor, slots: torch.Tensor
):
    if cache.old_log_probs is None:
        raise ValueError("ODE trajectories have no GRPO likelihood")
    if rows.ndim != 1 or slots.shape != rows.shape or len(rows) != len(velocity):
        raise ValueError("rows and slots must match the prediction batch")
    rows, slots = rows.cpu(), slots.cpu()
    if (
        (rows < 0).any()
        or (rows >= len(cache.endpoints)).any()
        or (slots < 0).any()
        or (slots >= len(cache.selected)).any()
    ):
        raise ValueError("replay index out of range")
    indices = cache.indices(rows, slots)
    return flow_transition(
        cache.states[rows, slots].to(velocity.device),
        velocity,
        cache.sigmas[indices],
        cache.sigmas[indices + 1],
        kind=cache.transition_kind,
        noise_level=cache.noise_level,
        sigma_max=float(cache.sigmas[1]),
        likelihood=cache.likelihood,
        observed=cache.next_states[rows, slots].to(velocity.device),
    )


@torch.no_grad()
def endpoint_renoising(endpoints: torch.Tensor, sigmas: torch.Tensor, noise: torch.Tensor):
    """NFT training states are re-noised endpoints, not cached ODE states."""
    if endpoints.shape != noise.shape or endpoints.ndim < 2:
        raise ValueError("endpoints and noise must have equal [batch, ...] shapes")
    if sigmas.shape != (len(endpoints),) or not ((sigmas > 0) & (sigmas <= 1)).all():
        raise ValueError("one finite sigma in (0, 1] is required per training item")
    if not torch.isfinite(endpoints).all() or not torch.isfinite(noise).all():
        raise ValueError("nonfinite NFT training input")
    t = sigmas.detach().to(endpoints.device).float().reshape((-1,) + (1,) * (endpoints.ndim - 1))
    return (1 - t) * endpoints.detach().float() + t * noise.detach().to(endpoints.device).float()
