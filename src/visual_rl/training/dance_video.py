"""Model-independent native DanceGRPO video sampling and likelihood replay.

References: DanceGRPO 15cc71d, fastvideo/train_grpo_hunyuan.py::flux_step;
flow_grpo_video_baseline 3ca2a36f, hunyuan_public_dancegrpo.py. This is NOT the
FlowGRPO SDE. Conditioning, model timesteps and autocast belong to ``predict``.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import torch

from ..algorithms import clipped_grpo_loss
from ..sampling import validate_schedule
from .objectives import _indices
from .rollout import Rollout


def dance_inputs(
    *, seed: int, rank: int, attempt: int, prompts_per_rank: int, k: int,
    latent_shape: Sequence[int], pool: int | Sequence[int], count: int, steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Plan complete prompt-major K groups without touching the global RNG.

    Each group draws one BF16 initial latent, K x steps FP32 innovations, the
    unapplied K==bestofn shuffle, then K unordered timestep permutations. Groups
    consume this entire sequence in order on one CPU generator. Thus p=1/K=8
    is bitwise canonical, and adding groups leaves earlier groups unchanged.
    ``pool`` is an ordered sequence of eligible steps, or a prefix length.
    No candidates are pruned or reordered, for any K.
    """
    for name, value in (("seed", seed), ("rank", rank), ("attempt", attempt)):
        if type(value) is not int or (name != "seed" and value < 0):
            raise ValueError(f"{name} must be an integer (rank/attempt nonnegative)")
    for name, value in (("prompts_per_rank", prompts_per_rank), ("k", k),
                        ("steps", steps), ("count", count)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    latent_shape = tuple(latent_shape)
    if not latent_shape or any(type(n) is not int or n < 1 for n in latent_shape):
        raise ValueError("latent_shape must contain positive integer dimensions")
    pool = tuple(range(pool)) if type(pool) is int else tuple(pool)
    if (not pool or len(set(pool)) != len(pool) or count > len(pool)
            or any(type(i) is not int or not 0 <= i < steps - 1 for i in pool)):
        raise ValueError("pool must contain unique nonterminal steps and fit count")
    generator = torch.Generator(device="cpu").manual_seed(
        (seed + 1_000_003 * attempt + 10_007 * rank) % (2**63 - 1)
    )
    eligible = torch.tensor(pool, dtype=torch.int64, device="cpu")
    noises, selections, innovations = [], [], []
    for _ in range(prompts_per_rank):
        noises.append(torch.randn(
            (1, *latent_shape), generator=generator, dtype=torch.bfloat16, device="cpu"
        ).repeat(k, *([1] * len(latent_shape))))
        innovations.append(torch.randn(
            (k, steps, *latent_shape), generator=generator, dtype=torch.float32, device="cpu"
        ))
        # The native equal-K branch consumes this shuffle but does not apply it.
        torch.randperm(k, generator=generator, device="cpu")
        selections.append(torch.stack([
            eligible[torch.randperm(len(pool), generator=generator, device="cpu")[:count]]
            for _ in range(k)
        ]))
    return torch.cat(noises), torch.cat(selections), torch.cat(innovations)


def _dance_transition(sigmas, velocity, steps, state, *, eta, innovation=None, observed=None):
    """Native scalar-sigma operation order, including the BF16 field products."""
    if (innovation is None) == (observed is None):
        raise ValueError("provide exactly one innovation or observed next state")
    samples, predictions, log_probs, means, stds = [], [], [], [], []
    for row, index in enumerate(steps.detach().cpu().tolist()):
        # A broadcast FP32 sigma would upcast BF16 products before rounding.
        sigma = sigmas[index]
        dsigma = sigmas[index + 1] - sigma
        latents, output = state[row:row + 1].float(), velocity[row:row + 1]
        mean = latents + dsigma * output
        prediction = latents - sigma * output
        std = eta * math.sqrt(sigma - sigmas[index + 1])
        score = -(latents - prediction * (1 - sigma)) / sigma**2
        mean = mean + (-0.5 * eta**2 * score) * dsigma
        sample = (mean + innovation[row:row + 1].detach().to(mean) * std
                  if innovation is not None else observed[row:row + 1].detach().to(mean))
        log_prob = (
            -(sample.detach().float() - mean.float())**2 / (2 * std**2)
            - math.log(std)
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi))).to(mean.device)
        ).mean(dim=tuple(range(1, mean.ndim)))
        samples.append(sample)
        predictions.append(prediction)
        log_probs.append(log_prob)
        means.append(mean)
        stds.append(std)
    return (torch.cat(samples), torch.cat(predictions), torch.cat(log_probs),
            torch.cat(means), torch.tensor(stds, dtype=torch.float32, device=state.device))


@torch.no_grad()
def collect_dance_rollout(
    predict: Callable, noise: torch.Tensor, sigmas: torch.Tensor, *,
    row_selected: torch.Tensor, innovations: torch.Tensor, microbatch: int, eta: float = .25,
) -> Rollout:
    """Execute all stochastic steps; store selected transitions and last x0 only.

    ``noise.device`` determines execution device; caches are detached CPU FP32.
    The first prediction sees BF16 noise, subsequent predictions FP32 states.
    ``predict(state, sigma_batch, global_rows)`` owns precision and conditioning
    and must be deterministic and row-separable for microbatch invariance.
    ``selected`` is representative only; use ``step_indices``/``indices``.
    """
    if not callable(predict):
        raise ValueError("predict must be callable")
    if (noise.ndim < 2 or noise.numel() == 0 or noise.dtype != torch.bfloat16
            or not torch.isfinite(noise).all()):
        raise ValueError("noise must be a nonempty finite BF16 [batch, ...] tensor")
    if type(microbatch) is not int or microbatch < 1:
        raise ValueError("microbatch must be positive")
    if isinstance(eta, bool) or not math.isfinite(eta) or eta <= 0:
        raise ValueError("eta must be finite and positive")
    if not sigmas.is_floating_point():
        raise ValueError("sigmas must be floating point")
    schedule = sigmas.detach().to(device="cpu", dtype=torch.float32).clone()
    validate_schedule(schedule)
    batch, steps = len(noise), len(schedule) - 1
    if (row_selected.dtype != torch.int64 or row_selected.ndim != 2
            or row_selected.shape[0] != batch or row_selected.shape[1] == 0
            or (row_selected < 0).any() or (row_selected >= steps - 1).any()
            or any(row.unique().numel() != len(row) for row in row_selected)):
        raise ValueError("row_selected needs distinct nonterminal integer steps per row")
    if (innovations.shape != (batch, steps, *noise.shape[1:])
            or innovations.dtype != torch.float32 or not torch.isfinite(innovations).all()):
        raise ValueError("innovations must be finite FP32 [batch, steps, ...] noise")
    selection = row_selected.detach().cpu().clone()
    slots = selection.shape[1]
    shape = (batch, slots, *noise.shape[1:])
    states, velocities, following, means = [
        torch.empty(shape, dtype=torch.float32, device="cpu") for _ in range(4)
    ]
    log_probs = torch.empty((batch, slots), dtype=torch.float32, device="cpu")
    stds = torch.empty_like(log_probs)
    endpoints = torch.empty(noise.shape, dtype=torch.float32, device="cpu")
    for start in range(0, batch, microbatch):
        end = min(start + microbatch, batch)
        rows = torch.arange(start, end, device=noise.device)
        state = noise[start:end].detach().clone()
        for index in range(steps):
            times = schedule[index].to(state.device).expand(end - start)
            velocity = predict(state, times, rows)
            if (velocity.shape != state.shape or velocity.device != state.device
                    or not velocity.is_floating_point() or not torch.isfinite(velocity).all()):
                raise ValueError("predict must return a finite floating field matching state")
            next_state, prediction, logp, mean, std = _dance_transition(
                schedule, velocity, torch.full((end - start,), index, dtype=torch.int64),
                state, eta=eta, innovation=innovations[start:end, index],
            )
            if not all(torch.isfinite(x).all() for x in (next_state, prediction, logp)):
                raise ValueError("nonfinite Dance rollout transition")
            local, slot = (selection[start:end] == index).nonzero(as_tuple=True)
            if len(local):
                target, source = local + start, local.to(state.device)
                for destination, value in ((states, state), (velocities, velocity),
                                           (following, next_state), (means, mean)):
                    destination[target, slot] = value[source].float().cpu()
                log_probs[target, slot] = logp[source].cpu()
                stds[target, slot] = std[source].cpu()
            state = next_state
        endpoints[start:end].copy_(prediction.cpu())
    return Rollout(
        endpoints=endpoints, selected=tuple(selection[0].tolist()), sigmas=schedule,
        states=states, old_velocities=velocities, next_states=following,
        old_log_probs=log_probs, old_means=means, transition_stds=stds,
        stochastic=tuple(range(steps)), transition_kind="dance", likelihood="gaussian",
        noise_level=float(eta), step_indices=selection,
    )


@dataclass
class DanceObjective:
    """Per-item native replay; caller owns advantage preparation and denominator."""

    cache: Rollout
    predict: Callable
    advantages: torch.Tensor | Mapping[str, torch.Tensor]
    device: torch.device
    options: dict = field(default_factory=lambda: {"ratio_clip": 1e-4, "advantage_clip": 5.0})

    def __post_init__(self):
        if (self.cache.transition_kind != "dance" or self.cache.step_indices is None
                or self.cache.old_log_probs is None):
            raise ValueError("DanceObjective requires a Dance rollout with row-specific steps")
        values = (self.advantages.values() if isinstance(self.advantages, Mapping)
                  else (self.advantages,))
        if any(v.shape != (len(self.cache.endpoints),) for v in values):
            raise ValueError("advantages must contain one frozen value per rollout row")

    def __call__(self, items: Sequence[tuple[int, int]]) -> torch.Tensor:
        rows, slots, steps = _indices(items, self.cache)
        state = self.cache.states[rows, slots].detach().to(self.device)
        velocity = self.predict(
            state, self.cache.sigmas[steps].to(self.device), rows.to(self.device)
        )
        _, _, log_prob, _, _ = _dance_transition(
            self.cache.sigmas, velocity, steps, state, eta=self.cache.noise_level,
            observed=self.cache.next_states[rows, slots].detach().to(self.device),
        )
        old_log_prob = self.cache.old_log_probs[rows, slots].detach().to(self.device)
        if isinstance(self.advantages, Mapping):
            advantage = {k: v.detach()[rows.to(v.device)].to(self.device)
                         for k, v in self.advantages.items()}
        else:
            advantage = self.advantages.detach()[rows.to(self.advantages.device)].to(self.device)
        return clipped_grpo_loss(
            log_prob, old_log_prob, advantage, **self.options,
        )["per_item_loss"]
