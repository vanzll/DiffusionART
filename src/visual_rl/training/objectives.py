"""Algorithm-specific replay objectives over frozen rollout/training caches.

These adapters connect pure losses to model forwards, without choosing sampling
budgets, optimizer boundaries, reward normalization or EMA timing for the caller.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import torch

from ..algorithms import art_loss, clipped_grpo_loss, flowdppo_loss, nft_loss
from .rollout import Rollout, replay_transition


def disjoint_windows(rows: int, selected_count: int, updates: int = 2):
    if any(type(x) is not int or x <= 0 for x in (rows, selected_count, updates)) or rows % updates:
        raise ValueError("positive row count must divide evenly into update windows")
    size = rows // updates
    return tuple(
        tuple(
            (row, slot)
            for row in range(window * size, (window + 1) * size)
            for slot in range(selected_count)
        )
        for window in range(updates)
    )


@torch.no_grad()
def group_advantages(
    rewards, groups, *, mode: str, eps: float, global_rewards: torch.Tensor | None = None
):
    """Statistics are explicit; callers gather distributed prompt groups first.

    Group indices identify prompts, not physical microbatches or local ranks.
    ``global_rewards`` is the uncentered rollout population for global std.
    """
    if rewards.ndim != 1 or not rewards.numel() or groups.shape != rewards.shape:
        raise ValueError("rewards and groups must be equal nonempty vectors")
    if groups.dtype not in (torch.int32, torch.int64) or (groups < 0).any():
        raise ValueError("groups must be nonnegative integer indices")
    if (
        not torch.isfinite(rewards).all()
        or not isinstance(eps, (int, float))
        or not 0 < eps < float("inf")
    ):
        raise ValueError("rewards and epsilon must be finite")
    if mode not in ("group_sample_std", "global_population_std"):
        raise ValueError("advantage normalization mode must be explicit")
    values = rewards.detach().float()
    groups = groups.to(values.device)
    result = torch.empty_like(values)
    if mode == "global_population_std":
        population = (
            values if global_rewards is None else global_rewards.detach().to(values).reshape(-1)
        )
        if not population.numel() or not torch.isfinite(population).all():
            raise ValueError("invalid global reward population")
        denominator = population.std(unbiased=False) + eps
    elif global_rewards is not None:
        raise ValueError("group sample std must not receive a global denominator")
    for group in groups.unique():
        selection = groups == group
        members = values[selection]
        if len(members) < 2:
            raise ValueError("group-based advantages require at least two samples per prompt")
        if mode == "group_sample_std":
            denominator = members.std(unbiased=True) + eps
        result[selection] = (members - members.mean()) / denominator
    return result


def _indices(items: Sequence[tuple[int, int]], cache: Rollout):
    if not items or any(len(item) != 2 for item in items):
        raise ValueError("items must be nonempty (row, slot) pairs")
    indices = torch.tensor(items)
    if indices.dtype != torch.int64:
        raise ValueError("replay indices must be integers")
    rows, slots = indices.unbind(1)
    if (
        (rows < 0).any()
        or (rows >= len(cache.endpoints)).any()
        or (slots < 0).any()
        or (slots >= len(cache.selected)).any()
    ):
        raise ValueError("replay index out of range")
    steps = cache.indices(rows, slots)
    return rows, slots, steps


@dataclass
class ARTObjective:
    cache: Rollout
    predict: Callable
    row_directions: torch.Tensor
    row_magnitudes: torch.Tensor
    row_anchors: torch.Tensor
    device: torch.device
    options: dict = field(default_factory=dict)
    reconstruct_from_mean: bool = False
    reconstruct_old_from_mean: bool = False

    @property
    def has_guidance(self):
        rms = self.row_directions.float().flatten(1).square().mean(1).sqrt()
        return bool(((self.row_magnitudes > 0) & (rms > 0)).any())

    def __call__(self, items):
        rows, slots, steps = _indices(items, self.cache)
        state = self.cache.states[rows, slots].to(self.device)
        sigma = self.cache.sigmas[steps].to(self.device)
        velocity = self.predict(state, sigma, rows.to(self.device))
        old = self.cache.old_velocities[rows, slots].to(self.device)
        dt = (self.cache.sigmas[steps + 1] - self.cache.sigmas[steps]).to(self.device)
        shape = (-1,) + (1,) * (state.ndim - 1)
        current_mean = state + dt.reshape(shape) * (velocity.float() if self.reconstruct_from_mean else velocity)
        old_mean = state + dt.reshape(shape) * old
        if self.reconstruct_from_mean or self.reconstruct_old_from_mean:
            if (dt.abs() < 1e-8).any():
                raise ValueError("ART mean reconstruction requires non-degenerate ODE steps")
            old_mean = self.cache.old_means[rows, slots].to(self.device)
            if self.reconstruct_from_mean:
                velocity = (current_mean - state) / dt.reshape(shape)
            old = (old_mean - state) / dt.reshape(shape)
        terms = art_loss(
            velocity,
            old,
            sigma,
            self.row_directions[rows].to(self.device),
            self.row_magnitudes[rows].to(self.device),
            state=state,
            ode_velocity_coefficient=dt,
            current_ode_mean=current_mean,
            old_ode_mean=old_mean,
            restore_anchor_mask=self.row_anchors[rows].to(self.device),
            **self.options,
        )
        return terms["per_item_loss"]


@dataclass
class GRPOObjective:
    cache: Rollout
    predict: Callable
    advantages: torch.Tensor | Mapping[str, torch.Tensor]
    device: torch.device
    algorithm: str
    options: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.algorithm not in ("flow_grpo", "flow_grpo_fast", "dancegrpo", "flow_dppo"):
            raise ValueError("unsupported likelihood algorithm")

    def __call__(self, items):
        rows, slots, steps = _indices(items, self.cache)
        state = self.cache.states[rows, slots].to(self.device)
        velocity = self.predict(
            state, self.cache.sigmas[steps].to(self.device), rows.to(self.device)
        )
        replay = replay_transition(self.cache, velocity, rows, slots)
        old_logp = self.cache.old_log_probs[rows, slots].to(self.device)
        if isinstance(self.advantages, Mapping):
            advantage = {k: v[rows].to(self.device) for k, v in self.advantages.items()}
        else:
            advantage = self.advantages[rows].to(self.device)
        if self.algorithm == "flow_dppo":
            if isinstance(advantage, Mapping):
                raise ValueError("FlowDPPO requires explicit scalarized advantages")
            terms = flowdppo_loss(
                replay.log_prob,
                old_logp,
                replay.mean,
                self.cache.old_means[rows, slots].to(self.device),
                replay.std,
                advantage,
                **self.options,
            )
        else:
            terms = clipped_grpo_loss(replay.log_prob, old_logp, advantage, **self.options)
        return terms["per_item_loss"]


@dataclass
class NFTObjective:
    endpoints: torch.Tensor
    states: torch.Tensor
    sigmas: torch.Tensor
    old_predictions: torch.Tensor
    row_indices: torch.Tensor
    advantages: torch.Tensor
    predict: Callable
    device: torch.device
    options: dict = field(default_factory=dict)
    upcast_predictions: bool = False

    def __post_init__(self):
        if (
            self.endpoints.shape != self.states.shape
            or self.states.shape != self.old_predictions.shape
            or self.sigmas.shape != (len(self.states),)
            or self.row_indices.shape != self.sigmas.shape
            or self.advantages.shape != self.sigmas.shape
        ):
            raise ValueError(
                "NFT cache needs one endpoint/state/old prediction/sigma/advantage per item"
            )

    def __call__(self, items: Sequence[int]):
        indices = torch.tensor(items, dtype=torch.int64)
        state, sigma = self.states[indices].to(self.device), self.sigmas[indices].to(self.device)
        prediction = self.predict(state, sigma, self.row_indices[indices].to(self.device))
        if self.upcast_predictions:
            prediction = prediction.float()
        terms = nft_loss(
            prediction,
            self.old_predictions[indices].to(self.device),
            self.endpoints[indices].to(self.device),
            state,
            sigma,
            self.advantages[indices].to(self.device),
            **self.options,
        )
        return terms["per_item_loss"]
