"""Routed AV ART/NFT and shared-ratio H3 Fast, with frozen complete groups."""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import torch

from ..algorithms import art_loss, art_scalar_covariance, clipped_grpo_loss, nft_loss
from ..rewards.av_profiles import THREE_REWARD_WEIGHTS
from .audio_video_rollout import (
    AVRollout,
    feature_mask,
    feature_mean,
    gather_slots,
    h3_cps_transition,
    h3_fast_transition,
    validate_state,
)
from .audio_video_types import AVState
from .objectives import group_advantages
from .rollout import endpoint_renoising

ROUTING = {
    "video": {"video_align": 1., "hpsv3": 1.5, "audiobox": 0., "clap": 0., "desync": 1.},
    "audio": {"video_align": 0., "hpsv3": 0., "audiobox": .5, "clap": 1., "desync": 1.},
    "shared": {"video_align": 1., "hpsv3": 1.5, "audiobox": .5, "clap": 1., "desync": 1.},
}


@torch.no_grad()
def route_rewards(components: Mapping[str, torch.Tensor], *, groups=None, weights=None,
                  prompt_group_std=False) -> dict[str, torch.Tensor]:
    """Accept transformed training components, never raw scorer units.

    DeSync enters both modality routes and enters the Fast scalar exactly once.
    """
    expected = set(ROUTING["shared"])
    preference = weights is not None and set(weights) == set(THREE_REWARD_WEIGHTS)
    if weights is not None:
        expected = set(THREE_REWARD_WEIGHTS) if preference else expected | {"imagebind"}
        if set(weights) != expected:
            raise ValueError("AV routes require all six or all three explicit weights")
    if set(components) != expected:
        raise ValueError("AV reward routing requires exactly the configured scorer components")
    values = {key: value.detach().float() for key, value in components.items()}
    shape = next(iter(values.values())).shape
    if len(shape) != 1 or not shape[0] or any(v.shape != shape or not torch.isfinite(v).all() for v in values.values()):
        raise ValueError("AV scorer components must be finite equal nonempty vectors")
    if prompt_group_std:
        if groups is None:
            raise ValueError("prompt-group reward normalization requires complete K8 group identities")
        inverse, count = validate_groups(groups, shape[0], even=True)
        counts = inverse.bincount()
        if not bool((counts == 8).all()):
            raise ValueError("prompt-group reward normalization requires complete K8 groups before splitting")
        for name, value in values.items():
            normalized = torch.empty_like(value)
            for group in range(count):
                selected = (inverse == group).to(value.device)
                raw = value[selected]
                normalized[selected] = (torch.zeros_like(raw) if bool((raw == raw[0]).all()) else
                                        (raw - raw.mean()) / (raw.std(correction=1) + 1e-4))
            values[name] = normalized
    if weights is not None:
        names = ({"video": {"pickscore", "imagebind"}, "audio": {"clap", "imagebind"}, "shared": expected}
                 if preference else
                 {"video": {"video_align", "hpsv3", "desync", "imagebind"},
                  "audio": {"audiobox", "clap", "desync", "imagebind"}, "shared": expected})
        return {route: sum(float(weights[key]) * values[key] for key in sorted(keys))
                for route, keys in names.items()}
    return {route: sum(weight * values[key] for key, weight in weights.items())
            for route, weights in ROUTING.items()}


def validate_groups(groups: torch.Tensor, rows: int, *, even: bool = False):
    if groups.shape != (rows,) or groups.dtype not in (torch.int32, torch.int64) or (groups < 0).any():
        raise ValueError("AV groups require one nonnegative prompt ID per row")
    _, inverse, counts = groups.detach().cpu().unique(sorted=True, return_inverse=True, return_counts=True)
    if not len(counts) or (counts < 2).any() or not (counts == counts[0]).all() or (even and int(counts[0]) % 2):
        raise ValueError("AV rollout requires complete equally sized prompt groups; U2 requires even K")
    return inverse, len(counts)


def _indices(items, cache):
    indices = torch.as_tensor(items, dtype=torch.long)
    if indices.ndim != 2 or indices.shape[1] != 2 or not len(indices):
        raise ValueError("AV replay items must be (original row, selected slot) pairs")
    rows, slots = indices.unbind(1)
    if (rows < 0).any() or (rows >= len(cache.endpoints)).any() or (slots < 0).any() or (slots >= len(cache.selected)).any():
        raise ValueError("AV replay item out of bounds")
    steps = torch.tensor(cache.selected)[slots]
    return rows, slots, steps


def _record_diagnostics(objective, items, metrics):
    # Key by immutable item identity, so scaler retries overwrite failed-attempt
    # telemetry instead of averaging repeated microbatches into the final run.
    if not hasattr(objective, "_diagnostic_items"):
        objective._diagnostic_items = {}
    for offset, item in enumerate(items):
        key = tuple(item) if isinstance(item, (tuple, list)) else int(item)
        objective._diagnostic_items.setdefault(key, {}).update(
            {name: float(value[offset].detach()) for name, value in metrics.items()})
    for name in metrics:
        values = [value[name] for value in objective._diagnostic_items.values() if name in value]
        mean = sum(values) / len(values)
        if name.endswith("_advantage"):
            objective.diagnostics[name + "_mean"] = mean
            objective.diagnostics[name + "_std"] = max(0., sum(x * x for x in values) / len(values) - mean * mean) ** .5
        else:
            objective.diagnostics[name] = mean


def _kernel_rows(kernel, prediction, old, source, sigma, *arguments, **options):
    """Apply existing kernels only to valid features; never forward twice.

    The native mask can vary per row. Flattening valid features separately
    preserves each kernel's RMS, residual normalization and mean denominator.
    """
    valid = feature_mask(source[0], source[1])
    results = []
    for row in range(len(prediction)):
        mask = valid[row].reshape(-1)
        def vector(value):
            return value[row].reshape(-1)[mask].reshape(1, -1)
        extra = []
        for value in arguments:
            extra.append(vector(value) if value.ndim == prediction.ndim and value.shape == prediction.shape else value[row:row + 1])
        kwargs = {}
        for key, value in options.items():
            if isinstance(value, torch.Tensor):
                kwargs[key] = vector(value) if value.shape == prediction.shape else value[row:row + 1]
            else:
                kwargs[key] = value
        results.append(kernel(vector(prediction), vector(old), sigma[row:row + 1], *extra, **kwargs))
    return {key: torch.cat([r[key].reshape(-1) for r in results])
            for key in results[0] if results[0][key].numel() == 1}


@dataclass
class AVARTObjective:
    cache: AVRollout
    predict: Callable
    directions: AVState
    video_magnitudes: torch.Tensor
    audio_magnitudes: torch.Tensor
    device: object
    options: dict = field(default_factory=dict)
    statistics: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    reference_predictions: tuple[AVState, ...] | None = None

    @property
    def items(self):
        return self.cache.items

    @property
    def has_guidance(self):
        return bool((self.video_magnitudes > 0).any() or (self.audio_magnitudes > 0).any())

    @property
    def has_reference(self):
        return self.options.get("reference_mse_beta", 0.) > 0

    def __call__(self, items):
        rows, slots, steps = _indices(items, self.cache)
        state = gather_slots(self.cache.states, rows, slots).to(self.device)
        old = gather_slots(self.cache.old_velocities, rows, slots).to(self.device)
        sv, sa = [schedule[steps].to(self.device) for schedule in (self.cache.video_sigmas, self.cache.audio_sigmas)]
        prediction = self.predict(state, sv, sa, rows.to(self.device))
        validate_state(prediction, like=state)
        reference = (None if self.reference_predictions is None else
                     gather_slots(self.reference_predictions, rows, slots).to(self.device))
        losses = []
        for name, sigma, schedule, magnitude in (("video", sv, self.cache.video_sigmas, self.video_magnitudes),
                                                ("audio", sa, self.cache.audio_sigmas, self.audio_magnitudes)):
            current = getattr(prediction, name).float()
            frozen = getattr(old, name)
            x = getattr(state, name)
            dt = (schedule[steps + 1] - schedule[steps]).to(self.device)
            expanded = dt.reshape((-1,) + (1,) * (x.ndim - 1))
            reference_options = ({} if reference is None else
                                 {"reference_prediction": getattr(reference, name)})
            terms = _kernel_rows(art_loss, current, frozen, (x, getattr(state, name + "_mask")), sigma,
                getattr(self.directions.select(rows), name).to(self.device), magnitude[rows].to(self.device),
                state=x, ode_velocity_coefficient=dt, current_ode_mean=x + expanded * current,
                old_ode_mean=x + expanded * frozen, **reference_options, **self.options)
            selected = getattr(self.cache, name + "_selected")
            active = torch.ones_like(sigma, dtype=torch.bool) if selected is None else torch.tensor(
                [int(step) in selected for step in steps], device=sigma.device)
            # The owner averages over union slots. Restore each modality's own
            # sample*time denominator before the equal-modal loss combination.
            factor = 1. if selected is None else len(self.cache.selected) / len(selected)
            policy = torch.where(active, terms["per_item_loss"], 0.)
            losses.append(policy * factor)
            diagnostics = {f"av/{name}_loss": policy,
                f"av/{name}_mask_fraction": torch.where(active, terms["masked"].float(), 0.),
                f"av/{name}_current_old_rms": feature_mean((current.detach() - frozen).square(),
                    getattr(state, name + "_mask")).sqrt()}
            if selected is not None:
                diagnostics[f"av/{name}_selected"] = active.float()
            if self.has_reference:
                diagnostics[f"av/{name}_reference_mse"] = torch.where(
                    active, terms["per_item_reference_mse"], 0.)
            _record_diagnostics(self, items, diagnostics)
        return .5 * losses[0] + .5 * losses[1]


@torch.no_grad()
def prepare_art_objective(cache, predict, video_rewards, audio_rewards, groups, device, *,
                          covariance_options=None, loss_options=None, reference_predictions=None):
    inverse, group_count = validate_groups(groups, len(cache.endpoints), even=True)
    if not cache.selected or (len(cache.video_sigmas) - 2) in cache.selected:
        raise ValueError("ART must select positions and exclude the final transition")
    options = dict(loss_options or {})
    if any(key.startswith("reference_") and key != "reference_mse_beta" for key in options):
        raise ValueError("AV objectives accept only reference_mse_beta and explicit reference predictions")
    beta = options.get("reference_mse_beta", 0.)
    if isinstance(beta, bool) or not isinstance(beta, (int, float)) or not math.isfinite(beta) or beta < 0:
        raise ValueError("reference_mse_beta must be finite and nonnegative")
    if (beta > 0) != (reference_predictions is not None):
        raise ValueError("positive reference_mse_beta requires cached reference predictions and vice versa")
    if reference_predictions is not None:
        if not isinstance(reference_predictions, tuple) or len(reference_predictions) != len(cache.selected):
            raise ValueError("reference predictions must match selected cached positions")
        for reference, source in zip(reference_predictions, cache.states):
            validate_state(reference, like=source)
            for name in ("video", "audio"):
                if getattr(reference, name).requires_grad:
                    raise ValueError("reference predictions must be detached")
                mask = getattr(reference, name + "_mask")
                source_mask = getattr(source, name + "_mask")
                if (mask is None) != (source_mask is None) or (mask is not None and not torch.equal(mask, source_mask)):
                    raise ValueError("reference prediction feature masks differ from cached states")
    directions, magnitudes, statistics = [], [], {}
    for name, rewards in (("video", video_rewards), ("audio", audio_rewards)):
        x = getattr(cache.endpoints, name).float().cpu()
        valid = feature_mask(x, getattr(cache.endpoints, name + "_mask"))
        direction, magnitude = torch.zeros_like(x), torch.zeros(len(x))
        modality_stats = []
        for group in range(group_count):
            indices = (inverse == group).nonzero().flatten()
            mask = valid[indices[0]].reshape(-1)
            if not (valid[indices].flatten(1) == mask).all():
                raise ValueError("samples of one prompt must share valid latent features")
            selected = x[indices].flatten(1)[:, mask]
            result = art_scalar_covariance(selected, rewards.detach().float().cpu()[indices],
                torch.zeros(len(indices), dtype=torch.long), 1, **dict(covariance_options or {}))
            # Scatter only the valid features; invalid storage remains zero.
            out = direction.flatten(1)
            for row in indices:
                out[row, mask] = result["group_directions"][0]
            magnitude[indices] = result["group_guidance_magnitudes"][0] * result["group_active"][0]
            modality_stats.append(result)
        directions.append(direction)
        magnitudes.append(magnitude)
        statistics[name] = modality_stats
    direction_state = AVState(*directions, cache.endpoints.video_mask, cache.endpoints.audio_mask)
    return AVARTObjective(cache, predict, direction_state, *magnitudes, device, options, statistics,
                          reference_predictions=reference_predictions)


@dataclass
class AVNFTObjective:
    endpoints: AVState
    states: AVState
    old_predictions: AVState
    video_sigmas: torch.Tensor
    audio_sigmas: torch.Tensor
    rows: torch.Tensor
    video_advantages: torch.Tensor
    audio_advantages: torch.Tensor
    predict: Callable
    device: object
    diagnostics: dict = field(default_factory=dict)

    @property
    def items(self):
        return tuple(range(len(self.states)))

    @property
    def has_guidance(self):
        return bool((self.video_advantages != 0).any() or (self.audio_advantages != 0).any())

    def __call__(self, items):
        indices = torch.as_tensor(items, dtype=torch.long)
        state, old, endpoint = [value.select(indices).to(self.device) for value in (self.states, self.old_predictions, self.endpoints)]
        sigmas = [value[indices].to(self.device) for value in (self.video_sigmas, self.audio_sigmas)]
        prediction = self.predict(state, *sigmas, self.rows[indices].to(self.device))
        validate_state(prediction, like=state)
        losses = []
        for name, sigma, advantages in (("video", sigmas[0], self.video_advantages), ("audio", sigmas[1], self.audio_advantages)):
            valid = feature_mask(getattr(state, name), getattr(state, name + "_mask"))
            values = []
            for row in range(len(indices)):
                mask = valid[row].reshape(-1)
                def vector(value):
                    return getattr(value, name)[row].reshape(-1)[mask].reshape(1, -1).float()
                terms = nft_loss(vector(prediction), vector(old), vector(endpoint), vector(state),
                    sigma[row:row + 1], advantages[indices[row:row + 1]].to(self.device),
                    nft_beta=1., adv_clip_max=5., eps=1e-5)
                values.append(terms["per_item_loss"])
            losses.append(torch.cat(values))
            _record_diagnostics(self, items, {f"av/{name}_loss": losses[-1],
                f"av/{name}_current_old_rms": feature_mean(
                    (getattr(prediction, name).detach().float() - getattr(old, name)).square(),
                    getattr(state, name + "_mask")).sqrt(),
                f"av/{name}_advantage": advantages[indices].to(self.device)})
        return .5 * losses[0] + .5 * losses[1]


@torch.no_grad()
def prepare_nft_objective(endpoints, predict, old_predict, video_rewards, audio_rewards, groups,
                          train_sigmas, noise: AVState, device, *, microbatch=1,
                          global_video_rewards=None, global_audio_rewards=None):
    """Freeze fresh times/noise/advantages/old predictions before the U1 pass.

    ``train_sigmas`` is [N,S] or shared [S], drawn from the upstream VIDEO
    shifted grid[:-1], floor-quantized to 1/1000 by select_nft_fresh_times.
    Audio receives the same fresh sigma, intentionally.
    ``noise`` is [N,S,...] per modality, generated once before this function.
    The callback owns activation of the same frozen EMA-old used for rollout.
    """
    validate_state(endpoints)
    validate_groups(groups, len(endpoints))
    if type(microbatch) is not int or microbatch < 1:
        raise ValueError("NFT microbatch must be positive")
    times = torch.as_tensor(train_sigmas, dtype=torch.float32).detach().cpu()
    if times.ndim == 1:
        times = times.unsqueeze(0).expand(len(endpoints), -1)
    if times.ndim != 2 or times.shape[0] != len(endpoints) or times.shape[1] < 1 or not torch.isfinite(times).all() or not ((times > 0) & (times <= 1)).all():
        raise ValueError("NFT fresh times must be finite [rows,positions] in (0,1]")
    count, slots = times.shape
    for name in ("video", "audio"):
        if getattr(noise, name).shape != (count, slots, *getattr(endpoints, name).shape[1:]):
            raise ValueError("NFT frozen noise requires [rows,positions,...] per modality")
    rows = torch.arange(count).repeat_interleave(slots)
    sigmas = times.reshape(-1).clone()
    endpoint = endpoints.select(rows).detached_cpu()
    states = []
    for name in ("video", "audio"):
        x0 = getattr(endpoint, name)
        states.append(endpoint_renoising(x0, sigmas, getattr(noise, name).detach().cpu().reshape_as(x0)))
    state = AVState(*states, endpoint.video_mask, endpoint.audio_mask)
    frozen = AVState(torch.empty_like(state.video), torch.empty_like(state.audio), state.video_mask, state.audio_mask)
    for start in range(0, len(rows), microbatch):
        index = torch.arange(start, min(start + microbatch, len(rows)))
        local = state.select(index).to(device)
        prediction = old_predict(local, sigmas[index].to(device), sigmas[index].to(device), rows[index].to(device))
        validate_state(prediction, like=local)
        frozen.video[index] = prediction.video.float().cpu()
        frozen.audio[index] = prediction.audio.float().cpu()
    advantages = [group_advantages(rewards.detach().cpu(), groups.detach().cpu(), mode="global_population_std", eps=1e-4,
                  global_rewards=None if global_rewards is None else global_rewards.detach().cpu())
                  for rewards, global_rewards in ((video_rewards, global_video_rewards), (audio_rewards, global_audio_rewards))]
    return AVNFTObjective(endpoint, state, frozen, sigmas, sigmas.clone(), rows,
                          advantages[0][rows], advantages[1][rows], predict, device)


@dataclass
class AVFastObjective:
    cache: AVRollout
    predict: Callable
    advantages: torch.Tensor
    device: object
    ratio_clip: float = 1e-4
    advantage_clip: float = 5.
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.cache.old_log_probs is None or self.advantages.shape != (len(self.cache.endpoints),):
            raise ValueError("H3 Fast requires cached CPS logp and scalar row advantages")
        if not torch.isfinite(self.advantages).all():
            raise ValueError("nonfinite H3 Fast advantages")

    @property
    def items(self):
        return self.cache.items

    @property
    def has_guidance(self):
        return bool((self.advantages != 0).any())

    def __call__(self, items):
        rows, slots, steps = _indices(items, self.cache)
        state = gather_slots(self.cache.states, rows, slots).to(self.device)
        following = gather_slots(self.cache.next_states, rows, slots).to(self.device)
        sv, sa = [schedule[steps].to(self.device) for schedule in (self.cache.video_sigmas, self.cache.audio_sigmas)]
        prediction = self.predict(state, sv, sa, rows.to(self.device))
        validate_state(prediction, like=state)
        logps = []
        for name, sigma, schedule in (("video", sv, self.cache.video_sigmas), ("audio", sa, self.cache.audio_sigmas)):
            replay = h3_cps_transition(getattr(state, name), getattr(prediction, name), sigma, schedule[steps + 1],
                observed=getattr(following, name), noise_level=self.cache.noise_level, mask=getattr(state, name + "_mask"))
            logps.append(replay.log_prob)
        terms = clipped_grpo_loss(.5 * logps[0] + .5 * logps[1], self.cache.old_log_probs[rows, slots].to(self.device),
            self.advantages[rows].to(self.device), ratio_clip=self.ratio_clip, advantage_clip=self.advantage_clip)
        _record_diagnostics(self, items, {"grpo/ratio": terms["ratio"],
                                         "grpo/clip_fraction": terms["clip_fraction"]})
        return terms["per_item_loss"]


@torch.no_grad()
def prepare_fast_objective(cache, predict, shared_rewards, groups, device, *, global_shared_rewards=None,
                           ratio_clip=1e-4, advantage_clip=5., advantage_epsilon=1e-4):
    validate_groups(groups, len(cache.endpoints), even=True)
    advantages = group_advantages(shared_rewards.detach().cpu(), groups.detach().cpu(),
        mode="global_population_std", eps=advantage_epsilon,
        global_rewards=None if global_shared_rewards is None else global_shared_rewards.detach().cpu())
    return AVFastObjective(cache, predict, advantages, device, ratio_clip, advantage_clip)


@dataclass
class AVRoutedFastObjective(AVFastObjective):
    """Independent cached-policy ratios and PPO clipping, then an equal-modal mean.

    ``advantages`` and ``audio_advantages`` are the already normalized/routed
    complete-group rewards. Do not apply another cross-group standard deviation.
    """

    audio_advantages: torch.Tensor | None = None
    reference_predictions: tuple[AVState, ...] | None = None
    reference_mse_beta: float = 0.

    def __post_init__(self):
        super().__post_init__()
        if self.cache.transition_kind not in ("cps", "sde"):
            raise ValueError("routed Fast requires an explicit supported transition policy")
        count, slots = len(self.cache.endpoints), len(self.cache.selected)
        if (self.audio_advantages is None or self.audio_advantages.shape != (count,)
                or not torch.isfinite(self.audio_advantages).all()):
            raise ValueError("routed Fast requires finite audio row advantages")
        if not slots or self.cache.stochastic != self.cache.selected:
            raise ValueError("routed Fast requires selected stochastic CPS positions")
        for name in ("video", "audio"):
            selected = getattr(self.cache, name + "_selected")
            logps = getattr(self.cache, "old_" + name + "_log_probs")
            if (not selected or tuple(sorted(set(selected))) != selected
                    or any(type(step) is not int or step not in self.cache.selected for step in selected)
                    or logps is None or logps.shape != (count, slots)
                    or logps.requires_grad or not torch.isfinite(logps).all()):
                raise ValueError("routed Fast requires detached per-modality CPS replay caches")
            for slot, step in enumerate(self.cache.selected):
                if step not in selected and bool((logps[:, slot] != 0).any()):
                    raise ValueError("inactive modality log probabilities must be zero placeholders")
        if self.cache.selected != tuple(sorted(set(self.cache.video_selected) | set(self.cache.audio_selected))):
            raise ValueError("routed Fast caches must be the modality selection union")
        beta = self.reference_mse_beta
        if isinstance(beta, bool) or not isinstance(beta, (int, float)) or not math.isfinite(beta) or beta < 0:
            raise ValueError("reference_mse_beta must be finite and nonnegative")
        if (beta > 0) != (self.reference_predictions is not None):
            raise ValueError("positive reference_mse_beta requires cached reference predictions and vice versa")
        if self.reference_predictions is not None:
            if not isinstance(self.reference_predictions, tuple) or len(self.reference_predictions) != slots:
                raise ValueError("reference predictions must match selected cached positions")
            for reference, source in zip(self.reference_predictions, self.cache.states):
                validate_state(reference, like=source)
                for name in ("video", "audio"):
                    if getattr(reference, name).requires_grad:
                        raise ValueError("reference predictions must be detached")
                    mask, source_mask = getattr(reference, name + "_mask"), getattr(source, name + "_mask")
                    if (mask is None) != (source_mask is None) or (mask is not None and not torch.equal(mask, source_mask)):
                        raise ValueError("reference prediction feature masks differ from cached states")
        # Validate scalar clipping options before the first training forward.
        clipped_grpo_loss(torch.zeros(1), torch.zeros(1), torch.zeros(1),
                          ratio_clip=self.ratio_clip, advantage_clip=self.advantage_clip)

    @property
    def has_guidance(self):
        return bool((self.advantages != 0).any() or (self.audio_advantages != 0).any())

    @property
    def has_reference(self):
        return self.reference_mse_beta > 0

    def __call__(self, items):
        rows, slots, steps = _indices(items, self.cache)
        state = gather_slots(self.cache.states, rows, slots).to(self.device)
        following = gather_slots(self.cache.next_states, rows, slots).to(self.device)
        sv, sa = [schedule[steps].to(self.device) for schedule in (self.cache.video_sigmas, self.cache.audio_sigmas)]
        prediction = self.predict(state, sv, sa, rows.to(self.device))
        validate_state(prediction, like=state)
        reference = (None if self.reference_predictions is None else
                     gather_slots(self.reference_predictions, rows, slots).to(self.device))
        losses, diagnostics = [], {}
        for name, sigma, schedule, advantages in (
                ("video", sv, self.cache.video_sigmas, self.advantages),
                ("audio", sa, self.cache.audio_sigmas, self.audio_advantages)):
            selected = getattr(self.cache, name + "_selected")
            active = torch.tensor([int(step) in selected for step in steps], device=self.device)
            current = getattr(prediction, name).float()
            mask = getattr(state, name + "_mask")
            # Keep a zero-gradient connection to both output heads even when a
            # physical microbatch contains only one modality's selected slots.
            # This preserves the FSDP backward participation of inactive heads.
            policy = current.flatten(1)[:, 0] * 0.
            ratio, clip_fraction = torch.zeros_like(sigma), torch.zeros_like(sigma)
            if bool(active.any()):
                replay = h3_fast_transition(getattr(state, name)[active], current[active], sigma[active],
                    schedule[steps[active.cpu()] + 1].to(self.device),
                    kind=self.cache.transition_kind, sigma_max=float(schedule[1]),
                    observed=getattr(following, name)[active], noise_level=self.cache.noise_level,
                    mask=None if mask is None else mask[active])
                old = getattr(self.cache, "old_" + name + "_log_probs")[rows[active.cpu()], slots[active.cpu()]]
                terms = clipped_grpo_loss(replay.log_prob, old.to(self.device),
                    advantages[rows[active.cpu()]].to(self.device), ratio_clip=self.ratio_clip,
                    advantage_clip=self.advantage_clip)
                policy[active], ratio[active], clip_fraction[active] = (
                    terms["per_item_loss"], terms["ratio"], terms["clip_fraction"])
            reference_mse = torch.zeros_like(sigma)
            if reference is not None:
                valid = feature_mask(current, mask)
                difference = torch.where(valid, current - getattr(reference, name).detach(), 0.)
                reference_mse = torch.where(active, feature_mean(difference.square(), mask), 0.)
            loss = policy + self.reference_mse_beta * reference_mse
            # The update owner averages union slots. Restore each branch's own
            # sample*selected-position mean before combining modalities equally.
            losses.append(loss * len(self.cache.selected) / len(selected))
            diagnostics.update({f"av/{name}_selected": active.float(), f"av/{name}_loss": loss,
                f"av/{name}_policy_loss": policy, f"av/{name}_ratio": ratio,
                f"av/{name}_clip_fraction": clip_fraction})
            if self.has_reference:
                diagnostics[f"av/{name}_reference_mse"] = reference_mse
        _record_diagnostics(self, items, diagnostics)
        return .5 * losses[0] + .5 * losses[1]


@torch.no_grad()
def prepare_routed_fast_objective(cache, predict, video_rewards, audio_rewards, groups, device, *,
                                  ratio_clip=1e-4, advantage_clip=5., reference_mse_beta=0.,
                                  reference_predictions=None):
    """Accept frozen per-scorer-normalized routes without renormalizing again."""
    validate_groups(groups, len(cache.endpoints), even=True)
    return AVRoutedFastObjective(cache, predict, video_rewards.detach().float().cpu().clone(), device,
        ratio_clip=ratio_clip, advantage_clip=advantage_clip,
        audio_advantages=audio_rewards.detach().float().cpu().clone(),
        reference_mse_beta=reference_mse_beta, reference_predictions=reference_predictions)


@torch.no_grad()
def select_nft_fresh_times(video_sigmas: torch.Tensor, rows: int, *, positions: int | None = None,
                           generator: torch.Generator) -> torch.Tensor:
    """Source-matched per-row permutation of video schedule[:-1], no replacement.

    H3 adapter _build_train_timesteps exposes this pool; DiffusionNFTAlgorithm
    _select_training_timesteps shuffles every row before truncating. Shared
    fresh sigma is sent to both modalities, independent of audio rollout grid.
    """
    # Upstream shuffles sigma*1000 then casts selected timesteps to long.
    pool = (video_sigmas.detach().float().cpu()[:-1] * 1000).floor() / 1000
    if pool.ndim != 1 or not len(pool) or not torch.isfinite(pool).all() or not ((pool > 0) & (pool <= 1)).all():
        raise ValueError("invalid NFT video fresh-time pool")
    positions = len(pool) if positions is None else positions
    if type(rows) is not int or rows < 1 or type(positions) is not int or not 1 <= positions <= len(pool):
        raise ValueError("NFT fresh-time count out of bounds")
    return torch.stack([pool[torch.randperm(len(pool), generator=generator)[:positions]] for _ in range(rows)])
