"""Explicit dual-sigma AV rollouts and CPS/Flow-SDE observation replay.

CPS arithmetic follows verl-omni PR744 93c8a747, flow_match_sde.py and
minimax_h3_flow_grpo/common.py (Apache-2.0); see algorithm provenance.
The opt-in Flow-SDE policy uses the common scheduler's Gaussian density.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch

from ..sampling import Transition, flow_transition, validate_schedule
from .audio_video_types import AVState


def feature_mask(value: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return torch.ones_like(value, dtype=torch.bool)
    if mask.dtype != torch.bool or mask.shape != value.shape[:mask.ndim]:
        raise ValueError("AV masks must be boolean feature prefixes of the latent")
    expanded = mask.to(value.device).reshape(*mask.shape, *((1,) * (value.ndim - mask.ndim)))
    expanded = expanded.expand_as(value)
    if not expanded.flatten(1).any(1).all():
        raise ValueError("each AV row needs at least one valid feature")
    return expanded


def feature_mean(value: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    valid = feature_mask(value, mask)
    return torch.where(valid, value, 0).flatten(1).sum(1) / valid.flatten(1).sum(1)


def validate_state(state: AVState, *, like: AVState | None = None):
    if not isinstance(state, AVState) or len(state.video) != len(state.audio) or not len(state):
        raise ValueError("AV state requires equal nonempty video/audio batches")
    for name in ("video", "audio"):
        value = getattr(state, name)
        if value.ndim < 2 or not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError("AV latent must be finite floating [batch,...]")
        feature_mask(value, getattr(state, name + "_mask"))
        if like is not None and (value.shape != getattr(like, name).shape or value.device != getattr(like, name).device):
            raise ValueError("AV prediction must preserve latent shape/device")


def _state(video, audio, source):
    return AVState(video, audio, source.video_mask, source.audio_mask)


def gather_slots(values: Sequence[AVState], rows: torch.Tensor, slots: torch.Tensor) -> AVState:
    """Gather arbitrary row/slot identities without changing original row IDs."""
    rows, slots = rows.cpu().tolist(), slots.cpu().tolist()
    output = []
    for name in ("video", "audio", "video_mask", "audio_mask"):
        if getattr(values[0], name) is None:
            output.append(None)
        else:
            output.append(torch.stack([getattr(values[slot], name)[row] for row, slot in zip(rows, slots)]))
    return AVState(*output)


def h3_cps_transition(state, velocity, sigma, sigma_next, *, noise_level=.8,
                      innovation=None, observed=None, mask=None) -> Transition:
    """H3's negative-MSE CPS surrogate, including a deterministic terminal step.

    Unlike a Gaussian log density, this surrogate is defined at sigma_next=0.
    No epsilon is injected into its zero standard deviation. Padding is omitted
    from the feature mean; no per-item/optimizer denominator changes.
    """
    if (state.shape != velocity.shape or state.ndim < 2 or
            not math.isfinite(noise_level) or not 0 < noise_level <= 1 or
            (innovation is None) == (observed is None)):
        raise ValueError("H3 CPS requires equal latents and one innovation or observed next state")
    x, v = state.float(), velocity.float()
    def time(value):
        value = torch.as_tensor(value, device=x.device, dtype=torch.float32).reshape(-1)
        if value.numel() not in (1, len(x)):
            raise ValueError("one sigma per row or one shared sigma required")
        return value.reshape((-1,) + (1,) * (x.ndim - 1))
    sigma, nxt = time(sigma), time(sigma_next)
    supplied = innovation if innovation is not None else observed
    if (supplied.shape != x.shape or not all(torch.isfinite(z).all() for z in (x, v, supplied, sigma, nxt))
            or not ((sigma > nxt) & (sigma <= 1) & (nxt >= 0)).all()):
        raise ValueError("invalid H3 CPS transition")
    # Reuse the shared transition on nonterminal steps. Its Gaussian-oriented
    # variance guard deliberately rejects the terminal step, handled explicitly.
    if (nxt > 0).all():
        result = flow_transition(x, v, sigma, nxt, kind="cps", likelihood="mse",
                                 noise_level=noise_level, innovation=innovation, observed=observed)
        sample, mean, std = result.sample, result.mean, result.std
    else:
        std = nxt * math.sin(noise_level * math.pi / 2)
        mean = (x - sigma * v) * (1 - nxt) + (x + (1 - sigma) * v) * (nxt.square() - std.square()).clamp_min(0).sqrt()
        sample = mean + std * supplied.to(x) if innovation is not None else supplied.detach().to(x)
    logp = -feature_mean((sample.detach() - mean).square(), mask)
    return Transition(sample, mean, std, logp)


def h3_fast_transition(state, velocity, sigma, sigma_next, *, kind="cps", sigma_max=None,
                       noise_level=.8, innovation=None, observed=None, mask=None) -> Transition:
    """Explicit H3 Fast policy: legacy CPS/MSE or Flow-SDE/Gaussian.

    SDE uses the actual modality schedule's second boundary for the t=1
    diffusion guard, identically in sampling and observed-state replay.
    Likelihoods average valid features, preserving the existing modality mean.
    """
    if kind == "cps":
        return h3_cps_transition(state, velocity, sigma, sigma_next, noise_level=noise_level,
                                 innovation=innovation, observed=observed, mask=mask)
    if kind != "sde":
        raise ValueError("H3 Fast transition must be cps or sde")
    result = flow_transition(state, velocity, sigma, sigma_next, kind="sde", likelihood="gaussian",
        sigma_max=sigma_max, noise_level=noise_level, innovation=innovation, observed=observed)
    logp = feature_mean(-(result.sample.detach() - result.mean).square() / (2 * result.std.square())
                        - result.std.log() - math.log(2 * math.pi) / 2, mask)
    return Transition(result.sample, result.mean, result.std, logp)


def select_fast_window(transition_count: int, *, rollout_index: int = 0, seed: int = 42,
                       size: int = 3, window_range: tuple[int, int] = (0, 8)) -> tuple[int, ...]:
    """Pinned H3 [low,high) range; start in [low, high-size], inclusive.

    rollout_index is zero-based, matching upstream seed42 + max(global_step-1,0).
    The terminal transition is permitted when it belongs to the actual range.
    """
    if any(type(x) is not int for x in (transition_count, rollout_index, seed, size)) or rollout_index < 0:
        raise ValueError("Fast window arguments must be integers with nonnegative rollout index")
    low, high = window_range
    high = min(high, transition_count)
    if low < 0 or size <= 0 or high - low < size:
        raise ValueError("invalid H3 Fast window")
    generator = torch.Generator().manual_seed(seed + rollout_index)
    start = int(torch.randint(low, high - size + 1, (1,), generator=generator))
    return tuple(range(start, start + size))


@dataclass(frozen=True)
class AVRollout:
    endpoints: AVState
    selected: tuple[int, ...]
    video_sigmas: torch.Tensor
    audio_sigmas: torch.Tensor
    states: tuple[AVState, ...]
    old_velocities: tuple[AVState, ...]
    next_states: tuple[AVState, ...]
    old_log_probs: torch.Tensor | None
    stochastic: tuple[int, ...]
    noise_level: float = .8
    video_selected: tuple[int, ...] | None = None
    audio_selected: tuple[int, ...] | None = None
    old_video_log_probs: torch.Tensor | None = None
    old_audio_log_probs: torch.Tensor | None = None
    transition_kind: str = "cps"

    @property
    def items(self):
        return tuple((row, slot) for row in range(len(self.endpoints)) for slot in range(len(self.selected)))


def validate_schedules(video_sigmas, audio_sigmas):
    for sigmas in (video_sigmas, audio_sigmas):
        validate_schedule(sigmas)
        if not torch.isclose(sigmas[0], sigmas.new_tensor(1.0)) or not torch.isclose(sigmas[-1], sigmas.new_tensor(0.0)):
            raise ValueError("AV rollout schedules must run from one to zero")
    if len(video_sigmas) != len(audio_sigmas):
        raise ValueError("AV schedule boundary counts must match")


@torch.no_grad()
def collect_av_rollout(predict: Callable, noise: AVState, video_sigmas: torch.Tensor,
                       audio_sigmas: torch.Tensor, *, selected: Sequence[int] | None = None,
                       stochastic: Sequence[int] = (), innovations: Mapping[int, AVState] | None = None,
                       microbatch: int = 1, noise_level: float = .8, unselected_kind: str = "ode",
                       device=None, modality_selected=None, modality_stochastic=None,
                       transition_kind="cps") -> AVRollout:
    """One joint forward/step with independent schedules; detached CPU caches.

    Caller activates NFT's old policy around collection. ART defaults to all
    transitions except the last. Fast supplies its selected stochastic window
    and uses cps_zero_noise for unselected transitions, including the last.
    The opt-in routed SDE policy uses ODE at unselected transitions instead.
    Innovations are supplied upfront and independent of physical microbatching.
    Explicit modality_stochastic selects independent noisy positions and
    requires the matching modality_selected replay union. Its inactive logp
    entries are zero placeholders, not likelihoods of a noisy transition.
    """
    validate_state(noise)
    validate_schedules(video_sigmas, audio_sigmas)
    if transition_kind not in ("cps", "sde"):
        raise ValueError("H3 Fast transition must be cps or sde")
    if transition_kind == "sde" and (modality_stochastic is None or unselected_kind != "ode"):
        raise ValueError("H3 SDE requires routed stochastic positions and ODE elsewhere")
    count = len(video_sigmas) - 1
    selected = tuple(range(count - 1)) if selected is None else tuple(selected)
    video_selected = audio_selected = None
    if modality_selected is not None:
        if set(modality_selected) != {"video", "audio"}:
            raise ValueError("explicit AV selection requires both modalities")
        video_selected, audio_selected = (tuple(modality_selected[n]) for n in ("video", "audio"))
        for positions in (video_selected, audio_selected):
            if (not positions or tuple(sorted(set(positions))) != positions
                    or any(type(p) is not int or not 0 <= p < count - 1 for p in positions)):
                raise ValueError("modality positions must be distinct chronological nonfinal transitions")
        if selected != tuple(sorted(set(video_selected) | set(audio_selected))):
            raise ValueError("cached positions must be the union of modality selections")
    stochastic = tuple(stochastic)
    if type(microbatch) is not int or microbatch < 1 or unselected_kind not in ("ode", "cps_zero_noise"):
        raise ValueError("invalid AV rollout microbatch or deterministic transition")
    for positions in (selected, stochastic):
        if len(set(positions)) != len(positions) or any(type(x) is not int or not 0 <= x < count for x in positions):
            raise ValueError("AV positions must be distinct valid transition indices")
    if stochastic and (not selected or not set(selected) <= set(stochastic)):
        raise ValueError("Fast replay positions must be stochastic")
    noisy_by_modality = None
    if modality_stochastic is not None:
        if (not isinstance(modality_stochastic, Mapping)
                or set(modality_stochastic) != {"video", "audio"}
                or modality_selected is None):
            raise ValueError("modality CPS requires explicit stochastic and replay positions for both modalities")
        try:
            noisy_by_modality = {name: tuple(modality_stochastic[name]) for name in ("video", "audio")}
        except TypeError as error:
            raise ValueError("modality CPS positions must be chronological integer sequences") from error
        for positions in noisy_by_modality.values():
            if (not positions or any(type(p) is not int or not 0 <= p < count - 1 for p in positions)
                    or tuple(sorted(set(positions))) != positions):
                raise ValueError("modality CPS positions must be chronological nonfinal integer transitions")
        if (noisy_by_modality["video"] != video_selected or noisy_by_modality["audio"] != audio_selected
                or stochastic != selected or unselected_kind != ("ode" if transition_kind == "sde" else "cps_zero_noise")):
            raise ValueError("modality CPS requires matching selected/stochastic unions and zero-noise unselected steps")
    innovations = {} if innovations is None else innovations
    if set(innovations) != set(stochastic):
        raise ValueError("supply exactly one frozen AV innovation per stochastic position")
    for innovation in innovations.values():
        validate_state(innovation)
        if innovation.video.shape != noise.video.shape or innovation.audio.shape != noise.audio.shape:
            raise ValueError("AV innovation shape mismatch")
    # Copy only mask metadata once. Copying the full initial noise for every
    # empty cache allocation would add O(batch*positions*latent_size) CPU I/O.
    video_mask = None if noise.video_mask is None else noise.video_mask.detach().cpu().clone()
    audio_mask = None if noise.audio_mask is None else noise.audio_mask.detach().cpu().clone()
    def allocate():
        return AVState(torch.empty_like(noise.video, dtype=torch.float32, device="cpu"),
                       torch.empty_like(noise.audio, dtype=torch.float32, device="cpu"), video_mask, audio_mask)
    states, velocities, following = [tuple(allocate() for _ in selected) for _ in range(3)]
    endpoints = allocate()
    logp = torch.empty((len(noise), len(selected)), dtype=torch.float32) if stochastic else None
    modal_logp = ({name: torch.zeros_like(logp) for name in ("video", "audio")}
                 if noisy_by_modality is not None else None)
    lookup = {step: slot for slot, step in enumerate(selected)}
    device = noise.video.device if device is None else torch.device(device)
    for start in range(0, len(noise), microbatch):
        rows = torch.arange(start, min(start + microbatch, len(noise)), device=device)
        state = noise.select(rows).to(device)
        for step in range(count):
            sv = video_sigmas[step].to(state.video.device).expand(len(rows))
            sa = audio_sigmas[step].to(state.audio.device).expand(len(rows))
            velocity = predict(state, sv, sa, rows)
            validate_state(velocity, like=state)
            transitions = []
            for name, sigma, schedule in (("video", sv, video_sigmas), ("audio", sa, audio_sigmas)):
                value, prediction = getattr(state, name), getattr(velocity, name)
                noisy = step in (stochastic if noisy_by_modality is None else noisy_by_modality[name])
                if noisy:
                    result = h3_fast_transition(value, prediction, sigma, schedule[step + 1],
                        kind=transition_kind, sigma_max=float(schedule[1]),
                        noise_level=noise_level, innovation=getattr(innovations[step].select(rows), name).to(value),
                        mask=getattr(state, name + "_mask"))
                else:
                    result = flow_transition(value, prediction, sigma, schedule[step + 1], kind=unselected_kind)
                transitions.append(result)
            next_state = _state(transitions[0].sample, transitions[1].sample, state)
            validate_state(next_state)
            if step in lookup:
                slot = lookup[step]
                for storage, value in ((states, state), (velocities, velocity), (following, next_state)):
                    for name in ("video", "audio"):
                        getattr(storage[slot], name)[rows.cpu()] = getattr(value, name).float().cpu()
                if logp is not None:
                    if modal_logp is None:
                        logp[rows.cpu(), slot] = (.5 * transitions[0].log_prob + .5 * transitions[1].log_prob).cpu()
                    else:
                        for name, transition in zip(("video", "audio"), transitions):
                            if step in noisy_by_modality[name]:
                                modal_logp[name][rows.cpu(), slot] = transition.log_prob.cpu()
                        logp[rows.cpu(), slot] = (.5 * modal_logp["video"][rows.cpu(), slot]
                                                  + .5 * modal_logp["audio"][rows.cpu(), slot])
            state = next_state
        endpoints.video[rows.cpu()] = state.video.float().cpu()
        endpoints.audio[rows.cpu()] = state.audio.float().cpu()
    return AVRollout(endpoints, selected, video_sigmas.detach().float().cpu().clone(),
        audio_sigmas.detach().float().cpu().clone(), states, velocities, following, logp, stochastic, noise_level,
        video_selected, audio_selected,
        None if modal_logp is None else modal_logp["video"],
        None if modal_logp is None else modal_logp["audio"], transition_kind)
