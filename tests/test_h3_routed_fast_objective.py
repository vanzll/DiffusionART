"""Analytical routed CPS ratios, equal-modal means and reference gradients."""

import copy
import math
from dataclasses import replace

import pytest
import torch

from visual_rl.algorithms import clipped_grpo_loss
from visual_rl.training.audio_video import run_av_updates
from visual_rl.training.audio_video_objectives import (
    AVFastObjective,
    prepare_routed_fast_objective,
)
from visual_rl.training.audio_video_rollout import (
    collect_av_rollout,
    feature_mean,
    h3_fast_transition,
)
from visual_rl.training.audio_video_types import AVState


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.video = torch.nn.Parameter(torch.tensor(.1))
        self.audio = torch.nn.Parameter(torch.tensor(.2))

    def forward(self, state, sv, sa, rows):
        return AVState(state.video * self.video * (1 + sv[:, None]),
                       state.audio * self.audio * (1 + sa[:, None]))


def fixture(selection=None, *, padding=0, kind="cps"):
    selection = selection or {"video": (0, 3), "audio": (1, 2)}
    generator = torch.Generator().manual_seed(819)
    model = Policy()
    video, audio = torch.randn(8, 4, generator=generator), torch.randn(8, 3, generator=generator)
    vm = am = None
    if padding:
        video = torch.cat((video, torch.full((8, padding), 1000.)), 1)
        audio = torch.cat((audio, torch.full((8, padding), -1000.)), 1)
        vm = torch.arange(video.shape[1]).expand(8, -1) < 4
        am = torch.arange(audio.shape[1]).expand(8, -1) < 3
    noise = AVState(video, audio, vm, am)
    sv, sa = torch.tensor([1., .99, .98, .9, .7, 0.]), torch.tensor([1., .97, .94, .8, .5, 0.])
    union = tuple(sorted(set(selection["video"]) | set(selection["audio"])))
    innovations = {step: AVState(torch.randn(video.shape, generator=generator),
                                 torch.randn(audio.shape, generator=generator)) for step in union}
    cache = collect_av_rollout(model, noise, sv, sa, selected=union, stochastic=union,
        innovations=innovations, modality_selected=selection, modality_stochastic=selection,
        unselected_kind="ode" if kind == "sde" else "cps_zero_noise", transition_kind=kind, microbatch=2)
    return model, cache


def objective(model, cache, video=None, audio=None, *, beta=0., reference=None):
    video = torch.linspace(-2, 2, 8) if video is None else video
    audio = torch.linspace(1, -3, 8) if audio is None else audio
    return prepare_routed_fast_objective(cache, model, video, audio, torch.zeros(8, dtype=torch.long), "cpu",
        reference_mse_beta=beta, reference_predictions=reference)


def manual(model, cache, video, audio, *, reference=None, beta=0.):
    modal = []
    for name, reward, schedule in (("video", video, cache.video_sigmas), ("audio", audio, cache.audio_sigmas)):
        positions = []
        for step in getattr(cache, name + "_selected"):
            slot = cache.selected.index(step)
            state, following = cache.states[slot], cache.next_states[slot]
            prediction = model(state, cache.video_sigmas[step].expand(8),
                               cache.audio_sigmas[step].expand(8), torch.arange(8))
            replay = h3_fast_transition(getattr(state, name), getattr(prediction, name), schedule[step],
                schedule[step + 1], observed=getattr(following, name), mask=getattr(state, name + "_mask"),
                kind=cache.transition_kind, sigma_max=float(schedule[1]))
            term = clipped_grpo_loss(replay.log_prob, getattr(cache, "old_" + name + "_log_probs")[:, slot],
                reward, ratio_clip=1e-4, advantage_clip=5.)["per_item_loss"]
            if reference is not None:
                term = term + beta * feature_mean(
                    (getattr(prediction, name) - getattr(reference[slot], name)).square(), getattr(state, name + "_mask"))
            positions.append(term)
        modal.append(torch.stack(positions).mean())
    return .5 * modal[0] + .5 * modal[1]


@pytest.mark.parametrize("selection", [
    {"video": (0, 1), "audio": (0, 1)},
    {"video": (0, 3), "audio": (0, 1)},
    {"video": (0, 3), "audio": (1, 2)},
])
@pytest.mark.parametrize("padding", [0, 7])
@pytest.mark.parametrize("kind", ["cps", "sde"])
def test_loss_and_gradient_are_separate_modal_means(selection, padding, kind):
    model, cache = fixture(selection, padding=padding, kind=kind)
    obj = objective(model, cache)
    assert isinstance(obj, AVFastObjective)
    with torch.no_grad():
        model.video.add_(.01)
        model.audio.sub_(.02)
    actual = obj(obj.items).mean()
    expected = manual(model, cache, obj.advantages, obj.audio_advantages)
    torch.testing.assert_close(actual, expected)
    ga = torch.autograd.grad(actual, tuple(model.parameters()), retain_graph=True)
    gb = torch.autograd.grad(expected, tuple(model.parameters()))
    for left, right in zip(ga, gb):
        torch.testing.assert_close(left, right)


@pytest.mark.parametrize("kind", ["cps", "sde"])
def test_initial_active_ratios_are_one_and_rewards_are_not_renormalized(kind):
    model, cache = fixture(kind=kind)
    video, audio = torch.arange(8).float() + 3, torch.arange(8).float() * -2
    obj = objective(model, cache, video, audio)
    torch.testing.assert_close(obj.advantages, video)
    torch.testing.assert_close(obj.audio_advantages, audio)
    video.add_(100.)
    assert obj.advantages[0] == 3  # Complete-group routes are frozen copies.
    obj(obj.items)
    for (row, slot), values in obj._diagnostic_items.items():
        step = cache.selected[slot]
        for name in ("video", "audio"):
            expected = float(step in getattr(cache, name + "_selected"))
            assert values[f"av/{name}_ratio"] == expected
            assert values[f"av/{name}_clip_fraction"] == 0.


def test_clipping_is_independent_before_equal_modal_mixing():
    model, cache = fixture({"video": (0, 1), "audio": (0, 1)})
    cache = replace(cache, old_video_log_probs=cache.old_video_log_probs - math.log(1.2),
                    old_audio_log_probs=cache.old_audio_log_probs - math.log(.8))
    obj = objective(model, cache, torch.full((8,), 2.), torch.full((8,), -1.))
    value = obj(obj.items).mean()
    expected = .5 * (-2 * 1.0001 + .9999)
    torch.testing.assert_close(value, torch.tensor(expected))
    value.backward()
    assert model.video.grad == model.audio.grad == 0
    assert obj.diagnostics["av/video_clip_fraction"] == obj.diagnostics["av/audio_clip_fraction"] == 1.


@pytest.mark.parametrize("active", ["video", "audio"])
@pytest.mark.parametrize("kind", ["cps", "sde"])
def test_route_reward_has_no_direct_gradient_on_other_modal_output(active, kind):
    model, cache = fixture(kind=kind)
    rewards = {"video": torch.zeros(8), "audio": torch.zeros(8)}
    rewards[active] = torch.ones(8)
    obj = objective(model, cache, **rewards)
    obj(obj.items).mean().backward()
    assert getattr(model, active).grad.abs() > 0
    other = "audio" if active == "video" else "video"
    assert getattr(model, other).grad == 0


def test_inactive_position_has_no_direct_policy_loss():
    model, cache = fixture()
    obj = objective(model, cache, torch.ones(8), torch.zeros(8))
    # Slot1 is audio-only; the positive video reward cannot train it.
    slot = cache.selected.index(1)
    values = obj(tuple((row, slot) for row in range(8)))
    assert torch.equal(values, torch.zeros(8))
    values.sum().backward()
    assert model.video.grad == model.audio.grad == 0


@pytest.mark.parametrize("padding", [0, 5])
def test_reference_mse_is_raw_unclipped_and_survives_zero_policy_reward(padding):
    model, cache = fixture(padding=padding)
    reference = cache.old_velocities
    with torch.no_grad():
        model.video.add_(.3)
        model.audio.sub_(.4)
    zeros = torch.zeros(8)
    obj = objective(model, cache, zeros, zeros, beta=1e-3, reference=reference)
    assert not obj.has_guidance and obj.has_reference
    actual = obj(obj.items).mean()
    expected = manual(model, cache, zeros, zeros, reference=reference, beta=1e-3)
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual, tuple(model.parameters()), retain_graph=True)
    expected_grad = torch.autograd.grad(expected, tuple(model.parameters()))
    for left, right in zip(actual_grad, expected_grad):
        torch.testing.assert_close(left, right)
        assert left.abs() > 0


@pytest.mark.parametrize("kind", ["cps", "sde"])
def test_full_u2_is_invariant_to_physical_microbatch(kind):
    first, cache = fixture(kind=kind)
    second = copy.deepcopy(first)
    routes = (torch.linspace(-2, 2, 8), torch.linspace(2, -1, 8))
    metrics = []
    for model, microbatch in ((first, 1), (second, 5)):
        obj = objective(model, cache, *routes, beta=1e-3, reference=cache.old_velocities)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        metrics.append(run_av_updates(model, optimizer, obj, torch.zeros(8, dtype=torch.long),
            algorithm="flow_grpo_fast", microbatch=microbatch, max_grad_norm=100.))
    assert metrics[0]["train/optimizer_steps_per_rollout"] == metrics[1]["train/optimizer_steps_per_rollout"] == 2
    for left, right in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(left, right)
    for key in ("train/loss", "av/video_reference_mse", "av/audio_reference_mse", "av/video_ratio", "av/audio_ratio"):
        assert metrics[0][key] == pytest.approx(metrics[1][key], rel=1e-5, abs=1e-7)


@pytest.mark.parametrize("case", ["missing_logp", "inactive_logp", "missing_reference", "unexpected_reference", "gradient_reference", "bad_beta"])
def test_invalid_routed_replay_or_reference_is_rejected(case):
    model, cache = fixture()
    beta, reference = 0., None
    if case == "missing_logp":
        cache = replace(cache, old_audio_log_probs=None)
    elif case == "inactive_logp":
        value = cache.old_video_log_probs.clone()
        value[:, cache.selected.index(1)] = 1.
        cache = replace(cache, old_video_log_probs=value)
    elif case == "missing_reference":
        beta = 1e-3
    elif case == "unexpected_reference":
        reference = cache.old_velocities
    elif case == "gradient_reference":
        beta = 1e-3
        reference = tuple(AVState(v.video.clone().requires_grad_(), v.audio, v.video_mask, v.audio_mask)
                          for v in cache.old_velocities)
    elif case == "bad_beta":
        beta = float("nan")
    with pytest.raises(ValueError):
        objective(model, cache, beta=beta, reference=reference)
