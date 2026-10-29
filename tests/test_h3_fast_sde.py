"""Flow-SDE Gaussian arithmetic, t=1 guard and dual-schedule trajectory proof."""

from dataclasses import replace

import pytest
import torch
from test_h3_fast_dual_rollout import CoupledPolicy, fixture
from test_minimax_h3_runtime import config_factory

from visual_rl.backends import minimax_h3 as backend
from visual_rl.config import ConfigError
from visual_rl.training.audio_video_rollout import (
    collect_av_rollout,
    feature_mask,
    h3_fast_transition,
)
from visual_rl.training.audio_video_types import AVState


def analytic(x, velocity, t, nxt, second, innovation=None, observed=None):
    # Independent tensor/distribution expression for the Flow-GRPO SDE.
    dt = nxt - t
    diffusion_squared = .8**2 * t / (1 - (second if t == 1 else t))
    mean = x + dt * (velocity + diffusion_squared / (2 * t) * (x + (1 - t) * velocity))
    std = torch.full_like(x, (diffusion_squared * -dt) ** .5)
    sample = mean + std * innovation if observed is None else observed
    return sample, torch.distributions.Normal(mean, std).log_prob(sample.detach())


@pytest.mark.parametrize("t,nxt,second", [(1., .990825688, .990825688), (1., .964285714, .964285714),
                                        (.979591837, .965517241, .990825688), (.923076923, .875, .964285714)])
def test_sde_sample_density_and_gradient_match_gaussian(t, nxt, second):
    x = torch.tensor([[.2, -.5, 1000.], [1., .7, -1000.]])
    velocity = torch.tensor([[.1, -.3, 500.], [-.4, .2, -500.]], requires_grad=True)
    innovation = torch.tensor([[.3, -.4, 30.], [-.2, .6, -30.]])
    mask = torch.tensor([[True, True, False], [True, True, False]])
    actual = h3_fast_transition(x, velocity, t, nxt, kind="sde", sigma_max=second,
                                innovation=innovation, mask=mask)
    expected, logp = analytic(x, velocity, t, nxt, second, innovation=innovation)
    torch.testing.assert_close(actual.sample, expected, atol=2e-4, rtol=2e-5)
    torch.testing.assert_close(actual.log_prob, logp[:, :2].mean(1), atol=2e-5, rtol=2e-5)
    replay = h3_fast_transition(x, velocity, t, nxt, kind="sde", sigma_max=second,
                                observed=actual.sample.detach(), mask=mask)
    torch.testing.assert_close(replay.log_prob, actual.log_prob, rtol=0, atol=0)
    _, independent_logp = analytic(x, velocity, t, nxt, second, observed=actual.sample.detach())
    left = torch.autograd.grad(replay.log_prob.sum(), velocity, retain_graph=True)[0]
    right = torch.autograd.grad(independent_logp[:, :2].mean(1).sum(), velocity)[0]
    torch.testing.assert_close(left, right, atol=1e-6, rtol=2e-5)
    assert torch.isfinite(left).all() and torch.equal(left[:, 2], torch.zeros(2))


def test_dual_schedule_sde_chain_uses_noise_only_at_own_positions():
    noise, video, audio, selections, union, innovations = fixture()
    actual = collect_av_rollout(CoupledPolicy(), noise, video, audio, selected=union, stochastic=union,
        innovations=innovations, modality_selected=selections, modality_stochastic=selections,
        transition_kind="sde", unselected_kind="ode", microbatch=2)
    state, policy = noise, CoupledPolicy()
    for step in range(len(video) - 1):
        velocity = policy(state, video[step].expand(len(noise)), audio[step].expand(len(noise)), torch.arange(len(noise)))
        following = []
        for name, schedule in (("video", video), ("audio", audio)):
            x, v = getattr(state, name), getattr(velocity, name)
            if step in selections[name]:
                sample, logp = analytic(x, v, float(schedule[step]), float(schedule[step + 1]), float(schedule[1]),
                                       innovation=getattr(innovations[step], name))
                mask = feature_mask(x, getattr(state, name + "_mask"))
                expected = torch.where(mask, logp, 0).flatten(1).sum(1) / mask.flatten(1).sum(1)
                torch.testing.assert_close(getattr(actual, "old_" + name + "_log_probs")[:, union.index(step)], expected,
                                           atol=3e-5, rtol=3e-5)
            else:
                sample = x + (schedule[step + 1] - schedule[step]) * v
            following.append(sample)
        state = AVState(*following, noise.video_mask, noise.audio_mask)
    torch.testing.assert_close(actual.endpoints.video, state.video, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(actual.endpoints.audio, state.audio, atol=2e-6, rtol=2e-5)
    assert actual.transition_kind == "sde"


@pytest.mark.parametrize("algorithm,kind", [("art", "sde"), ("fast", "unknown"), ("fast", "sde")])
def test_transition_switch_cannot_silently_change_legacy_algorithms(tmp_path, monkeypatch, algorithm, kind):
    factory = config_factory.__wrapped__(tmp_path, monkeypatch)
    config = factory(algorithm, "full")  # No routed post-training mode.
    config = replace(config, sampling=replace(config.sampling, options={**config.sampling.options, "transition_kind": kind}))
    with pytest.raises(ConfigError, match="transition_kind"):
        backend.validate_config(config)
