"""Independent native Dance equation and SD3 integration contracts."""

import math
from dataclasses import replace

import pytest
import torch
from test_native_grpo import grpo_config
from test_sd3_runner import threads, tiny_parts  # noqa: F401

from visual_rl.backends import sd3
from visual_rl.config import ConfigError, load_config
from visual_rl.registry import get_backend
from visual_rl.sampling import flow_transition
from visual_rl.training.grpo import dance_advantages
from visual_rl.training.grpo_sampling import collect_grpo_rollout
from visual_rl.training.image_contracts import sd3_schedule
from visual_rl.training.rollout import collect_rollout, replay_transition


def native_step(x, v, t, nxt, eta, *, noise=None, observed=None):
    # DanceGRPO 15cc71d train_grpo_flux_lora.py::flux_step, FP32 field.
    dt = nxt - t
    mean = x + dt * v
    x0 = x - t * v
    std = eta * math.sqrt(t - nxt)
    score = -(x - x0 * (1 - t)) / t**2
    mean = mean + (-.5 * eta**2 * score) * dt
    sample = mean + noise * std if observed is None else observed
    logp = (-(sample.detach().float() - mean.float())**2 / (2 * std**2)
            - math.log(std) - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi))))
    return sample, mean, logp.flatten(1).mean(1)


@pytest.mark.parametrize("sigma,next_sigma", [(1., .83), (.83, .21), (.001, 0.)])
def test_dance_transition_native_values_and_replay_gradient(sigma, next_sigma):
    rng = torch.Generator().manual_seed(31)
    x, noise = [torch.randn((3, 2, 4, 4), generator=rng) for _ in range(2)]
    v = torch.randn(x.shape, generator=rng).requires_grad_()
    actual = flow_transition(x, v, sigma, next_sigma, kind="dance", noise_level=.8, innovation=noise)
    sample, mean, logp = native_step(x, v, sigma, next_sigma, .8, noise=noise)
    torch.testing.assert_close(actual.sample, sample, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(actual.mean, mean, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(actual.log_prob, logp, rtol=2e-6, atol=2e-6)
    replay = flow_transition(x, v, sigma, next_sigma, kind="dance", noise_level=.8,
                             observed=actual.sample.detach())
    _, _, oracle = native_step(x, v, sigma, next_sigma, .8, observed=actual.sample.detach())
    ga = torch.autograd.grad(replay.log_prob.sum(), v, retain_graph=True)[0]
    gb = torch.autograd.grad(oracle.sum(), v)[0]
    # Tensor FP32 std vs native Python scalar std rounds differently near zero.
    torch.testing.assert_close(ga, gb, rtol=3e-5, atol=5e-6)
    assert torch.isfinite(ga).all() and ga.abs().sum() > 0


def test_dance_full_rollout_row_selection_and_physical_batch_invariance():
    sigmas = sd3_schedule(4, 3.)
    noise = torch.randn((8, 2, 4, 4), generator=torch.Generator().manual_seed(4))

    def predict(state, time, rows):
        return state * .125 + rows[:, None, None, None] * .01 + time[:, None, None, None] * .05

    caches = [collect_grpo_rollout(
        predict, noise, sigmas, pool=(0, 1, 2, 3), count=3, selection="random_per_row",
        logical_batch_size=4, microbatch=mb, transition_kind="dance", likelihood="gaussian",
        noise_level=.8, unselected_noise="all", generator=torch.Generator().manual_seed(2),
    ) for mb in (1, 3, 8)]
    a = caches[0]
    assert a.stochastic == (0, 1, 2, 3)
    assert any(not torch.equal(row, a.step_indices[0]) for row in a.step_indices[1:])
    assert (a.step_indices == 3).any()  # terminal Dance variance remains positive
    for b in caches[1:]:
        for key in ("endpoints", "states", "old_velocities", "next_states", "old_log_probs", "step_indices"):
            torch.testing.assert_close(getattr(a, key), getattr(b, key), rtol=0, atol=0)
    rows = torch.arange(8).repeat_interleave(3)
    slots = torch.arange(3).repeat(8)
    replay = replay_transition(a, a.old_velocities[rows, slots], rows, slots)
    torch.testing.assert_close(replay.log_prob, a.old_log_probs[rows, slots], rtol=0, atol=0)


def test_dance_scores_last_pretransition_x0_not_sampled_terminal():
    sigmas = torch.tensor([1., .6, .2, 0.])
    noise = torch.randn(3, 2, 4, 4, generator=torch.Generator().manual_seed(5))
    innovations = {i: torch.randn(noise.shape, generator=torch.Generator().manual_seed(i))
                   for i in range(3)}

    def predict(state, time, rows):
        return state * .2 + time[:, None, None, None] * .1

    state = noise
    for i in range(3):
        v = predict(state, sigmas[i].expand(3), torch.arange(3))
        endpoint = state - sigmas[i] * v
        state, _, _ = native_step(state, v, float(sigmas[i]), float(sigmas[i + 1]), .8,
                                  noise=innovations[i])
    inputs = dict(predict=predict, noise=noise, sigmas=sigmas, selected=(0, 1, 2),
                  stochastic=(0, 1, 2), innovations=innovations,
                  transition_kind="dance", noise_level=.8, microbatch=2)
    scored = collect_rollout(**inputs, endpoint_kind="last_x0")
    sampled = collect_rollout(**inputs)
    torch.testing.assert_close(scored.endpoints, endpoint)
    assert not torch.equal(scored.endpoints, sampled.endpoints)
    for key in ("states", "old_velocities", "next_states", "old_log_probs"):
        torch.testing.assert_close(getattr(scored, key), getattr(sampled, key), rtol=0, atol=0)
    torch.testing.assert_close(sampled.endpoints, state)


@pytest.mark.parametrize("mode", ["group_sample_std", "prompt_global_std"])
def test_dance_advantage_modes_explicit_and_independent(tmp_path, mode):
    c = grpo_config(tmp_path, "dancegrpo")
    c = replace(c, algorithm=replace(c.algorithm, options={**c.algorithm.options, "advantage_mode": mode}))
    values = torch.tensor([1., 2., 4., 8., 2., 4., 6., 9.])
    prompts = ["same"] * 8  # repeated groups still have distinct group-local stats
    actual = dance_advantages(c, values, prompts)
    if mode == "group_sample_std":
        groups = values.reshape(2, 4)
        oracle = ((groups - groups.mean(1, keepdim=True)) /
                  (groups.std(1, unbiased=True, keepdim=True) + 1e-8)).flatten()
    else:
        oracle = ((values.double() - values.double().mean()) /
                  (values.double().std(unbiased=False) + 1e-4)).float()
    torch.testing.assert_close(actual, oracle, rtol=0, atol=0)


@pytest.mark.parametrize("shared", [False, True])
def test_dance_initial_noise_shared_only_within_each_prompt(tmp_path, monkeypatch, shared):
    import visual_rl.training.grpo_sampling as sampling

    c = grpo_config(tmp_path, "dancegrpo")
    c = replace(c, sampling=replace(c.sampling, options={**c.sampling.options, "same_latent": shared}))
    monkeypatch.setattr(sd3, "load_components", tiny_parts)
    original = sampling.collect_grpo_rollout
    seen = []

    def observe(predict, noise, sigmas, **kwargs):
        groups = noise.reshape(2, 4, *noise.shape[1:])
        for group in groups:
            assert torch.equal(group[0], group[1]) == shared
        assert not torch.equal(groups[0, 0], groups[1, 0])
        seen.append(noise.clone())
        return original(predict, noise, sigmas, **kwargs)

    monkeypatch.setattr(sampling, "collect_grpo_rollout", observe)
    sd3.run(c)
    assert len(seen) == 2


def test_dance_config_isolated_and_examples_validate(tmp_path):
    for name in ("sd3_dance_pickscore", "sd3_dance_ddp"):
        c = load_config(f"configs/examples/{name}.yaml")
        assert get_backend(c).name == "sd3"
        assert c.algorithm.name == "dancegrpo" and c.sampling.options["same_latent"]
        assert c.training.precision == "fp16"
    c = grpo_config(tmp_path, "dancegrpo")
    with pytest.raises(ConfigError, match="Dance"):
        sd3.validate_config(replace(c, model=replace(c.model, name="flux")))
    s = {**c.sampling.options, "likelihood": "mse"}
    with pytest.raises(ConfigError, match="Dance"):
        sd3.validate_config(replace(c, sampling=replace(c.sampling, options=s)))
    with pytest.raises(ValueError, match="Gaussian"):
        flow_transition(torch.ones(2, 3), torch.ones(2, 3), .5, .3,
                        kind="dance", likelihood="mse", innovation=torch.ones(2, 3))
