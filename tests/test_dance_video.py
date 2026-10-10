"""CPU parity against the pinned native DanceGRPO equations and RNG plan."""

import math
from dataclasses import fields, replace

import pytest
import torch

from visual_rl.training.dance_video import (
    DanceObjective,
    _dance_transition,
    collect_dance_rollout,
    dance_inputs,
)


def native_step(output, latents, sigmas, index, *, innovation=None, observed=None, eta=.25):
    # Direct flux_step equations from DanceGRPO 15cc71d. Only randn_like is
    # replaced by explicit innovation; sampling passes FP32 latents upstream.
    sigma = sigmas[index]
    dsigma = sigmas[index + 1] - sigma
    prev_sample_mean = latents + dsigma * output
    pred_original_sample = latents - sigma * output
    delta_t = sigma - sigmas[index + 1]
    std_dev_t = eta * math.sqrt(delta_t)
    score_estimate = -(latents - pred_original_sample * (1 - sigma)) / sigma**2
    log_term = -0.5 * eta**2 * score_estimate
    prev_sample_mean = prev_sample_mean + log_term * dsigma
    prev_sample = (prev_sample_mean + innovation * std_dev_t
                   if observed is None else observed)
    log_prob = (
        -((prev_sample.detach().to(torch.float32) - prev_sample_mean.to(torch.float32))**2)
        / (2 * std_dev_t**2)
        - math.log(std_dev_t) - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    ).mean(dim=tuple(range(1, latents.ndim)))
    return prev_sample, pred_original_sample, log_prob, prev_sample_mean, std_dev_t


def plan(**overrides):
    options = dict(seed=42, rank=3, attempt=2, prompts_per_rank=1, k=8,
                   latent_shape=(2, 2, 3), pool=tuple(range(15)), count=9, steps=16)
    return dance_inputs(**{**options, **overrides})


def schedule():
    t = torch.linspace(1, 0, 17)
    return 5 * t / (1 + 4 * t)


def predict(state, sigma, rows):
    shape = (-1,) + (1,) * (state.ndim - 1)
    return (state.float() * .17 + sigma.reshape(shape) * .31
            + rows.reshape(shape) * .013).to(torch.bfloat16)


def rollout(microbatch=8, predictor=predict, **overrides):
    noise, selected, innovations = plan(**overrides)
    cache = collect_dance_rollout(
        predictor, noise, schedule(), row_selected=selected,
        innovations=innovations, microbatch=microbatch,
    )
    return cache, (noise, selected, innovations)


def exact(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(4,), (2, 2, 3)])
def test_p1_k8_exact_canonical_rng_order_and_unapplied_shuffle(shape):
    before = torch.get_rng_state().clone()
    generator = torch.Generator(device="cpu").manual_seed(42 + 1_000_003 * 2 + 10_007 * 3)
    noise = torch.randn((1, *shape), generator=generator, dtype=torch.bfloat16).repeat(
        8, *([1] * len(shape))
    )
    innovations = torch.randn((8, 16, *shape), generator=generator, dtype=torch.float32)
    after_innovations = generator.get_state()
    torch.randperm(8, generator=generator)
    selected = torch.stack([torch.randperm(15, generator=generator)[:9] for _ in range(8)])
    actual = plan(latent_shape=shape)
    for value, expected in zip(actual, (noise, selected, innovations)):
        exact(value, expected)
        assert value.device.type == "cpu"
    generator.set_state(after_innovations)
    skipped_shuffle = torch.stack([
        torch.randperm(15, generator=generator)[:9] for _ in range(8)
    ])
    assert not torch.equal(actual[1], skipped_shuffle)
    assert not torch.equal(actual[1], actual[1].sort(dim=1).values)
    assert actual[0].dtype == torch.bfloat16
    assert actual[1].dtype == torch.int64
    assert actual[2].dtype == torch.float32
    assert actual[2].flatten(2).reshape(8 * 16, -1).unique(dim=0).shape[0] == 8 * 16
    exact(torch.get_rng_state(), before)
    for changed in (dict(rank=4), dict(attempt=3)):
        other = plan(latent_shape=shape, **changed)
        assert all(not torch.equal(a, b) for a, b in zip(actual, other))


def test_multiple_complete_groups_follow_group_major_stream_without_pruning():
    noise, selected, innovations = plan(prompts_per_rank=3)
    one = plan()
    for actual, expected in zip((noise, selected, innovations), one):
        exact(actual[:8], expected)
    generator = torch.Generator().manual_seed(42 + 1_000_003 * 2 + 10_007 * 3)
    for group in range(3):
        rows = slice(group * 8, (group + 1) * 8)
        expected_noise = torch.randn((1, 2, 2, 3), generator=generator, dtype=torch.bfloat16)
        exact(noise[rows], expected_noise.repeat(8, 1, 1, 1))
        exact(innovations[rows], torch.randn((8, 16, 2, 2, 3), generator=generator))
        torch.randperm(8, generator=generator)
        exact(selected[rows], torch.stack([
            torch.randperm(15, generator=generator)[:9] for _ in range(8)
        ]))
    assert noise[::8].unique(dim=0).shape[0] == 3
    _, sparse, _ = plan(k=3, prompts_per_rank=2, pool=(12, 2, 9), count=3)
    assert sparse.shape == (6, 3)
    exact(sparse.sort(dim=1).values, torch.tensor([2, 9, 12]).expand(6, 3))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_transition_equation_and_observed_likelihood_gradient_match_native(dtype):
    generator = torch.Generator().manual_seed(8)
    state = torch.randn((4, 2, 3), generator=generator)
    velocity = torch.randn(state.shape, generator=generator).to(dtype).requires_grad_()
    innovations = torch.randn(state.shape, generator=generator)
    steps = torch.tensor([0, 14, 5, 15])
    sigmas = schedule()
    sampled = _dance_transition(
        sigmas, velocity, steps, state, eta=.25, innovation=innovations,
    )
    expected = [native_step(
        velocity[r:r + 1], state[r:r + 1], sigmas, i,
        innovation=innovations[r:r + 1],
    ) for r, i in enumerate(steps.tolist())]
    for index in range(4):
        exact(sampled[index], torch.cat([row[index] for row in expected]))
    exact(sampled[4], torch.tensor([row[4] for row in expected]))
    observed = sampled[0].detach().requires_grad_()
    replay = _dance_transition(sigmas, velocity, steps, state, eta=.25, observed=observed)
    exact(replay[2], sampled[2])
    oracle_logp = torch.cat([native_step(
        velocity[r:r + 1], state[r:r + 1], sigmas, i, observed=observed[r:r + 1],
    )[2] for r, i in enumerate(steps.tolist())])
    gradient, observed_gradient = torch.autograd.grad(
        replay[2].sum(), (velocity, observed), allow_unused=True,
    )
    exact(gradient, torch.autograd.grad(oracle_logp.sum(), velocity)[0])
    assert observed_gradient is None
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    if dtype == torch.bfloat16:
        broadcast = sigmas[steps].reshape(-1, 1, 1)
        assert not torch.equal(sampled[1], state - broadcast * velocity)


def test_all_steps_native_endpoint_not_terminal_noisy_state_and_selected_only_cache():
    calls = []

    def recording_predict(state, times, rows):
        calls.append((state.dtype, torch.is_grad_enabled(), rows.clone(), times.clone()))
        return predict(state, times, rows)

    before = torch.get_rng_state().clone()
    cache, (noise, selected, innovations) = rollout(predictor=recording_predict)
    state = noise
    assert len(calls) == 16 and calls[0][0] == torch.bfloat16
    assert all(dtype == torch.float32 for dtype, *_ in calls[1:])
    assert all(not grad for _, grad, *_ in calls)
    assert cache.states.shape == (8, 9, 2, 2, 3)
    assert cache.old_log_probs.shape == (8, 9)
    assert cache.transition_kind == "dance" and cache.stochastic == tuple(range(16))
    assert cache.noise_level == .25 and cache.likelihood == "gaussian"
    exact(cache.step_indices, selected)
    for step in range(16):
        velocity = predict(state, schedule()[step].expand(8), torch.arange(8))
        sample, x0, logp, mean, std = native_step(
            velocity, state.float(), schedule(), step, innovation=innovations[:, step],
        )
        for row, slot in (selected == step).nonzero().tolist():
            exact(cache.states[row, slot], state[row].float())
            exact(cache.old_velocities[row, slot], velocity[row].float())
            exact(cache.next_states[row, slot], sample[row])
            exact(cache.old_log_probs[row, slot], logp[row])
            exact(cache.old_means[row, slot], mean[row])
            exact(cache.transition_stds[row, slot], torch.tensor(std))
        state = sample
    exact(cache.endpoints, x0)
    assert not torch.equal(cache.endpoints, state)
    exact(torch.get_rng_state(), before)
    for f in fields(cache):
        value = getattr(cache, f.name)
        if isinstance(value, torch.Tensor):
            assert value.device.type == "cpu" and not value.requires_grad
    frozen = cache.step_indices.clone()
    selected.zero_()
    noise.zero_()
    innovations.zero_()
    exact(cache.step_indices, frozen)


@pytest.mark.parametrize("microbatch", [1, 3, 8, 17])
def test_microbatch_invariance_and_global_prompt_rows(microbatch):
    expected, _ = rollout(prompts_per_rank=2)
    actual, _ = rollout(microbatch=microbatch, prompts_per_rank=2)
    for f in fields(expected):
        a, b = getattr(actual, f.name), getattr(expected, f.name)
        if isinstance(a, torch.Tensor):
            exact(a, b)
        else:
            assert a == b


def test_old_policy_replay_unit_ratio_with_unordered_row_slot_items(monkeypatch):
    import visual_rl.training.dance_video as adapter

    cache, _ = rollout(microbatch=3)
    losses = adapter.clipped_grpo_loss
    ratios = []

    def inspect(logp, old, advantages, **options):
        result = losses(logp, old, advantages, **options)
        ratios.append(result["ratio"])
        return result

    monkeypatch.setattr(adapter, "clipped_grpo_loss", inspect)
    items = [(r, s) for r in reversed(range(8)) for s in range(9)]
    advantages = torch.arange(8).float() - 3.5
    objective = DanceObjective(cache, predict, advantages, "cpu", {"ratio_clip": 1e-4})
    before = torch.get_rng_state().clone()
    actual = torch.cat([objective(items[i:i + 5]) for i in range(0, len(items), 5)])
    exact(torch.cat(ratios), torch.ones(72))
    exact(actual, -advantages[torch.tensor([r for r, _ in items])])
    exact(torch.get_rng_state(), before)


@pytest.mark.parametrize("componentwise", [False, True])
def test_objective_native_gradient_component_clipping_frozen_targets_and_denominator(componentwise):
    cache, _ = rollout()
    # Deliberately mark cached targets as differentiable: replay must detach them.
    for name in ("states", "next_states", "old_log_probs"):
        getattr(cache, name).requires_grad_()
    old = cache.old_log_probs.detach().clone()
    a = torch.tensor([8., -9., .2, -.3, 1., -2., 0., 2.], requires_grad=True)
    b = torch.tensor([-12., 7., -.8, .4, -2., 1., 3., 0.], requires_grad=True)
    advantages = {"VQ": a, "MQ": b} if componentwise else a
    options = {"ratio_clip": 1e-4, "advantage_clip": 5.0}
    if componentwise:
        options["component_weights"] = {"VQ": .7, "MQ": .3}
    parameter = torch.tensor(.015, requires_grad=True)

    def updated(state, times, rows):
        return (predict(state, times, rows).float() + parameter).to(torch.bfloat16)

    items = [(r, s) for r in range(4) for s in range(9)]
    objective = DanceObjective(cache, updated, advantages, "cpu", options)
    actual = objective(items)
    expected = []
    for row, slot in items:
        step = int(cache.step_indices[row, slot])
        state = cache.states[row:row + 1, slot].detach()
        velocity = updated(state, cache.sigmas[step:step + 1], torch.tensor([row]))
        logp = native_step(velocity, state, cache.sigmas, step,
                           observed=cache.next_states[row:row + 1, slot])[2]
        ratio = (logp - old[row, slot]).exp()

        def term(adv):
            value = adv[row].detach().clamp(-5, 5)
            return torch.maximum(-value * ratio, -value * ratio.clamp(1 - 1e-4, 1 + 1e-4))

        expected.append(.7 * term(a) + .3 * term(b) if componentwise else term(a))
    expected = torch.cat(expected)
    assert actual.shape == (36,)
    exact(actual, expected)
    # The caller retains native 4 rows * 9 selected steps, even for partial chunks.
    actual_gradient = torch.autograd.grad(actual.sum() / 36, parameter)[0]
    expected_gradient = torch.autograd.grad(expected.sum() / 36, parameter)[0]
    exact(actual_gradient, expected_gradient)
    assert actual_gradient.abs() > 0
    for size in (1, 5, 36):
        parameter.grad = None
        for start in range(0, 36, size):
            (objective(items[start:start + size]).sum() / 36).backward()
        torch.testing.assert_close(parameter.grad, expected_gradient, rtol=2e-6, atol=1e-7)
    assert a.grad is None and b.grad is None
    assert all(getattr(cache, name).grad is None
               for name in ("states", "next_states", "old_log_probs"))
    exact(cache.old_log_probs, old)


@pytest.mark.parametrize("overrides", [
    {"pool": (0, 15)}, {"pool": (1, 1)}, {"count": 16}, {"steps": 1},
    {"prompts_per_rank": 0}, {"k": 0}, {"latent_shape": (0,)}, {"attempt": -1},
])
def test_invalid_random_plan_rejected(overrides):
    with pytest.raises(ValueError):
        plan(**overrides)


@pytest.mark.parametrize("bad", ["terminal", "duplicate", "dtype", "innovation", "eta"])
def test_invalid_rollout_rejected(bad):
    noise, selected, innovations = plan()
    options = {}
    if bad == "terminal":
        selected[0, 0] = 15
    elif bad == "duplicate":
        selected[0, 0] = selected[0, 1]
    elif bad == "dtype":
        selected = selected.float()
    elif bad == "innovation":
        innovations = innovations[:, :-1]
    else:
        options["eta"] = 0
    with pytest.raises(ValueError):
        collect_dance_rollout(predict, noise, schedule(), row_selected=selected,
                              innovations=innovations, microbatch=3, **options)


def test_invalid_objective_cache_and_slot_rejected():
    cache, _ = rollout()
    with pytest.raises(ValueError, match="Dance rollout"):
        DanceObjective(replace(cache, transition_kind="sde"), predict, torch.ones(8), "cpu")
    objective = DanceObjective(cache, predict, torch.ones(8), "cpu")
    for items in ([], [(0, 9)], [(-1, 0)], [(8, 0)], [(0., 1.)]):
        with pytest.raises(ValueError):
            objective(items)
