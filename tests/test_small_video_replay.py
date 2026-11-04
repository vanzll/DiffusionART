"""Algebra/cache proofs, distinct from pretrained GPU parity or science approval."""

from dataclasses import replace

import pytest
import torch

from test_wan_t2v_schedule import native_scheduler_class as native_scheduler_class

from visual_rl.algorithms import art_loss
from visual_rl.models.ltx_video_sampling import (
    draw_ltx_transition_innovation,
    ltx_video_schedule,
    ltx_video_stochastic_transition,
)
from visual_rl.training.small_video_replay import (
    NATIVE_AFFINE_ART_CONTRACT,
    NativeARTObjective,
    collect_native_video,
    native_window_items,
)


def prediction(state, sigma, rows, step):
    return state * 0.1 + rows.to(state).reshape(-1, 1, 1, 1, 1) * 0.01 + step * 0.001


def test_ltx_complete_native_trajectory_and_draw_sequence_not_euler():
    noise = torch.linspace(-1, 1, 16).reshape(2, 2, 2, 1, 2).bfloat16()
    torch.manual_seed(123)
    before = torch.get_rng_state()
    cache = collect_native_video(
        prediction, noise, model="ltx_video_2b_distilled", selected=(0, 2), microbatch=2
    )
    after = torch.get_rng_state()
    torch.set_rng_state(before)
    state = noise.clone()
    schedule = ltx_video_schedule()
    for step in range(8):
        v = prediction(state, schedule[step].expand(2), torch.arange(2), step)
        innovation = draw_ltx_transition_innovation(state)
        native = ltx_video_stochastic_transition(
            state, v, schedule[step], schedule[step + 1], innovation=innovation
        )
        torch.testing.assert_close(cache.states[:, step], state.float(), atol=0, rtol=0)
        torch.testing.assert_close(cache.innovations[:, step], innovation.float(), atol=0, rtol=0)
        torch.testing.assert_close(cache.old_velocities[:, step], v.float(), atol=0, rtol=0)
        torch.testing.assert_close(cache.old_means[:, step], native.mean, atol=0, rtol=0)
        torch.testing.assert_close(cache.next_states[:, step], native.sample, atol=0, rtol=0)
        state = native.sample
    assert torch.equal(after, torch.get_rng_state())  # last zero-weight draw was consumed
    assert cache.states.shape[1] == 8 and cache.eligible == tuple(range(8))
    assert not torch.allclose(cache.velocity_coefficients, schedule[1:] - schedule[:-1])
    torch.testing.assert_close(cache.endpoints, state)


def test_wan_full_frozen_history_collector_uses_real_official_solver(native_scheduler_class):
    noise = torch.tensor([0.3, -0.2, 0.4, -0.6]).reshape(2, 1, 1, 1, 2)
    seen = []

    def predict(state, sigma, rows, index):
        seen.append((len(state), index))
        return prediction(state, sigma, rows, index)

    cache = collect_native_video(
        predict,
        noise,
        model="wan_t2v_1_3b",
        selected=(0, 3),
        microbatch=2,
        scheduler_factory=native_scheduler_class,
    )
    native = native_scheduler_class(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
    native.set_timesteps(50, device="cpu", shift=8)
    state = noise.clone()
    for index, timestep in enumerate(native.timesteps):
        velocity = prediction(state, cache.sigmas[index].expand(2), torch.arange(2), index)
        state = native.step(velocity, timestep, state, return_dict=False)[0]
        torch.testing.assert_close(cache.next_states[:, index], state, atol=0, rtol=0)
    assert seen == [(2, i) for i in range(50)]  # real physical batch2; no per-prompt forward
    assert cache.eligible == tuple(range(50)) and cache.history_audit_max_error < 3e-5
    torch.testing.assert_close(cache.endpoints, state, atol=0, rtol=0)


@pytest.fixture
def cache():
    torch.manual_seed(42)
    return collect_native_video(
        prediction,
        torch.randn(4, 2, 1, 1, 2),
        model="ltx_video_2b_distilled",
        selected=(0, 3),
        microbatch=4,
    )


@pytest.mark.parametrize("slot", [0, 1])
@pytest.mark.parametrize("restoration", [False, True])
def test_native_normalized_mean_gradient_target_weight_and_mask_equal_velocity_art(
    cache, slot, restoration
):
    rows = torch.arange(4)
    steps = torch.full((4,), cache.selected[slot])
    state = cache.states[rows, steps]
    old = cache.old_velocities[rows, steps]
    direction = torch.ones_like(state)
    magnitude = torch.tensor([3.0, 0.0, 3.0, 3.0])
    anchors = torch.tensor([False, restoration, False, False])
    # Include an outward masked row and target-correcting row.
    movement = torch.tensor([1.0, 0.01, -1.0, 0.0]).reshape(-1, 1, 1, 1, 1)
    current = (old + movement).requires_grad_()
    sigma = cache.sigmas[steps]
    dt = cache.sigmas[steps + 1] - sigma
    terms = art_loss(
        current,
        old,
        sigma,
        direction,
        magnitude,
        state=state,
        ode_velocity_coefficient=dt,
        restore_anchor_mask=anchors,
    )
    mean = cache.conditional_mean(current, rows, steps)
    target_mean = cache.conditional_mean(terms["target"], rows, steps).detach()
    a = cache.velocity_coefficients[steps].reshape(-1, 1, 1, 1, 1)
    normalized = sigma * ((mean - target_mean) / a).square().flatten(1).mean(1)
    normalized = torch.where(terms["keep_mask"], normalized, 0.0)
    torch.testing.assert_close(terms["per_item_loss"], normalized, atol=2e-7, rtol=1e-4)
    (direct_grad,) = torch.autograd.grad(terms["per_item_loss"].sum(), current, retain_graph=True)
    (mean_grad,) = torch.autograd.grad(normalized.sum(), current)
    torch.testing.assert_close(direct_grad, mean_grad, atol=2e-6, rtol=1e-4)
    assert terms["masked"][2] and not terms["masked"][0]
    assert bool(terms["keep_mask"][1]) == restoration


def test_objective_uses_native_position_batch_and_keeps_reference_metric_separate(cache):
    parameter = torch.nn.Parameter(torch.tensor(0.02))
    calls = []

    def predict(state, sigma, rows, index):
        calls.append((len(state), index))
        return prediction(state, sigma, rows, index) + parameter

    objective = NativeARTObjective(
        cache,
        predict,
        torch.ones_like(cache.endpoints),
        torch.full((4,), 0.1),
        torch.zeros(4, dtype=torch.bool),
        torch.device("cpu"),
        NATIVE_AFFINE_ART_CONTRACT,
    )
    losses = objective(native_window_items([0, 1, 2, 3], 1))
    losses.sum().backward()
    assert calls == [(4, 0)] and torch.isfinite(parameter.grad)
    assert objective.observations[0]["items"] == 4
    with pytest.raises(ValueError, match="one position"):
        objective([(0, 0), (1, 1)])


@pytest.mark.parametrize(
    "options",
    [
        {"mask_space": "ode_mean"},
        {"velocity_mask_metric": "q2_sde_kl"},
        {"loss_space": "velocity_uniform"},
        {"reference_mse_beta": 0.1},
    ],
)
def test_explicit_scientific_contract_cannot_silently_change_mask_or_weights(cache, options):
    with pytest.raises(ValueError):
        NativeARTObjective(
            cache,
            prediction,
            torch.ones_like(cache.endpoints),
            torch.ones(4),
            torch.zeros(4, dtype=torch.bool),
            torch.device("cpu"),
            NATIVE_AFFINE_ART_CONTRACT,
            options,
        )


def test_native_cache_rejects_nonfinite_missing_innovations_and_zero_sigma_slot(cache):
    with pytest.raises(ValueError, match="innovation"):
        replace(cache, innovations=None)
    with pytest.raises(ValueError, match="eligible"):
        replace(cache, selected=(8,))
    bad = cache.old_means.clone()
    bad[0, 0, 0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        replace(cache, old_means=bad)
