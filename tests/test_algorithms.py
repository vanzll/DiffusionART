"""Independent equations and gradients, with no research-repository imports."""

import pytest
import torch

from visual_rl.algorithms import (
    art_loss,
    art_pairwise_covariance,
    art_scalar_covariance,
    clipped_grpo_loss,
    flowdppo_loss,
    nft_loss,
    old_policy_decay,
    refl_loss,
    transition_kl_per_sample,
)


def leaf(values):
    return torch.tensor(values, dtype=torch.float32, requires_grad=True)


@pytest.mark.parametrize("reduction,denominator", [("population", 3), ("unbiased", 2)])
@pytest.mark.parametrize(
    "scale",
    [
        "unit_normalized_covariance",
        "raw_covariance",
        "reward_mad",
        "reward_std_normalized_covariance",
    ],
)
def test_scalar_covariance_independent_formula(reduction, denominator, scale):
    x = leaf([[1, 4], [4, -2], [-1, 3], [2, 1]])
    r = leaf([0, 2, 4, 1])
    groups = torch.tensor([0, 0, 0, 1])
    terms = art_scalar_covariance(x, r, groups, 3, reduction=reduction, scale_mode=scale)
    # E[(R-E[R])X], with the selected sample/population denominator.
    cov = torch.tensor([-4.0, -2.0]) / denominator
    rms = cov.square().mean().sqrt()
    expected_magnitude = {
        "unit_normalized_covariance": torch.tensor(1.0),
        "raw_covariance": rms,
        "reward_mad": torch.tensor(4 / 3),
        "reward_std_normalized_covariance": rms / ((torch.tensor(8 / 3)).sqrt() + 1e-5),
    }[scale]
    torch.testing.assert_close(terms["group_covariance"][0], cov)
    torch.testing.assert_close(terms["group_directions"][0], cov / rms)
    torch.testing.assert_close(terms["group_guidance_magnitudes"][0], expected_magnitude)
    assert terms["group_active"].tolist() == [True, False, False]
    assert not terms["group_covariance"][1:].any()
    assert not any(t.requires_grad for t in terms.values())
    assert x.grad is None and r.grad is None


def test_scalar_zero_rows_keep_original_denominator_and_active_restoration():
    v = leaf([[.5], [.5], [-.5], [-.5]])
    old = torch.zeros_like(v)
    result = art_loss(v, old, torch.ones(4), torch.tensor([[0.], [0.], [1.], [1.]]),
                      torch.tensor([0., 0., 1., 1.]), corrective_mask_enabled=False,
                      velocity_mask_metric="fixed_mse")
    assert result["keep_mask"].tolist() == [False, False, True, True]
    result["per_item_loss"].sum().div(4).backward()
    torch.testing.assert_close(v.grad, torch.tensor([[0.], [0.], [.25], [.25]]))
    # Active rows still restore their output toward the frozen target after drift.
    assert result["per_item_loss"].tolist() == [0., 0., .25, .25]


def test_zero_signal_modes_and_normalized_aggregate_are_explicit():
    x = torch.tensor([[0.0, 0.0], [2e-8, -2e-8]])
    r = torch.tensor([0.0, 1.0])
    groups = torch.tensor([0, 0])
    historical = art_scalar_covariance(x, r, groups, 1)
    normalized = art_scalar_covariance(x, r, groups, 1, direction_mode="normalized_aggregate")
    torch.testing.assert_close(historical["group_directions"], torch.tensor([[0.002, -0.002]]))
    torch.testing.assert_close(normalized["group_directions"], torch.tensor([[1.0, -1.0]]))
    gated = art_scalar_covariance(
        x, r, groups, 1, zero_signal_mode="threshold", zero_signal_atol=1e-7
    )
    assert not gated["group_active"].any()
    assert not gated["group_guidance_magnitudes"].any()
    zero = art_scalar_covariance(torch.zeros_like(x), r, groups, 1)
    assert not zero["group_active"].any()
    assert zero["group_guidance_magnitudes"].item() == 1  # Historical inactive magnitude.
    assert not zero["group_directions"].any()
    with pytest.raises(ValueError):
        art_scalar_covariance(x, r, groups, 1, zero_signal_atol=1e-7)


def test_identical_rewards_thresholds_roundoff_before_normalization():
    x = torch.arange(14, dtype=torch.float32).reshape(7, 2)
    r = torch.full((7,), 0.1)
    groups = torch.zeros(7, dtype=torch.long)
    result = art_scalar_covariance(
        x, r, groups, 1, zero_signal_mode="threshold", zero_signal_atol=1e-12
    )
    assert not result["group_active"].any()
    assert not result["group_directions"].any()


def test_exact_zero_signal_preserves_real_weak_rewards():
    x = torch.arange(14, dtype=torch.float32).reshape(7, 2)
    r = torch.full((7,), 0.1)
    groups = torch.zeros(7, dtype=torch.long)
    constant = art_scalar_covariance(x, r, groups, 1, zero_signal_mode="exact")
    for key in ("group_covariance", "group_directions", "group_guidance_magnitudes",
                "group_reward_stds", "group_reward_abs_means"):
        assert not constant[key].any()
    assert not constant["group_active"].any()
    r[-1] = torch.nextafter(r[-1], torch.tensor(float("inf")))
    exact = art_scalar_covariance(x, r, groups, 1, zero_signal_mode="exact")
    historical = art_scalar_covariance(x, r, groups, 1)
    assert exact["group_active"].all()
    for key in historical:
        torch.testing.assert_close(exact[key], historical[key], rtol=0, atol=0)
    with pytest.raises(ValueError):
        art_scalar_covariance(x, r, groups, 1, zero_signal_mode="exact", zero_signal_atol=1e-7)


def test_exact_zero_for_different_constants_and_cancelled_covariance():
    x = torch.tensor([[0.], [1.], [0.], [2.], [4.], [6.], [1.], [3.], [5.]])
    r = torch.tensor([0., 1., 2., .1, .1, .1, .7, .7, .7])
    result = art_scalar_covariance(x, r, torch.arange(3).repeat_interleave(3), 3,
                                   zero_signal_mode="exact")
    assert not result["group_active"].any()
    assert not result["group_covariance"].any()
    assert not result["group_guidance_magnitudes"].any()
    assert not result["group_directions"].any()


def test_pairwise_ties_rng_offsets_and_global_anchor():
    x = torch.tensor([[1.0], [3.0], [0.0], [2.0]])
    scores = leaf([2, 2, 0, 9])
    rng = torch.random.get_rng_state().clone()
    result = art_pairwise_covariance(x, scores, group_size=2, seed=73)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert result["binary_rewards"].tolist() == [0.5, 0.5, 0, 1]
    assert result["row_restore_anchor"].tolist() == [True, True, False, False]
    assert result["row_active"].tolist() == [False, False, True, True]
    torch.testing.assert_close(result["group_covariance"], torch.tensor([[0.0], [1.0]]))
    remote = art_pairwise_covariance(x[:2], scores[:2], 2, any_preference_active=True)
    local = art_pairwise_covariance(x[:2], scores[:2], 2, any_preference_active=False)
    assert remote["row_trainable"].all() and not local["row_trainable"].any()
    shard = art_pairwise_covariance(x[2:], scores[2:], 2, seed=73, group_offset=1)
    torch.testing.assert_close(shard["pair_rows"] + 2, result["pair_rows"][1:])
    assert not any(t.requires_grad for t in result.values())


@pytest.mark.parametrize("group_size", [2, 4, 8])
def test_pairwise_seed_matches_independent_comparisons(group_size):
    x = torch.arange(group_size * 4).float().reshape(group_size * 2, 2)
    scores = torch.arange(group_size * 2).float() // 2
    result = art_pairwise_covariance(x, scores, group_size, seed=9, round_index=3, group_offset=7)
    for group in range(2):
        generator = torch.Generator().manual_seed(9 + 1_000_003 * (10 + group))
        rows = torch.randperm(group_size, generator=generator) + group_size * group
        for a, b in rows.reshape(-1, 2).tolist():
            expected = 0.5 if scores[a] == scores[b] else float(scores[a] > scores[b])
            assert result["binary_rewards"][a] == expected
            assert result["binary_rewards"][b] == 1 - expected


@pytest.mark.parametrize(
    "space,weight", [("velocity_uniform", 1.0), ("velocity_linear_sigma", 0.4), ("x0", 0.16)]
)
def test_art_frozen_targets_values_and_gradients(space, weight):
    current = leaf([[0.4, -0.2], [0.1, 0.3]])
    old = leaf([[0.1, 0.1], [0.1, 0.1]])
    direction = leaf([[1, -1], [1, -1]])
    magnitude = leaf([0.3, 0.3])
    state = leaf([[2, 3], [4, 5]])
    sigma = leaf([0.4, 0.4])
    result = art_loss(
        current,
        old,
        sigma,
        direction,
        magnitude,
        state=state,
        loss_space=space,
        velocity_mask_metric="fixed_mse",
        corrective_mask_enabled=False,
    )
    displacement = 0.3 if space != "x0" else 0.3 / 0.4
    velocity_target = torch.tensor([[0.1 - displacement, 0.1 + displacement]]).expand(2, 2)
    expected = weight * (current.detach() - velocity_target).square().mean(1)
    torch.testing.assert_close(result["per_item_loss"], expected)
    result["per_item_loss"].sum().backward()
    torch.testing.assert_close(current.grad, weight * (current.detach() - velocity_target))
    assert all(value.grad is None for value in (old, direction, magnitude, state, sigma))
    assert not result["target"].requires_grad


def test_art_corrective_mask_overshoot_anchor_and_zero_sigma():
    current = leaf([[-0.2], [-1.3], [0.4], [0.4], [0.4]])
    result = art_loss(
        current,
        torch.zeros_like(current),
        torch.tensor([0.5, 0.5, 0.5, 0.5, 0.0]),
        torch.tensor([[1.0], [1.0], [0.0], [0.0], [1.0]]),
        torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0]),
        velocity_mask_metric="fixed_mse",
        movement_mask_threshold=0.01,
        restore_anchor_mask=torch.tensor([False, False, True, False, True]),
    )
    assert result["keep_mask"].tolist() == [False, True, True, False, False]
    torch.testing.assert_close(result["per_item_loss"], torch.tensor([0, 0.045, 0.08, 0, 0]))
    result["per_item_loss"].sum().backward()
    torch.testing.assert_close(current.grad.flatten(), torch.tensor([0, -0.3, 0.4, 0, 0]))


@pytest.mark.parametrize("metric", ["q2_sde_kl", "scheduler_reference_metric"])
def test_art_scheduler_mask_initial_and_noninitial(metric):
    current = leaf([[-0.1], [-0.1], [-0.1]])
    sigma = torch.tensor([1.0, 0.5, 0.1])
    dt = torch.tensor([-0.2, -0.1, -0.05])
    result = art_loss(
        current,
        torch.zeros_like(current),
        sigma,
        torch.ones_like(current),
        torch.ones(3),
        velocity_mask_metric=metric,
        ode_velocity_coefficient=dt,
        q2_reference_kl_threshold=0.002,
    )
    std_squared = torch.tensor([0.8**2, 0.8**2 * 0.1, 0.8**2 * 0.1 / 0.9 * 0.05])
    coefficient = torch.tensor([-0.2, -0.1 * 1.32, -0.05 * 1.32])
    expected = coefficient.square() * 0.01 / (2 * std_squared)
    torch.testing.assert_close(result["movement_score"], expected)
    assert result["keep_mask"].tolist() == [True, True, False]


def test_art_other_mask_metrics_and_reference_penalty():
    v = leaf([[-0.2], [-0.2]])
    kwargs = dict(
        current_velocity=v,
        old_velocity=torch.zeros_like(v),
        sigma=torch.tensor([0.5, 0.1]),
        guidance_direction=torch.ones_like(v),
        guidance_magnitude=torch.ones(2),
    )
    normalized = art_loss(
        **kwargs, velocity_mask_metric="noise_normalized_v", noise_normalized_v_radius=0.4
    )
    assert normalized["keep_mask"].tolist() == [True, False]
    reference = leaf([[0.0], [0.0]])
    ode = art_loss(
        **kwargs,
        mask_space="ode_mean",
        velocity_mask_metric="fixed_mse",
        ode_velocity_coefficient=torch.tensor([-0.1, -0.1]),
        current_ode_mean=-0.1 * v,
        old_ode_mean=torch.zeros_like(v),
        movement_mask_threshold=0.0001,
        reference_prediction=reference,
        reference_mse_beta=2,
    )
    assert not ode["keep_mask"].any()
    torch.testing.assert_close(ode["per_item_loss"], torch.tensor([0.08, 0.08]))
    ode["per_item_loss"].sum().backward()
    torch.testing.assert_close(v.grad, 4 * v.detach())
    assert reference.grad is None


def test_nft_branch_normalizers_are_frozen_and_independent():
    prediction = leaf([[0.2, -0.4], [0.6, 0.1], [0.1, 0.1]])
    old = leaf([[0.1, 0.2], [-0.1, 0.3], [0.1, 0.1]])
    x0 = leaf([[0, 0], [0.1, 0.2], [0, 0]])
    xt = leaf([[0.7, -0.3], [0.2, -0.4], [0.05, 0.05]])
    sigma = leaf([0.4, 0.7, 0.5])
    advantage = leaf([7, -1, 0])
    result = nft_loss(prediction, old, x0, xt, sigma, advantage, nft_beta=0.7, adv_clip_max=3)
    expected_values, expected_gradients = [], []
    for i in range(3):
        p = (min(max(float(advantage.detach()[i]), -3), 3) / 3) / 2 + 0.5
        plus = (
            xt.detach()[i]
            - sigma.detach()[i] * (0.7 * prediction.detach()[i] + 0.3 * old.detach()[i])
            - x0.detach()[i]
        )
        minus = (
            xt.detach()[i]
            - sigma.detach()[i] * (1.7 * old.detach()[i] - 0.7 * prediction.detach()[i])
            - x0.detach()[i]
        )
        wp = max(float(plus.double().abs().mean()), 1e-5)
        wm = max(float(minus.double().abs().mean()), 1e-5)
        expected_values.append(
            3
            / 0.7
            * (
                p * plus.square().double().mean() / wp
                + (1 - p) * minus.square().double().mean() / wm
            )
        )
        expected_gradients.append(3 * sigma.detach()[i] * (-p * plus / wp + (1 - p) * minus / wm))
    torch.testing.assert_close(result["per_item_loss"], torch.stack(expected_values))
    result["per_item_loss"].sum().backward()
    torch.testing.assert_close(
        prediction.grad, torch.stack(expected_gradients), atol=1e-6, rtol=1e-5
    )
    assert all(value.grad is None for value in (old, x0, xt, sigma, advantage))
    assert not result["positive_weight"].requires_grad
    assert result["positive_weight"].dtype == torch.float64


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("all", [0.0, 0.25, 0.5, 1.0]),
        ("positive_only", [0.5, 0.5, 0.5, 1.0]),
        ("negative_only", [0.0, 0.25, 0.5, 0.5]),
        ("one_only", [0.5, 0.5, 0.5, 0.75]),
        ("binary", [0.25, 0.25, 0.5, 0.75]),
    ],
)
def test_nft_advantage_modes(mode, expected):
    p = torch.zeros(4, 1, requires_grad=True)
    result = nft_loss(
        p,
        p.detach(),
        p.detach(),
        torch.ones_like(p),
        torch.ones(4),
        torch.tensor([-4.0, -1.0, 0.0, 3.0]),
        adv_mode=mode,
        adv_clip_max=2,
    )
    torch.testing.assert_close(result["positive_probability"], torch.tensor(expected))


def test_nft_bfloat16_prediction_preserves_float32_reference():
    p = torch.tensor([[0.5, 1.0]], dtype=torch.bfloat16, requires_grad=True)
    ref = leaf([[0.5003, 1.002]])
    result = nft_loss(
        p,
        p.detach(),
        torch.zeros(1, 2),
        torch.zeros(1, 2),
        torch.zeros(1),
        torch.zeros(1),
        reference_prediction=ref,
        reference_mse_beta=3,
    )
    expected = (p.float() - ref.detach()).square().mean(1)
    torch.testing.assert_close(result["per_item_reference_mse"], expected)
    result["per_item_loss"].sum().backward()
    torch.testing.assert_close(p.grad, (3 * (p.float() - ref.detach())).to(torch.bfloat16))
    assert ref.grad is None


def test_art_x0_projection_radius():
    v = leaf([[0.0, 0.0]])
    result = art_loss(
        v,
        v.detach(),
        torch.tensor([0.5]),
        torch.tensor([[3.0, 4.0]]),
        torch.ones(1),
        state=torch.zeros_like(v),
        loss_space="x0",
        target_projection_radius=0.2,
        velocity_mask_metric="fixed_mse",
        corrective_mask_enabled=False,
    )
    torch.testing.assert_close(result["target"].square().mean().sqrt(), torch.tensor(0.2))
    torch.testing.assert_close(result["per_item_loss"], torch.tensor([0.04]))


def test_kl_actual_std_feature_broadcast_and_epsilon_floor():
    current = leaf([[1.0, 2.0], [3.0, 4.0]])
    std = torch.tensor([[0.5, 1.0], [1.5, 2.0]])
    torch.testing.assert_close(
        transition_kl_per_sample(current, torch.zeros_like(current), std), torch.tensor([2.0, 2.0])
    )
    floor = transition_kl_per_sample(current * 1e-8, torch.zeros_like(current), torch.tensor(0.0))
    torch.testing.assert_close(floor, torch.tensor([1.25, 6.25]))


def test_clipped_grpo_values_gradients_and_componentwise_mixture():
    logp = leaf([0.3, -0.3, 0.3, -0.3, 0])
    old = leaf([0, 0, 0, 0, 0])
    advantage = leaf([2, -2, -2, 2, 9])
    result = clipped_grpo_loss(logp, old, advantage, ratio_clip=0.1)
    ratio = logp.detach().exp()
    expected = torch.tensor([-2.2, 1.8, 2 * ratio[2], -2 * ratio[3], -5])
    torch.testing.assert_close(result["per_item_loss"], expected)
    result["per_item_loss"].sum().backward()
    torch.testing.assert_close(logp.grad, torch.tensor([0, 0, 2 * ratio[2], -2 * ratio[3], -5]))
    assert old.grad is None and advantage.grad is None
    components = clipped_grpo_loss(
        logp,
        old,
        {"a": torch.ones(5), "b": -torch.ones(5)},
        ratio_clip=0.1,
        component_weights={"a": 0.5, "b": 0.5},
    )
    assert components["per_item_loss"][0] > 0  # Clipping a zero mixed advantage would give zero.


def test_flowdppo_threshold_equality_signs_and_gradients():
    logp = leaf([0.2, -0.2, -0.2, 0.2, 0, 0.2])
    old_logp = leaf([0] * 6)
    means = leaf([[1.0, -1.0]] * 6)
    old_mean = leaf([[0.0, 0.0]] * 6)
    std = leaf([1] * 6)
    adv = leaf([2, -2, 2, -2, 2, 0])
    result = flowdppo_loss(logp, old_logp, means, old_mean, std, adv, kl_mask_threshold=0.5)
    assert result["keep_mask"].tolist() == [False, False, True, True, True, True]
    expected = torch.tensor([0.0, 0.0, -2.0, 2.0, -2.0, 0.0]) * logp.detach().exp()
    torch.testing.assert_close(result["per_item_loss"], expected)
    result["per_item_loss"].sum().backward()
    torch.testing.assert_close(logp.grad, expected)
    assert all(t.grad is None for t in (old_logp, old_mean, std, adv, means))
    kl = transition_kl_per_sample(means, old_mean, std)
    torch.testing.assert_close(kl, torch.full((6,), 0.5))
    kl.sum().backward()
    torch.testing.assert_close(means.grad, means.detach() / 2)
    assert std.grad is None and old_mean.grad is None


def test_refl_gradient_is_negative_weight_not_detached_or_centered():
    rewards = {"VQ": leaf([2, -1]), "MQ": leaf([0, 3]), "TA": leaf([1, 1])}
    result = refl_loss(rewards, {"VQ": 0.2, "MQ": 0.3, "TA": 0.5})
    torch.testing.assert_close(result["per_item_loss"], torch.tensor([-0.9, -1.2]))
    result["per_item_loss"].sum().backward()
    for key, weight in {"VQ": 0.2, "MQ": 0.3, "TA": 0.5}.items():
        torch.testing.assert_close(rewards[key].grad, torch.full((2,), -weight))


def test_decay_schedule_zero_based_and_validation():
    assert [old_policy_decay(i, "slow-ema") for i in (0, 1, 480, 999)] == [0, 0.00125, 0.6, 0.6]
    assert old_policy_decay(1000, "no-ema") == 0
    for step in (-1, True, 0.5):
        with pytest.raises(ValueError):
            old_policy_decay(step, "slow-ema")
    with pytest.raises(ValueError):
        old_policy_decay(0, "slow-ema", maximum=1)


def test_invalid_shapes_and_unsupported_modes_fail_explicitly():
    v = torch.ones(3, 2)
    with pytest.raises(ValueError):
        art_loss(v, v, torch.ones(3), v, torch.ones(3), loss_space="lateral_restore")
    with pytest.raises(ValueError):
        art_loss(v, v, torch.ones(3), v, torch.ones(3))  # Scheduler mask needs dt.
    with pytest.raises(ValueError):
        nft_loss(v, v, v, v, torch.ones(3), torch.ones(2))
    with pytest.raises(ValueError):
        clipped_grpo_loss(v, v, torch.ones(3), ratio_clip=1e-4)
    with pytest.raises(ValueError):
        art_pairwise_covariance(v, torch.ones(3), 2)
    with pytest.raises(ValueError):
        refl_loss({"a": torch.ones(3)}, {"a": 2.0})
