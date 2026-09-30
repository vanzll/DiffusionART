"""Independent formula checks; no dependency on the private reference trainer."""

import math
from copy import deepcopy

import numpy as np
import pytest
import torch

from visual_rl.training.multi_reward import combined_group_advantages


def formula_oracle(rewards, prompts, weights, eps, renormalize):
    # Scalar Python arithmetic keeps this independent of vectorized NumPy grouping.
    components = {}
    for name, reward in rewards.items():
        rows = reward.detach().double().reshape(len(prompts), -1).tolist()
        result = [[0.0] * len(rows[0]) for _ in rows]
        for i, prompt in enumerate(prompts):
            for column in range(len(rows[0])):
                values = [row[column] for row, p in zip(rows, prompts) if p == prompt]
                mean = sum(values) / len(values)
                std = math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))
                result[i][column] = (rows[i][column] - mean) / (std + eps)
        components[name] = torch.tensor(result, dtype=torch.float64).reshape(reward.shape)
    total = sum(weights.values())
    combined = sum(components[name] * (weights[name] / total) for name in sorted(weights))
    if renormalize:
        rows = combined.reshape(len(prompts), -1).tolist()
        for column in range(len(rows[0])):
            mean = sum(row[column] for row in rows) / len(rows)
            centered = [row[column] - mean for row in rows]
            centered_mean = sum(centered) / len(centered)
            std = math.sqrt(sum((x - centered_mean) ** 2 for x in centered) / len(centered))
            for i, row in enumerate(rows):
                row[column] = centered[i] / (std + eps)
        combined = torch.tensor(rows, dtype=torch.float64).reshape(combined.shape)
    return combined, components


@pytest.mark.parametrize("matrix", [False, True])
@pytest.mark.parametrize("renormalize", [False, True])
def test_independent_formula_repeated_prompts_and_disagreement(matrix, renormalize):
    a = torch.tensor([1, 8, 5, 2, 4, 9, 13], dtype=torch.float64)
    b = torch.tensor([9, 2, 1, 2, 3, 4, 7], dtype=torch.float64)
    if matrix:
        a = torch.stack((a, a.square(), torch.ones_like(a)), dim=1)
        b = torch.stack((b, -b.square(), torch.ones_like(b)), dim=1)
    rewards = {"z": a, "a": b}
    prompts = ["repeat", "other", "repeat", "other", "repeat", "other", "singleton"]
    weights = {"z": 2.0, "a": 5.0}
    expected, expected_parts = formula_oracle(rewards, prompts, weights, 0.03, renormalize)
    combined, parts, metrics = combined_group_advantages(
        rewards, prompts, weights, eps=0.03, renormalize_combined=renormalize
    )
    torch.testing.assert_close(combined, expected, atol=2e-15, rtol=2e-15)
    for name in parts:
        torch.testing.assert_close(parts[name], expected_parts[name], atol=2e-15, rtol=2e-15)
    for value in (combined, *parts.values()):
        assert value.dtype == torch.float64 and value.device.type == "cpu"
        assert not value.requires_grad and value.grad_fn is None
    assert all(type(value) is float and math.isfinite(value) for value in metrics.values())


@pytest.mark.parametrize("renormalize", [False, True])
def test_exact_zeros_for_singletons_and_representable_ties(renormalize):
    rewards = {"a": torch.tensor([0.25, 0.25, 0.25, 77], dtype=torch.float64)}
    combined, parts, metrics = combined_group_advantages(
        rewards, ["p", "p", "p", "q"], {"a": 3}, renormalize_combined=renormalize
    )
    assert torch.count_nonzero(combined) == torch.count_nonzero(parts["a"]) == 0
    assert metrics["all_zero_fraction"] == metrics["a/zero_fraction"] == 1
    assert metrics["scalarization_retention"] == 0
    assert "correlation_mean" not in metrics


@pytest.mark.parametrize("matrix", [False, True])
@pytest.mark.parametrize("renormalize", [False, True])
def test_nonrepresentable_ties_preserve_fp64_rounding_residual(matrix, renormalize):
    values = torch.tensor([0.1, 0.1, 0.1, 77], dtype=torch.float64)
    if matrix:
        values = torch.stack((values, torch.full_like(values, 0.25)), dim=1)
    rewards, prompts, weights = {"a": values}, ["p", "p", "p", "q"], {"a": 3}
    expected, expected_parts = formula_oracle(rewards, prompts, weights, 1e-4, renormalize)
    combined, parts, metrics = combined_group_advantages(
        rewards, prompts, weights, renormalize_combined=renormalize
    )
    torch.testing.assert_close(parts["a"], expected_parts["a"], atol=0, rtol=0)
    torch.testing.assert_close(combined, expected, atol=0, rtol=0)
    # Three repeated 0.1 values round their mean away from the input value.
    residual = parts["a"][:3, 0] if matrix else parts["a"][:3]
    assert torch.all(residual != 0)
    assert torch.count_nonzero(parts["a"][-1]) == 0
    assert metrics["all_zero_fraction"] == metrics["a/zero_fraction"] == (0.625 if matrix else 0.25)


def test_conflicting_objectives_and_original_diagnostics():
    a = torch.tensor([-1, 1, -1, 1, 4], dtype=torch.float64)
    combined, parts, metrics = combined_group_advantages(
        {"a": a, "b": -a, "c": torch.ones(5)},
        ["p", "p", "q", "q", "s"],
        {"a": 2, "b": 2, "c": 4},
        eps=0.25,
    )
    assert torch.count_nonzero(combined) == 0
    torch.testing.assert_close(parts["a"], a.new_tensor([-0.8, 0.8, -0.8, 0.8, 0]))
    assert metrics["conflict_fraction"] == 0.8
    assert metrics["all_zero_fraction"] == 0.2
    assert metrics["scalarization_retention"] == 0
    assert metrics["a/weight"] == metrics["b/weight"] == 0.25
    assert metrics["c/weight"] == 0.5
    assert metrics["a/abs_mean"] == pytest.approx(0.64)
    assert metrics["a/group_std_mean"] == pytest.approx(2 / 3)
    assert metrics["a/group_zero_std_fraction"] == pytest.approx(1 / 3)
    assert metrics["a/raw_mean"] == 0.8
    assert metrics["a/raw_std"] == pytest.approx(np.std(a.numpy()))
    assert metrics["correlation/a__b"] == pytest.approx(-1)
    assert metrics["correlation/a__c"] == metrics["correlation/b__c"] == 0
    assert metrics["correlation_min"] == pytest.approx(-1)
    assert metrics["correlation_mean"] == pytest.approx(-1 / 3)


def test_population_std_eps_after_std_and_normalized_weights():
    rewards = {"a": torch.tensor([0.0, 4.0]), "b": torch.tensor([100.0, 100.0])}
    result, parts, metrics = combined_group_advantages(rewards, ["p", "p"], {"a": 2, "b": 6}, eps=2)
    torch.testing.assert_close(parts["a"], torch.tensor([-0.5, 0.5], dtype=torch.float64))
    torch.testing.assert_close(result, parts["a"] / 4)
    scaled = combined_group_advantages(rewards, ["p", "p"], {"a": 20, "b": 60}, eps=2)
    assert torch.equal(result, scaled[0]) and metrics == scaled[2]
    renorm, _, _ = combined_group_advantages(
        rewards, ["p", "p"], {"a": 2, "b": 6}, eps=2, renormalize_combined=True
    )
    torch.testing.assert_close(renorm, result / (0.125 + 2))


def test_large_offset_tiny_spread_preserves_fp64():
    values = torch.tensor([1e12 - 0.001, 1e12, 1e12 + 0.001], dtype=torch.float64)
    actual, parts, _ = combined_group_advantages({"a": values}, ["p"] * 3, {"a": 1})
    expected, _ = formula_oracle({"a": values}, ["p"] * 3, {"a": 1}, 1e-4, False)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual[0] < -0.5 and actual[2] > 0.5
    assert torch.equal(parts["a"], actual)


@pytest.mark.parametrize("renormalize", [False, True])
def test_columns_are_independent_and_groups_use_exact_text(renormalize):
    rewards = {
        "a": torch.tensor([[1, 100, 0], [4, 400, 0], [10, 2, 0], [15, 4, 0]], dtype=torch.float64),
        "b": torch.tensor([[2, 3, 8], [1, 5, 8], [3, 8, 8], [9, 9, 8]], dtype=torch.float64),
    }
    prompts = ["p", "p", "p ", "p "]
    weights = {"a": 2, "b": 1}
    combined, parts, metrics = combined_group_advantages(
        rewards, prompts, weights, renormalize_combined=renormalize
    )
    for column in range(3):
        separate, separate_parts, _ = combined_group_advantages(
            {name: value[:, column] for name, value in rewards.items()},
            prompts,
            weights,
            renormalize_combined=renormalize,
        )
        torch.testing.assert_close(combined[:, column], separate, atol=1e-15, rtol=1e-15)
        for name in parts:
            assert torch.equal(parts[name][:, column], separate_parts[name])
    # Preserve the canonical diagnostic's flattened group std, not per-column std.
    assert metrics["a/group_std_mean"] == pytest.approx(
        (np.std(rewards["a"][:2].numpy()) + np.std(rewards["a"][2:].numpy())) / 2
    )


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.int64]
)
def test_no_input_mutation_or_autograd_and_no_output_aliasing(dtype):
    source = torch.arange(12).reshape(3, 4).to(dtype).T
    if source.is_floating_point():
        source.requires_grad_()
    assert not source.is_contiguous()
    rewards, prompts, weights = {"a": source}, ["p"] * 4, {"a": 3.0}
    before, old_prompts, old_weights = source.detach().clone(), prompts[:], deepcopy(weights)
    combined, parts, _ = combined_group_advantages(rewards, prompts, weights)
    assert combined.dtype == parts["a"].dtype == torch.float64
    assert not combined.requires_grad and not parts["a"].requires_grad
    assert source.grad is None and torch.equal(source, before)
    combined.fill_(99)
    assert not torch.all(parts["a"] == 99)
    parts["a"].fill_(77)
    assert torch.equal(source, before) and prompts == old_prompts and weights == old_weights
    assert rewards["a"] is source


def test_sorted_component_accumulation_is_insertion_order_independent():
    generator = torch.Generator().manual_seed(17)
    rewards = {
        name: torch.randn(12, 3, generator=generator, dtype=torch.float64) for name in "zabc"
    }
    weights = {"z": 1e16, "a": 1.0, "b": 1.0, "c": 1e16}
    prompts = ["p", "q", "r"] * 4
    forward = combined_group_advantages(rewards, prompts, weights)
    reverse = combined_group_advantages(
        dict(reversed(list(rewards.items()))), prompts, dict(reversed(list(weights.items())))
    )
    assert torch.equal(forward[0], reverse[0]) and forward[2] == reverse[2]
    assert list(forward[1]) == sorted(rewards)
    for name in rewards:
        assert torch.equal(forward[1][name], reverse[1][name])


@pytest.mark.parametrize(
    "value", [True, np.bool_(True), 0, -1, float("nan"), float("inf"), -float("inf"), "0.1", None]
)
def test_invalid_eps(value):
    with pytest.raises(ValueError, match="eps"):
        combined_group_advantages({"a": torch.ones(2)}, ["p", "p"], {"a": 1}, eps=value)


@pytest.mark.parametrize("value", [1, 0, "false", None, np.bool_(True)])
def test_invalid_renormalization_flag(value):
    with pytest.raises(ValueError, match="bool"):
        combined_group_advantages(
            {"a": torch.ones(2)}, ["p", "p"], {"a": 1}, renormalize_combined=value
        )


@pytest.mark.parametrize(
    "value",
    [True, np.bool_(True), 0, -1, float("nan"), float("inf"), -float("inf"), "1", None, 10**400],
)
def test_invalid_weights(value):
    with pytest.raises(ValueError, match="weights"):
        combined_group_advantages({"a": torch.ones(2)}, ["p", "p"], {"a": value})


def test_weight_sum_overflow():
    with pytest.raises(ValueError, match="weight sum"):
        combined_group_advantages(
            {"a": torch.ones(2), "b": torch.ones(2)}, ["p", "p"], {"a": 1e308, "b": 1e308}
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_reward(value):
    with pytest.raises(ValueError, match="finite"):
        combined_group_advantages({"a": torch.tensor([0, value])}, ["p", "p"], {"a": 1})


@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize(
    "values, calculated",
    [([1e308, 1e308], "advantages"), ([-1e308, 1e308], "metrics")],
)
def test_finite_rewards_with_overflowed_calculations_are_rejected(values, calculated, renormalize):
    rewards = {"a": torch.tensor(values, dtype=torch.float64)}
    assert torch.isfinite(rewards["a"]).all()
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        with pytest.raises(ValueError, match=f"calculated {calculated} must be finite"):
            combined_group_advantages(
                rewards, ["p", "p"], {"a": 1}, renormalize_combined=renormalize
            )


@pytest.mark.parametrize(
    "reward",
    [
        torch.tensor(1),
        torch.empty(0),
        torch.empty(2, 0),
        torch.ones(2, 1, 1),
        torch.ones(3),
        torch.ones(2, dtype=torch.complex128),
        torch.ones(2, dtype=torch.bool),
        [1, 2],
        np.ones(2),
    ],
)
def test_malformed_rewards(reward):
    with pytest.raises(ValueError):
        combined_group_advantages({"a": reward}, ["p", "p"], {"a": 1})


@pytest.mark.parametrize(
    "prompts", [[], ["p"], ["p", ""], ["p", "  "], ["p", None], ["p", 1], "pp", None, {"p", "q"}]
)
def test_malformed_prompts(prompts):
    with pytest.raises(ValueError):
        combined_group_advantages({"a": torch.ones(2)}, prompts, {"a": 1})


@pytest.mark.parametrize("weights", [{}, {"b": 1}, {"a": 1, "b": 1}, [], None])
def test_weight_keys_must_match(weights):
    with pytest.raises(ValueError, match="keys"):
        combined_group_advantages({"a": torch.ones(2)}, ["p", "p"], weights)


@pytest.mark.parametrize(
    "rewards, weights",
    [({}, {}), ([], {}), (None, {}), ({1: torch.ones(2)}, {1: 1}), ({"": torch.ones(2)}, {"": 1})],
)
def test_empty_or_invalid_reward_mapping(rewards, weights):
    with pytest.raises(ValueError):
        combined_group_advantages(rewards, ["p", "p"], weights)


@pytest.mark.parametrize("shape", [(2, 1), (2, 2)])
def test_components_must_have_identical_shapes(shape):
    with pytest.raises(ValueError, match="matching shapes"):
        combined_group_advantages(
            {"a": torch.ones(2), "b": torch.ones(shape)}, ["p", "p"], {"a": 1, "b": 1}
        )
