from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from visual_rl.algorithms import art_scalar_covariance
from visual_rl.backends import hunyuan, sd3, wan
from visual_rl.config import ConfigError, load_config
from visual_rl.training import video_pairwise
from visual_rl.training.objectives import disjoint_windows, prompt_disjoint_windows
from visual_rl.training.update import accumulated_step


@pytest.mark.parametrize("world", [1, 8, 16, 32])
@pytest.mark.parametrize("k", [2, 4, 8, 16, 18, 24])
@pytest.mark.parametrize("p", [32, 64])
def test_global_prompt_coverage_is_independent_of_rank_partition(world, k, p):
    slots = 3
    groups = torch.arange(p).repeat_interleave(k)
    expected = prompt_disjoint_windows(groups, slots)
    assembled = [[], []]
    local_rows = p * k // world
    for rank in range(world):
        start = rank * local_rows
        local = prompt_disjoint_windows(groups[start:start + local_rows], slots)
        for window, items in enumerate(local):
            assert len(items) == local_rows * slots // 2
            assembled[window].extend((row + start, slot) for row, slot in items)
    assert tuple(tuple(items) for items in assembled) == expected
    for items in expected:
        counts = Counter(int(groups[row]) for row, slot in items if slot == 0)
        assert counts == {group: k // 2 for group in range(p)}
        assert len(set(items)) == p * k * slots // 2
    assert set(expected[0]).isdisjoint(expected[1])
    assert set(expected[0]) | set(expected[1]) == {
        (row, slot) for row in range(p * k) for slot in range(slots)
    }


def test_permuted_interleaved_groups_keep_all_slots_and_one_update_order():
    groups = torch.tensor([4, 9, 4, 9, 4, 9, 4, 9])
    order = torch.randperm(8, generator=torch.Generator().manual_seed(42)).tolist()
    windows = prompt_disjoint_windows(groups, 5, order=order)
    for items in windows:
        rows = [row for row, slot in items if slot == 0]
        assert Counter(int(groups[row]) for row in rows) == {4: 2, 9: 2}
        assert rows == [row for row in order if row in rows]
        assert all({slot for r, slot in items if r == row} == set(range(5)) for row in rows)
    assert prompt_disjoint_windows(groups, 5, updates=1, order=order) == (
        tuple((row, slot) for row in order for slot in range(5)),
    )


def test_one_prompt_video_keeps_historical_window_order():
    assert prompt_disjoint_windows(torch.zeros(8, dtype=torch.long), 9) == disjoint_windows(8, 9)


@pytest.mark.parametrize("groups,slots,updates,order", [
    (torch.tensor([0, 0, 0, 1, 1, 1]), 2, 2, None),
    (torch.tensor([], dtype=torch.long), 2, 2, None),
    (torch.tensor([0., 0.]), 2, 2, None),
    (torch.tensor([-1, -1]), 2, 2, None),
    (torch.tensor([0, 0]), 0, 2, None),
    (torch.tensor([0, 0]), 2, 0, None),
    (torch.tensor([0, 0]), 2, 2, [0, 0]),
])
def test_invalid_or_incomplete_groups_fail(groups, slots, updates, order):
    with pytest.raises(ValueError):
        prompt_disjoint_windows(groups, slots, updates, order=order)


@pytest.mark.parametrize("kind", ["image", "video"])
def test_odd_k_rejected_even_when_total_row_count_is_even(tmp_path, kind):
    if kind == "image":
        from test_sd3_runner import config_file
        backend = sd3
    else:
        from test_hunyuan_runner import config_file
        backend = hunyuan
    c = load_config(config_file(tmp_path))
    c = replace(c, sampling=replace(c.sampling, prompts_per_rollout=2, samples_per_prompt=3))
    with pytest.raises(ConfigError, match="samples_per_prompt"):
        backend.validate_config(c)


def test_single_update_image_art_still_accepts_odd_k(tmp_path):
    from test_sd3_runner import config_file

    c = load_config(config_file(tmp_path))
    c = replace(c, sampling=replace(c.sampling, samples_per_prompt=3),
                algorithm=replace(c.algorithm, options={**c.algorithm.options, "updates_per_rollout": 1}))
    sd3.validate_config(c)
    groups = torch.tensor([0, 0, 0, 1, 1, 1])
    order = [2, 5, 0, 1, 4, 3]
    assert prompt_disjoint_windows(groups, 2, updates=1, order=order) == (
        tuple((row, slot) for row in order for slot in range(2)),
    )


def inputs(p=4, k=8):
    cache = SimpleNamespace(endpoints=torch.arange(p * k * 3).reshape(p * k, 3).float(),
                            selected=(0, 1, 2))
    rewards = torch.arange(k).float().repeat(p)
    rewards[:k] = 1  # A real zero-guidance group must stay in both denominators.
    scores = SimpleNamespace(weighted=rewards)
    c = SimpleNamespace(
        seed=42, model=SimpleNamespace(name="sd3"), training=SimpleNamespace(precision="fp32"),
        sampling=SimpleNamespace(samples_per_prompt=k),
        algorithm=SimpleNamespace(name="diffusionart", options={
            "covariance": {}, "loss": {}, "restore_zero_signal": False}),
    )
    return c, cache, scores


@pytest.mark.parametrize("backend", [hunyuan, wan])
@pytest.mark.parametrize("restore", [False, True])
def test_video_scalar_uses_full_covariance_and_balanced_windows(backend, restore):
    c, cache, scores = inputs()
    c.algorithm.options["restore_zero_signal"] = restore
    parts = SimpleNamespace(adapter=SimpleNamespace(transformer=torch.nn.Linear(1, 1)))
    obj, windows = backend.prepare_objective(c, parts, cache, None, scores, scores, torch.device("cpu"))
    groups = torch.arange(32) // 8
    stats = art_scalar_covariance(cache.endpoints, scores.weighted, groups, 4)
    torch.testing.assert_close(obj.row_directions, stats["group_directions"][groups])
    assert obj.row_anchors[:8].tolist() == [restore] * 8
    assert windows == prompt_disjoint_windows(groups, 3)
    assert all(len(w) == 48 for w in windows)


@pytest.mark.parametrize("model", ["sd3", "flux"])
@pytest.mark.parametrize("updates", [1, 2])
def test_image_routing_full_covariance_rng_and_one_update_parity(model, updates):
    c, cache, scores = inputs()
    c.model.name = model
    c.algorithm.options["updates_per_rollout"] = updates
    groups = torch.arange(32) // 8
    generator = torch.Generator().manual_seed(42)
    reference = torch.Generator().manual_seed(42)
    order = torch.randperm(32, generator=reference).tolist()
    obj, windows = sd3._art_update(c, cache, None, SimpleNamespace(scores=scores),
                                 groups, 4, 8, cache.selected, generator, torch.device("cpu"))
    stats = art_scalar_covariance(cache.endpoints, scores.weighted, groups, 4)
    torch.testing.assert_close(obj.row_directions, stats["group_directions"][groups])
    assert torch.equal(generator.get_state(), reference.get_state())
    assert windows == prompt_disjoint_windows(groups, 3, updates=updates, order=order)
    assert all(len(w) == 96 // updates for w in windows)


def test_pairwise_keeps_feedback_anchors_and_inactive_rows():
    c, cache, _ = inputs()
    preference = torch.ones(32, dtype=torch.bool)
    preference[:8] = False
    feedback = {"row_preference_active": preference,
                "group_directions": torch.ones(4, 3),
                "row_guidance_magnitudes": preference.float()}
    obj, windows, _ = video_pairwise.finalize(
        c, cache, None, feedback, torch.tensor([1, 1, 1]), torch.device("cpu")
    )
    assert torch.equal(obj.row_anchors, ~preference)
    assert torch.equal(obj.row_magnitudes, preference.float())
    assert windows == prompt_disjoint_windows(torch.arange(32) // 8, 3)
    assert all(len(w) == 48 for w in windows)


@pytest.mark.parametrize("microbatch", [1, 3, 8, 100])
def test_two_updates_keep_original_denominator_and_frozen_targets(microbatch):
    groups = torch.arange(4).repeat_interleave(8)
    windows = prompt_disjoint_windows(groups, 3)
    model = torch.nn.Linear(2, 1, bias=False).double()
    model.weight.data.fill_(0.1)
    reference = torch.nn.Linear(2, 1, bias=False).double()
    reference.load_state_dict(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=0)
    oracle = torch.optim.AdamW(reference.parameters(), lr=.01, weight_decay=0)
    features = torch.arange(32 * 3 * 2, dtype=torch.float64).reshape(32, 3, 2) / 100
    targets = reference(features).detach().squeeze(-1) + 0.2
    frozen = targets.clone()

    def loss(policy, items):
        rows, slots = torch.tensor(items).unbind(1)
        active = (groups[rows] != 0).double()
        return (policy(features[rows, slots]).squeeze(-1) - targets[rows, slots]).square() * active

    for window in windows:
        oracle.zero_grad()
        loss(reference, window).mean().backward()
        oracle.step()
        accumulated_step(model, optimizer, window, lambda items: loss(model, items),
                         microbatch=microbatch, denominator=len(window), max_grad_norm=1e6)
        torch.testing.assert_close(model.weight, reference.weight, rtol=1e-10, atol=1e-12)
    assert torch.equal(targets, frozen)
