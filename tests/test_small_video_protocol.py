import io

import pytest
import torch

from visual_rl.training.small_video_protocol import (
    ActualStepBudget,
    RolloutSelection,
    group_video_reward,
    group_windows,
)


def test_per_reward_population_normalization_before_sum_with_ta_monitor_only():
    groups = torch.arange(16).repeat_interleave(8)
    x = torch.arange(128, dtype=torch.float32).reshape(16, 8)
    raw = {
        "VQ": (x.sin() * 2 + x * 0.02).flatten(),
        "MQ": (x.cos() * 7 - x * 0.03).flatten(),
        "TA": x.flatten() * 1e4,
    }
    output = group_video_reward(raw, groups)
    expected = []
    for name in ("VQ", "MQ"):
        r = raw[name].double().reshape(16, 8)
        expected.append(
            (r - r.mean(1, keepdim=True)) / (r.std(1, unbiased=False, keepdim=True) + 1e-4)
        )
    torch.testing.assert_close(
        output["target"], ((expected[0] + expected[1]) / 2).flatten().float()
    )
    combined = (raw["VQ"] + raw["MQ"]).reshape(16, 8) / 2
    wrong = (
        (combined - combined.mean(1, keepdim=True))
        / (combined.std(1, unbiased=False, keepdim=True) + 1e-4)
    ).flatten()
    assert not torch.allclose(output["target"], wrong)
    assert output["metrics"]["reward/raw/TA_mean"] > 1000
    assert output["metrics"]["reward/raw/VQ_MQ_equal_mean"] == pytest.approx(
        float(((raw["VQ"] + raw["MQ"]) / 2).double().mean())
    )
    assert output["target"].reshape(16, 8).mean(1).abs().max() < 1e-6


@pytest.mark.parametrize("invalid", ["missing", "nan", "partial"])
def test_reward_fails_closed_instead_of_imputing_or_rank_local_normalizing(invalid):
    groups = torch.zeros(8, dtype=torch.long)
    raw = {k: torch.arange(8, dtype=torch.float32) for k in ("VQ", "MQ", "TA")}
    if invalid == "missing":
        del raw["TA"]
    elif invalid == "nan":
        raw["MQ"][2] = float("nan")
    else:
        groups = groups[:4]
        raw = {k: v[:4] for k, v in raw.items()}
    with pytest.raises(ValueError):
        group_video_reward(raw, groups)


def test_group_statistics_do_not_depend_on_dp_assignment_of_complete_groups():
    groups = torch.arange(16).repeat_interleave(8)
    raw = {
        k: torch.arange(128, dtype=torch.float32).sin() * (i + 1)
        for i, k in enumerate(("VQ", "MQ", "TA"))
    }
    expected = group_video_reward(raw, groups)["target"]
    for rank in range(8):
        selected = groups % 8 == rank
        actual = group_video_reward({k: v[selected] for k, v in raw.items()}, groups[selected])
        torch.testing.assert_close(actual["target"], expected[selected], rtol=0, atol=0)


def test_every_prompt_k4_k4_windows_are_disjoint_and_cover_complete_k8():
    groups = torch.arange(16).repeat_interleave(8)
    a, b = group_windows(groups)
    assert len(a) == len(b) == 64
    assert not set(a.tolist()) & set(b.tolist())
    assert set(a.tolist()) | set(b.tolist()) == set(range(128))
    for rows in (a, b):
        assert torch.bincount(groups[rows]).tolist() == [4] * 16


@pytest.mark.parametrize("selected", [2, 7])
@pytest.mark.parametrize("microbatch", [1, 4, 8])
@pytest.mark.parametrize("dp", [1, 2, 8])
def test_masked_selected_pair_denominator_gradient_equivalence(selected, microbatch, dp):
    # One64-video window, shared parameters and nonparallel per-pair gradients.
    generator = torch.Generator().manual_seed(71)
    inputs = torch.randn(64 * selected, 3, generator=generator)
    target = torch.randn(64 * selected, generator=generator)
    mask = torch.arange(len(target)) % 3 != 0
    p = torch.tensor([0.3, -0.2, 0.1], requires_grad=True)
    expected = torch.autograd.grad((((inputs @ p - target) ** 2) * mask).mean(), p)[0]
    total = torch.zeros_like(p)
    for rank in range(dp):
        rows = torch.arange(len(target))[rank::dp]
        q = p.detach().clone().requires_grad_()
        for chunk in rows.split(microbatch):
            loss = (
                (((inputs[chunk] @ q - target[chunk]) ** 2) * mask[chunk]).sum() * dp / len(target)
            )
            loss.backward()
        total += q.grad / dp
    torch.testing.assert_close(total, expected, rtol=1e-5, atol=1e-6)


def test_flash_private_rng_resume_shared_windows_and_original_full_positions():
    torch.manual_seed(912)
    before = torch.get_rng_state().clone()
    selection = RolloutSelection(42)
    first = selection.choose(tuple(range(8)), flash=True)
    assert len(first) == len(set(first)) == 2 and set(first) <= set(range(4))
    state = selection.state_dict()
    expected = [selection.choose(tuple(range(8)), flash=True) for _ in range(5)]
    restored = RolloutSelection(0)
    restored.load_state_dict(state)
    assert expected == [restored.choose(tuple(range(8)), flash=True) for _ in range(5)]
    assert torch.equal(before, torch.get_rng_state())
    assert restored.choose(tuple(range(8)), flash=False) == tuple(range(8))


def test_actual_adam_calls_resume_and_checkpoint_share_one_ten_step_budget():
    groups = torch.arange(16).repeat_interleave(8)
    windows = group_windows(groups)
    parameter = torch.nn.Parameter(torch.tensor([0.5]))
    optimizer = torch.optim.AdamW([parameter], lr=0.01)
    budget = ActualStepBudget(optimizer)

    def step():
        rollout, window = divmod(budget.completed, 2)
        budget.begin_window(
            rollout=rollout,
            window=window,
            rows=windows[window].tolist(),
            groups=groups,
            selected_positions=[0, 2],
        )
        optimizer.zero_grad()
        parameter.square().sum().backward()
        optimizer.step()

    for _ in range(4):
        step()
    buffer = io.BytesIO()
    torch.save(
        {
            "parameter": parameter.detach(),
            "optimizer": optimizer.state_dict(),
            "budget": budget.state_dict(),
        },
        buffer,
    )
    budget.close()
    buffer.seek(0)
    checkpoint = torch.load(buffer, weights_only=False)
    parameter = torch.nn.Parameter(checkpoint["parameter"].clone())
    optimizer = torch.optim.AdamW([parameter], lr=0.01)
    optimizer.load_state_dict(checkpoint["optimizer"])
    budget = ActualStepBudget.from_state_dict(optimizer, checkpoint["budget"])
    for _ in range(6):
        step()
    assert budget.completed == 10 and budget.state_dict()["completed_rollouts"] == 5
    assert len(budget.memberships) == 10
    assert optimizer.state[parameter]["step"] == 10
    assert all(x["global_loss_denominator"] == 128 for x in budget.memberships)
    with pytest.raises(RuntimeError):
        optimizer.step()
    with pytest.raises(RuntimeError):
        budget.begin_window(
            rollout=5, window=0, rows=windows[0].tolist(), groups=groups, selected_positions=[0, 2]
        )
    budget.close()
