import pytest
import torch

from visual_rl.sampling import shifted_schedule
from visual_rl.training.rollout import collect_rollout, endpoint_renoising, replay_transition


@pytest.mark.parametrize("stochastic", [(), (0, 2)])
def test_rollout_microbatch_and_replay(stochastic):
    torch.manual_seed(9)
    noise = torch.randn(8, 2, 3)
    innovations = {step: torch.randn_like(noise) for step in stochastic}
    sigmas = shifted_schedule(4, 3)
    selected = stochastic or (0, 1, 2)
    weight = torch.tensor(0.2, requires_grad=True)

    def predict(state, sigma, rows):
        assert not torch.is_grad_enabled()
        return state * weight + rows[:, None, None] * 0.01

    one = collect_rollout(
        predict,
        noise,
        sigmas,
        selected=selected,
        stochastic=stochastic,
        innovations=innovations,
        microbatch=8,
    )
    many = collect_rollout(
        predict,
        noise,
        sigmas,
        selected=selected,
        stochastic=stochastic,
        innovations=innovations,
        microbatch=3,
    )
    assert weight.grad is None
    for field in (
        "endpoints",
        "states",
        "next_states",
        "old_velocities",
        "old_means",
        "transition_stds",
    ):
        assert torch.equal(getattr(one, field), getattr(many, field))
        assert not getattr(one, field).requires_grad
    if stochastic:
        rows, slots = torch.tensor([1, 3]), torch.tensor([0, 1])
        pred = one.old_velocities[rows, slots].clone().requires_grad_(True)
        replay = replay_transition(one, pred, rows, slots)
        torch.testing.assert_close(replay.log_prob, one.old_log_probs[rows, slots])
        replay.log_prob.sum().backward()
        assert pred.grad is not None and torch.isfinite(pred.grad).all()
    else:
        with pytest.raises(ValueError, match="no GRPO"):
            replay_transition(one, one.old_velocities[:, 0], torch.arange(8), torch.zeros(8).long())


def test_mismatched_fast_window_cannot_silently_replay_ode():
    with pytest.raises(ValueError, match="only stochastic"):
        collect_rollout(
            lambda x, t, rows: x,
            torch.ones(2, 3),
            shifted_schedule(4, 1),
            selected=(1,),
            stochastic=(0,),
            innovations={0: torch.ones(2, 3)},
        )


def test_endpoint_renoising_not_trajectory_reuse():
    x, z = torch.ones(2, 3), torch.zeros(2, 3)
    torch.testing.assert_close(
        endpoint_renoising(x, torch.tensor([0.2, 0.8]), z),
        torch.tensor([[0.8, 0.8, 0.8], [0.2, 0.2, 0.2]]),
        atol=3e-8,
        rtol=0,
    )


def test_row_specific_steps_match_full_trajectory_cache():
    noise = torch.randn(4, 2, 3, generator=torch.Generator().manual_seed(12)).bfloat16()
    schedule = shifted_schedule(5, 5)
    selected = torch.tensor([[0, 3], [2, 0], [1, 3], [3, 2]])
    def predict(x, t, rows):
        return x * .17 + t[:, None, None]
    full = collect_rollout(predict, noise, schedule, selected=range(5), microbatch=2)
    chosen = collect_rollout(predict, noise, schedule, selected=selected[0].tolist(), row_selected=selected, microbatch=2)
    for field in ("states", "next_states", "old_velocities", "old_means", "transition_stds"):
        expected = getattr(full, field)[torch.arange(4)[:, None], selected]
        torch.testing.assert_close(getattr(chosen, field), expected, rtol=0, atol=0)
    torch.testing.assert_close(chosen.endpoints, full.endpoints, rtol=0, atol=0)
    assert torch.equal(chosen.step_indices, selected)
