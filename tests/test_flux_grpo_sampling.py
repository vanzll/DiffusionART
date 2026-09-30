"""FLUX latent rounding and packed innovations across stochastic windows."""

import pytest
import torch

from visual_rl.models.flux import unpack_latents
from visual_rl.sampling import flow_transition
from visual_rl.training.grpo_sampling import collect_grpo_rollout
from visual_rl.training.rollout import replay_transition


@pytest.mark.parametrize("kind,unselected", [("sde", "all"), ("sde", "ode"), ("cps", "ode")])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_flux_stochastic_recurrence_and_physical_microbatch(kind, unselected, dtype):
    noise = torch.linspace(-1, 1, 128).reshape(4, 2, 4, 4).to(dtype)
    schedule = torch.tensor([1., .81731, .61373, .21971, 0.])
    selected = (1, 2)
    stochastic = tuple(range(4)) if unselected == "all" else selected
    likelihood = "gaussian" if kind == "sde" else "mse"

    def predict(state, times, rows):
        assert state.dtype == dtype
        return (state * .123 + rows[:, None, None, None].to(dtype) * .031).to(dtype)

    snapshots = []
    for microbatch in (1, 2, 4):
        generator = torch.Generator().manual_seed(81)
        cache = collect_grpo_rollout(
            predict, noise, schedule, pool=selected, count=2, selection="fixed",
            logical_batch_size=2, microbatch=microbatch, transition_kind=kind,
            likelihood=likelihood, noise_level=.7, unselected_noise=unselected,
            generator=generator, flux_precision=True)
        snapshots.append(cache)
        reference_rng = torch.Generator().manual_seed(81)
        for start in (0, 2):
            innovations = {step: unpack_latents(torch.randn(2, 4, 8, generator=reference_rng), 4, 4)
                           for step in stochastic}
            state = noise[start:start + 2]
            rows = torch.arange(start, start + 2)
            for step in range(4):
                times = schedule[step].expand(2)
                velocity = predict(state, times, rows)
                mode = kind if step in stochastic else ("cps_zero_noise" if kind == "cps" else "ode")
                transition = flow_transition(
                    state, velocity, times, schedule[step + 1], kind=mode, noise_level=.7,
                    sigma_max=float(schedule[1]), likelihood=likelihood,
                    innovation=innovations.get(step))
                if step in selected:
                    slot = selected.index(step)
                    torch.testing.assert_close(cache.states[start:start + 2, slot], state.float(), atol=0, rtol=0)
                    torch.testing.assert_close(cache.old_log_probs[start:start + 2, slot],
                                               transition.log_prob, atol=0, rtol=0)
                    torch.testing.assert_close(cache.old_means[start:start + 2, slot],
                                               transition.mean, atol=0, rtol=0)
                state = transition.sample.to(dtype)
            torch.testing.assert_close(cache.endpoints[start:start + 2], state.float(), atol=0, rtol=0)
        assert torch.equal(generator.get_state(), reference_rng.get_state())
    for other in snapshots[1:]:
        for field in ("endpoints", "states", "next_states", "old_means", "old_log_probs"):
            torch.testing.assert_close(getattr(other, field), getattr(snapshots[0], field), atol=0, rtol=0)
    cache = snapshots[0]
    velocity = cache.old_velocities[:, 0].clone().requires_grad_()
    replay = replay_transition(cache, velocity, torch.arange(4), torch.zeros(4, dtype=torch.long))
    replay.log_prob.sum().backward()
    assert velocity.grad is not None and torch.isfinite(velocity.grad).all()
    if dtype == torch.float32:
        torch.testing.assert_close(replay.log_prob, cache.old_log_probs[:, 0], atol=0, rtol=0)
    else:
        assert not torch.equal(replay.log_prob, cache.old_log_probs[:, 0])


def test_zero_noise_cps_matches_expanded_reference_not_rounded_euler():
    state = torch.linspace(-3, 3, 2048).reshape(2, 1024)
    velocity = state.sin()
    t, nxt = torch.tensor(.81271), torch.tensor(.41791)
    result = flow_transition(state, velocity, t, nxt, kind="cps_zero_noise")
    expected = (state - t * velocity) * (1 - nxt) + (state + (1 - t) * velocity) * nxt
    torch.testing.assert_close(result.sample, expected, atol=0, rtol=0)
    assert result.log_prob is None and (result.std == 0).all()
    assert not torch.equal(result.sample, state + (nxt - t) * velocity)


@pytest.mark.parametrize("flag,shape", [("yes", (2, 2, 4, 4)), (True, (2, 2, 3, 4)), (True, (2, 2))])
def test_invalid_flux_geometry_or_mode_before_rng(flag, shape):
    generator = torch.Generator().manual_seed(3)
    before = generator.get_state()
    with pytest.raises(ValueError, match="FLUX|flux_precision"):
        collect_grpo_rollout(lambda x, t, r: x, torch.ones(shape), torch.tensor([1., .5, 0.]),
                             pool=(0,), count=1, selection="fixed", logical_batch_size=2,
                             microbatch=1, transition_kind="sde", likelihood="gaussian",
                             noise_level=.7, unselected_noise="all", generator=generator,
                             flux_precision=flag)
    assert torch.equal(before, generator.get_state())
