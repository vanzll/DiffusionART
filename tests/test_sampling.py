import math

import pytest
import torch

from visual_rl.sampling import (
    flow_transition,
    select_timesteps,
    shifted_schedule,
    validate_schedule,
)


@pytest.mark.parametrize("sigma", [1.0, 0.8, 0.2])
def test_sde_matches_reference_and_replay_gradient(sigma):
    torch.manual_seed(12)
    x, noise = torch.randn(3, 2, 4), torch.randn(3, 2, 4)
    v = torch.randn_like(x, requires_grad=True)
    nxt, level = sigma / 2, 0.7
    tr = flow_transition(
        x, v, sigma, nxt, kind="sde", noise_level=level, sigma_max=0.9, innovation=noise
    )
    t = torch.tensor(sigma)
    dt = torch.tensor(nxt) - t
    diffusion = (t / (1 - (0.9 if sigma == 1 else t))).sqrt() * level
    mean = x * (1 + diffusion**2 / (2 * t) * dt) + v * (1 + diffusion**2 * (1 - t) / (2 * t)) * dt
    std = diffusion * (-dt).sqrt()
    expected = mean + std * noise
    torch.testing.assert_close(tr.sample, expected)
    replay = flow_transition(
        x, v, sigma, nxt, kind="sde", noise_level=level, sigma_max=0.9, observed=tr.sample.detach()
    )
    torch.testing.assert_close(replay.log_prob, tr.log_prob)
    expected_logp = (
        (-((expected.detach() - mean) ** 2) / (2 * std**2) - std.log() - math.log(2 * math.pi) / 2)
        .flatten(1)
        .mean(1)
    )
    torch.testing.assert_close(replay.log_prob, expected_logp)
    (actual_grad,) = torch.autograd.grad(replay.log_prob.sum(), v, retain_graph=True)
    (expected_grad,) = torch.autograd.grad(expected_logp.sum(), v)
    torch.testing.assert_close(actual_grad, expected_grad)


def test_ode_is_deterministic_and_has_no_fake_log_probability():
    x, v = torch.ones(2, 3), torch.ones(2, 3) * 2
    result = flow_transition(x, v, 0.8, 0.4)
    torch.testing.assert_close(result.sample, torch.full_like(x, 0.2))
    assert result.log_prob is None
    with pytest.raises(ValueError, match="ODE"):
        flow_transition(x, v, 0.8, 0.4, observed=x)


def test_cps_negative_mse_and_zero_variance_rejected():
    x, v = torch.randn(2, 3), torch.randn(2, 3)
    result = flow_transition(
        x, v, 0.8, 0.4, kind="cps", likelihood="mse", innovation=torch.ones_like(x)
    )
    torch.testing.assert_close(result.log_prob, -(result.sample - result.mean).square().mean(1))
    with pytest.raises(ValueError, match="zero-variance"):
        flow_transition(x, v, 0.8, 0.0, kind="cps", innovation=torch.ones_like(x))


def test_shift_is_explicit_and_indices_are_not_uniform_time():
    values = shifted_schedule(16, 7)
    validate_schedule(values)
    assert values[0] == 1 and values[-1] == 0
    assert not torch.allclose(values[:-1] - values[1:], torch.full((16,), 1 / 16))
    selection = select_timesteps(tuple(range(6)), 2, torch.Generator().manual_seed(42))
    assert len(selection) == 2 and max(selection) < 6
