"""Mean-space regression preserves gradients only with per-transition scaling."""

import pytest
import torch

from visual_rl.sampling import flow_transition


@pytest.mark.parametrize("kind", ["ode", "sde"])
def test_affine_transition_regression_direction_magnitude_and_timestep_weights(kind):
    # Shared parameters give two nonparallel item gradients. Different step
    # lengths then make timestep reweighting observable in parameter space.
    parameters = torch.tensor([.2, -.4], requires_grad=True)
    velocity = torch.stack((parameters, torch.stack((2 * parameters[1], -parameters[0]))))
    target = torch.tensor([[.8, .3], [-.5, .9]])
    state = torch.tensor([[.7, -.2], [1.1, .5]])
    sigma, following = torch.tensor([1., .4]), torch.tensor([.7, .2])
    noise_level = .7

    def mean(prediction):
        options = {} if kind == "ode" else {
            "innovation": torch.zeros_like(state),
            "noise_level": noise_level,
            "sigma_max": float(following[0]),
        }
        return flow_transition(state, prediction, sigma, following, kind=kind, **options).mean

    # Independent analytic velocity coefficients. At sigma=1 the SDE drift's
    # velocity correction vanishes, while the actual next boundary guards its
    # otherwise singular diffusion coefficient.
    coefficient = following - sigma
    if kind == "sde":
        coefficient = coefficient * torch.tensor([1., 1 + .5 * noise_level**2])
    intercept = mean(torch.zeros_like(state))
    torch.testing.assert_close(
        mean(torch.ones_like(state)) - intercept, coefficient[:, None].expand_as(state),
        atol=1e-7, rtol=1e-6,
    )
    torch.testing.assert_close(
        mean(velocity), intercept + coefficient[:, None] * velocity,
        atol=1e-7, rtol=1e-6,
    )
    if kind == "sde":
        first = flow_transition(state[:1], velocity[:1], 1., float(following[0]),
            kind="sde", innovation=torch.zeros_like(state[:1]), noise_level=noise_level,
            sigma_max=float(following[0]))
        assert torch.isfinite(first.mean).all() and torch.isfinite(first.std).all()
        torch.testing.assert_close(first.std.square(), torch.full_like(first.std, noise_level**2))

    velocity_losses = (velocity - target).square().mean(1)
    mean_losses = (mean(velocity) - mean(target).detach()).square().mean(1)

    def gradient(loss):
        return torch.autograd.grad(loss, parameters, retain_graph=True)[0]

    for item in range(2):
        original = gradient(velocity_losses[item])
        raw = gradient(mean_losses[item])
        assert original.norm() > 0 and coefficient[item] != 0
        # A positive scalar changes magnitude but preserves this item's direction.
        torch.testing.assert_close(raw, coefficient[item].square() * original)

    aggregate = gradient(velocity_losses.mean())
    normalized = gradient((mean_losses / coefficient.square()).mean())
    torch.testing.assert_close(normalized, aggregate, atol=1e-6, rtol=2e-6)

    raw_aggregate = gradient(mean_losses.mean())
    # Nonzero 2-D cross product proves the aggregate directions differ; a
    # per-item direction equivalence does not imply a summed direction identity.
    cross = aggregate[0] * raw_aggregate[1] - aggregate[1] * raw_aggregate[0]
    assert cross.abs() > 1e-3
