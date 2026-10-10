import pytest
import torch

from visual_rl.training.rollout import collect_rollout
from visual_rl.training.update import accumulated_step
from visual_rl.training.video_refl import ReFLObjective


def test_final_step_stopped_vjp_equals_direct_policy_gradient():
    model = torch.nn.Linear(3, 3, bias=False)
    torch.manual_seed(9)
    noise = torch.randn(2, 3)
    sigmas = torch.tensor([1., .8, .3, 0.])

    def predict(state, time, rows):
        return model(state)

    cache = collect_rollout(predict, noise, sigmas, selected=(2,), microbatch=2)
    endpoint = cache.endpoints.detach().requires_grad_()
    losses = endpoint.square().mean(1)
    gradient, = torch.autograd.grad(losses.sum(), endpoint)
    objective = ReFLObjective(cache, predict, gradient, losses.detach(), torch.device("cpu"))
    actual = objective([0, 1])
    states = cache.states[:, 0]
    direct_endpoint = states - .3 * model(states)
    expected = direct_endpoint.square().mean(1)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(torch.autograd.grad(actual.mean(), model.weight)[0],
                              torch.autograd.grad(expected.mean(), model.weight)[0], atol=1e-7, rtol=1e-6)
    # No unrequested normalization of either the endpoint cotangent or pixels.
    scaled = ReFLObjective(cache, predict, 3 * gradient, 3 * losses.detach(), torch.device("cpu"))
    expected_gradient = torch.autograd.grad(objective([0, 1]).mean(), model.weight)[0]
    torch.testing.assert_close(torch.autograd.grad(scaled([0, 1]).mean(), model.weight)[0], 3 * expected_gradient)
    with torch.no_grad():
        model.weight.add_(1)
    with pytest.raises(RuntimeError, match="changed"):
        objective([0, 1])


def test_zero_global_refl_gradient_rejected_before_optimizer_update():
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.1, weight_decay=.1)
    initial = model.weight.detach().clone()
    with pytest.raises(RuntimeError, match="nonzero policy gradient"):
        accumulated_step(model, optimizer, [0], lambda items: model.weight.flatten() * 0,
                         microbatch=1, denominator=1, max_grad_norm=1, require_nonzero_grad=True)
    assert torch.equal(model.weight, initial)
    assert not optimizer.state
