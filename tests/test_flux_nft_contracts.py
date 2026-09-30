"""Fresh packed noise and FP32 mirror branches of the historical FLUX NFT path."""

from dataclasses import replace

import pytest
import torch
from test_native_nft import nft_config

from visual_rl.algorithms import nft_loss
from visual_rl.models.flux import pack_latents, unpack_latents
from visual_rl.training.nft import NFTState, prepare_nft_objective
from visual_rl.training.objectives import NFTObjective


def test_flux_nft_mirror_branches_are_computed_after_fp32_prediction_cast():
    generator = torch.Generator().manual_seed(38)
    shape = (3, 2, 4, 4)
    output = torch.randn(shape, generator=generator).bfloat16().requires_grad_()
    old = torch.randn(shape, generator=generator).bfloat16().float()
    x0, xt = (torch.randn(shape, generator=generator) for _ in range(2))
    t, advantages = torch.tensor([.91731, .52373, .19491]), torch.tensor([-2., 0., 3.])
    objective = NFTObjective(x0, xt, t, old, torch.arange(3), advantages,
                             lambda x, s, r: output[r], torch.device("cpu"),
                             upcast_predictions=True)
    actual = objective([0, 1, 2])
    # standard_flux_loss casts both predictions before constructing its branches.
    v, sigma = output.float(), t[:, None, None, None]
    positive_delta = xt - sigma * v - x0
    negative_delta = xt - sigma * (2 * old - v) - x0
    dims = (1, 2, 3)
    positive_weight = positive_delta.detach().double().abs().mean(dims, keepdim=True).clamp_min(1e-5)
    negative_weight = negative_delta.detach().double().abs().mean(dims, keepdim=True).clamp_min(1e-5)
    positive = (positive_delta.square() / positive_weight).mean(dims)
    negative = (negative_delta.square() / negative_weight).mean(dims)
    probability = (advantages.clamp(-5, 5) / 5) / 2 + .5
    expected = (probability * positive + (1 - probability) * negative) * 5
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual_grad = torch.autograd.grad(actual.sum(), output, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.sum(), output)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=0, rtol=0)
    direct_low_precision = nft_loss(output, old, x0, xt, t, advantages)["per_item_loss"]
    assert not torch.equal(actual, direct_low_precision)


@pytest.mark.parametrize("microbatch", [1, 4])
def test_flux_nft_fresh_noise_has_packed_coordinate_order(tmp_path, microbatch):
    config = nft_config(tmp_path)
    config = replace(config, model=replace(config.model, name="flux"),
                     training=replace(config.training, precision="bf16", microbatch=microbatch),
                     sampling=replace(config.sampling, options={
                         "solver": "euler", "train_timestep_count": 2,
                         "timestep_selection": "prefix"}))
    model = torch.nn.Linear(1, 1, bias=False)
    state = NFTState(model, config.algorithm.options)
    endpoints = torch.linspace(-1, 1, 128).reshape(4, 2, 4, 4)
    schedule = torch.tensor([1., .72137, .41237, 0.])
    seen = []

    def predict(x, t, rows):
        seen.append(x.dtype)
        return (x * model.weight.flatten()[0]).bfloat16()

    actual_generator = torch.Generator().manual_seed(19)
    objective = prepare_nft_objective(
        config, state, predict, endpoints, schedule, torch.zeros(4),
        actual_generator, torch.device("cpu"))
    generator = torch.Generator().manual_seed(19)
    rows = torch.randperm(4, generator=generator).repeat_interleave(2)
    times = (schedule[:2] * 1000 / 1000).repeat(4)
    torch.testing.assert_close(objective.row_indices, rows)
    torch.testing.assert_close(objective.sigmas, times, atol=0, rtol=0)
    for index, row in enumerate(rows):
        noise = torch.randn(1, 4, 8, generator=generator)
        packed_x0 = pack_latents(endpoints[row:row + 1])
        expected = (1 - times[index]) * packed_x0 + times[index] * noise
        torch.testing.assert_close(objective.states[index:index + 1],
                                   unpack_latents(expected, 4, 4), atol=0, rtol=0)
    assert seen and all(dtype == torch.float32 for dtype in seen)
    assert objective.upcast_predictions
    assert objective.states.dtype == objective.old_predictions.dtype == torch.float32
    assert torch.equal(generator.get_state(), actual_generator.get_state())
