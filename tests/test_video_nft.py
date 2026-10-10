import hashlib
from dataclasses import replace

import pytest
import torch
from test_hunyuan_runner import config_file

from visual_rl.algorithms import nft_loss
from visual_rl.backends import hunyuan
from visual_rl.config import load_config
from visual_rl.rewards.scalar import weighted_score
from visual_rl.training import video_nft
from visual_rl.training.image_contracts import sd3_schedule


def predict(state, time, rows):
    shape = (-1,) + (1,) * (state.ndim - 1)
    return (state.float() * .21 + time.reshape(shape) * .13 + rows.reshape(shape) * .01).bfloat16()


def test_plan_euler_dtype_and_terminal_training_position(tmp_path):
    c = load_config(config_file(tmp_path, "diffusionnft"))
    c = replace(c, sampling=replace(c.sampling, options={**c.sampling.options, "train_timestep_count": 4}))
    plan = hunyuan.training_inputs(c, 3, rank=2)
    generator = torch.Generator().manual_seed(42 + 1_000_003 * 3 + 10_007 * 2)
    noise = torch.randn(plan[0].shape, generator=generator, dtype=torch.bfloat16)
    selected = torch.stack([torch.randperm(4, generator=generator) for _ in range(len(noise))])
    assert torch.equal(noise, plan[0]) and torch.equal(selected, plan[1])
    sigmas = sd3_schedule(4, 5)
    dtypes = []

    def recording(state, time, rows):
        dtypes.append(state.dtype)
        return predict(state, time, rows)

    cache = video_nft.collect(recording, plan, sigmas, microbatch=2, device=torch.device("cpu"))
    assert dtypes == [torch.bfloat16, torch.float32, torch.float32, torch.float32]
    state = noise
    for i in range(4):
        field = predict(state, sigmas[i].expand(len(noise)), torch.arange(len(noise)))
        state = state.float() + (sigmas[i + 1] - sigmas[i]) * field.float()
    torch.testing.assert_close(cache.endpoints, state, atol=0, rtol=0)
    other = video_nft.collect(predict, plan, sigmas, microbatch=1, device=torch.device("cpu"))
    assert torch.equal(cache.endpoints, other.endpoints)
    assert torch.equal(cache.selected, selected)


def test_renoising_private_seed_rows_and_original_objective(tmp_path):
    c = load_config(config_file(tmp_path, "diffusionnft"))
    plan = hunyuan.training_inputs(c, 3, rank=2)
    sigmas = sd3_schedule(4, 5)
    cache = video_nft.collect(predict, plan, sigmas, microbatch=2, device=torch.device("cpu"))
    scores = weighted_score({"VQ": torch.tensor([0., 2.])}, {"VQ": 1.}, space="raw")
    global_scores = weighted_score({"VQ": torch.tensor([0., 2., 10., 12.])}, {"VQ": 1.}, space="raw")
    objective = video_nft.prepare_objective(c, cache, predict, scores, global_scores,
                                          torch.device("cpu"), attempt=3, rank=2)
    video_nft.predict_old(objective, 3)
    for item, row in enumerate(objective.row_indices.tolist()):
        index = int(cache.selected[row, item % 2])
        seed = int.from_bytes(hashlib.sha256(f"42:3:2:{2 * len(cache.endpoints) + row}:{index}".encode()).digest()[:8], "big") & ((1 << 63) - 1)
        noise = torch.randn(cache.endpoints.shape[1:], generator=torch.Generator().manual_seed(seed))
        expected = (1 - sigmas[index]) * cache.endpoints[row] + sigmas[index] * noise
        torch.testing.assert_close(objective.states[item], expected, atol=0, rtol=0)
    torch.testing.assert_close(objective.advantages, torch.tensor([-1., -1., 1., 1.]) / (26 ** .5 + 1e-4))
    old = objective.old_predictions.clone()
    assert not torch.equal(objective.states[:2], cache.endpoints[:2])
    scale = torch.nn.Parameter(torch.tensor(.2))
    # Compare accumulation algebra in FP32, not BF16 reduction-order rounding.
    objective.predict = lambda state, time, rows: predict(state, time, rows).float() * (1 + scale)
    current = objective.predict(objective.states, objective.sigmas, objective.row_indices)
    expected = nft_loss(current, old, objective.endpoints, objective.states, objective.sigmas,
                        objective.advantages, **c.algorithm.options["loss"])["per_item_loss"]
    actual = torch.cat([objective([i]) for i in range(len(objective.states))])
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), scale)[0],
                              torch.autograd.grad(expected.sum(), scale)[0], atol=1e-6, rtol=1e-6)
    assert torch.equal(objective.old_predictions, old)


@pytest.mark.parametrize("mode", ["no-ema", "slow-ema"])
def test_zero_based_shadow_refresh_and_current_policy_restoration(mode):
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1)
    state = video_nft.VideoNFTState(model, {"old_policy": {"mode": mode, "rate": .1, "maximum": .6}})
    model.weight.data.fill_(2)
    with state.use_old():
        assert model.weight.item() == 1
    assert model.weight.item() == 2
    assert state.after_update(1)["nft/old_policy_decay"] == 0
    model.weight.data.fill_(4)
    decay = state.after_update(2)["nft/old_policy_decay"]
    assert decay == (0 if mode == "no-ema" else .1)
    with state.use_old():
        assert model.weight.item() == pytest.approx(2 * decay + 4 * (1 - decay))
    assert model.weight.item() == 4


def test_nft_odd_update_budget_and_terminal_sigma_allowed(tmp_path):
    c = load_config(config_file(tmp_path, "diffusionnft"))
    c = replace(c, training=replace(c.training, max_updates=3),
                evaluation=replace(c.evaluation, every_updates=1), runtime=replace(c.runtime, checkpoint_every=1))
    hunyuan.validate_config(c)
    assert hunyuan.updates_per_rollout(c) == 1
