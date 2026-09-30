import math

import pytest
import torch
from test_hunyuan_runner import config_file

from visual_rl.config import load_config
from visual_rl.rewards.scalar import weighted_score
from visual_rl.sampling import shifted_schedule
from visual_rl.training import video_grpo


def test_fast_rng_and_microbatch_cache(tmp_path):
    from dataclasses import replace

    c = load_config(config_file(tmp_path, "flow_grpo_fast"))
    shape = (2, 2, 3)
    plan = video_grpo.inputs(c, shape, attempt=7, rank=3)
    generator = torch.Generator().manual_seed(c.seed + 1_000_003 * 7 + 10_007 * 3)
    selected = torch.randperm(3, generator=generator)[:2].sort().values
    noise = torch.randn(shape, generator=generator, dtype=torch.bfloat16)
    innovations = torch.randn((2, 2, 2, 3), generator=generator)
    for a, b in zip(plan, (noise, selected, innovations)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    seen = []
    def predict(x, t, rows):
        seen.append(x.dtype)
        return (.2 * x).to(torch.bfloat16)
    sigmas = shifted_schedule(4, 5)
    full = video_grpo.collect(c, predict, plan, sigmas, torch.device("cpu"))
    partial = video_grpo.collect(replace(c, sampling=replace(c.sampling, microbatch=1)),
                                  predict, plan, sigmas, torch.device("cpu"))
    assert set(seen) == {torch.float32}
    for key in ("states", "endpoints", "next_states", "old_log_probs"):
        torch.testing.assert_close(getattr(full, key), getattr(partial, key), rtol=0, atol=0)
    objective = video_grpo.FastObjective(full, predict, torch.tensor([1., -1.]), torch.device("cpu"),
                                         {"ratio_clip": 1e-5, "advantage_clip": 5})
    result = objective([(0, 0), (1, 1)])
    torch.testing.assert_close(result, torch.tensor([-1., 1.]), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fast_native_transition_and_gradient(dtype):
    torch.manual_seed(44)
    x = torch.randn(2, 3, 4)
    velocity = torch.randn_like(x).to(dtype).requires_grad_()
    t = torch.tensor([1., .8]).reshape(2, 1, 1)
    nxt = torch.tensor([.9, .6]).reshape(2, 1, 1)
    dt = nxt - t
    diffusion = torch.sqrt(t / (1 - torch.where(t == 1, .9, t))) * .8
    mean = x * (1 + diffusion**2 / (2 * t) * dt) + velocity.float() * (1 + diffusion**2 * (1 - t) / (2 * t)) * dt
    std = diffusion * torch.sqrt(-dt)
    innovation = torch.randn_like(x)
    sample = mean + std * innovation
    expected_logp = (-(sample.detach() - mean)**2 / (2 * std**2) - std.log()
                    - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))).mean((1, 2))
    actual = video_grpo.fast_transition(x, velocity, t.flatten(), nxt.flatten(), sigma_max=.9,
                                       noise_level=.8, innovation=innovation)
    torch.testing.assert_close(actual.sample, sample, rtol=0, atol=0)
    torch.testing.assert_close(actual.log_prob, expected_logp, rtol=0, atol=0)
    expected_grad = torch.autograd.grad(expected_logp.sum(), velocity, retain_graph=True)[0]
    actual_grad = torch.autograd.grad(actual.log_prob.sum(), velocity)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


def test_fast_uses_global_uncentered_population_variance(tmp_path):
    c = load_config(config_file(tmp_path, "flow_grpo_fast"))
    def score(values):
        values = torch.tensor(values, dtype=torch.float32)
        return weighted_score({"VQ": values, "MQ": values * 0, "TA": values * 0},
                              {"VQ": 1., "MQ": 0., "TA": 0.}, space="raw")
    plan = video_grpo.inputs(c, (2, 2, 3), attempt=1)
    cache = video_grpo.collect(c, lambda x, t, rows: x * .2, plan, shifted_schedule(4, 5), torch.device("cpu"))
    local, population = score([0, 2]), score([0, 2, 10, 12])
    objective, windows = video_grpo.prepare_objective(c, cache, None, local, population, torch.device("cpu"))
    expected = torch.tensor([-1., 1.]) / (torch.tensor(26.).sqrt() + 1e-4)
    torch.testing.assert_close(objective.advantages, expected)
    assert windows == (((0, 0), (0, 1)), ((1, 0), (1, 1)))


def test_dance_keeps_component_advantages_before_clipping(tmp_path):
    from visual_rl.training.dance_video import collect_dance_rollout

    c = load_config(config_file(tmp_path, "dancegrpo"))
    plan = video_grpo.inputs(c, (2, 2, 3), attempt=1)
    cache = collect_dance_rollout(lambda x, t, rows: x * .2, plan[0], shifted_schedule(4, 5),
                                 row_selected=plan[1], innovations=plan[2], microbatch=2, eta=.25)
    scores = weighted_score({"VQ": torch.tensor([0., 2.]), "MQ": torch.tensor([2., 0.]), "TA": torch.zeros(2)},
                            {"VQ": .5, "MQ": .5, "TA": 0.}, space="raw")
    objective, _ = video_grpo.prepare_objective(c, cache, None, scores, scores, torch.device("cpu"))
    assert set(objective.advantages) == {"VQ", "MQ"}
    torch.testing.assert_close(objective.advantages["VQ"], torch.tensor([-1., 1.]) / math.sqrt(2))
    torch.testing.assert_close(objective.advantages["MQ"], -objective.advantages["VQ"])
    assert objective.options["component_weights"] == {"VQ": .5, "MQ": .5}
    assert torch.equal(scores.weighted, torch.ones(2))
