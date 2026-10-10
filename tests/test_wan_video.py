import hashlib
import math
from types import SimpleNamespace

import pytest
import torch

from visual_rl.rewards.scalar import weighted_score
from visual_rl.sampling import shifted_schedule
from visual_rl.training import wan_video


@pytest.mark.parametrize("algorithm", wan_video.ALGORITHMS)
def test_native_rng_plan_and_rank_shared_selection(algorithm):
    rank, attempt, shape, k = 3, 7, (2, 3), 1 if algorithm == "refl" else 8
    actual = wan_video.inputs(algorithm=algorithm, seed=42, rank=rank, attempt=attempt,
                             k=k, latent_shape=shape)
    def seed(r, stream):
        return int.from_bytes(hashlib.sha256(f"42:{r}:7:{stream}".encode()).digest()[:8], "little") % (2**63 - 1)
    generator = torch.Generator().manual_seed(seed(rank, "noise"))
    expected = torch.randn((1 if algorithm == "dancegrpo" else k, *shape), generator=generator)
    if algorithm == "dancegrpo":
        expected = expected.repeat(k, 1, 1)
    torch.testing.assert_close(actual[0], expected, rtol=0, atol=0)
    if algorithm == "flow_grpo_fast":
        selected = torch.randperm(3, generator=torch.Generator().manual_seed(seed(0, "steps")))[:1]
        torch.testing.assert_close(actual[1], selected, rtol=0, atol=0)
        other = wan_video.inputs(algorithm=algorithm, seed=42, rank=0, attempt=attempt, k=k, latent_shape=shape)
        assert torch.equal(actual[1], other[1])
        assert not torch.equal(actual[0], other[0])
    elif algorithm == "refl":
        assert torch.equal(actual[1], torch.tensor([[3]]))
    else:
        assert torch.equal(actual[1], torch.arange(4 if algorithm == "diffusionnft" else 3).repeat(k, 1))
    if algorithm in ("dancegrpo", "flow_grpo_fast"):
        expected = torch.randn((k, 3 if algorithm == "dancegrpo" else 1, *shape), generator=generator)
        torch.testing.assert_close(actual[2], expected, rtol=0, atol=0)
    else:
        assert actual[2] is None


@pytest.mark.parametrize("algorithm", wan_video.ALGORITHMS)
def test_rollout_microbatch_invariant_and_fp32_arithmetic(algorithm):
    plan = wan_video.inputs(algorithm=algorithm, seed=42, rank=0, attempt=1,
                           k=1 if algorithm == "refl" else 8, latent_shape=(2, 3))
    seen = []
    def predict(x, times, rows):
        seen.append(x.dtype)
        return (x * .2 + times[:, None, None] * .01).to(torch.bfloat16)
    sigmas = shifted_schedule(4, 5)
    caches = [wan_video.collect(algorithm, predict, plan, sigmas, microbatch=mb, device=torch.device("cpu"))
              for mb in (1, 2, 4, 8)]
    assert set(seen) == {torch.float32}
    for cache in caches[1:]:
        for name in ("endpoints",) if algorithm == "diffusionnft" else ("states", "endpoints", "next_states", "old_velocities"):
            torch.testing.assert_close(getattr(cache, name), getattr(caches[0], name), rtol=0, atol=0)
    cache = caches[0]
    if algorithm == "dancegrpo":
        final_state = cache.next_states[:, -1]
        field = predict(final_state, sigmas[3].expand(8), torch.arange(8)).float()
        torch.testing.assert_close(cache.endpoints, final_state - sigmas[3] * field, rtol=0, atol=0)
        assert cache.stochastic == (0, 1, 2)
        assert cache.transition_kind == "wan_dance"
    if algorithm in ("dancegrpo", "flow_grpo_fast"):
        cls = wan_video.DanceObjective if algorithm == "dancegrpo" else wan_video.FastObjective
        advantages = torch.arange(8).float() - 3.5
        objective = cls(cache, predict, advantages, torch.device("cpu"),
                        {"ratio_clip": 1e-4, "advantage_clip": 5})
        items = [(row, slot) for row in range(8) for slot in range(len(cache.selected))]
        actual = objective(items)
        torch.testing.assert_close(actual, -advantages.repeat_interleave(len(cache.selected)), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_dance_transition_and_gradient(dtype):
    generator = torch.Generator().manual_seed(71)
    x, noise = [torch.randn(3, 2, 3, generator=generator) for _ in range(2)]
    v = torch.randn(x.shape, generator=generator).to(dtype).requires_grad_()
    sigmas = shifted_schedule(4, 5)
    means, samples, logps = [], [], []
    for i in range(3):
        t, dt = sigmas[i], sigmas[i + 1] - sigmas[i]
        state, field = x[i:i + 1], v[i:i + 1].float()
        x0 = state - t * field
        score = -(state - x0 * (1 - t)) / t**2
        mean = state + dt * field + (-.5 * .25**2 * score) * dt
        std = .25 * math.sqrt(-dt)
        sample = mean + noise[i:i + 1] * std
        logp = (-(sample.detach() - mean)**2 / (2 * std**2) - math.log(std)
                - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))).flatten(1).mean(1)
        means.append(mean)
        samples.append(sample)
        logps.append(logp)
    actual = wan_video.dance_transition(x, v, sigmas[:3], sigmas[1:4], innovation=noise)
    for left, right in ((actual.mean, torch.cat(means)), (actual.sample, torch.cat(samples)), (actual.log_prob, torch.cat(logps))):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    grad = torch.autograd.grad(actual.log_prob.sum(), v, retain_graph=True)[0]
    expected = torch.autograd.grad(torch.cat(logps).sum(), v)[0]
    torch.testing.assert_close(grad, expected, rtol=0, atol=0)


def test_wan_dance_scalarizes_before_advantage_and_clipping():
    c = SimpleNamespace(sampling=SimpleNamespace(samples_per_prompt=4), algorithm=SimpleNamespace(
        name="dancegrpo", options={"loss": {"ratio_clip": 1e-4, "advantage_clip": 5}, "advantage_epsilon": 1e-8}))
    plan = wan_video.inputs(algorithm="dancegrpo", seed=42, rank=0, attempt=1, k=4, latent_shape=(2, 3))
    cache = wan_video.collect("dancegrpo", lambda x, t, rows: x * .1, plan, shifted_schedule(4, 5), microbatch=4, device=torch.device("cpu"))
    score = weighted_score({"VQ": torch.tensor([0., 1., 2., 3.]), "MQ": torch.tensor([3., 1., 2., 0.]), "TA": torch.zeros(4)},
                           {"VQ": .5, "MQ": .5, "TA": 0.}, space="raw")
    objective, windows = wan_video.prepare_grpo(c, cache, None, score, score, torch.device("cpu"))
    scalar = score.weighted
    torch.testing.assert_close(objective.advantages, (scalar - scalar.mean()) / (scalar.std() + 1e-8), rtol=0, atol=0)
    assert windows == (((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)),
                       ((2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)))


def test_dance_terminal_likelihood_rejected():
    with pytest.raises(ValueError, match="nonterminal"):
        wan_video.dance_transition(torch.ones(2, 3), torch.zeros(2, 3), .625, 0., innovation=torch.ones(2, 3))
