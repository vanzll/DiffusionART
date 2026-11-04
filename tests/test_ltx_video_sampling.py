"""CPU algebra/replay and RNG-layout checks; no optional model dependencies."""

import pytest
import torch

from visual_rl.models.ltx_video_sampling import (
    LTX_VIDEO_ALLOWED_INFERENCE_STEPS,
    LTXVideoReplayStep,
    draw_ltx_transition_innovation,
    ltx_video_normalized_mean_mse,
    ltx_video_schedule,
    ltx_video_stochastic_transition,
    pack_ltx_latents,
    require_ltx_video_art_replay_support,
    unpack_ltx_latents,
)


def test_rounded_checkpoint_allowlist_is_not_the_physical_schedule():
    schedule = ltx_video_schedule()[:-1]
    rounded = torch.tensor(LTX_VIDEO_ALLOWED_INFERENCE_STEPS)
    assert not torch.equal(schedule, rounded)
    assert [round(value, 4) for value in schedule.tolist()] == list(
        LTX_VIDEO_ALLOWED_INFERENCE_STEPS
    )


def test_exact_checkpoint_schedule_and_coefficients_not_euler():
    schedule = ltx_video_schedule()
    torch.testing.assert_close(
        schedule,
        torch.tensor([1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]),
        rtol=0,
        atol=0,
    )
    state, v, noise = torch.tensor([[2.0]]), torch.tensor([[0.5]]), torch.tensor([[3.0]])
    first = ltx_video_stochastic_transition(state, v, 1.0, 0.99375, innovation=noise)
    torch.testing.assert_close(first.mean, torch.tensor([[0.009375]]))
    torch.testing.assert_close(first.sample, torch.tensor([[2.990625]]))
    torch.testing.assert_close(first.velocity_coefficient, torch.tensor([-0.00625]))
    # At the second step, the native velocity coefficient differs from delta-sigma.
    second = ltx_video_stochastic_transition(state, v, 0.99375, 0.9875, innovation=noise)
    assert not torch.isclose(second.velocity_coefficient, schedule[2] - schedule[1]).all()
    terminal = ltx_video_stochastic_transition(state, v, 0.4219, 0.0, innovation=noise)
    torch.testing.assert_close(terminal.sample, state - 0.4219 * v)
    assert terminal.std.item() == 0
    assert torch.equal(terminal.innovation, noise)  # zero-weight draw is still retained


def test_native_packed_reference_arithmetic_and_mixed_dtype_promotion():
    state = torch.arange(2 * 3 * 2 * 2 * 3).reshape(2, 3, 2, 2, 3).to(torch.bfloat16) / 10
    velocity = (state / 3).to(torch.bfloat16)
    noise = torch.ones_like(state)
    result = ltx_video_stochastic_transition(state, velocity, 1.0, 0.9937, innovation=noise)
    # Independent upstream packed-path operations with FP32 per-token timestep.
    packed, v, eps = map(pack_ltx_latents, (state, velocity, noise))
    sigma = torch.tensor([[1.0]], dtype=torch.float32)
    dt = sigma[..., None] - torch.tensor(0.9937)
    clean = packed - sigma[..., None] * v
    following = sigma[..., None] - dt
    official = (1 - following) * clean + following * eps
    torch.testing.assert_close(pack_ltx_latents(result.sample), official, atol=0, rtol=0)
    assert result.sample.dtype == torch.float32
    next_velocity = result.sample.to(torch.bfloat16)
    following = ltx_video_stochastic_transition(
        result.sample, next_velocity, 0.9937, 0.9875, innovation=torch.ones_like(result.sample)
    )
    assert following.sample.dtype == torch.float32


def test_packing_and_actual_randn_like_memory_order():
    state = torch.zeros(2, 3, 2, 2, 4)
    packed = pack_ltx_latents(state)
    assert not packed.is_contiguous()
    torch.testing.assert_close(unpack_ltx_latents(packed, state.shape), state)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(917)
        official_noise = torch.randn_like(packed)
        torch.manual_seed(917)
        ours = draw_ltx_transition_innovation(state)
        torch.testing.assert_close(pack_ltx_latents(ours), official_noise, atol=0, rtol=0)
    generator = torch.Generator().manual_seed(917)
    explicit = draw_ltx_transition_innovation(state, generator=generator)
    torch.testing.assert_close(pack_ltx_latents(explicit), official_noise, atol=0, rtol=0)


def test_frozen_native_cache_and_resume_draws():
    state, old = torch.randn(2, 3, 2, 2, 2), torch.randn(2, 3, 2, 2, 2)
    generator = torch.Generator().manual_seed(42)
    eps = draw_ltx_transition_innovation(state, generator=generator)
    rng = generator.get_state().clone()
    cache = LTXVideoReplayStep.capture(state, old, 0.9875, 0.9812, eps)
    result = cache.replay(old)
    torch.testing.assert_close(result.sample, cache.next_state, atol=0, rtol=0)
    torch.testing.assert_close(result.mean, cache.old_mean, atol=0, rtol=0)
    state.add_(100)
    old.add_(100)
    eps.add_(100)
    assert cache.state.max() < 20 and cache.old_velocity.max() < 20
    generator2 = torch.Generator().set_state(rng)
    draw1 = draw_ltx_transition_innovation(cache.state, generator=generator)
    draw2 = draw_ltx_transition_innovation(cache.state, generator=generator2)
    torch.testing.assert_close(draw1, draw2, atol=0, rtol=0)


def test_native_mean_regression_a_squared_preserves_aggregate_gradient():
    parameters = torch.tensor([0.2, -0.4], requires_grad=True)
    velocity = torch.stack((parameters, torch.stack((2 * parameters[1], -parameters[0]))))
    target = torch.tensor([[0.8, 0.3], [-0.5, 0.9]])
    state = torch.tensor([[0.7, -0.2], [1.1, 0.5]])
    sigma, next_sigma = torch.tensor([1.0, 0.4219]), torch.tensor([0.9937, 0.0])
    noise = torch.zeros_like(state)
    current = ltx_video_stochastic_transition(state, velocity, sigma, next_sigma, innovation=noise)
    frozen = ltx_video_stochastic_transition(state, target, sigma, next_sigma, innovation=noise)
    # Frozen timestep weights + inactive mask remain identical in both spaces.
    weights = torch.tensor([1.0, 0.4219])

    def grad(loss):
        return torch.autograd.grad(loss, parameters, retain_graph=True)[0]

    direct = grad(((velocity - target).square().mean(1) * weights).mean())
    normalized = grad(
        (
            ltx_video_normalized_mean_mse(current.mean, frozen.mean, current.velocity_coefficient)
            * weights
        ).mean()
    )
    torch.testing.assert_close(normalized, direct, rtol=3e-6, atol=1e-6)
    raw = grad(((current.mean - frozen.mean.detach()).square().mean(1) * weights).mean())
    assert abs(direct[0] * raw[1] - direct[1] * raw[0]) > 1e-4


@pytest.mark.parametrize("sigma,next_sigma", [(0.0, 0.0), (0.4, 0.4), (0.2, 0.3), (1.1, 0.0)])
def test_invalid_native_boundaries_fail(sigma, next_sigma):
    x = torch.ones(2, 3)
    with pytest.raises(ValueError):
        ltx_video_stochastic_transition(x, x, sigma, next_sigma, innovation=x)


def test_no_implicit_generic_art_compatibility():
    for native, mask in [(False, False), (False, True), (True, False)]:
        with pytest.raises(ValueError, match="generic Euler/Q2"):
            require_ltx_video_art_replay_support(native_coefficients=native, explicit_mask=mask)
    require_ltx_video_art_replay_support(native_coefficients=True, explicit_mask=True)
