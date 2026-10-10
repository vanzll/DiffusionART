"""FLUX schedule and stage-specific numerical contracts, without model downloads."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from visual_rl.algorithms import art_loss, art_scalar_covariance
from visual_rl.backends import sd3
from visual_rl.models import FluxAdapter, FluxConditioning, FluxConfig
from visual_rl.sampling import flow_transition
from visual_rl.training.flux_contracts import flux_schedule, validate_asset_scheduler
from visual_rl.training.image import ConditionedImagePolicy
from visual_rl.training.objectives import ARTObjective
from visual_rl.training.rollout import collect_rollout, replay_transition

SETTINGS = dict(use_dynamic_shifting=True, shift=1.0, base_image_seq_len=256,
                max_image_seq_len=4096, base_shift=0.5, max_shift=1.15)


@pytest.mark.parametrize("change", [{"shift": 2}, {"use_dynamic_shifting": False},
                                  {"invert_sigmas": True}, {"time_shift_type": "linear"},
                                  {"shift_terminal": .05}, {"num_train_timesteps": 500}])
def test_asset_scheduler_must_match_configuration(change):
    validate_asset_scheduler(SETTINGS, SETTINGS)
    with pytest.raises(ValueError, match="scheduler asset"):
        validate_asset_scheduler(dict(SETTINGS, **change), SETTINGS)


@pytest.mark.parametrize("size", [256, 512, 1024])
@pytest.mark.parametrize("steps", [1, 6, 28])
@pytest.mark.parametrize("dynamic", [True, False])
def test_resolution_schedule_matches_pipeline_formula(size, steps, dynamic):
    diffusers = pytest.importorskip("diffusers")
    from diffusers.pipelines.flux.pipeline_flux import calculate_shift

    settings = dict(SETTINGS, use_dynamic_shifting=dynamic, shift=3.0)
    actual = flux_schedule(steps, height=size, width=size, spatial_factor=8, scheduler=settings)
    scheduler = diffusers.FlowMatchEulerDiscreteScheduler(**settings)
    scheduler.set_timesteps(sigmas=np.linspace(1., 1 / steps, steps),
                            mu=calculate_shift((size // 16) ** 2))
    torch.testing.assert_close(actual, scheduler.sigmas, atol=0, rtol=0)


@pytest.mark.parametrize("change", [{"shift": 0}, {"shift": float("nan")},
                                  {"base_image_seq_len": 4096}, {"max_shift": True},
                                  {"use_dynamic_shifting": 1}, {"unknown": 1}])
def test_bad_schedule_configuration_rejected(change):
    pytest.importorskip("diffusers")
    with pytest.raises(ValueError):
        flux_schedule(6, height=512, width=512, spatial_factor=8,
                      scheduler=dict(SETTINGS, **change))


@pytest.mark.parametrize("microbatch", [1, 3])
def test_bf16_ode_preserves_each_step_and_unrounded_mean(microbatch):
    noise = torch.linspace(-1, 1, 48).reshape(3, 2, 2, 4).bfloat16()
    sigmas = torch.tensor([1., .7271, .3789, 0.])

    def predict(x, t, rows):
        assert x.dtype == torch.bfloat16
        return x * .271 + t[:, None, None, None].to(x.dtype)

    cache = collect_rollout(predict, noise, sigmas, selected=(0, 1, 2),
                            microbatch=microbatch, preserve_state_dtype=True)
    state = noise
    for i in range(3):
        v = predict(state, sigmas[i].expand(3), torch.arange(3))
        mean = state.float() + (sigmas[i + 1] - sigmas[i]) * v.float()
        torch.testing.assert_close(cache.states[:, i], state.float(), atol=0, rtol=0)
        torch.testing.assert_close(cache.old_means[:, i], mean, atol=0, rtol=0)
        state = mean.bfloat16()
        torch.testing.assert_close(cache.next_states[:, i], state.float(), atol=0, rtol=0)
    torch.testing.assert_close(cache.endpoints, state.float(), atol=0, rtol=0)
    assert not torch.equal(cache.old_means[:, -1], cache.endpoints)


def test_stochastic_state_cast_retains_historical_precast_logprob():
    noise = torch.ones(2, 3, dtype=torch.bfloat16)
    innovation = torch.linspace(-1, 1, 6).reshape(2, 3)
    sigmas = torch.tensor([1., .5, 0.])
    cache = collect_rollout(lambda x, t, r: x * .1234, noise, sigmas,
                            selected=(0,), stochastic=(0,), innovations={0: innovation},
                            preserve_state_dtype=True)
    transition = flow_transition(noise, noise * .1234, 1., .5, kind="sde",
                                 sigma_max=.5, innovation=innovation)
    torch.testing.assert_close(cache.old_log_probs[:, 0], transition.log_prob, atol=0, rtol=0)
    torch.testing.assert_close(cache.next_states[:, 0], transition.sample.bfloat16().float(),
                               atol=0, rtol=0)
    replay = replay_transition(cache, cache.old_velocities[:, 0], torch.arange(2),
                                torch.zeros(2, dtype=torch.long))
    assert not torch.equal(replay.log_prob, cache.old_log_probs[:, 0])


def test_default_rollout_still_promotes_state_after_first_transition():
    dtypes = []

    def predict(x, t, rows):
        dtypes.append(x.dtype)
        return x * .17

    collect_rollout(predict, torch.ones(2, 3, dtype=torch.bfloat16),
                    torch.tensor([1., .5, 0.]), selected=(0, 1))
    assert dtypes == [torch.bfloat16, torch.float32]


class CaptureTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.25))
        self.config = SimpleNamespace(guidance_embeds=True, in_channels=8)
        self.calls = []

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        return (kwargs["hidden_states"] * self.weight,)


@pytest.mark.parametrize("dtype", [None, torch.float32, torch.bfloat16])
def test_flux_vae_affine_precision_is_explicit(dtype):
    adapter = FluxAdapter(CaptureTransformer(), nn.Linear(1, 1),
                          FluxConfig(.3611, .1159, latent_channels=2, vae_affine_dtype=dtype))
    latent = torch.linspace(-2, 2, 64).reshape(2, 2, 4, 4).bfloat16().float()
    actual = adapter.to_vae_latents(latent)
    expected = latent.to(dtype or torch.float32) / .3611 + .1159
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    if dtype == torch.bfloat16:
        assert not torch.equal(actual.float(), latent / .3611 + .1159)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_flux_rollout_and_update_timestep_casts_are_explicit(dtype):
    model = CaptureTransformer()
    adapter = FluxAdapter(model, nn.Linear(1, 1), FluxConfig(2., .25, latent_channels=2))
    condition = FluxConditioning(torch.ones(2, 3, 4, dtype=dtype),
                                 torch.ones(2, 8, dtype=dtype), torch.zeros(3, 3, dtype=dtype))
    policy = ConditionedImagePolicy(adapter, condition,
                                    autocast_dtype=dtype if dtype == torch.bfloat16 else None)
    state = torch.ones(2, 2, 2, 2)
    sigma, rows = torch.tensor([.7271, .3789]), torch.arange(2)
    policy.rollout(state, sigma, rows)
    policy(state, sigma, rows).sum().backward()
    rollout, update = model.calls
    assert rollout["hidden_states"].dtype == update["hidden_states"].dtype == dtype
    assert rollout["timestep"].dtype == dtype
    assert update["timestep"].dtype == torch.float32
    torch.testing.assert_close(rollout["timestep"], (sigma * 1000).to(dtype) / 1000,
                               atol=0, rtol=0)
    torch.testing.assert_close(update["timestep"], sigma * 1000 / 1000, atol=0, rtol=0)
    assert model.weight.grad is not None and model.weight.grad.abs() > 0


def test_flux_art_reconstructs_old_and_current_velocity_from_fp32_means():
    state = torch.tensor([[1.125, -1.875], [.75, -.25]]).bfloat16()
    sigmas = torch.tensor([1., .7123, 0.])
    cache = collect_rollout(lambda x, t, rows: x * .317, state, sigmas,
                            selected=(0, 1), preserve_state_dtype=True)
    current = torch.tensor(.219, requires_grad=True)
    direction, magnitude = torch.ones(2, 2), torch.ones(2)
    options = dict(corrective_mask_enabled=False)
    objective = ARTObjective(cache, lambda x, t, rows: x * current, direction, magnitude,
                              torch.zeros(2, dtype=torch.bool), torch.device("cpu"),
                              options=options, reconstruct_from_mean=True)
    actual = objective([(0, 1), (1, 0)])
    rows, slots = torch.tensor([0, 1]), torch.tensor([1, 0])
    x = cache.states[rows, slots]
    dt = (sigmas[slots + 1] - sigmas[slots])[:, None]
    mean = x + dt * (x * current)
    old_mean = cache.old_means[rows, slots]
    expected = art_loss((mean - x) / dt, (old_mean - x) / dt, sigmas[slots],
                        direction, magnitude, state=x, ode_velocity_coefficient=dt.flatten(),
                        current_ode_mean=mean, old_ode_mean=old_mean,
                        restore_anchor_mask=torch.zeros(2, dtype=torch.bool), **options)["per_item_loss"]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    a_grad = torch.autograd.grad(actual.sum(), current, retain_graph=True)[0]
    b_grad = torch.autograd.grad(expected.sum(), current)[0]
    torch.testing.assert_close(a_grad, b_grad, atol=0, rtol=0)


@pytest.mark.parametrize("name,precision,expected_dtype", [
    ("flux", "bf16", torch.bfloat16), ("flux", "fp32", torch.float32),
    ("sd3", "bf16", torch.float32), ("sd3", "fp32", torch.float32),
])
def test_historical_direction_cast_is_flux_bf16_only(name, precision, expected_dtype):
    c = SimpleNamespace(model=SimpleNamespace(name=name),
                        training=SimpleNamespace(precision=precision),
                        algorithm=SimpleNamespace(options=dict(
                            covariance={}, loss={}, restore_zero_signal=False)))
    generator = torch.Generator().manual_seed(123)
    endpoints = torch.randn(4, 2, 4, 4, generator=generator).bfloat16().float()
    rewards, groups = torch.tensor([0., 1., 0., 1.]), torch.tensor([0, 0, 1, 1])
    expected = art_scalar_covariance(endpoints, rewards, groups, 2)["group_directions"][groups]
    objective, _ = sd3._art_update(
        c, SimpleNamespace(endpoints=endpoints), None,
        SimpleNamespace(scores=SimpleNamespace(weighted=rewards)), groups,
        2, 2, (0,), generator, torch.device("cpu"))
    assert objective.row_directions.dtype == expected_dtype
    torch.testing.assert_close(objective.row_directions, expected.to(expected_dtype), atol=0, rtol=0)
    assert objective.reconstruct_from_mean == (name == "flux")
    if expected_dtype == torch.bfloat16:
        assert not torch.equal(objective.row_directions.float(), expected)


def test_bf16_direction_rounding_can_change_corrective_mask():
    endpoints = torch.randn(8, 16, 8, 8, generator=torch.Generator().manual_seed(71)).bfloat16()
    rewards = torch.tensor([.2, .6, -.1, 1.1, .8, .3, -.5, 1.4])
    direction = art_scalar_covariance(endpoints.float(), rewards,
                                      torch.zeros(8, dtype=torch.long), 1)["group_directions"]
    direction = direction / direction.square().mean().sqrt()
    rounded = direction.bfloat16().float()
    rounded = rounded / rounded.square().mean().sqrt()
    error = rounded - direction
    # Place the movement between the two outward-alignment boundaries.
    tangent = error - (error * direction).mean() * direction
    tangent = tangent / tangent.square().mean().sqrt()
    scale = .02
    parallel = ((-1 + (1 - 4 * scale * scale) ** .5) / 2
                - scale * (tangent * error).mean().item() / 2)
    velocity = scale * tangent + parallel * direction
    settings = dict(old_velocity=torch.zeros_like(direction), sigma=torch.tensor([.7]),
                    guidance_magnitude=torch.ones(1), ode_velocity_coefficient=torch.tensor([-.1]))
    unrounded = art_loss(velocity, guidance_direction=direction, **settings)
    historical = art_loss(velocity, guidance_direction=direction.bfloat16(), **settings)
    assert unrounded["keep_mask"].tolist() == [False]
    assert historical["keep_mask"].tolist() == [True]
    assert unrounded["per_item_loss"].item() == 0
    assert historical["per_item_loss"].item() > 0
