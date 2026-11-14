"""Schedule and replay math; optional direct tests of immutable official source.

Set WAN_T2V_NATIVE_SOURCE to a local pinned Wan2.1 checkout or the audit source
directory. No test downloads source, model weights, or optional dependencies.
"""

import importlib.util
import os
from pathlib import Path

import pytest
import torch

from visual_rl.models.wan_t2v_schedule import (
    NativeUniPCSnapshot,
    capture_native_unipc,
    native_unipc_velocity_coefficient,
    native_wan_schedule,
    replay_native_unipc,
    require_native_art_replay,
)


def test_native_schedule_separates_physical_sigma_from_integer_condition():
    result = native_wan_schedule()
    assert result.sigmas.shape == (51,) and result.timesteps.shape == (50,)
    assert result.sigmas.dtype == torch.float32 and result.timesteps.dtype == torch.int64
    assert 0 < result.sigmas[0] < 1 and result.sigmas[-1] == 0
    assert (result.sigmas[:-1] > result.sigmas[1:]).all()
    assert result.timesteps[0] == 999 and result.eligible == tuple(range(50))
    assert not torch.equal(result.timesteps.float(), result.sigmas[:-1] * 1000)
    assert (result.sigmas[:4] > 0.99).all()


@pytest.mark.parametrize("steps,shift", [(0, 8), (True, 8), (50, 0), (50, float("nan"))])
def test_schedule_rejects_invalid_settings(steps, shift):
    with pytest.raises(ValueError):
        native_wan_schedule(steps, shift)


def test_native_coefficients_have_correct_first_last_and_multistep_behavior():
    schedule = native_wan_schedule()
    coefficients = torch.stack(
        [
            native_unipc_velocity_coefficient(schedule.sigmas, i, previous_order=1 if i == 1 else 2)
            for i in range(50)
        ]
    )
    assert torch.isfinite(coefficients).all() and (coefficients < 0).all()
    delta = schedule.sigmas[1:] - schedule.sigmas[:-1]
    torch.testing.assert_close(coefficients[0], delta[0], atol=2e-7, rtol=1e-4)
    torch.testing.assert_close(coefficients[-1], -schedule.sigmas[-2])
    assert not torch.allclose(coefficients[1:-1], delta[1:-1])


def test_native_training_is_explicitly_blocked_without_approved_replay_contract():
    with pytest.raises(NotImplementedError, match="frozen solver history"):
        require_native_art_replay()


@pytest.fixture
def native_scheduler_class():
    source = os.environ.get("WAN_T2V_NATIVE_SOURCE")
    if not source:
        pytest.skip("WAN_T2V_NATIVE_SOURCE not set; immutable official source test is opt-in")
    root = Path(source)
    path = root / "wan/utils/fm_solvers_unipc.py"
    if not path.is_file():
        path = root / "wan__utils__fm_solvers_unipc.py"
    if not path.is_file():
        pytest.fail(f"native scheduler source is missing from {root}")
    pytest.importorskip("diffusers")
    spec = importlib.util.spec_from_file_location("_test_wan_native_unipc", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FlowUniPCMultistepScheduler


@pytest.mark.parametrize("shift", [8.0, 5.0])
def test_schedule_exactly_matches_real_official_unipc(native_scheduler_class, shift):
    native = native_scheduler_class(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
    native.set_timesteps(50, device="cpu", shift=shift)
    actual = native_wan_schedule(50, shift)
    torch.testing.assert_close(actual.sigmas, native.sigmas, atol=0, rtol=0)
    torch.testing.assert_close(actual.timesteps, native.timesteps, atol=0, rtol=0)


def test_real_native_history_replay_affine_derivative_and_snapshot_roundtrip(
    native_scheduler_class,
):
    native = native_scheduler_class(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
    native.set_timesteps(8, device="cpu", shift=8)
    state = torch.tensor([[0.3, -0.4], [0.7, 0.2]], dtype=torch.float32)
    for index, timestep in enumerate(native.timesteps):
        velocity = state * 0.17 + torch.tensor([[0.2, -0.1], [-0.3, 0.4]])
        snapshot = capture_native_unipc(native, state, timestep)
        restored = NativeUniPCSnapshot.from_state_dict(snapshot.state_dict())
        with torch.no_grad():
            expected = native.step(velocity, timestep, state, return_dict=False)[0]
        probe = velocity.detach().clone().requires_grad_()
        actual = replay_native_unipc(restored, probe, scheduler_factory=native_scheduler_class)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        coefficient = native_unipc_velocity_coefficient(
            native.sigmas, index, previous_order=snapshot.history.get("this_order", 1)
        )
        derivative = torch.autograd.grad(actual.sum(), probe)[0]
        torch.testing.assert_close(derivative, coefficient.expand_as(probe), atol=2e-6, rtol=2e-5)
        alternative = probe.detach() + torch.tensor([[0.7, -0.2], [-0.3, 0.8]])
        changed = replay_native_unipc(
            restored, alternative, scheduler_factory=native_scheduler_class
        )
        torch.testing.assert_close(
            changed - actual.detach(),
            coefficient * (alternative - probe.detach()),
            atol=2e-6,
            rtol=2e-5,
        )
        # Cached history is unchanged by either replay, and owns detached CPU copies.
        assert restored.history.get("_step_index") == snapshot.history.get("_step_index")
        assert all(x is None or not x.requires_grad for x in restored.history["model_outputs"])
        state = expected


def test_snapshot_copies_state_and_rejects_nonlinear_thresholding(native_scheduler_class):
    native = native_scheduler_class(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
    native.set_timesteps(8, device="cpu", shift=8)
    state = torch.ones(2, 2)
    snapshot = capture_native_unipc(native, state, native.timesteps[0])
    state.zero_()
    assert snapshot.state.eq(1).all()
    invalid = native_scheduler_class(thresholding=True)
    invalid.set_timesteps(8)
    with pytest.raises(ValueError, match="thresholding"):
        capture_native_unipc(invalid, state, invalid.timesteps[0])
