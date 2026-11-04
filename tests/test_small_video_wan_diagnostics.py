"""CPU diagnostic instrumentation; no claim about actual CUDA AMP behavior."""

import json

import pytest
import torch
from test_wan_t2v_schedule import native_scheduler_class as _native_scheduler_class_fixture

from visual_rl.models.wan_t2v_schedule import capture_native_unipc
from visual_rl.small_video_wan_diagnostics import (
    diagnose_solver,
    observe_native_einsum,
    run_wan_diagnostic,
)
from visual_rl.training import small_video_replay

native_scheduler_class = _native_scheduler_class_fixture


def test_real_native_solver_diagnostic_preserves_rng_schedule_identity_and_fp32_gate(
    native_scheduler_class,
):
    before = torch.get_rng_state()
    original = native_scheduler_class.step.__globals__["torch"]
    plain = diagnose_solver(native_scheduler_class, amp_enabled=False)
    cpu_amp = diagnose_solver(native_scheduler_class, amp_enabled=True)
    assert torch.equal(before, torch.get_rng_state())
    assert native_scheduler_class.step.__globals__["torch"] is original
    for report in (plain, cpu_amp):
        assert report["status"] == "diagnostic_notaccepted"
        assert report["all_same_velocity_exact"]
        assert report["all_original_gates_passed"]
        assert len(report["steps"]) == 50
        assert len(report["sigmas"]) == 51 and report["sigmas"][-1] == 0
        assert report["timesteps"][0] == 999
        calls = [call for step in report["steps"] for call in step["native_einsum_calls"]]
        assert calls and all(call["input_dtypes"] == ["torch.float32"] * 2 for call in calls)
        assert all(not call["test_only_emulated_cuda_policy"] for call in calls)
        # Record actual CPU dtype; it must never be labeled actual CUDA evidence.
        assert report["device"] == "cpu"
        assert all(step["threshold"] == 3e-5 for step in report["steps"])
    json.dumps(cpu_amp, allow_nan=False)


def test_test_only_cuda_einsum_policy_emulation_exposes_nonaffinity_without_identity_loss(
    native_scheduler_class,
):
    original = native_scheduler_class.step.__globals__["torch"]
    report = diagnose_solver(native_scheduler_class, amp_enabled=True, emulate_cuda_policy=True)
    assert native_scheduler_class.step.__globals__["torch"] is original
    assert report["status"] == "diagnostic_notaccepted"
    assert report["test_only_emulated_cuda_policy"] and report["all_same_velocity_exact"]
    failed = [step for step in report["steps"] if not step["passed_original_3e5_gate"]]
    assert failed, "BF16-quantized current-dependent predictor must fail the frozen affine gate"
    assert all(step["failure_kind"] == "finite_precision_affinity" for step in failed)
    assert all(step["identity_max_abs_error"] == 0 for step in failed)
    assert any(
        call["output_dtype"] == "torch.bfloat16"
        for step in failed
        for call in step["native_einsum_calls"]
    )
    json.dumps(report, allow_nan=False)


def test_einsum_observer_restores_native_module_on_failure(native_scheduler_class):
    original = native_scheduler_class.step.__globals__["torch"]
    with pytest.raises(RuntimeError, match="fixture"):
        with observe_native_einsum(native_scheduler_class):
            assert native_scheduler_class.step.__globals__["torch"] is not original
            raise RuntimeError("fixture")
    assert native_scheduler_class.step.__globals__["torch"] is original


def test_affine_failure_diagnostics_distinguish_broken_snapshot_and_keep_original_gate(
    native_scheduler_class,
    monkeypatch,
):
    scheduler = native_scheduler_class(
        num_train_timesteps=1000, shift=1, use_dynamic_shifting=False
    )
    scheduler.set_timesteps(50, device="cpu", shift=8)
    velocity = torch.zeros(2, 1, 1, 1, 2)
    snapshot = capture_native_unipc(scheduler, torch.ones_like(velocity), scheduler.timesteps[0])
    calls = []

    def replay(snapshot, value, *, scheduler_factory):
        calls.append(value.clone())
        return value + 1

    monkeypatch.setattr(small_video_replay, "replay_native_unipc", replay)
    # Passing audit must retain the old one-perturbation-call path.
    result = small_video_replay.audit_native_unipc_affine(
        snapshot,
        velocity,
        torch.ones_like(velocity),
        torch.tensor(1.0),
        index=0,
        sigma=0.9,
        scheduler_factory=native_scheduler_class,
    )
    assert result["threshold"] == 3e-5 and len(calls) == 1
    with pytest.raises(small_video_replay.NativeAffineAuditError) as failure:
        small_video_replay.audit_native_unipc_affine(
            snapshot,
            velocity,
            torch.zeros_like(velocity),
            torch.tensor(1.0),
            index=40,
            sigma=0.66,
            scheduler_factory=native_scheduler_class,
        )
    assert failure.value.diagnostics["failure_kind"] == "snapshot_identity"
    assert failure.value.diagnostics["index"] == 40
    assert failure.value.diagnostics["identity_max_abs_error"] == 1
    assert len(calls) == 3
    json.dumps(failure.value.diagnostics, allow_nan=False)


def test_diagnostic_requires_explicit_cuda_before_assets_or_optional_imports(tmp_path):
    with pytest.raises(ValueError, match="explicit --device cuda"):
        run_wan_diagnostic(
            mode="solver",
            asset_root=tmp_path / "missing",
            receipt=tmp_path / "missing",
            output=tmp_path / "out",
            owned_root=tmp_path,
        )
    assert not (tmp_path / "out").exists()
