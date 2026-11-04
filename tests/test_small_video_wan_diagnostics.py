"""CPU diagnostic instrumentation; no claim about actual CUDA AMP behavior."""

import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from test_small_video_wan_parity import ObservedScheduler, RawDecoder
from test_wan_t2v_schedule import native_scheduler_class as _native_scheduler_class_fixture

from visual_rl.models.wan_t2v_schedule import capture_native_unipc
from visual_rl.small_video_wan_diagnostics import (
    diagnose_first_step,
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


class PrefixField(torch.nn.Module):
    """Small real projections, native list/context/time API; no native GPU claim."""

    def __init__(self):
        super().__init__()
        self.model_type, self.in_dim, self.out_dim, self.text_len = "t2v", 16, 16, 512
        self.q, self.k, self.v, self.o = [torch.nn.Linear(16, 16) for _ in range(4)]
        self.text = torch.nn.Linear(3, 16)
        self.calls = []

    def forward(self, x, *, t, context, seq_len):
        self.calls.append(len(x))
        state = torch.stack(x)
        hidden = self.o(self.v(self.k(self.q(state.movedim(1, -1))))).movedim(-1, 1)
        padded = torch.stack([torch.nn.functional.pad(c, (0, 0, 0, 512 - len(c))) for c in context])
        text = self.text(padded).mean(1)
        output = hidden.float() + text.float()[:, :, None, None, None]
        output = output + t.float()[:, None, None, None, None] / 1000
        return list(output.unbind())


class PrefixPipe:
    def __init__(self):
        self.model = PrefixField().requires_grad_(False).eval()
        self.vae = SimpleNamespace(model=torch.nn.Module())
        self.vae.model.decoder = RawDecoder()
        self.seeds, self.generated_states = [], []

    def generate(self, prompt, **kwargs):
        self.seeds.append(kwargs["seed"])
        generator = torch.Generator().manual_seed(kwargs["seed"])
        state = torch.randn(16, 1, 2, 2, generator=generator)
        self.generated_states.append(state.clone())
        scheduler = ObservedScheduler()
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            for time in scheduler.timesteps:
                cond = self.model(
                    [state], t=time[None], context=[torch.ones(len(prompt), 3)], seq_len=1
                )[0]
                uncond = self.model([state], t=time[None], context=[torch.zeros(2, 3)], seq_len=1)[
                    0
                ]
                state = scheduler.step(
                    (uncond + 6 * (cond - uncond))[None], time, state[None], return_dict=False
                )[0][0]
        raise AssertionError("prefix observer must stop before second step/decode")


@pytest.mark.parametrize("finetune", ["lora", "full"])
def test_composed_prefix_real_observer_b1_replay_pre_post_peft_and_adapter_b4(finetune):
    if finetune == "lora":
        pytest.importorskip("peft")
    pipe = PrefixPipe()
    before = torch.get_rng_state()
    original_step = ObservedScheduler.step
    report = diagnose_first_step(
        pipe,
        SimpleNamespace(FlowUniPCMultistepScheduler=ObservedScheduler),
        ["a", "longer"],
        device="cpu",
        finetune=finetune,
        seed=42,
    )
    assert torch.equal(before, torch.get_rng_state())
    assert report["status"] == "diagnostic_notaccepted"
    assert report["all_frozen_tolerances_passed"]
    assert report["reference_physical_batch"] == 1 and report["candidate_cfg_branches"] == 4
    assert report["native_steps_per_reference"] == 1 and not report["decode_executed"]
    assert report["text_token_lengths"] == [1, 6]
    assert report["negative_token_lengths"] == [2, 2]
    assert pipe.seeds == [42, 43]
    assert report["captured_noise_equals_model_inputs"]
    assert report["initial_noise_sha256"] == [
        hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        for value in pipe.generated_states
    ]
    assert report["positive_context_sha256"][0] != report["positive_context_sha256"][1]
    assert pipe.model.calls == [1] * 8 + [4] * (4 if finetune == "lora" else 3)
    assert all(row["autocast_enabled"] for row in report["reference_forwards"])
    assert all(
        row["output_metadata"]["dtype"] == "torch.float32" for row in report["reference_forwards"]
    )
    assert all(row["t"]["dtype"] == "torch.int64" for row in report["raw_replay_calls"])
    assert all(row["seq_len"] == 1 for row in report["raw_replay_calls"])
    for comparison in report["comparisons"]:
        assert comparison["atol"] == 1 / 128 and comparison["rtol"] == 1 / 128
    assert ObservedScheduler.step is original_step
    assert not pipe.model._forward_hooks and not pipe.vae.model.decoder._forward_hooks
    assert set(report["parameters"]["trainable"]) == {"float32"}
    json.dumps(report, allow_nan=False)
