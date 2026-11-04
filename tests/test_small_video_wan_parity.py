"""CPU observation/finite/comparison contracts; no pretrained or GPU claims."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from visual_rl.small_video_wan_parity import (
    TOLERANCES,
    OfficialWanPrefixCaptured,
    compare_tensor,
    observe_official_wan,
    pad_contexts,
    parity_plan,
    run_wan_parity,
)


class ObservedScheduler:
    """Synthetic observer fixture, explicitly not a UniPC implementation."""

    def __init__(self):
        self.sigmas = torch.tensor([0.9, 0.4, 0.0])
        self.timesteps = torch.tensor([900, 400], dtype=torch.int64)
        self.index = 0

    def step(self, prediction, timestep, sample, return_dict=False):
        dt = self.sigmas[self.index + 1] - self.sigmas[self.index]
        self.index += 1
        return (sample + dt * prediction,)


class ObservedModel(nn.Module):
    def forward(self, x, *, t, context, seq_len=1):
        return [
            value * 0.2 + text.float().mean() + float(t[0]) / 10000
            for value, text in zip(x, context)
        ]


class RawDecoder(nn.Module):
    def __init__(self, nonfinite=False):
        super().__init__()
        self.nonfinite = nonfinite

    def forward(self, x):
        return torch.full_like(x, float("inf")) if self.nonfinite else x


class ObservedPipe:
    def __init__(self, nonfinite=False):
        self.model = ObservedModel()
        self.vae = SimpleNamespace(model=SimpleNamespace(decoder=RawDecoder(nonfinite)))

    def generate(self, *, tokens=2):
        # Mirrors the real call signatures, not native numerical science.
        scheduler = ObservedScheduler()
        state = torch.tensor([[[[0.7]]]])
        for time in scheduler.timesteps:
            conditional = self.model([state], t=time[None], context=[torch.ones(tokens, 3)])[0]
            unconditional = self.model([state], t=time[None], context=[torch.zeros(1, 3)])[0]
            state = scheduler.step(
                (unconditional + 6 * (conditional - unconditional))[None],
                time,
                state[None],
                return_dict=False,
            )[0][0]
        return self.vae.model.decoder(state).clamp(-1, 1)


def test_observer_records_actual_noise_text_branches_schedule_mean_and_restores_descriptors():
    pipe = ObservedPipe()
    original = ObservedScheduler.step
    with observe_official_wan(
        pipe, SimpleNamespace(FlowUniPCMultistepScheduler=ObservedScheduler)
    ) as trace:
        trace.decoded = pipe.generate(tokens=5)
    trace.validate(steps=2)
    assert ObservedScheduler.step is original and not pipe.model._forward_hooks
    assert not pipe.vae.model.decoder._forward_hooks
    assert trace.positive.shape == (5, 3) and trace.negative.shape == (1, 3)
    assert len(trace.forwards) == 4 and len(trace.steps) == 2
    assert trace.timesteps.tolist() == [900, 400]
    assert trace.sigmas.tolist() == pytest.approx([0.9, 0.4, 0.0])
    torch.testing.assert_close(trace.steps[0]["state"], torch.tensor([[[[0.7]]]]))
    torch.testing.assert_close(trace.steps[0]["mean"], trace.steps[1]["state"])
    assert trace.raw_decode[0]["maximum"] < -1  # Actual raw value precedes native clamp.
    assert not trace.steps[0]["state"].requires_grad


def test_observer_catches_raw_overflow_that_native_clamp_would_hide():
    pipe = ObservedPipe(nonfinite=True)
    assert torch.isfinite(pipe.generate()).all()  # Clamp maps Inf to finite endpoints.
    original = ObservedScheduler.step
    with pytest.raises(ValueError, match="before clamp"):
        with observe_official_wan(
            pipe, SimpleNamespace(FlowUniPCMultistepScheduler=ObservedScheduler)
        ):
            pipe.generate()
    assert ObservedScheduler.step is original and not pipe.model._forward_hooks
    assert not pipe.vae.model.decoder._forward_hooks


def test_diagnostic_prefix_stops_after_original_step_and_restores_all_hooks():
    pipe = ObservedPipe()
    original = ObservedScheduler.step
    with observe_official_wan(
        pipe,
        SimpleNamespace(FlowUniPCMultistepScheduler=ObservedScheduler),
        stop_after_steps=1,
        capture_inputs=True,
    ) as trace:
        with pytest.raises(OfficialWanPrefixCaptured):
            pipe.generate(tokens=5)
    assert len(trace.steps) == 1 and len(trace.forwards) == 2
    assert trace.steps[0]["mean"].abs().max() > 1  # Original step did run, before native clamp.
    assert not trace.raw_decode and trace.decoded is None
    assert len(trace.first_inputs) == 2
    assert trace.first_inputs[0]["context"][0].shape == (5, 3)
    assert trace.forwards[0]["output_metadata"]["dtype"] == "torch.float32"
    assert trace.forwards[0]["input_metadata"][0]["stride"] == [1, 1, 1, 1]
    assert ObservedScheduler.step is original and not pipe.model._forward_hooks
    assert not pipe.vae.model.decoder._forward_hooks


def test_comparator_frozen_bf16_vs_fp32_tolerance_and_nonfinite_rejection():
    reference = torch.ones(2, 3)
    near = reference + 0.01
    assert compare_tensor("bf16", reference, near, tolerance="bf16")["passed"]
    assert not compare_tensor("fp32", reference, near, tolerance="fp32")["passed"]
    assert not compare_tensor("beyond", reference, reference + 0.02, tolerance="bf16")["passed"]
    assert TOLERANCES["bf16"] == {"atol": 1 / 128, "rtol": 1 / 128}
    with pytest.raises(ValueError, match="nonfinite"):
        compare_tensor("overflow", reference, reference * float("inf"), tolerance="bf16")
    with pytest.raises(ValueError, match="shape"):
        compare_tensor("shape", reference, torch.ones(3), tolerance="exact")


def test_actual_variable_text_prefixes_pad_before_projection_without_implicit_repeat():
    short, long = torch.arange(6.0).reshape(2, 3), torch.arange(15.0).reshape(5, 3)
    values, masks = pad_contexts([short, long])
    assert values.shape == (2, 5, 3)
    torch.testing.assert_close(values[0, :2], short)
    assert values[0, 2:].count_nonzero() == 0
    assert masks.tolist() == [[True, True, False, False, False], [True] * 5]
    with pytest.raises(ValueError):
        pad_contexts([torch.ones(513, 3)])


@pytest.mark.parametrize("diagnostic", [None, "--solver-audit-only"])
def test_default_plan_runs_without_torch_native_or_optional_imports(tmp_path, diagnostic):
    code = """
import runpy,sys
class BlockModels:
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'torch','wan','diffusers','peft'}:
            raise AssertionError('plan imported model dependency: '+fullname)
sys.meta_path.insert(0,BlockModels())
sys.argv=['wan-parity','--plan'] + DIAGNOSTIC
runpy.run_module('visual_rl.small_video_wan_parity',run_name='__main__')
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code.replace("DIAGNOSTIC", repr([diagnostic] if diagnostic else [])),
        ],
        capture_output=True,
        text=True,
        timeout=10,
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONPATH": str(Path(__file__).parents[1] / "src"),
        },
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["reference_physical_batch"] == 1 and plan["candidate_physical_batch"] == 2
    assert plan["candidate_cfg_branches"] == 4 and not plan["training_acceptance"]
    if diagnostic:
        assert plan["status"] == "diagnostic_plan_notaccepted"


def test_real_execution_needs_explicit_cuda_before_asset_or_model_loading(tmp_path):
    with pytest.raises(ValueError, match="explicit --device cuda"):
        run_wan_parity(
            asset_root=tmp_path / "missing",
            receipt=tmp_path / "missing",
            output=tmp_path / "out",
            owned_root=tmp_path,
        )
    assert not (tmp_path / "out").exists()
    assert parity_plan()["optimizer_steps"] == 0 and not parity_plan()["reward_loaded"]
