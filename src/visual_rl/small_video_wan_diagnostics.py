"""Opt-in Wan failure localization; every result is diagnostic/notaccepted.

No production sampler, precision, RNG or acceptance gate is changed. Importing
this module loads neither Torch nor native Wan. CUDA execution is caller-owned.
"""

import contextlib
import importlib.util
from pathlib import Path


@contextlib.contextmanager
def observe_native_einsum(scheduler_class, *, emulate_cuda_policy=False):
    """Observe only this native module's einsum calls, restoring its globals.

    The CPU emulation is TEST ONLY: CPU autocast does not share CUDA's einsum
    policy. It cannot establish actual CUDA behavior and is unavailable in CLI.
    """
    import torch

    namespace = scheduler_class.step.__globals__
    original = namespace["torch"]
    records = []

    class TorchObserver:
        def __getattr__(self, name):
            return getattr(original, name)

        def einsum(self, equation, *operands):
            device = operands[0].device.type
            enabled = torch.is_autocast_enabled(device)
            if emulate_cuda_policy:
                if device != "cpu":
                    raise ValueError("test-only CUDA policy emulation requires CPU")
                with torch.autocast("cpu", enabled=False):
                    result = original.einsum(
                        equation, *[x.to(torch.bfloat16) if enabled else x for x in operands]
                    )
            else:
                result = original.einsum(equation, *operands)
            records.append(
                {
                    "equation": equation,
                    "input_dtypes": [str(x.dtype) for x in operands],
                    "input_shapes": [list(x.shape) for x in operands],
                    "output_dtype": str(result.dtype),
                    "autocast_enabled": enabled,
                    "autocast_dtype": str(torch.get_autocast_dtype(device)) if enabled else None,
                    "test_only_emulated_cuda_policy": emulate_cuda_policy,
                }
            )
            return result

    namespace["torch"] = TorchObserver()
    try:
        yield records
    finally:
        namespace["torch"] = original


def diagnose_solver(scheduler_class, *, device="cpu", amp_enabled=True, emulate_cuda_policy=False):
    """Actual native50step small graph; deterministic tensors, no random draws."""
    import torch

    from .models.wan_t2v_schedule import (
        capture_native_unipc,
        native_unipc_velocity_coefficient,
        native_wan_schedule,
        replay_native_unipc,
    )
    from .training.small_video_replay import NativeAffineAuditError, audit_native_unipc_affine

    target = torch.device(device)
    scheduler = scheduler_class(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
    scheduler.set_timesteps(50, device=target, shift=8)
    schedule = native_wan_schedule()
    if not torch.equal(schedule.sigmas, scheduler.sigmas.cpu()) or not torch.equal(
        schedule.timesteps, scheduler.timesteps.cpu()
    ):
        raise ValueError("diagnostic solver differs from frozen native50step schedule")
    initial = torch.linspace(-2.0, 2.0, 512, device=target).reshape(2, 2, 2, 8, 8)
    state = initial.clone()
    steps = []
    with observe_native_einsum(scheduler_class, emulate_cuda_policy=emulate_cuda_policy) as calls:
        with (
            torch.no_grad(),
            torch.autocast(target.type, dtype=torch.bfloat16, enabled=amp_enabled),
        ):
            for index, timestep in enumerate(scheduler.timesteps):
                velocity = torch.sin(initial * 2.7 + index * 0.13) * 0.7 + state * 0.17
                snapshot = capture_native_unipc(scheduler, state, timestep)
                begin = len(calls)
                mean = scheduler.step(velocity, timestep, state, return_dict=False)[0]
                native_end = len(calls)
                same = replay_native_unipc(snapshot, velocity, scheduler_factory=scheduler_class)
                coefficient = native_unipc_velocity_coefficient(
                    schedule.sigmas, index, previous_order=snapshot.history.get("this_order", 1)
                )
                try:
                    audit = audit_native_unipc_affine(
                        snapshot,
                        velocity,
                        mean,
                        coefficient,
                        index=index,
                        sigma=schedule.sigmas[index],
                        scheduler_factory=scheduler_class,
                    )
                    passed = True
                except NativeAffineAuditError as exc:
                    audit, passed = exc.diagnostics, False
                steps.append(
                    {
                        **audit,
                        "passed_original_3e5_gate": passed,
                        "same_velocity_max_abs_error": float((same - mean).abs().max()),
                        "same_velocity_exact": torch.equal(same, mean),
                        "native_einsum_calls": calls[begin:native_end],
                        "all_replay_einsum_calls": calls[native_end:],
                    }
                )
                state = mean
    return {
        "status": "diagnostic_notaccepted",
        "device": str(target),
        "amp_enabled": amp_enabled,
        "shape": list(initial.shape),
        "steps": steps,
        "sigmas": schedule.sigmas.tolist(),
        "timesteps": schedule.timesteps.tolist(),
        "test_only_emulated_cuda_policy": emulate_cuda_policy,
        "all_original_gates_passed": all(x["passed_original_3e5_gate"] for x in steps),
        "all_same_velocity_exact": all(x["same_velocity_exact"] for x in steps),
    }


def diagnose_first_step(pipe, pipeline_module, prompts, *, device, finetune, seed):
    """Two serial original B1 first steps, then raw B1/prePEFTB4/postPEFTB4.

    Raising after the original first scheduler.step skips remaining49 steps and
    decoding. The original generate's context managers still exit normally.
    """
    import hashlib

    import torch

    from .models.finetune import FineTuneConfig, parameter_inventory, prepare_policy
    from .models.wan_t2v import (
        WAN_T2V_NATIVE_LORA_TARGETS,
        WanT2VAdapter,
        WanT2VConditioning,
        WanT2VConfig,
    )
    from .small_video_wan_parity import (
        OfficialWanPrefixCaptured,
        compare_tensor,
        observe_official_wan,
        pad_contexts,
        tensor_metadata,
    )

    traces = []
    for row, prompt in enumerate(prompts):
        print(f"Wan diagnostic: original B1 first-step capture {row + 1}/2", flush=True)
        with observe_official_wan(
            pipe, pipeline_module, stop_after_steps=1, capture_inputs=True
        ) as trace:
            try:
                pipe.generate(
                    prompt,
                    size=(832, 480),
                    frame_num=81,
                    shift=8.0,
                    sample_solver="unipc",
                    sampling_steps=50,
                    guide_scale=6.0,
                    seed=seed + row,
                    offload_model=True,
                )
            except OfficialWanPrefixCaptured:
                pass
            else:
                raise ValueError("diagnostic prefix stop was not reached")
        if len(trace.steps) != 1 or len(trace.forwards) != 2 or trace.raw_decode:
            raise ValueError("diagnostic must capture exactly one native step without decode")
        traces.append(trace)
    base = pipe.model
    if any(p.dtype != torch.float32 for p in base.parameters()):
        raise ValueError("diagnostic requires unchanged official FP32 base parameters")
    metrics, observations = [], []

    def tensor_hash(value):
        return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

    # Conditional rows0/1 followed by unconditional rows0/1, matching adapter CFG.
    ordered = [(trace, branch) for branch in (0, 1) for trace in traces]
    expected = torch.stack([trace.forwards[branch]["velocity"] for trace, branch in ordered])
    captured = [trace.first_inputs[branch] for trace, branch in ordered]

    def raw_call(model, inputs):
        kwargs = {
            "t": inputs["t"].to(device),
            "context": [x.to(device) for x in inputs["context"]],
            "seq_len": inputs["seq_len"],
        }
        values = [x.to(device) for x in inputs["x"]]
        observations.append(
            {
                "inputs": [tensor_metadata(x) for x in values],
                "context": [tensor_metadata(x) for x in kwargs["context"]],
                "t": tensor_metadata(kwargs["t"]),
                "seq_len": kwargs["seq_len"],
                "autocast_enabled": torch.is_autocast_enabled(torch.device(device).type),
            }
        )
        output = model(values, **kwargs)
        observations[-1]["outputs"] = [tensor_metadata(x) for x in output]
        return torch.stack(output)

    with torch.no_grad(), torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
        for index, inputs in enumerate(captured):
            actual = raw_call(base, inputs)
            metrics.append(
                compare_tensor(
                    f"original_B1_replay/branch{index}",
                    expected[index : index + 1],
                    actual,
                    tolerance="bf16",
                )
            )
        if len({x["seq_len"] for x in captured}) != 1:
            raise ValueError("native diagnostic sequence lengths differ")
        combined = {
            "x": [x["x"][0] for x in captured],
            "context": [x["context"][0] for x in captured],
            "t": torch.cat([x["t"] for x in captured]),
            "seq_len": captured[0]["seq_len"],
        }
        before = raw_call(base, combined)
        metrics.append(compare_tensor("prePEFT_B4/branches", expected, before, tolerance="bf16"))
    fine = (
        FineTuneConfig("lora", rank=32, alpha=32, target_modules=WAN_T2V_NATIVE_LORA_TARGETS)
        if finetune == "lora"
        else FineTuneConfig("full")
    )
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        policy = prepare_policy(base, fine).to(device).eval()
    adapter = WanT2VAdapter(policy, pipe.vae.model, WanT2VConfig()).eval()
    positive, mask = pad_contexts([trace.positive for trace in traces])
    negative, negative_mask = pad_contexts([trace.negative for trace in traces])
    state = torch.stack([trace.steps[0]["state"] for trace in traces]).to(device)
    time = traces[0].timesteps[0].expand(2).to(device)
    conditioning = WanT2VConditioning(
        positive.to(device), mask.to(device), negative.to(device), negative_mask.to(device), time
    )
    branch_outputs = []
    handle = base.register_forward_hook(
        lambda module, args, values: branch_outputs.append(torch.stack(values).detach().cpu())
    )
    try:
        with torch.no_grad(), torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
            after = raw_call(policy, combined)
            metrics.append(
                compare_tensor("postPEFT_B4/branches_vs_prePEFT", before, after, tolerance="bf16")
            )
            guided = adapter.predict_velocity(state, traces[0].sigmas[0].expand(2), conditioning)
            metrics.append(
                compare_tensor(
                    "adapter_B4/branches_vs_postPEFT", after, branch_outputs[-1], tolerance="bf16"
                )
            )
            metrics.append(
                compare_tensor(
                    "adapter_B4/branches_vs_reference",
                    expected,
                    branch_outputs[-1],
                    tolerance="bf16",
                )
            )
            expected_guided = torch.stack([trace.steps[0]["velocity"] for trace in traces])
            metrics.append(
                compare_tensor(
                    "adapter_B4/guided_vs_reference", expected_guided, guided, tolerance="bf16"
                )
            )
            if finetune == "lora":
                with policy.disable_adapter():
                    disabled = adapter.predict_velocity(
                        state, traces[0].sigmas[0].expand(2), conditioning
                    )
                metrics.append(
                    compare_tensor("zero_lora/disabled", disabled, guided, tolerance="bf16")
                )
    finally:
        handle.remove()
    return {
        "status": "diagnostic_notaccepted",
        "comparisons": metrics,
        "all_frozen_tolerances_passed": all(x["passed"] for x in metrics),
        "parameters": parameter_inventory(policy),
        "raw_replay_calls": observations,
        "reference_forwards": [
            {k: v for k, v in entry.items() if k not in ("velocity", "timestep")}
            for trace in traces
            for entry in trace.forwards
        ],
        "reference_step_dtypes": [
            {
                k: step[k]
                for k in ("state_dtype", "velocity_dtype", "mean_dtype", "autocast_enabled")
            }
            for trace in traces
            for step in trace.steps
        ],
        "text_token_lengths": [len(trace.positive) for trace in traces],
        "negative_token_lengths": [len(trace.negative) for trace in traces],
        "initial_noise_sha256": [tensor_hash(trace.steps[0]["state"]) for trace in traces],
        "positive_context_sha256": [tensor_hash(trace.positive) for trace in traces],
        "negative_context_sha256": [tensor_hash(trace.negative) for trace in traces],
        "captured_noise_equals_model_inputs": all(
            torch.equal(trace.steps[0]["state"], inputs["x"][0])
            for trace in traces
            for inputs in trace.first_inputs
        ),
        "sigmas": traces[0].sigmas.tolist(),
        "timesteps": traces[0].timesteps.tolist(),
        "reference_physical_batch": 1,
        "candidate_cfg_branches": 4,
        "native_steps_per_reference": 1,
        "decode_executed": False,
    }


def run_wan_diagnostic(
    *,
    mode,
    asset_root,
    receipt,
    output,
    owned_root,
    device=None,
    finetune="lora",
    asset_manifest=None,
    seed=42,
    **unused,
):
    from .small_video_wan_parity import CODE_REVISION, _atomic_json, parity_plan

    if device != "cuda":
        raise ValueError("Wan diagnostic execution requires explicit --device cuda")
    if mode not in ("solver", "first_step") or finetune not in ("lora", "full"):
        raise ValueError("unsupported Wan diagnostic mode")
    if type(seed) is not int or seed < 0:
        raise ValueError("diagnostic seed must be a nonnegative integer")
    owned, output = Path(owned_root).resolve(strict=True), Path(output).absolute()
    if owned.parent == owned or not output.resolve().is_relative_to(owned) or output.exists():
        raise ValueError("diagnostic output must be new and inside its explicit owned root")
    from .small_video_assets import load_manifest, verify_bound_receipt

    spec = load_manifest(asset_manifest) if asset_manifest else load_manifest()
    verified = verify_bound_receipt(spec, asset_root, receipt, profile="wan")
    import torch

    sources = {row["id"]: row["path"] for row in verified["sources"]}
    report = {
        **parity_plan(),
        "status": "diagnostic_notaccepted",
        "diagnostic": mode,
        "asset_receipt_checksum": verified["checksum"],
        "finetune": finetune,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
    }
    output.mkdir(parents=True, exist_ok=False)
    _atomic_json(output / "metrics.json", report)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("explicit CUDA diagnostic requested but CUDA unavailable")
        target = torch.device("cuda", torch.cuda.current_device())
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if mode == "solver":
            # Receipt has verified source HEAD/cleanliness. Import only this file:
            # importing wan.__init__ would unnecessarily load model/FA packages.
            path = Path(sources["wan"]) / "wan/utils/fm_solvers_unipc.py"
            source = importlib.util.spec_from_file_location("_wan_diagnostic_unipc", path)
            module = importlib.util.module_from_spec(source)
            source.loader.exec_module(module)
            report["solver_runs"] = [
                diagnose_solver(
                    module.FlowUniPCMultistepScheduler, device=target, amp_enabled=enabled
                )
                for enabled in (False, True)
            ]
            report["weights_loaded"] = False
        else:
            from .models.source_pin import import_pinned_source
            from .models.wan_t2v_dependencies import check_native_wan_dependencies

            report["native_dependencies"] = check_native_wan_dependencies()
            import_pinned_source(sources["wan"], package="wan", revision=CODE_REVISION)
            pipeline_module = importlib.import_module("wan.text2video")
            config = importlib.import_module("wan.configs").t2v_1_3B
            pipe = pipeline_module.WanT2V(
                config, verified["assets"]["wan"], device_id=target.index, t5_cpu=True
            )
            import json

            rows = [
                json.loads(line) for line in Path(verified["data"]["path"]).read_text().splitlines()
            ]
            rows.sort(key=lambda row: (len(row["prompt"]), row["sample_id"]))
            report["first_step"] = diagnose_first_step(
                pipe,
                pipeline_module,
                [rows[0]["prompt"], rows[-1]["prompt"]],
                device=target,
                finetune=finetune,
                seed=seed,
            )
            report["weights_loaded"] = True
        _atomic_json(output / "metrics.json", report)
        return report
    except Exception as exc:
        report["diagnostic_error_type"] = type(exc).__name__
        _atomic_json(output / "metrics.json", report)
        raise
