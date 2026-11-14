"""Explicit CUDA Wan parity; default CLI plan imports no native packages/models.

Official reference is serial B1 generate(); candidate is real B2/CFG4. Native
fixed-input parity and conditional affine audits do not validate ART training.
"""

import argparse
import contextlib
import hashlib
import importlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

CODE_REVISION = "9737cba9c1c3c4d04b33fcad41c111989865d315"
MODEL_REVISION = "37ec512624d61f7aa208f7ea8140a131f93afc9a"
PROFILE = {
    "height": 480,
    "width": 832,
    "frames": 81,
    "fps": 16.0,
    "steps": 50,
    "cfg": 6.0,
    "shift": 8.0,
    "solver": "native_UniPC_order2_bh2_predict_x0",
}
TOLERANCES = {
    "exact": {"atol": 0.0, "rtol": 0.0},
    "fp32": {"atol": 1e-5, "rtol": 1e-5},
    "bf16": {"atol": 1 / 128, "rtol": 1 / 128},
}


def _cpu(value):
    return value.detach().cpu().clone()


def compare_tensor(name, reference, candidate, *, tolerance):
    import torch

    if reference.shape != candidate.shape:
        raise ValueError(f"parity shape mismatch: {name}")
    if not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():
        raise ValueError(f"nonfinite parity tensor: {name}")
    reference, candidate = reference.detach().double().cpu(), candidate.detach().double().cpu()
    limits = TOLERANCES[tolerance]
    delta = (candidate - reference).abs()
    passed = bool((delta <= limits["atol"] + limits["rtol"] * reference.abs()).all())
    return {
        "name": name,
        "tolerance": tolerance,
        "shape": list(reference.shape),
        "passed": passed,
        "max_abs_error": float(delta.max()),
        "rms_error": float(delta.square().mean().sqrt()),
        "max_reference_abs": float(reference.abs().max()),
        **limits,
    }


def pad_contexts(contexts):
    import torch

    if (
        not contexts
        or any(x.ndim != 2 or not 0 < len(x) <= 512 for x in contexts)
        or len({x.shape[1] for x in contexts}) != 1
    ):
        raise ValueError("native text contexts must be nonempty compatible real-token prefixes")
    maximum = max(len(x) for x in contexts)
    values = torch.stack(
        [torch.nn.functional.pad(x, (0, 0, 0, maximum - len(x))) for x in contexts]
    )
    masks = torch.stack([torch.arange(maximum) < len(x) for x in contexts])
    return values, masks


@dataclass
class OfficialWanTrace:
    positive: object = None
    negative: object = None
    forwards: list = field(default_factory=list)
    steps: list = field(default_factory=list)
    raw_decode: list = field(default_factory=list)
    sigmas: object = None
    timesteps: object = None
    decoded: object = None
    first_inputs: list = field(default_factory=list)

    def validate(self, *, steps=50):
        import torch

        if len(self.steps) != steps or len(self.forwards) != 2 * steps:
            raise ValueError(
                "official Wan reference must execute two serial CFG branches per native step"
            )
        if self.sigmas.shape != (steps + 1,) or self.timesteps.shape != (steps,):
            raise ValueError("official native physical/time schedule is incomplete")
        if self.timesteps.dtype != torch.int64 or not self.raw_decode or self.decoded is None:
            raise ValueError(
                "official reference lacks native INT64 time or finite raw decode evidence"
            )
        for index, step in enumerate(self.steps):
            conditional, unconditional = self.forwards[2 * index : 2 * index + 2]
            if (
                conditional["timestep"].numel() != 1
                or unconditional["timestep"].numel() != 1
                or int(conditional["timestep"][0]) != int(self.timesteps[index])
                or not torch.equal(conditional["timestep"], unconditional["timestep"])
            ):
                raise ValueError("official branch time conditions differ")
            if any(
                tensor.shape != self.steps[0]["state"].shape
                for tensor in (
                    step["state"],
                    step["velocity"],
                    step["mean"],
                    conditional["velocity"],
                    unconditional["velocity"],
                )
            ):
                raise ValueError("official reference latent/velocity geometry changed")
            for tensor in (
                step["state"],
                step["velocity"],
                step["mean"],
                conditional["velocity"],
                unconditional["velocity"],
                self.decoded,
            ):
                if not torch.isfinite(tensor).all():
                    raise ValueError("official native reference contains nonfinite values")


class OfficialWanPrefixCaptured(RuntimeError):
    """Diagnostic-only stop after a completed original native transition."""


def tensor_metadata(value):
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "stride": list(value.stride()),
        "device": str(value.device),
        "contiguous": value.is_contiguous(),
    }


@contextlib.contextmanager
def observe_official_wan(pipe, pipeline_module, *, stop_after_steps=None, capture_inputs=False):
    """Observe original model calls/native step without replacing computations."""
    import torch

    if stop_after_steps is not None and (type(stop_after_steps) is not int or stop_after_steps < 1):
        raise ValueError("diagnostic stop must be a positive native step count")
    trace = OfficialWanTrace()
    scheduler = pipeline_module.FlowUniPCMultistepScheduler
    original_step = scheduler.step
    own_step = "step" in vars(scheduler)

    def forward_hook(module, args, kwargs, output):
        inputs = args[0] if args else kwargs["x"]
        if len(inputs) != 1 or not isinstance(output, list) or len(output) != 1:
            raise ValueError("official generate reference is B1; unsupported reference batching")
        branch = len(trace.forwards) % 2
        if len(trace.forwards) < 2:
            context = kwargs["context"]
            if len(context) != 1:
                raise ValueError("official text context batch differs")
            if branch == 0:
                trace.positive = _cpu(context[0])
            else:
                trace.negative = _cpu(context[0])
            if capture_inputs:
                trace.first_inputs.append(
                    {
                        "x": [_cpu(x) for x in inputs],
                        "t": _cpu(kwargs["t"]),
                        "context": [_cpu(x) for x in context],
                        "seq_len": kwargs.get("seq_len"),
                    }
                )
        trace.forwards.append(
            {
                "timestep": _cpu(kwargs["t"]),
                "velocity": _cpu(output[0]),
                "input_metadata": [tensor_metadata(x) for x in inputs],
                "output_metadata": tensor_metadata(output[0]),
                "context_metadata": [tensor_metadata(x) for x in kwargs["context"]],
                "t_metadata": tensor_metadata(kwargs["t"]),
                "seq_len": kwargs.get("seq_len"),
                "autocast_enabled": torch.is_autocast_enabled(inputs[0].device.type),
                "autocast_dtype": str(torch.get_autocast_dtype(inputs[0].device.type)),
            }
        )

    def step_hook(self, model_output, timestep, sample, *args, **kwargs):
        if len(sample) != 1:
            raise ValueError("official reference scheduler must be B1")
        if trace.sigmas is None:
            trace.sigmas, trace.timesteps = _cpu(self.sigmas), _cpu(self.timesteps)
        state, velocity = _cpu(sample[0]), _cpu(model_output[0])
        result = original_step(self, model_output, timestep, sample, *args, **kwargs)
        mean = result[0] if isinstance(result, tuple) else result.prev_sample
        trace.steps.append(
            {
                "state": state,
                "velocity": velocity,
                "mean": _cpu(mean[0]),
                "timestep": int(timestep),
                "state_dtype": str(sample.dtype),
                "velocity_dtype": str(model_output.dtype),
                "mean_dtype": str(mean.dtype),
                "autocast_enabled": torch.is_autocast_enabled(sample.device.type),
            }
        )
        if stop_after_steps is not None and len(trace.steps) >= stop_after_steps:
            raise OfficialWanPrefixCaptured("diagnostic native prefix captured; no decode")
        return result

    def raw_decode_hook(module, args, output):
        if not isinstance(output, torch.Tensor) or not torch.isfinite(output).all():
            raise ValueError("native raw VAE decoder returned nonfinite pixels before clamp")
        trace.raw_decode.append(
            {
                "shape": list(output.shape),
                "dtype": str(output.dtype),
                "minimum": float(output.min()),
                "maximum": float(output.max()),
            }
        )

    forward = pipe.model.register_forward_hook(forward_hook, with_kwargs=True)
    decode = pipe.vae.model.decoder.register_forward_hook(raw_decode_hook)
    scheduler.step = step_hook
    try:
        yield trace
    finally:
        forward.remove()
        decode.remove()
        if own_step:
            scheduler.step = original_step
        else:
            delattr(scheduler, "step")


def _atomic_json(path, value):
    fd, temporary = tempfile.mkstemp(prefix=".parity-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def parity_plan():
    return {
        "schema_version": "wan-official-native-parity/v1",
        "model": "wan2.1-t2v-1.3b",
        "tool": "wan-official-native-parity",
        "status": "plan_only_no_model_loading",
        "code_revision": CODE_REVISION,
        "model_revision": MODEL_REVISION,
        "profile": PROFILE,
        "tolerances": TOLERANCES,
        "reference_physical_batch": 1,
        "reference_execution": "two serial official generate calls",
        "candidate_physical_batch": 2,
        "candidate_cfg_branches": 4,
        "training_acceptance": False,
        "ART_replay_validation": False,
        "reward_loaded": False,
        "optimizer_steps": 0,
        "required_execution": "--device cuda --asset-root ROOT --receipt RECEIPT --owned-root ROOT --output DIR",
        "tensor_trace": "optional; multi-GB CPU traces are not written by default",
    }


def run_wan_parity(
    *,
    asset_root,
    receipt,
    output,
    owned_root,
    device=None,
    finetune="lora",
    asset_manifest=None,
    seed=42,
    save_tensor_trace=False,
    ffmpeg="ffmpeg",
    ffprobe="ffprobe",
):
    if device != "cuda":
        raise ValueError("real pretrained Wan parity requires explicit --device cuda")
    if finetune not in ("lora", "full") or type(seed) is not int or seed < 0:
        raise ValueError("parity requires full/LoRA mode and a nonnegative seed")
    owned, output = Path(owned_root).resolve(strict=True), Path(output).absolute()
    if owned.parent == owned or not output.resolve().is_relative_to(owned) or output.exists():
        raise ValueError("parity output must be new and inside its explicit owned task root")
    from .small_video_assets import load_manifest
    from .small_video_assets import verify_bound_receipt as verify_receipt

    spec = load_manifest(asset_manifest) if asset_manifest else load_manifest()
    verified = verify_receipt(spec, asset_root, receipt, profile="wan")
    import torch

    from .evaluation.video import encode_video
    from .models.finetune import FineTuneConfig, parameter_inventory, prepare_policy
    from .models.source_pin import import_pinned_source
    from .models.wan_t2v import (
        WAN_T2V_NATIVE_LORA_TARGETS,
        WanT2VAdapter,
        WanT2VConditioning,
        WanT2VConfig,
    )
    from .models.wan_t2v_dependencies import NativeWanDependencyError, check_native_wan_dependencies
    from .models.wan_t2v_schedule import native_wan_schedule
    from .training.small_video_replay import collect_native_video

    try:
        dependencies = check_native_wan_dependencies()
    except NativeWanDependencyError as exc:
        output.mkdir(parents=True, exist_ok=False)
        _atomic_json(
            output / "metrics.json",
            {
                **parity_plan(),
                "status": "failed",
                "error_type": type(exc).__name__,
                "native_dependencies": exc.report,
                "asset_receipt_checksum": verified["checksum"],
                "finetune": finetune,
            },
        )
        raise
    if not torch.cuda.is_available():
        raise RuntimeError("explicit CUDA parity requested but CUDA unavailable")
    output.mkdir(parents=True, exist_ok=False)
    paths = {key: Path(value) for key, value in verified["assets"].items()}
    sources = {row["id"]: row["path"] for row in verified["sources"]}
    rows = [json.loads(line) for line in Path(verified["data"]["path"]).read_text().splitlines()]
    ordered = sorted(rows, key=lambda row: (len(row["prompt"]), row["sample_id"]))
    prompts = [ordered[0]["prompt"], ordered[-1]["prompt"]]
    report = {
        **parity_plan(),
        "status": "running",
        "finetune": finetune,
        "native_dependencies": dependencies,
        "asset_receipt_checksum": verified["checksum"],
        "seed": seed,
        "reference_seeds": [seed, seed + 1],
        "reference_sample_ids": [ordered[0]["sample_id"], ordered[-1]["sample_id"]],
        "conditioning": "captured official real-token prefixes; candidate padding/CFG independently exercised",
        "prompt_sha256": [hashlib.sha256(text.encode()).hexdigest() for text in prompts],
        "asset_files": [
            {
                key: row[key]
                for key in ("asset", "relative", "sha256", "git_blob_sha1")
                if key in row
            }
            for row in verified["files"]
        ],
        "comparisons": [],
    }
    _atomic_json(output / "metrics.json", report)
    try:
        print(
            "Wan parity: loading pinned official pipeline; serial B1 reference, then B2 candidate",
            flush=True,
        )
        import_pinned_source(sources["wan"], package="wan", revision=CODE_REVISION)
        pipeline_module = importlib.import_module("wan.text2video")
        from wan.configs import t2v_1_3B

        target = torch.device("cuda", torch.cuda.current_device())
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        hardware = torch.cuda.get_device_properties(target)
        report["hardware"] = {
            "name": hardware.name,
            "total_memory_bytes": hardware.total_memory,
            "devices_used": 1,
            "eight_gpu_acceptance": False,
        }
        pipe = pipeline_module.WanT2V(
            t2v_1_3B, str(paths["wan"]), device_id=target.index, t5_cpu=True
        )
        references = []
        for index, prompt in enumerate(prompts):
            print(
                f"Wan parity: official serial reference {index + 1}/2, 50 native steps", flush=True
            )
            with observe_official_wan(pipe, pipeline_module) as trace:
                video = pipe.generate(
                    prompt,
                    size=(832, 480),
                    frame_num=81,
                    shift=8.0,
                    sample_solver="unipc",
                    sampling_steps=50,
                    guide_scale=6.0,
                    seed=seed + index,
                    offload_model=True,
                )
                trace.decoded = _cpu(video)
            trace.validate()
            if tuple(trace.steps[0]["state"].shape) != (16, 21, 60, 104) or tuple(
                trace.decoded.shape
            ) != (3, 81, 480, 832):
                raise ValueError(
                    "official native reference differs from the frozen480P/81frame profile"
                )
            references.append(trace)
            encode_video(
                trace.decoded.permute(1, 0, 2, 3),
                output / f"reference-{index}.mp4",
                fps=16.0,
                ffmpeg=ffmpeg,
                ffprobe=ffprobe,
            )
        if len(references[0].positive) == len(references[1].positive):
            raise ValueError("selected real official prompts did not produce variable text lengths")
        schedule = native_wan_schedule()
        metrics = report["comparisons"]
        for index, trace in enumerate(references):
            metrics.extend(
                [
                    compare_tensor(
                        f"reference{index}/sigmas", trace.sigmas, schedule.sigmas, tolerance="exact"
                    ),
                    compare_tensor(
                        f"reference{index}/timesteps",
                        trace.timesteps,
                        schedule.timesteps,
                        tolerance="exact",
                    ),
                ]
            )
        report["sigmas"], report["timesteps"] = (
            schedule.sigmas.tolist(),
            schedule.timesteps.tolist(),
        )
        report["text_token_lengths"] = [len(trace.positive) for trace in references]
        report["negative_token_lengths"] = [len(trace.negative) for trace in references]
        report["initial_noise_sha256"] = [
            hashlib.sha256(
                trace.steps[0]["state"].contiguous().view(torch.uint8).numpy().tobytes()
            ).hexdigest()
            for trace in references
        ]
        positive, masks = pad_contexts([trace.positive for trace in references])
        negative, negative_masks = pad_contexts([trace.negative for trace in references])
        base = pipe.model
        report["base_parameters"] = parameter_inventory(base)
        if any(parameter.dtype != torch.float32 for parameter in base.parameters()):
            raise ValueError("official native Wan base must retain its FP32 parameter semantics")
        targets = {
            name: module
            for name, module in base.named_modules()
            if isinstance(module, torch.nn.Linear)
            and any(name.endswith("." + suffix) for suffix in WAN_T2V_NATIVE_LORA_TARGETS)
        }
        if len(targets) != 240:
            raise ValueError(
                "pinned native Wan1.3B must expose240 self/cross attention projections"
            )
        fine = (
            FineTuneConfig("lora", rank=32, alpha=32, target_modules=WAN_T2V_NATIVE_LORA_TARGETS)
            if finetune == "lora"
            else FineTuneConfig("full")
        )
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            policy = prepare_policy(base, fine).to(target).eval()
        adapter = WanT2VAdapter(policy, pipe.vae.model, WanT2VConfig()).eval()
        report["parameters"] = parameter_inventory(policy)
        report["lora_targets"] = {
            "count": len(targets),
            "suffixes": list(WAN_T2V_NATIVE_LORA_TARGETS),
            "rank": 32 if finetune == "lora" else None,
            "alpha": 32 if finetune == "lora" else None,
            "dropout": 0.0,
        }
        expected_trainable = (
            sum(32 * (layer.in_features + layer.out_features) for layer in targets.values())
            if finetune == "lora"
            else sum(p.numel() for p in policy.parameters())
        )
        if sum(p.numel() for p in policy.parameters() if p.requires_grad) != expected_trainable:
            raise ValueError("native fine-tuning parameter coverage differs")
        physical = []
        hook = base.blocks[0].register_forward_pre_hook(
            lambda module, args: physical.append(len(args[0]))
        )
        branch_output = {}

        def record_branches(module, args, values):
            if not isinstance(values, list) or len(values) != 4:
                raise ValueError("candidate native transformer must return4 physical CFG branches")
            branch_output["value"] = torch.stack([_cpu(value) for value in values])

        branch_hook = base.register_forward_hook(record_branches)
        phase = {"name": "frozen"}

        def predict(state, sigma, row_indices, step):
            if phase["name"] == "trajectory":
                print(f"Wan parity: full native B2 trajectory step {step + 1}/50", flush=True)
            rows = row_indices.cpu()
            c = WanT2VConditioning(
                positive[rows].to(target),
                masks[rows].to(target),
                negative[rows].to(target),
                negative_masks[rows].to(target),
                schedule.timesteps[step].expand(len(rows)).to(target),
            )
            return adapter.predict_velocity(state, sigma, c)

        print("Wan parity: candidate B2/CFG4 on each frozen official native state", flush=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for step in range(50):
                print(f"Wan parity: frozen official-state comparison {step + 1}/50", flush=True)
                state = torch.stack([trace.steps[step]["state"] for trace in references]).to(target)
                expected = torch.stack([trace.steps[step]["velocity"] for trace in references])
                actual = predict(state, schedule.sigmas[step].expand(2), torch.arange(2), step)
                metrics.append(
                    compare_tensor(f"frozen/velocity/{step}", expected, actual, tolerance="bf16")
                )
                expected_branches = torch.stack(
                    [
                        references[row].forwards[2 * step + branch]["velocity"]
                        for branch in (0, 1)
                        for row in (0, 1)
                    ]
                )
                metrics.append(
                    compare_tensor(
                        f"frozen/CFG_branches/{step}",
                        expected_branches,
                        branch_output["value"],
                        tolerance="bf16",
                    )
                )
                if finetune == "lora":
                    with policy.disable_adapter():
                        disabled = predict(
                            state, schedule.sigmas[step].expand(2), torch.arange(2), step
                        )
                    metrics.append(
                        compare_tensor(
                            f"lora_disable_base/{step}", disabled, actual, tolerance="bf16"
                        )
                    )
            noise = torch.stack([trace.steps[0]["state"] for trace in references]).to(target)
            phase["name"] = "trajectory"
            cache = collect_native_video(
                predict,
                noise,
                model="wan_t2v_1_3b",
                selected=tuple(range(50)),
                microbatch=2,
                scheduler_factory=pipeline_module.FlowUniPCMultistepScheduler,
            )
        hook.remove()
        branch_hook.remove()
        if not physical or any(batch != 4 for batch in physical):
            raise ValueError("candidate native transformer block did not execute real CFG4 batches")
        report["transformer_block_batches"] = sorted(set(physical))
        report["conditional_affine_all50_max_error"] = cache.history_audit_max_error
        for step in range(50):
            for label, candidate in (
                ("state", cache.states[:, step]),
                ("velocity", cache.old_velocities[:, step]),
                ("mean", cache.old_means[:, step]),
            ):
                expected = torch.stack([trace.steps[step][label] for trace in references])
                metrics.append(
                    compare_tensor(
                        f"trajectory/{label}/{step}", expected, candidate, tolerance="bf16"
                    )
                )
        raw_decode = []

        def check_raw(module, args, pixels):
            if not torch.isfinite(pixels).all():
                raise ValueError("candidate raw VAE pixels nonfinite before clamp")
            raw_decode.append({"shape": list(pixels.shape), "dtype": str(pixels.dtype)})

        hook = adapter.vae.decoder.register_forward_hook(check_raw)
        with torch.no_grad():
            original_endpoints = torch.stack([trace.steps[-1]["mean"] for trace in references]).to(
                target
            )
            expected_pixels = torch.stack([trace.decoded for trace in references])
            same_input_pixels = adapter.decode(original_endpoints)
            metrics.append(
                compare_tensor(
                    "same_input_VAE_decode", expected_pixels, same_input_pixels, tolerance="fp32"
                )
            )
            pixels = adapter.decode(cache.endpoints.to(target))
            metrics.append(
                compare_tensor("full_trajectory_decode", expected_pixels, pixels, tolerance="bf16")
            )
        hook.remove()
        report["raw_decode"] = {
            "reference": [trace.raw_decode for trace in references],
            "candidate": raw_decode,
        }
        for index, video in enumerate(pixels.cpu()):
            encode_video(
                video.permute(1, 0, 2, 3),
                output / f"candidate-{index}.mp4",
                fps=16.0,
                ffmpeg=ffmpeg,
                ffprobe=ffprobe,
            )
        if save_tensor_trace:
            torch.save(
                {"references": [vars(trace) for trace in references], "candidate": vars(cache)},
                output / "tensor-trace.pt",
            )
        report["status"] = "passed" if all(item["passed"] for item in metrics) else "failed"
        _atomic_json(output / "metrics.json", report)
        if report["status"] != "passed":
            raise ValueError(
                "native Wan parity exceeded frozen tolerances; inspect metrics, never substitute sampler"
            )
        return report
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__)
        if hasattr(exc, "diagnostics"):
            report["native_affine_audit_failure"] = exc.diagnostics
        _atomic_json(output / "metrics.json", report)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true")
    diagnostics = parser.add_mutually_exclusive_group()
    diagnostics.add_argument("--solver-audit-only", action="store_true")
    diagnostics.add_argument("--diagnostic-first-step", action="store_true")
    parser.add_argument("--device", choices=("cuda",))
    for key in ("asset-root", "receipt", "output", "owned-root", "asset-manifest"):
        parser.add_argument("--" + key)
    parser.add_argument("--finetune", choices=("lora", "full"), default="lora")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-tensor-trace", action="store_true")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args(argv)
    if args.plan or args.device is None:
        plan = parity_plan()
        if args.solver_audit_only or args.diagnostic_first_step:
            plan.update(
                status="diagnostic_plan_notaccepted",
                diagnostic=("solver_audit_only" if args.solver_audit_only else "first_step"),
            )
        print(json.dumps(plan, indent=2))
        return 0
    if any(
        getattr(args, name) is None for name in ("asset_root", "receipt", "output", "owned_root")
    ):
        parser.error("CUDA execution requires asset-root, receipt, output and owned-root")
    options = {
        key: value
        for key, value in vars(args).items()
        if key not in ("plan", "solver_audit_only", "diagnostic_first_step")
    }
    if args.solver_audit_only or args.diagnostic_first_step:
        from .small_video_wan_diagnostics import run_wan_diagnostic

        run_wan_diagnostic(mode="solver" if args.solver_audit_only else "first_step", **options)
    else:
        run_wan_parity(**options)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
