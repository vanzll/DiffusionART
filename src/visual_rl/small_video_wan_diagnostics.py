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
    from .small_video_wan_parity import _atomic_json, parity_plan

    if device != "cuda":
        raise ValueError("Wan diagnostic execution requires explicit --device cuda")
    if mode != "solver" or finetune not in ("lora", "full"):
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
        _atomic_json(output / "metrics.json", report)
        return report
    except Exception as exc:
        report["diagnostic_error_type"] = type(exc).__name__
        _atomic_json(output / "metrics.json", report)
        raise
