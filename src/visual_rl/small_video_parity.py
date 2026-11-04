"""Real pretrained LTX-Video parity against the pinned official pipeline.

The reference executes upstream LTXVideoPipeline/RectifiedFlowScheduler, not
this repository's sampling helper. Observation wrappers preserve all inputs,
outputs and random draws. CPU unit tests exercise instrumentation only; real
pretrained execution requires explicit --device cuda and verified local assets.
No reward model, optimizer, resource/lease control or download is involved.
"""

import argparse
import contextlib
import hashlib
import importlib
import inspect
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import torch

from .models.finetune import FineTuneConfig, parameter_inventory, prepare_policy
from .models.ltx_video import (
    LTX_VIDEO_LORA_TARGETS,
    encode_ltx_video_prompts,
    install_ltx_video_lora_precision,
    load_ltx_video_components,
    ltx_video_latent_shape,
    read_ltx_video_checkpoint_config,
)
from .models.ltx_video_sampling import (
    LTX_VIDEO_CHECKPOINT,
    LTX_VIDEO_CODE_REVISION,
    ltx_video_schedule,
    pack_ltx_latents,
    unpack_ltx_latents,
)
from .small_video_assets import load_manifest
from .small_video_assets import verify_bound_receipt as verify_receipt
from .training.small_video_replay import collect_native_video

TOLERANCES = {
    "exact": {"atol": 0.0, "rtol": 0.0},
    "fp32": {"atol": 1e-5, "rtol": 1e-5},
    "bf16": {"atol": 1 / 128, "rtol": 1 / 128},
}
PROFILE = {"height": 704, "width": 1216, "num_frames": 153, "frame_rate": 30.0}


def attention_targets(transformer, *, expected_layers=28):
    targets = {}
    for suffix in LTX_VIDEO_LORA_TARGETS:
        names = [
            (name, module)
            for name, module in transformer.named_modules()
            if name == suffix or name.endswith("." + suffix)
        ]
        if len(names) != expected_layers * 2 or any(
            not isinstance(module, torch.nn.Linear) for _, module in names
        ):
            raise ValueError(
                "native attention target count/type differs from two attentions per LTX layer"
            )
        targets[suffix] = [
            {
                "name": name,
                "shape": list(module.weight.shape),
                "parameters": sum(p.numel() for p in module.parameters()),
            }
            for name, module in names
        ]
    return targets


def prepare_parity_candidate(
    adapter, encoder, finetune, *, reference_captured, expected_attention_layers=28
):
    """Prepare ONLY after the independent official reference has finished."""
    if reference_captured is not True:
        raise ValueError("capture the unwrapped official reference before candidate preparation")
    if finetune not in {"base", "lora", "full"}:
        raise ValueError("parity finetune mode must be base, lora or full")
    if any(
        p.requires_grad or p.grad is not None
        for module in (adapter.vae, encoder)
        for p in module.parameters()
    ):
        raise ValueError("VAE and T5 must already be frozen with no gradients")
    if adapter.vae.training or encoder.training:
        raise ValueError("frozen VAE and T5 must be in evaluation mode")
    model = adapter.transformer
    original = {name: p.numel() for name, p in model.named_parameters()}
    targets = attention_targets(model, expected_layers=expected_attention_layers)
    before = parameter_inventory(model)
    if finetune == "base":
        model.requires_grad_(False)
    else:
        options = (
            FineTuneConfig(mode="full")
            if finetune == "full"
            else FineTuneConfig(
                mode="lora", rank=32, alpha=32, target_modules=LTX_VIDEO_LORA_TARGETS
            )
        )
        model = prepare_policy(model, options)
        if finetune == "lora":
            install_ltx_video_lora_precision(model)
        adapter.transformer = model
    model.eval()
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if any(p.dtype != torch.float32 or p.grad is not None for p in trainable.values()):
        raise ValueError("candidate trainable parameters must be FP32 with no gradients")
    if finetune == "full" and (
        set(trainable) != set(original)
        or sum(p.numel() for p in trainable.values()) != sum(original.values())
    ):
        raise ValueError(
            "full candidate did not make every original transformer parameter trainable"
        )
    lora = None
    if finetune == "lora":
        if not trainable or any("lora_" not in name for name in trainable):
            raise ValueError("LoRA candidate trained non-adapter parameters")
        b_parameters = {name: p for name, p in trainable.items() if ".lora_B." in name}
        layers = [
            module
            for module in model.modules()
            if hasattr(module, "lora_A") and hasattr(module, "lora_B")
        ]
        expected = sum(len(values) for values in targets.values())
        if (
            len(layers) != expected
            or len(b_parameters) != expected
            or any(torch.count_nonzero(p).item() != 0 for p in b_parameters.values())
        ):
            raise ValueError("LoRA target coverage or zero-B initialization differs")
        for layer in layers:
            dropout = layer.lora_dropout["default"]
            if (
                layer.r["default"] != 32
                or layer.lora_alpha["default"] != 32
                or layer.scaling["default"] != 1.0
                or not (
                    isinstance(dropout, torch.nn.Identity)
                    or isinstance(dropout, torch.nn.Dropout)
                    and dropout.p == 0.0
                )
            ):
                raise ValueError("LoRA rank32/alpha32/dropout0/scaling1 contract differs")
        if not callable(getattr(model, "disable_adapter", None)):
            raise ValueError("LoRA candidate cannot expose disable_adapter base parity")
        lora = {
            "rank": 32,
            "alpha": 32,
            "dropout": 0.0,
            "scaling": 1.0,
            "initialization": "gaussian_A_zero_B",
            "zero_B_verified": True,
            "adapter_layers": len(layers),
            "B_parameter_names": list(b_parameters),
            "A_B_gemm_precision": "bf16_autocast_only_inside_existing_linear_branches",
        }
    if any(
        p.requires_grad or p.grad is not None
        for module in (adapter.vae, encoder)
        for p in module.parameters()
    ):
        raise ValueError("candidate preparation changed frozen VAE/T5 parameters")
    return {
        "mode": finetune,
        "reference_captured_before_preparation": True,
        "before": before,
        "after": parameter_inventory(model),
        "original_transformer_parameters": sum(original.values()),
        "attention_targets": targets,
        "attention_target_counts": {key: len(values) for key, values in targets.items()},
        "trainable_names": list(trainable),
        "lora": lora,
        "vae": parameter_inventory(adapter.vae),
        "text_encoder": parameter_inventory(encoder),
        "vae_eval": not adapter.vae.training,
        "text_encoder_eval": not encoder.training,
        "optimizer_calls": 0,
        "candidate_autocast": "disabled_native_bf16",
        "full_forward": (
            "parity_only_no_grad_bf16_parameter_views" if finetune == "full" else None
        ),
        "full_backward_validated": False,
        "distributed_validated": False,
    }


def predict_full_bf16_views(adapter, state, sigma, conditioning):
    """Parity-only simulation of BF16 compute parameters with FP32 masters.

    Never a production training fallback. Actual full backward/sharding and
    checkpoint recomputation require the separately validated FSDP path.
    """
    native = adapter.transformer
    if (
        torch.is_grad_enabled()
        or adapter.training
        or native.training
        or adapter.native_forward_dtype != torch.bfloat16
        or any(
            getattr(module, "gradient_checkpointing", False)
            or type(module).__module__.startswith(("torch.distributed.fsdp", "torch.nn.parallel"))
            for module in native.modules()
        )
    ):
        raise ValueError(
            "BF16 parameter views are parity-only: eval/no_grad/no checkpoint/DDP/FSDP"
        )
    parameters = dict(native.named_parameters())
    buffers = dict(native.named_buffers())
    if not parameters or any(
        p.dtype != torch.float32 or p.grad is not None for p in parameters.values()
    ):
        raise ValueError("full parity requires untouched FP32 master parameters with no gradients")
    before = {
        name: (id(value), value.data_ptr(), value.dtype)
        for name, value in {**parameters, **buffers}.items()
    }
    views = {
        name: value.to(torch.bfloat16) if value.is_floating_point() else value
        for name, value in {**parameters, **buffers}.items()
    }

    class ComputeProxy(torch.nn.Module):
        def forward(self, *args, **kwargs):
            # The real transformer Module owns the observed official boundaries.
            return torch.func.functional_call(native, views, args, kwargs, strict=True)

    # A proxy changes only this single-threaded no-grad call. No hooks or views
    # survive it, and real transformer master identities must be restored even
    # when a native forward raises. A/B precision wrappers are not used in full.
    proxy = ComputeProxy().eval()
    adapter.transformer = proxy
    try:
        return adapter.predict_velocity(state, sigma, conditioning)
    finally:
        adapter.transformer = native
        after = {
            name: (id(value), value.data_ptr(), value.dtype)
            for name, value in {
                **dict(native.named_parameters()),
                **dict(native.named_buffers()),
            }.items()
        }
        if after != before:
            raise ValueError("parity BF16 views did not restore FP32 master identities")


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu(item) for item in value)
    return value


def tensor_metadata(value):
    """Observe actual call precision/layout without copying large activations."""
    if isinstance(value, torch.dtype):
        # Native adaln_single passes hidden_dtype as a keyword value, not a
        # Tensor attribute. Keep that observed context JSON-safe as well.
        return str(value)
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "stride": list(value.stride()),
            "dtype": str(value.dtype),
            "device_type": value.device.type,
        }
    if isinstance(value, dict):
        return {key: tensor_metadata(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [tensor_metadata(item) for item in value]
    return value


def write_error_report(path, report):
    """Retain strict error evidence even if one report field cannot encode.

    Unknown tensors/objects and nonfinite values remain rejected, with their
    field names and serialization failures recorded; never stringify them.
    """
    try:
        text = json.dumps(report, indent=2, allow_nan=False)
    except (TypeError, ValueError) as error:
        retained, rejected = {}, {}
        for key, value in report.items():
            try:
                json.dumps({key: value}, allow_nan=False)
            except (TypeError, ValueError) as field_error:
                rejected[key] = {
                    "error_type": type(field_error).__name__,
                    "error": str(field_error),
                }
            else:
                retained[key] = value
        retained.update(
            status="error",
            report_serialization_error={"error_type": type(error).__name__, "error": str(error)},
            rejected_report_fields=rejected,
        )
        text = json.dumps(retained, indent=2, allow_nan=False)
    Path(path).write_text(text + "\n")


def _reward_owned_path(path, root, *, existing=False):
    path = Path(path)
    if (
        not path.is_absolute()
        or not path.is_relative_to(root)
        or not path.resolve().is_relative_to(root)
        or any(item.is_symlink() for item in (path, *path.parents))
    ):
        raise ValueError(
            "reward diagnostic paths must stay under the explicit nonsymlink owned root"
        )
    if existing and not path.is_file():
        raise ValueError("reward diagnostic requires existing regular MP4 files")
    return path


def _reward_video_profile(path, *, ffprobe):
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if (
        len(streams) != 1
        or streams[0].get("width") != PROFILE["width"]
        or streams[0].get("height") != PROFILE["height"]
        or int(streams[0].get("nb_read_frames", "-1")) != PROFILE["num_frames"]
        or not math.isclose(
            float(Fraction(streams[0]["avg_frame_rate"])),
            PROFILE["frame_rate"],
            rel_tol=0,
            abs_tol=1e-6,
        )
    ):
        raise ValueError("reward-only input must retain native LTX704x1216/153frames/30fps")
    return streams[0]


def _reward_device_identity():
    from .rewards.small_video_process import physical_visible_device

    if not torch.cuda.is_available():
        raise RuntimeError("explicit CUDA reward diagnostic requested but CUDA is unavailable")
    device = torch.device("cuda", torch.cuda.current_device())
    identifier = str(torch.cuda.get_device_properties(device).uuid)
    if not identifier.startswith(("GPU-", "MIG-")):
        raise ValueError("reward diagnostic requires an actual GPU UUID")
    return device, {"gpu_uuid": identifier, "visible_device": physical_visible_device(device)}


@torch.no_grad()
def run_reward_only(
    *,
    asset_root,
    receipt,
    output,
    owned_root,
    videos,
    prompts,
    device="cpu",
    manifest=None,
    ffprobe="ffprobe",
):
    """Score two existing native videos; no generator/policy/optimizer is loaded."""
    if str(device) != "cuda":
        raise ValueError("reward-only diagnostic requires explicit --device cuda")
    root = Path(owned_root)
    if not root.is_absolute() or not root.is_dir() or root.is_symlink() or root.resolve() != root:
        raise ValueError("reward-only requires an existing absolute nonsymlink owned root")
    if (
        not isinstance(videos, (list, tuple))
        or len(videos) != 2
        or not isinstance(prompts, (list, tuple))
        or len(prompts) != 2
        or any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts)
    ):
        raise ValueError(
            "reward-only requires exactly two explicit videos and matching nonempty prompts"
        )
    paths = [_reward_owned_path(path, root, existing=True) for path in videos]
    if len(set(paths)) != 2 or any(path.suffix.lower() != ".mp4" for path in paths):
        raise ValueError("reward-only requires two distinct existing MP4 paths")
    output = _reward_owned_path(output, root)
    if output.exists():
        raise FileExistsError(output)
    profiles = [_reward_video_profile(path, ffprobe=ffprobe) for path in paths]
    python = os.environ.get("SMALL_VIDEO_REWARD_PYTHON")
    if not python or not Path(python).is_file() or not os.access(python, os.X_OK):
        raise ValueError(
            "export SMALL_VIDEO_REWARD_PYTHON to the existing canonical reward interpreter"
        )
    from .rewards.small_video_process import (
        REWARD_REVISION,
        REWARD_SOURCE_REVISION,
        PhasedVideoReward,
    )

    spec = load_manifest() if manifest is None else load_manifest(manifest)
    snapshots = {entry["id"]: entry["revision"] for entry in spec["snapshots"]}
    sources = {entry["id"]: entry["revision"] for entry in spec["sources"]}
    backbone_revision = "895c3a49bc3fa70a340399125c650a463535e71c"
    if (
        snapshots.get("reward") != REWARD_REVISION
        or snapshots.get("reward_backbone") != backbone_revision
        or sources.get("reward") != REWARD_SOURCE_REVISION
    ):
        raise ValueError("reward-only requires pinned canonical reward/source/backbone")
    verified = verify_receipt(spec, asset_root, receipt, profile="ltx")
    verified_sources = {row["id"]: row["path"] for row in verified["sources"]}
    # All path/video/asset preflight above precedes even a CUDA availability query.
    target, hardware = _reward_device_identity()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": "ltx-existing-video-reward-diagnostic/v1",
        "status": "running",
        "diagnostic_only": True,
        "training_acceptance": False,
        "optimizer_calls": 0,
        "profile": PROFILE,
        "hardware": hardware,
        "asset_receipt_checksum": verified["checksum"],
        "reward_revision": REWARD_REVISION,
        "reward_source_revision": REWARD_SOURCE_REVISION,
        "backbone_revision": backbone_revision,
        "reward_fps": 4.0,
        "use_norm": False,
        "space": "raw",
        "weights": {"VQ": 0.5, "MQ": 0.5, "TA": 0.0},
        "inputs": [
            {
                "path": str(path),
                "profile": profile,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            }
            for path, profile, prompt in zip(paths, profiles, prompts)
        ],
    }
    scorer = None
    try:
        scorer = (
            PhasedVideoReward(
                python=python,
                native_root=verified_sources["reward"],
                checkpoint=verified["assets"]["reward"],
                backbone=verified["assets"]["reward_backbone"],
                revision=REWARD_REVISION,
                video_fps=PROFILE["frame_rate"],
                reward_fps=4.0,
                ffprobe=ffprobe,
                scratch=output / "reward-phases",
            )
            .to(target)
            .eval()
        )
        scores = scorer.score_paths(paths, prompts)
        if set(scores.raw) != {"VQ", "MQ", "TA"} or scores.space != "raw":
            raise ValueError("canonical reward diagnostic requires raw VQ/MQ/TA")
        for key, values in scores.raw.items():
            if (
                not isinstance(values, torch.Tensor)
                or values.shape != (2,)
                or not torch.isfinite(values).all()
            ):
                raise ValueError(f"reward-only nonfinite/missing raw component: {key}")
        report["rows"] = [
            {
                "index": row,
                **{key: float(values[row]) for key, values in scores.raw.items()},
                "raw_VQ_MQ_mean": float((scores.raw["VQ"][row] + scores.raw["MQ"][row]) / 2),
            }
            for row in range(2)
        ]
        report.update(
            status="passed", reward_phase=str(scorer.last_phase), metrics=scorer.last_metrics
        )
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        return report
    except BaseException as error:
        report.update(status="error", error_type=type(error).__name__, error=str(error))
        if scorer is not None:
            report["reward_phase"] = str(scorer.last_phase)
        write_error_report(output / "report.json", report)
        raise
    finally:
        # Existing one-shot score_paths owns bounded child drainage on errors
        # and returns only after exit. There is no persistent service to close.
        if scorer is not None:
            scorer.to("cpu")


def forward_precision(args, kwargs):
    device = args[0].device.type
    return {
        "autocast_enabled": torch.is_autocast_enabled(device),
        "autocast_dtype": str(torch.get_autocast_dtype(device)),
        "args": tensor_metadata(args),
        "kwargs": tensor_metadata(kwargs),
    }


@contextlib.contextmanager
def observe_precision_boundaries(transformer):
    """Metadata-only hooks on native normalization/projection boundaries."""
    native = transformer.get_base_model() if hasattr(transformer, "get_base_model") else transformer
    names = {
        "adaln_single",
        "caption_projection",
        "norm_out",
        "proj_out",
        "transformer_blocks.0.norm1",
        "transformer_blocks.0.norm2",
    }
    names.update(
        f"transformer_blocks.0.attn1.{name}"
        for name in (
            "to_q",
            "q_norm",
            "to_k",
            "k_norm",
            "to_v",
            "to_out.0",
            "to_q.lora_A.default",
            "to_q.lora_B.default",
        )
    )
    records, handles = [], []
    for name, module in native.named_modules():
        if name not in names:
            continue

        def observed(module, args, kwargs, result, name=name):
            records.append(
                {
                    "module": name,
                    **forward_precision(args, kwargs),
                    "output": tensor_metadata(result),
                    "parameter_dtypes": {
                        key: str(value.dtype)
                        for key, value in module.named_parameters(recurse=False)
                    },
                }
            )

        handles.append(module.register_forward_hook(observed, with_kwargs=True))
    try:
        yield records
    finally:
        for handle in handles:
            handle.remove()


def _rng(device):
    value = {"cpu": torch.random.get_rng_state().clone()}
    if device.type == "cuda":
        value["cuda"] = torch.cuda.get_rng_state(device).cpu().clone()
    return value


def _set_rng(value, device):
    torch.random.set_rng_state(value["cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(value["cuda"], device)


def _fork_devices(device):
    return (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )


@contextlib.contextmanager
def _replace(obj, name, value):
    # Restore the original descriptor lookup, not an additional bound-method
    # instance attribute, when the attribute originally belonged to the class.
    own = name in vars(obj)
    previous = vars(obj).get(name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        if own:
            setattr(obj, name, previous)
        else:
            delattr(obj, name)


class _TorchDrawObserver:
    """Proxy only this pipeline module's Torch namespace; never replace Torch."""

    def __init__(self, module, observe):
        self._module, self._observe = module, observe

    def __getattr__(self, name):
        return getattr(self._module, name)

    def randn_like(self, value, *args, **kwargs):
        before = _rng(value.device)
        result = self._module.randn_like(value, *args, **kwargs)
        self._observe(value, result, before)
        return result


@dataclass
class OfficialLTXTrace:
    initial: torch.Tensor | None = None
    rng_after_initial: dict | None = None
    text: tuple | None = None
    forwards: list = field(default_factory=list)
    steps: list = field(default_factory=list)
    decode_noise: list = field(default_factory=list)
    decoded: torch.Tensor | None = None
    decode_kwargs: dict | None = None
    scheduler_timesteps: torch.Tensor | None = None
    precision_boundaries: list = field(default_factory=list)


@contextlib.contextmanager
def observe_official_ltx(pipe, pipeline_module, *, device):
    """Observe an official call; preserve native methods, solver and random draws.

    add_noise inputs expose the actual innovation and x0 used by the scheduler.
    Its conditional mean is observed as (1-next_sigma)*actual_x0, independently
    of the adapter's transition implementation. The final zero-weight draw is
    retained. VAE output is captured before image-processor clipping.
    """
    trace = OfficialLTXTrace()
    prepare, encode = pipe.prepare_latents, pipe.encode_prompt
    step, add_noise, decode = pipe.scheduler.step, pipe.scheduler.add_noise, pipe.vae.decode
    pending = []

    def prepare_observed(*args, **kwargs):
        result = prepare(*args, **kwargs)
        if trace.initial is not None:
            raise ValueError("official pipeline prepared initial latents more than once")
        trace.initial, trace.rng_after_initial = _cpu(result), _rng(device)
        return result

    def encode_observed(*args, **kwargs):
        result = encode(*args, **kwargs)
        trace.text = _cpu(result)
        return result

    def before_transformer(module, args, kwargs):
        if len(args) != 1:
            raise ValueError("unsupported official transformer calling convention")
        trace.forwards.append(
            {
                "hidden_states": _cpu(args[0]),
                "kwargs": _cpu(kwargs),
                "precision": forward_precision(args, kwargs),
            }
        )

    def after_transformer(module, args, kwargs, result):
        trace.forwards[-1]["velocity"] = _cpu(result[0])

    def add_noise_observed(original_samples, noise, timesteps):
        if len(pending) != 1:
            raise ValueError("unexpected native add_noise invocation outside a captured step")
        result = add_noise(original_samples, noise, timesteps)
        pending[0].update(
            x0=_cpu(original_samples),
            innovation=_cpu(noise),
            next_sigma=_cpu(timesteps),
            mean=_cpu((1 - timesteps) * original_samples),
        )
        return result

    def step_observed(model_output, timestep, sample, **kwargs):
        if kwargs.get("stochastic_sampling") is not True or pending:
            raise ValueError("official LTX parity requires its native stochastic step")
        item = {
            "state": _cpu(sample),
            "state_stride": list(sample.stride()),
            "velocity": _cpu(model_output),
            "sigma": _cpu(timestep),
        }
        pending.append(item)
        try:
            result = step(model_output, timestep, sample, **kwargs)
        finally:
            pending.clear()
        if "innovation" not in item:
            raise ValueError("native stochastic step did not expose its actual innovation")
        item["next_state"] = _cpu(result[0] if isinstance(result, tuple) else result.prev_sample)
        trace.steps.append(item)
        print(
            f"Official LTX native step {len(trace.steps)}/8, physical batch={len(sample)}",
            file=sys.stderr,
            flush=True,
        )
        return result

    def decode_observed(latents, **kwargs):
        result = decode(latents, **kwargs)
        pixels = result[0]
        if not torch.isfinite(pixels).all():
            raise ValueError("official VAE returned nonfinite pixels before postprocessing")
        trace.decoded, trace.decode_kwargs = _cpu(pixels), _cpu(kwargs)
        return result

    def decode_draw(value, result, before):
        trace.decode_noise.append(
            {
                "state": _cpu(value),
                "innovation": _cpu(result),
                "rng_before": before,
                "stride": list(value.stride()),
            }
        )

    with contextlib.ExitStack() as stack:
        trace.precision_boundaries = stack.enter_context(
            observe_precision_boundaries(pipe.transformer)
        )
        for obj, name, value in (
            (pipe, "prepare_latents", prepare_observed),
            (pipe, "encode_prompt", encode_observed),
            (pipe.scheduler, "step", step_observed),
            (pipe.scheduler, "add_noise", add_noise_observed),
            (pipe.vae, "decode", decode_observed),
        ):
            stack.enter_context(_replace(obj, name, value))
        stack.enter_context(
            _replace(
                pipeline_module, "torch", _TorchDrawObserver(pipeline_module.torch, decode_draw)
            )
        )
        pre = pipe.transformer.register_forward_pre_hook(before_transformer, with_kwargs=True)
        post = pipe.transformer.register_forward_hook(after_transformer, with_kwargs=True)
        stack.callback(pre.remove)
        stack.callback(post.remove)
        yield trace
        trace.scheduler_timesteps = _cpu(pipe.scheduler.timesteps)


def compare_tensor(name, reference, candidate, *, tolerance):
    if tolerance not in TOLERANCES or reference.shape != candidate.shape:
        raise ValueError(f"parity tensor shape/tolerance differs: {name}")
    if not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():
        raise ValueError(f"nonfinite parity tensor: {name}")
    ref, actual = reference.detach().cpu().float(), candidate.detach().cpu().float()
    limits = TOLERANCES[tolerance]
    difference = (actual - ref).abs()
    allowed = limits["atol"] + limits["rtol"] * ref.abs()
    return {
        "name": name,
        "shape": list(reference.shape),
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
        "tolerance": tolerance,
        **limits,
        "passed": bool((difference <= allowed).all()),
        "max_absolute_error": float(difference.max()),
        "rms_error": float(difference.square().mean().sqrt()),
        "max_tolerance_fraction": float((difference / allowed.clamp_min(1e-30)).max()),
    }


def validate_official_trace(trace, *, shape, steps=8):
    if (
        trace.initial is None
        or tuple(trace.initial.shape) != tuple(shape)
        or len(shape) != 5
        or shape[0] != 2
        or trace.text is None
        or len(trace.steps) != steps
        or len(trace.forwards) != steps
        or len(trace.decode_noise) != 1
        or trace.decoded is None
        or trace.rng_after_initial is None
        or trace.scheduler_timesteps is None
    ):
        raise ValueError("incomplete official physical-batch2 native trace")
    mask = trace.text[1]
    if mask.ndim != 2 or mask.shape[0] != 2 or mask.sum(1)[0] == mask.sum(1)[1]:
        raise ValueError("parity requires two prompts with different native token lengths")
    if any(item["hidden_states"].shape[0] != 2 for item in trace.forwards):
        raise ValueError("official transformer did not execute a true physical batch2")
    if not torch.isfinite(trace.decoded).all() or any(
        not torch.isfinite(item[key]).all()
        for item in trace.steps
        for key in ("state", "velocity", "mean", "next_state", "innovation")
    ):
        raise ValueError("nonfinite official native trace")


def official_pipeline_kwargs(prompts, generator):
    if (
        not isinstance(prompts, (list, tuple))
        or len(prompts) != 2
        or any(not isinstance(p, str) or not p.strip() for p in prompts)
    ):
        raise ValueError("parity requires exactly two nonempty prompts")
    return {
        **PROFILE,
        "prompt": list(prompts),
        "num_inference_steps": 8,
        "guidance_scale": 1.0,
        "stg_scale": 0.0,
        "rescaling_scale": 1.0,
        "generator": generator,
        "output_type": "pt",
        "return_dict": True,
        "decode_timestep": 0.05,
        "decode_noise_scale": 0.025,
        "mixed_precision": False,
        "offload_to_cpu": False,
        "enhance_prompt": False,
        "text_encoder_max_tokens": 256,
        "stochastic_sampling": True,
        "tone_map_compression_ratio": 0.0,
        "is_video": True,
        "vae_per_channel_normalize": True,
    }


def validate_official_kwargs(pipeline_class, kwargs):
    # Only these two kwargs-only names are read by the pinned primary code.
    supported = set(inspect.signature(pipeline_class.__call__).parameters) | {
        "is_video",
        "vae_per_channel_normalize",
    }
    if set(kwargs) - supported:
        raise ValueError(
            "unsupported official pipeline keyword; do not rely on **kwargs swallowing it"
        )


def compare_native_cache(trace, cache, shape):
    metrics = [
        compare_tensor(
            "physical_sigmas", trace.scheduler_timesteps, cache.sigmas[:-1], tolerance="exact"
        )
    ]
    for index, item in enumerate(trace.steps):
        for official_key, cache_key, tolerance in (
            ("state", "states", "fp32"),
            ("velocity", "old_velocities", "bf16"),
            ("mean", "old_means", "fp32"),
            ("next_state", "next_states", "fp32"),
            ("innovation", "innovations", "exact"),
        ):
            reference = unpack_ltx_latents(item[official_key], shape)
            actual = getattr(cache, cache_key)[:, index]
            metrics.append(
                compare_tensor(
                    f"step/{index}/{official_key}", reference, actual, tolerance=tolerance
                )
            )
    metrics.append(
        compare_tensor(
            "endpoint",
            unpack_ltx_latents(trace.steps[-1]["next_state"], shape),
            cache.endpoints,
            tolerance="fp32",
        )
    )
    return metrics


def _native_module(name, source):
    module = importlib.import_module(name)
    expected = Path(source).resolve().joinpath(*name.split(".")).with_suffix(".py")
    if Path(module.__file__).resolve() != expected:
        raise ValueError("official parity module origin differs from verified native source")
    return module


@torch.no_grad()
def run_ltx_parity(
    *,
    asset_root,
    receipt,
    output,
    device="cpu",
    manifest=None,
    seed=42,
    ffmpeg="ffmpeg",
    ffprobe="ffprobe",
    finetune="base",
    save_tensors=False,
):
    if str(device) != "cuda":
        raise ValueError(
            "real pretrained parity requires explicit --device cuda; use --plan for no inference"
        )
    if type(seed) is not int or seed < 0:
        raise ValueError("parity seed must be a nonnegative integer")
    if finetune not in {"base", "lora", "full"} or type(save_tensors) is not bool:
        raise ValueError("invalid candidate finetune mode or tensor artifact option")
    spec = load_manifest() if manifest is None else load_manifest(manifest)
    verified = verify_receipt(spec, asset_root, receipt, profile="ltx")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    paths = {key: Path(value) for key, value in verified["assets"].items()}
    sources = {row["id"]: row["path"] for row in verified["sources"]}
    prompts = [
        json.loads(line)["prompt"]
        for line in Path(verified["data"]["path"]).read_text().splitlines()
    ][:2]
    if not torch.cuda.is_available():
        raise RuntimeError("explicit CUDA parity requested but CUDA is unavailable")
    target = torch.device("cuda", torch.cuda.current_device())
    hardware = torch.cuda.get_device_properties(target)
    torch.cuda.reset_peak_memory_stats(target)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report = {
        "schema_version": "ltx-official-native-parity/v1",
        "status": "running",
        "model": "ltxv-2b-0.9.6-distilled-04-25",
        "code_revision": LTX_VIDEO_CODE_REVISION,
        "asset_receipt_checksum": verified["checksum"],
        "seed": seed,
        "physical_batch": 2,
        "profile": PROFILE,
        "tolerances": TOLERANCES,
        "prompt_sha256": [hashlib.sha256(p.encode()).hexdigest() for p in prompts],
        "reference": "pinned official LTXVideoPipeline and RectifiedFlowScheduler",
        "mean_observation": "(1-next_sigma)*actual original_samples passed to native add_noise",
        "training_acceptance": False,
        "reward_loaded": False,
        "finetune": finetune,
        "save_tensors_requested": save_tensors,
        "tensor_artifact_written": False,
    }
    report["hardware"] = {
        "name": hardware.name,
        "total_memory_bytes": hardware.total_memory,
        "devices_used": 1,
        "eight_gpu_acceptance": False,
    }
    try:
        print(
            "Loading receipt-verified pretrained LTX/T5/VAE; no reward or optimizer",
            file=sys.stderr,
            flush=True,
        )
        checkpoint = paths["ltx"] / LTX_VIDEO_CHECKPOINT
        loaded = load_ltx_video_components(
            checkpoint, paths["ltx_text"], native_root=sources["ltx"], device="cpu"
        )
        adapter = loaded.adapter.eval().requires_grad_(False)
        transformer, vae, encoder = adapter.transformer, adapter.vae, loaded.text_encoder
        pipeline_module = _native_module("ltx_video.pipelines.pipeline_ltx_video", sources["ltx"])
        scheduler_module = _native_module("ltx_video.schedulers.rf", sources["ltx"])
        patch_module = _native_module(
            "ltx_video.models.transformers.symmetric_patchifier", sources["ltx"]
        )
        checkpoint_config = read_ltx_video_checkpoint_config(checkpoint)
        # This is the same official class/config path used by its from_pretrained
        # method, without loading all checkpoint tensors merely to read metadata.
        scheduler = scheduler_module.RectifiedFlowScheduler.from_config(
            checkpoint_config["scheduler"]
        )
        pipe = pipeline_module.LTXVideoPipeline(
            tokenizer=loaded.tokenizer,
            text_encoder=encoder,
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
            patchifier=patch_module.SymmetricPatchifier(patch_size=1),
            prompt_enhancer_image_caption_model=None,
            prompt_enhancer_image_caption_processor=None,
            prompt_enhancer_llm_model=None,
            prompt_enhancer_llm_tokenizer=None,
            allowed_inference_steps=checkpoint_config["allowed_inference_steps"],
        ).to(target)
        initial_generator = torch.Generator(device=target).manual_seed(seed)
        initial_generator_before = initial_generator.get_state().clone()
        kwargs = official_pipeline_kwargs(prompts, initial_generator)
        validate_official_kwargs(type(pipe), kwargs)
        shape = ltx_video_latent_shape(2)
        with torch.random.fork_rng(devices=_fork_devices(target)):
            torch.random.default_generator.manual_seed(seed)
            with torch.cuda.device(target):
                torch.cuda.manual_seed(seed)
            with observe_official_ltx(pipe, pipeline_module, device=target) as trace:
                pipe(**kwargs)
        validate_official_trace(trace, shape=shape)
        expected_pixels = (2, 3, PROFILE["num_frames"], PROFILE["height"], PROFILE["width"])
        if trace.decoded.shape != expected_pixels:
            raise ValueError("official decoded video does not match the full native profile")
        # The official run above is complete and frozen before any PEFT wrapping
        # or FP32 parameter mutation. Never run a wrapped model as the reference.
        with torch.random.fork_rng(devices=_fork_devices(target)):
            torch.random.default_generator.manual_seed(seed)
            with torch.cuda.device(target):
                torch.cuda.manual_seed(seed)
            report["parameters"] = prepare_parity_candidate(
                adapter,
                encoder,
                finetune,
                reference_captured=True,
                expected_attention_layers=checkpoint_config["transformer"]["num_layers"],
            )
        report["candidate_initialization_seed"] = seed
        transformer = adapter.transformer

        def candidate_prediction(state, sigma, conditioning):
            if finetune == "full":
                return predict_full_bf16_views(adapter, state, sigma, conditioning)
            return adapter.predict_velocity(state, sigma, conditioning)

        metrics = [
            compare_tensor(
                "native_physical_schedule",
                trace.scheduler_timesteps,
                ltx_video_schedule()[:-1],
                tolerance="exact",
            )
        ]
        report["metrics"] = metrics
        print(
            "Checking independent adapter text/mask and frozen official inputs",
            file=sys.stderr,
            flush=True,
        )
        c = encode_ltx_video_prompts(prompts, loaded.tokenizer, encoder, device=target)
        metrics.extend(
            (
                compare_tensor("text_embeddings", trace.text[0], c.prompt_embeds, tolerance="bf16"),
                compare_tensor("text_mask", trace.text[1], c.attention_mask, tolerance="exact"),
            )
        )
        disabled_predictions, disabled_batches, disabled_precision = [], [], []
        if finetune == "lora":
            print(
                "Checking disabled LoRA against the already-captured official base",
                file=sys.stderr,
                flush=True,
            )

            def disabled_pre(module, args, kwargs):
                disabled_batches.append(len(args[0]))
                disabled_precision.append(forward_precision(args, kwargs))

            disabled_hook = transformer.register_forward_pre_hook(disabled_pre, with_kwargs=True)
            try:
                with (
                    transformer.disable_adapter(),
                    observe_precision_boundaries(transformer) as disabled_boundaries,
                ):
                    for index, item in enumerate(trace.steps):
                        prediction = candidate_prediction(
                            unpack_ltx_latents(item["state"], shape).to(target),
                            item["sigma"].flatten()[0].to(target).expand(2),
                            c,
                        )
                        disabled_predictions.append(_cpu(prediction))
                        metrics.append(
                            compare_tensor(
                                f"lora_disabled/{index}/official_base",
                                unpack_ltx_latents(item["velocity"], shape),
                                prediction,
                                tolerance="bf16",
                            )
                        )
            finally:
                disabled_hook.remove()
            if disabled_batches != [2] * 8:
                raise ValueError("disabled LoRA did not preserve physical batch2 forwards")
        frozen_predictions, adapter_forwards = [], []

        def adapter_pre(module, args, kwargs):
            adapter_forwards.append(
                {
                    "hidden_states": _cpu(args[0]),
                    "kwargs": _cpu(kwargs),
                    "precision": forward_precision(args, kwargs),
                }
            )

        observer = transformer.register_forward_pre_hook(adapter_pre, with_kwargs=True)
        try:
            with observe_precision_boundaries(transformer) as candidate_boundaries:
                for index, item in enumerate(trace.steps):
                    state = unpack_ltx_latents(item["state"], shape).to(target)
                    sigma = item["sigma"].flatten()[0].to(target).expand(2)
                    prediction = candidate_prediction(state, sigma, c)
                    frozen_predictions.append(_cpu(prediction))
                    metrics.append(
                        compare_tensor(
                            f"teacher_forced/{index}/velocity",
                            unpack_ltx_latents(item["velocity"], shape),
                            prediction,
                            tolerance="bf16",
                        )
                    )
                    if finetune == "lora":
                        metrics.append(
                            compare_tensor(
                                f"lora_zero_enabled/{index}/disabled",
                                disabled_predictions[index],
                                prediction,
                                tolerance="exact",
                            )
                        )
                    forwarded, candidate = trace.forwards[index], adapter_forwards[index]
                    # Full parity explicitly supplies BF16 compute views;
                    # record raw dtype/layout separately, including inner norms.
                    if (
                        candidate["hidden_states"].dtype != forwarded["hidden_states"].dtype
                        or candidate["precision"]["args"][0]["stride"]
                        != forwarded["precision"]["args"][0]["stride"]
                    ):
                        raise ValueError("candidate actual packed model input dtype/stride differs")
                    metrics.append(
                        compare_tensor(
                            f"teacher_forced/{index}/effective_bf16_model_input",
                            forwarded["hidden_states"],
                            candidate["hidden_states"],
                            tolerance="exact",
                        )
                    )
                    for key in (
                        "indices_grid",
                        "encoder_hidden_states",
                        "encoder_attention_mask",
                        "timestep",
                    ):
                        if candidate["kwargs"][key].dtype != forwarded["kwargs"][key].dtype:
                            raise ValueError(f"candidate actual {key} dtype differs from official")
                        metrics.append(
                            compare_tensor(
                                f"teacher_forced/{index}/{key}",
                                forwarded["kwargs"][key],
                                candidate["kwargs"][key],
                                tolerance="exact",
                            )
                        )
        finally:
            observer.remove()
        (
            rollout_layouts,
            rollout_forward_batches,
            rollout_state_precision,
            rollout_forward_precision,
        ) = [], [], [], []

        def predict(state, sigma, rows, index):
            print(
                f"Adapter native rollout step {index+1}/8, physical batch={len(state)}",
                file=sys.stderr,
                flush=True,
            )
            rollout_layouts.append(list(pack_ltx_latents(state).stride()))
            rollout_state_precision.append(tensor_metadata(state))
            return candidate_prediction(state, sigma, c.select(rows))

        def rollout_pre(module, args, kwargs):
            rollout_forward_batches.append(len(args[0]))
            rollout_forward_precision.append(forward_precision(args, kwargs))

        rollout_hook = transformer.register_forward_pre_hook(rollout_pre, with_kwargs=True)
        try:
            with torch.random.fork_rng(devices=_fork_devices(target)):
                _set_rng(trace.rng_after_initial, target)
                cache = collect_native_video(
                    predict,
                    trace.initial.to(target),
                    model="ltx_video_2b_distilled",
                    selected=tuple(range(8)),
                    microbatch=2,
                )
                rng_after_rollout = _rng(target)
        finally:
            rollout_hook.remove()
        if rollout_forward_batches != [2] * 8:
            raise ValueError(
                "adapter rollout transformer did not execute physical batch2 at every step"
            )
        metrics.extend(compare_native_cache(trace, cache, shape))
        for name, value in rng_after_rollout.items():
            metrics.append(
                compare_tensor(
                    f"global_rng_after_transitions/{name}",
                    trace.decode_noise[0]["rng_before"][name],
                    value,
                    tolerance="exact",
                )
            )
        if rollout_layouts != [item["state_stride"] for item in trace.steps]:
            raise ValueError("adapter packed rollout strides differ from actual official states")
        print(
            "Checking adapter default decode draw against the actual official RNG/layout",
            file=sys.stderr,
            flush=True,
        )
        endpoint = cache.endpoints.to(target)
        metrics.append(
            compare_tensor(
                "decode/endpoint_before_noise",
                trace.decode_noise[0]["state"],
                endpoint,
                tolerance="fp32",
            )
        )
        decode_draws = []
        adapter_module = importlib.import_module(adapter.__class__.__module__)

        def adapter_decode_draw(value, result, before):
            decode_draws.append({"innovation": _cpu(result), "stride": list(value.stride())})

        with torch.random.fork_rng(devices=_fork_devices(target)):
            _set_rng(trace.decode_noise[0]["rng_before"], target)
            with _replace(
                adapter_module,
                "torch",
                _TorchDrawObserver(adapter_module.torch, adapter_decode_draw),
            ):
                decoded = adapter.decode(endpoint)
        if len(decode_draws) != 1:
            raise ValueError(
                "adapter default decode did not make exactly one actual native noise draw"
            )
        metrics.append(
            compare_tensor(
                "decode/default_actual_innovation",
                trace.decode_noise[0]["innovation"],
                decode_draws[0]["innovation"],
                tolerance="exact",
            )
        )
        dedicated = torch.Generator(device=target)
        dedicated.set_state(trace.decode_noise[0]["rng_before"]["cuda"])
        dedicated_noise = torch.empty_like(endpoint).normal_(generator=dedicated)
        metrics.append(
            compare_tensor(
                "decode/dedicated_generator_innovation",
                trace.decode_noise[0]["innovation"],
                dedicated_noise,
                tolerance="exact",
            )
        )
        if trace.decode_noise[0]["stride"] != decode_draws[0]["stride"]:
            raise ValueError("adapter endpoint storage strides differ from official decode input")
        if decoded.shape != expected_pixels:
            raise ValueError("adapter decoded geometry differs from native full profile")
        metrics.append(
            compare_tensor("raw_decoded_pixels", trace.decoded, decoded, tolerance="bf16")
        )
        passed = all(metric["passed"] for metric in metrics)
        report.update(
            status="passed" if passed else "failed",
            metrics=metrics,
            physical_sigmas=trace.scheduler_timesteps.tolist(),
            initial_dtype=str(trace.initial.dtype),
            decoded_dtype=str(trace.decoded.dtype),
            transformer_physical_batches=[len(item["hidden_states"]) for item in trace.forwards],
        )
        report["adapter_teacher_forced_physical_batches"] = [
            len(item["hidden_states"]) for item in adapter_forwards
        ]
        report["candidate_raw_model_input_dtypes"] = [
            str(item["hidden_states"].dtype) for item in adapter_forwards
        ]
        report["lora_disabled_physical_batches"] = disabled_batches
        report["forward_precision"] = {
            "reference": [item["precision"] for item in trace.forwards],
            "candidate_teacher_forced": [item["precision"] for item in adapter_forwards],
            "lora_disabled": disabled_precision,
            "candidate_rollout": rollout_forward_precision,
            "collector_states": rollout_state_precision,
        }
        report["precision_boundaries"] = {
            "reference": trace.precision_boundaries,
            "candidate_teacher_forced": candidate_boundaries,
            "lora_disabled": disabled_boundaries if finetune == "lora" else [],
        }
        if report["adapter_teacher_forced_physical_batches"] != [2] * 8:
            raise ValueError("adapter did not execute one physical batch2 call per transition")
        report["decode_stride"] = decode_draws[0]["stride"]
        report["packed_rollout_strides"] = rollout_layouts
        report["adapter_rollout_physical_batches"] = rollout_forward_batches
        report["peak_memory"] = {
            "allocated_bytes": torch.cuda.max_memory_allocated(target),
            "reserved_bytes": torch.cuda.max_memory_reserved(target),
        }
        # JSON-safe scalar decode kwargs; the timestep tensor is stored in trace.pt.
        report["reference_vae_decode_kwargs"] = {
            key: value.tolist() if isinstance(value, torch.Tensor) else value
            for key, value in trace.decode_kwargs.items()
        }
        if parameter_inventory(transformer) != report["parameters"]["after"] or any(
            p.grad is not None
            for module in (transformer, vae, encoder)
            for p in module.parameters()
        ):
            raise ValueError("no-grad parity changed parameter trainability or produced gradients")
        if save_tensors:
            artifact = {
                "reference": vars(trace),
                "candidate_cache": vars(cache),
                "teacher_forced_predictions": frozen_predictions,
                "adapter_decode_draws": decode_draws,
                "lora_disabled_predictions": disabled_predictions,
                "initial_generator_before": initial_generator_before,
                "initial_generator_after": initial_generator.get_state(),
                "prompts": prompts,
            }
            print("Saving optional observed tensor trajectories", file=sys.stderr, flush=True)
            torch.save(artifact, output / "trace.pt")
            report["tensor_artifact_written"] = True
        print("Saving matched native video artifacts", file=sys.stderr, flush=True)
        from .evaluation.video import encode_video

        candidate_pixels = decoded.detach().cpu()
        for label, pixels in (("official", trace.decoded), ("adapter", candidate_pixels)):
            for row in range(2):
                encode_video(
                    pixels[row].clamp(-1, 1).permute(1, 0, 2, 3),
                    output / f"{label}-{row}.mp4",
                    fps=PROFILE["frame_rate"],
                    ffmpeg=ffmpeg,
                    ffprobe=ffprobe,
                )
        report["video_artifacts"] = [
            f"{label}-{row}.mp4" for label in ("official", "adapter") for row in range(2)
        ]
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        return report
    except BaseException as error:
        report.update(status="error", error_type=type(error).__name__, error=str(error))
        if "trace" in locals():
            report["partial_reference_steps"] = len(trace.steps)
            report["partial_reference_physical_batches"] = [
                len(x["hidden_states"]) for x in trace.forwards
            ]
        write_error_report(output / "report.json", report)
        if save_tensors and "trace" in locals():
            torch.save({"reference": vars(trace), "partial": True}, output / "failed-trace.pt")
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--finetune", choices=("base", "lora", "full"), default="base")
    parser.add_argument("--save-tensors", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--reward-only", action="store_true")
    parser.add_argument("--owned-root", type=Path)
    parser.add_argument("--reward-videos", nargs=2, type=Path)
    parser.add_argument("--reward-prompts", nargs=2)
    args = parser.parse_args(argv)
    if not args.reward_only and any(
        value is not None for value in (args.owned_root, args.reward_videos, args.reward_prompts)
    ):
        parser.error("reward input options require explicit --reward-only")
    if args.reward_only and (args.finetune != "base" or args.save_tensors):
        parser.error("reward-only does not prepare a policy or save model trajectories")
    if args.plan:
        if args.reward_only:
            print(
                json.dumps(
                    {
                        "reward_only": True,
                        "diagnostic_only": True,
                        "training_acceptance": False,
                        "optimizer_calls": 0,
                        "gpu_execution": False,
                        "required": [
                            "--device cuda",
                            "--owned-root",
                            "--reward-videos PATH1 PATH2",
                            "--reward-prompts TEXT1 TEXT2",
                            "--output",
                            "SMALL_VIDEO_REWARD_PYTHON",
                        ],
                        "profile": PROFILE,
                        "reward_fps": 4.0,
                        "use_norm": False,
                    },
                    indent=2,
                )
            )
            return 0
        print(
            json.dumps(
                {
                    "model": "LTX-Video2B0.9.6 distilled only",
                    "profile": PROFILE,
                    "physical_batch": 2,
                    "device": args.device,
                    "pretrained_inference": False,
                    "finetune": args.finetune,
                    "save_tensors": args.save_tensors,
                    "gpu_execution_requires": "explicit --device cuda, verified assets, new output directory",
                    "tolerances": TOLERANCES,
                },
                indent=2,
            )
        )
        return 0
    if args.device != "cuda" or args.output is None:
        parser.error(
            "real pretrained parity requires explicit --device cuda and a new --output; use --plan otherwise"
        )
    if args.reward_only:
        if args.owned_root is None or args.reward_videos is None or args.reward_prompts is None:
            parser.error(
                "reward-only requires explicit --owned-root, two videos and two matching prompts"
            )
        result = run_reward_only(
            asset_root=args.asset_root,
            receipt=args.receipt or args.asset_root / "preflight-ltx.json",
            output=args.output,
            owned_root=args.owned_root,
            videos=args.reward_videos,
            prompts=args.reward_prompts,
            device=args.device,
            manifest=args.manifest,
            ffprobe=args.ffprobe,
        )
        print(json.dumps({"status": result["status"], "report": str(args.output / "report.json")}))
        return 0 if result["status"] == "passed" else 1
    result = run_ltx_parity(
        asset_root=args.asset_root,
        receipt=args.receipt or args.asset_root / "preflight-ltx.json",
        output=args.output,
        device=args.device,
        manifest=args.manifest,
        seed=args.seed,
        ffmpeg=args.ffmpeg,
        ffprobe=args.ffprobe,
        finetune=args.finetune,
        save_tensors=args.save_tensors,
    )
    print(json.dumps({"status": result["status"], "report": str(args.output / "report.json")}))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
