"""Image training with native SD3 and FLUX model adapters.

This integrated development runner is not a frozen paper reproduction profile.
The legacy module path is retained for existing SD3 callers and checkpoints.
ART/NFT DDP dispatch uses shared kernels and evaluation. Deadline-isolated
artifact publication remains separate; do not use unreliable network storage.
"""

import hashlib
import inspect
import json
import math
import random
import uuid
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from ..algorithms import art_loss, art_scalar_covariance
from ..config import ConfigError
from ..data import PromptDataset, PromptStream
from ..evaluation import EvaluationManifest, SampleRecord, audit_manifest
from ..models import FluxAdapter, FluxConfig, SD3Adapter, SD3Config
from ..models.finetune import FineTuneConfig, parameter_inventory, prepare_policy
from ..models.sd3_text import SD3TextEncoder
from ..rewards.image_factory import (
    IMAGE_REWARDS,
    load_image_reward,
    validate_image_reward,
    validate_reward_prompts,
)
from ..rewards.pickscore import quantized_rgb
from ..rewards.scalar import weighted_score
from ..sampling import shifted_schedule
from ..training.checkpoint import _atomic_json, load_checkpoint, save_checkpoint
from ..training.grpo import ALGORITHMS as GRPO_ALGORITHMS
from ..training.grpo import GRPOState, prepare_grpo_update, validate_grpo
from ..training.image import ConditionedImagePolicy, decode_and_score_images
from ..training.image_contracts import prompt_advantages, sd3_schedule, tf32_policy
from ..training.nft import (
    NFTState,
    nft_windows,
    prepare_nft_objective,
    validate_nft,
)
from ..training.objectives import ARTObjective, disjoint_windows
from ..training.rollout import collect_rollout
from ..training.state import CostClock, Progress
from ..training.update import accumulated_step
from ..wandb_export import open_metrics


def _keys(options, allowed, label, required=()):
    if not isinstance(options, dict) or set(options) - set(allowed) or set(required) - set(options):
        raise ConfigError(
            f"invalid {label} keys; allowed={sorted(allowed)}, required={sorted(required)}"
        )


def validate_config(c):
    is_flux = c.model.name == "flux"
    if c.model.name not in ("sd3", "flux"):
        raise ConfigError("native image runner requires an SD3 or FLUX model")
    if c.evaluation.shift is not None:
        raise ConfigError("image evaluation uses its model scheduler; evaluation.shift is video-only")
    if is_flux and c.algorithm.name not in ("diffusionart", "diffusionnft", *GRPO_ALGORITHMS):
        raise ConfigError("native FLUX integrates ART/NFT and the Flow-GRPO/Fast/DPPO family")
    if (
        c.task != "t2i"
        or c.algorithm.name not in {"diffusionart", "diffusionnft", *GRPO_ALGORITHMS}
        or c.reward.name not in IMAGE_REWARDS
    ):
        raise ConfigError("native image training supports T2I ART/NFT/GRPO family with image rewards")
    is_nft = c.algorithm.name == "diffusionnft"
    is_grpo = c.algorithm.name in GRPO_ALGORITHMS
    is_art = c.algorithm.name == "diffusionart"
    if is_flux and c.algorithm.name == "dancegrpo":
        raise ConfigError("image DanceGRPO currently integrates SD3 only")
    if (
        c.runtime.distributed not in {"none", "ddp"}
        or (is_flux and c.training.precision == "fp16")
    ):
        raise ConfigError("native image training supports none/ddp; FLUX requires fp32/bf16")
    if c.data is None:
        raise ConfigError("native image training requires a hashed data manifest")
    if not c.model.checkpoint or not c.model.revision:
        raise ConfigError("model checkpoint/revision must be explicit")
    validate_image_reward(c.reward)
    m, a, s = c.model.options, c.algorithm.options, c.sampling.options
    _keys(
        m,
        {
            "scaling_factor",
            "shift_factor",
            "latent_channels",
            "spatial_factor",
            "max_sequence_length",
            "finetune",
            "decode_microbatch",
            "local_files_only",
            "gradient_checkpointing",
        } | ({"scheduler"} if is_flux else set()),
        "model.options",
        {
            "scaling_factor",
            "shift_factor",
            "latent_channels",
            "spatial_factor",
            "max_sequence_length",
            "finetune",
            "decode_microbatch",
            "gradient_checkpointing",
        } | ({"scheduler"} if is_flux else set()),
    )
    if is_nft:
        validate_nft(c)
    elif is_grpo:
        validate_grpo(c)
    else:
        _keys(
            a,
            {"covariance", "loss", "restore_zero_signal", "multi_reward_advantage", "updates_per_rollout"},
            "algorithm.options",
            {"covariance", "loss", "restore_zero_signal"},
        )
        sampling_keys = {"train_timesteps"} | (set() if is_flux else {"shift"})
        _keys(s, sampling_keys, "sampling.options", sampling_keys)
        _validate_multi_reward(c)
    for key in ("latent_channels", "spatial_factor", "decode_microbatch", "max_sequence_length"):
        if type(m[key]) is not int or m[key] < 1:
            raise ConfigError(f"model {key} must be a positive integer")
    if m["max_sequence_length"] > 512:
        raise ConfigError("image max_sequence_length must be <=512")
    if type(m["gradient_checkpointing"]) is not bool:
        raise ConfigError("gradient_checkpointing must be boolean")
    if type(m.get("local_files_only", True)) is not bool:
        raise ConfigError("model local_files_only must be boolean")
    if is_art and type(a["restore_zero_signal"]) is not bool:
        raise ConfigError("restore_zero_signal must be boolean")
    if any(size % m["spatial_factor"] for size in (c.sampling.height, c.sampling.width)):
        raise ConfigError("image dimensions must divide by spatial_factor")
    if c.sampling.samples_per_prompt < 2 or c.sampling.prompts_per_rollout < 1:
        raise ConfigError("group-based training requires P>=1 and K>=2")
    art_updates = a.get("updates_per_rollout", 2) if is_art else 1
    if is_art and (type(art_updates) is not int or art_updates not in (1, 2)):
        raise ConfigError("ART updates_per_rollout must be 1 or 2")
    if is_art and c.sampling.prompts_per_rollout * c.sampling.samples_per_prompt % art_updates:
        raise ConfigError("disjoint updates require a divisible rollout row count")
    if is_art and (
        c.training.max_updates % art_updates
        or c.runtime.checkpoint_every % art_updates
        or c.evaluation.every_updates % art_updates
        or any(step % art_updates for step in (*c.evaluation.at_updates, *c.runtime.checkpoint_at_updates))
    ):
        raise ConfigError("update budget and artifact intervals must align to two-update windows")
    if c.evaluation.enabled:
        if (
            not c.evaluation.manifest
            or not c.evaluation.manifest_sha256
            or not c.evaluation.retain_all
        ):
            raise ConfigError("evaluation requires a hashed manifest and retain_all=true")
    elif c.evaluation.manifest is not None or c.evaluation.manifest_sha256 is not None:
        raise ConfigError("disabled evaluation must not silently ignore a manifest")
    try:
        FineTuneConfig(**m["finetune"])
        (FluxConfig if is_flux else SD3Config)(m["scaling_factor"], m["shift_factor"], m["latent_channels"])
    except (ValueError, TypeError) as exc:
        raise ConfigError(f"invalid {c.model.name} model settings: {exc}") from exc
    if not is_art:
        return
    cov_keys = {
        k
        for k, p in inspect.signature(art_scalar_covariance).parameters.items()
        if p.kind == inspect.Parameter.KEYWORD_ONLY
    }
    loss_keys = {
        "loss_space",
        "direction_normalization",
        "mask_space",
        "velocity_mask_metric",
        "corrective_mask_enabled",
        "movement_mask_threshold",
        "noise_normalized_v_radius",
        "q2_reference_sde_noise_level",
        "q2_reference_kl_threshold",
        "eps",
    }
    _keys(a["covariance"], cov_keys, "covariance")
    _keys(a["loss"], loss_keys, "loss")
    try:
        FineTuneConfig(**m["finetune"])
        _schedule(c, c.sampling.steps)
        indices = s["train_timesteps"]
        if (
            not isinstance(indices, list)
            or not indices
            or len(indices) != len(set(indices))
            or any(type(i) is not int or not 0 <= i < c.sampling.steps for i in indices)
        ):
            raise ValueError("train_timesteps must be distinct valid indices")
        dummy = torch.ones(2, 1)
        art_scalar_covariance(
            dummy, torch.tensor([0.0, 1.0]), torch.zeros(2, dtype=torch.long), 1, **a["covariance"]
        )
        art_loss(
            dummy,
            dummy,
            torch.tensor([1.0, 0.5]),
            dummy,
            torch.ones(2),
            state=dummy,
            ode_velocity_coefficient=torch.full((2,), -0.1),
            current_ode_mean=dummy,
            old_ode_mean=dummy,
            **a["loss"],
        )
    except (ValueError, TypeError) as exc:
        raise ConfigError(f"invalid {c.model.name} algorithm/model settings: {exc}") from exc


@dataclass
class Components:
    adapter: SD3Adapter | FluxAdapter
    text: object
    scorer: torch.nn.Module


def _validate_multi_reward(c):
    options = c.algorithm.options.get("multi_reward_advantage")
    if "multi_reward_advantage" not in c.algorithm.options:
        return
    keys = {"weights", "score_space", "eps", "renormalize_combined"}
    _keys(options, keys, "multi_reward_advantage", keys)
    weights = options["weights"]
    if (
        not isinstance(weights, dict)
        or not weights
        or set(weights) - set(c.reward.weights)
        or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
            for v in weights.values()
        )
        or not math.isfinite(sum(weights.values()))
    ):
        raise ConfigError(
            "multi-reward guidance needs positive finite weights for scored components"
        )
    eps = options["eps"]
    if (
        isinstance(eps, bool)
        or not isinstance(eps, (int, float))
        or not math.isfinite(eps)
        or eps <= 0
    ):
        raise ConfigError("multi-reward eps must be finite and positive")
    if type(options["renormalize_combined"]) is not bool:
        raise ConfigError("renormalize_combined must be boolean")
    if options["score_space"] not in ("raw", "fixed_normalized"):
        raise ConfigError("multi-reward score_space must be raw or fixed_normalized")
    if options["score_space"] == "fixed_normalized" and not (
        c.reward.name in {"pickscore", "clipscore"}
        or (c.reward.name == "composite" and c.reward.options.get("fixed_statistics") is not None)
    ):
        raise ConfigError("multi-reward fixed score_space requires component fixed statistics")


def _art_guidance_scores(c, result, prompts):
    options = c.algorithm.options.get("multi_reward_advantage")
    if options is None:
        return result.scores.weighted, {}
    from ..training.multi_reward import combined_group_advantages

    values = getattr(result.scores, options["score_space"])
    scores, _ = combined_group_advantages(
        {name: values[name] for name in options["weights"]},
        prompts,
        options["weights"],
        eps=options["eps"],
        renormalize_combined=options["renormalize_combined"],
    )
    return scores, {}


def _precision(c):
    return {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[c.training.precision]


def _scaler(c, device):
    if c.training.precision != "fp16":
        return None
    if device.type not in ("cpu", "cuda"):
        raise ConfigError("FP16 loss scaling requires a CPU or CUDA device")
    return torch.amp.GradScaler(device.type)


def load_components(c):
    m = c.model.options
    device = torch.device("cuda", torch.cuda.current_device()) if c.runtime.device == "cuda" and torch.cuda.is_available() else torch.device(c.runtime.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ConfigError("CUDA unavailable; no implicit CPU fallback")
    precision = _precision(c)
    if device.type == "cuda" and precision == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise ConfigError("this device does not support bf16")
    is_flux = c.model.name == "flux"
    if is_flux:
        from diffusers import FlowMatchEulerDiscreteScheduler

        from ..training.flux_contracts import validate_asset_scheduler

        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            c.model.checkpoint, subfolder="scheduler", revision=c.model.revision,
            local_files_only=m.get("local_files_only", True),
        )
        validate_asset_scheduler(scheduler.config, m["scheduler"])
    adapter_config = (
        FluxConfig(m["scaling_factor"], m["shift_factor"], m["latent_channels"],
                   embedded_guidance_scale=c.sampling.guidance_scale,
                   vae_affine_dtype=precision)
        if is_flux else SD3Config(
            m["scaling_factor"], m["shift_factor"], m["latent_channels"],
            cfg_scale=c.sampling.guidance_scale if c.sampling.guidance_scale != 1 else None,
        )
    )
    adapter = (FluxAdapter if is_flux else SD3Adapter).from_pretrained(
        c.model.checkpoint,
        revision=c.model.revision,
        local_files_only=m.get("local_files_only", True),
        torch_dtype=torch.float32,
        config=adapter_config,
    ).to(device)
    if m["gradient_checkpointing"]:
        adapter.transformer.enable_gradient_checkpointing()
    adapter.transformer = prepare_policy(adapter.transformer, FineTuneConfig(**m["finetune"]))
    adapter.eval()
    vae_factor = 2 ** (len(adapter.vae.config.block_out_channels) - 1)
    if vae_factor != m["spatial_factor"]:
        raise ConfigError("spatial_factor differs from loaded VAE")
    patch = 2 if is_flux else adapter.transformer.config.patch_size
    if any((size // vae_factor) % patch for size in (c.sampling.height, c.sampling.width)):
        raise ConfigError("latent dimensions do not divide by transformer patch_size")
    text_cls = SD3TextEncoder
    if is_flux:
        from ..models.flux_text import FluxTextEncoder

        text_cls = FluxTextEncoder
    text = text_cls.from_pretrained(
        c.model.checkpoint,
        revision=c.model.revision,
        device=device,
        torch_dtype=precision,
        local_files_only=m.get("local_files_only", True),
    )
    scorer = load_image_reward(c.reward, device=device)
    return Components(adapter, text, scorer)


def _predict(c, parts, prompts, repeats):
    options = {} if c.model.name == "flux" else {"cfg_scale": parts.adapter.config.cfg_scale}
    condition = parts.text.encode(
        prompts,
        max_sequence_length=c.model.options["max_sequence_length"],
        num_images_per_prompt=repeats,
        **options,
    )
    return ConditionedImagePolicy(
        parts.adapter,
        condition,
        autocast_dtype=_precision(c) if c.training.precision != "fp32" else None,
        replay_input_dtype=torch.float32
        if c.model.name == "flux" and c.algorithm.name == "diffusionnft" else None,
    )


def _schedule(c, steps):
    if c.model.name == "flux":
        from ..training.flux_contracts import flux_schedule

        return flux_schedule(steps, height=c.sampling.height, width=c.sampling.width,
                             spatial_factor=c.model.options["spatial_factor"],
                             scheduler=c.model.options["scheduler"])
    return (shifted_schedule if c.algorithm.name == "diffusionart" else sd3_schedule)(
        steps, c.sampling.options["shift"]
    )


def _noise_shape(c, rows):
    m = c.model.options
    return (
        rows,
        m["latent_channels"],
        c.sampling.height // m["spatial_factor"],
        c.sampling.width // m["spatial_factor"],
    )


def _score(c, parts, endpoints, prompts, *, metadata=None, evaluation=False):
    return decode_and_score_images(
        parts.adapter,
        endpoints,
        prompts,
        parts.scorer,
        decode_microbatch=c.model.options["decode_microbatch"],
        score_microbatch=c.reward.options["microbatch"],
        metadata=metadata,
        evaluation=evaluation,
    )


def _reward_metrics(scores, prefix):
    metrics = {f"{prefix}/reward": float(scores.weighted.mean())}
    for space in ("raw", "fixed_normalized"):
        values = getattr(scores, space)
        if values is not None:
            metrics.update(
                {f"{prefix}/reward/{space}/{k}": float(v.mean()) for k, v in values.items()}
            )
    if scores.diagnostics is not None:
        from ..rewards.geneval import summarize_diagnostics

        metrics.update({f"{prefix}/diagnostics/{k}": v
                        for k, v in summarize_diagnostics(scores.diagnostics).items()})
    return metrics


def _check_run_identity(c, root):
    try:
        record = json.loads((root / "run.json").read_text())
        resolved = json.loads((root / "resolved-config.json").read_text())
    except (OSError, ValueError) as exc:
        raise ConfigError("resume output is missing valid run identity/config") from exc
    digest = hashlib.sha256(
        json.dumps(resolved, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        record.get("kind") != f"{c.model.name}_native_development"
        or record.get("config_sha256") != c.fingerprint()
        or digest != c.fingerprint()
    ):
        raise ConfigError("resume output belongs to a different run/config")


def _restore_cost(c, root, progress):
    for name in ("cost.json", "status.json"):
        path = root / name
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        if record.get("config_sha256") != c.fingerprint():
            raise ConfigError("stored cost/status belongs to a different run/config")
        saved = Progress.from_state_dict(
            record["progress"]
            if name == "cost.json"
            else {k: record[k] for k in Progress.__dataclass_fields__}
        )
        if saved.optimizer_steps > progress.optimizer_steps:
            raise ConfigError(
                "stored cost/status is newer than checkpoint; refuse history splicing"
            )
        progress.elapsed_seconds = max(progress.elapsed_seconds, saved.elapsed_seconds)
        progress.training_seconds = max(progress.training_seconds, saved.training_seconds)


def _publish_cost(c, root, clock):
    clock.tick()
    _atomic_json(
        root / "cost.json",
        {"config_sha256": c.fingerprint(), "progress": clock.progress.state_dict()},
    )


@torch.no_grad()
def evaluate(c, parts, dataset, step, root, *, indices=None, directory=None, publish=True):
    c = replace(c, seed=c.seed if c.evaluation.seed is None else c.evaluation.seed,
                sampling=replace(c.sampling, height=c.evaluation.height or c.sampling.height,
                                 width=c.evaluation.width or c.sampling.width))
    directory = directory or root / "evaluation" / f"step-{step:09d}" / uuid.uuid4().hex
    directory.mkdir(parents=True, exist_ok=not publish)
    indices = tuple(range(len(dataset.rows))) if indices is None else tuple(indices)
    if len(set(indices)) != len(indices) or any(type(i) is not int or not 0 <= i < len(dataset.rows) for i in indices):
        raise ValueError("evaluation indices must be distinct valid global rows")
    samples = []
    steps = c.evaluation.sampling_steps or c.sampling.steps
    schedule = _schedule(c, steps)
    device = next(parts.adapter.trainable_module.parameters()).device
    for start in range(0, len(indices), c.sampling.microbatch):
        batch_indices = indices[start : start + c.sampling.microbatch]
        rows = [dataset.rows[i] for i in batch_indices]
        prompts = [row.text for row in rows]
        predict = _predict(c, parts, prompts, 1)
        seeds = [c.seed + i for i in batch_indices]
        noise = torch.cat(
            [
                torch.randn(
                    _noise_shape(c, 1),
                    generator=torch.Generator(device=device).manual_seed(seed),
                    device=device,
                    dtype=torch.bfloat16 if c.model.name == "flux" and c.training.precision == "bf16" else torch.float32,
                )
                for seed in seeds
            ]
        )
        cache = collect_rollout(
            predict.rollout if c.model.name == "flux" else predict,
            noise, schedule, selected=(steps - 1,), microbatch=c.sampling.microbatch,
            preserve_state_dtype=c.model.name == "flux",
            unselected_kind="cps_zero_noise"
            if c.model.name == "flux" and c.sampling.options.get("transition_kind") == "cps"
            else "ode",
        )
        result = _score(c, parts, cache.endpoints, prompts,
                        metadata=[row.metadata for row in rows], evaluation=True)
        for i, image in enumerate(quantized_rgb(result.pixels)):
            path = directory / f"{batch_indices[i]:06d}.png"
            image.save(path, format="PNG")
            score = weighted_score(
                {k: float(v[i]) for k, v in result.scores.raw.items()},
                result.scores.weights,
                space=result.scores.space,
                normalization=result.scores.normalization,
                diagnostics=None if result.scores.diagnostics is None else {
                    k: float(v[i]) for k, v in result.scores.diagnostics.items()
                },
            )
            samples.append(
                SampleRecord.from_file(
                    media_root=root,
                    path=path.relative_to(root).as_posix(),
                    sample_id=rows[i].sample_id,
                    prompt=prompts[i],
                    seed=seeds[i],
                    optimizer_step=step,
                    framecount=1,
                    fps=1.0,
                    scores=score,
                )
            )
    if not publish:
        return tuple(samples)
    return _publish_evaluation(c, dataset, step, root, directory, samples)


def _publish_evaluation(c, dataset, step, root, directory, samples):
    manifest = EvaluationManifest(
        optimizer_step=step,
        expected_sample_ids=tuple(r.sample_id for r in dataset.rows),
        samples=tuple(samples),
    )
    _atomic_json(directory / "manifest.json", json.loads(manifest.to_json()))
    audit = audit_manifest(manifest, media_root=root)
    if not audit.complete:
        raise RuntimeError("evaluation artifacts incomplete")
    summary = {
        "eval/optimizer_step": step,
        "eval/samples": len(samples),
        "eval/reward": sum(s.scores.weighted for s in samples) / len(samples),
        "manifest": str((directory / "manifest.json").resolve()),
    }
    for space in ("raw", "fixed_normalized"):
        values = getattr(samples[0].scores, space)
        if values is not None:
            for key in values:
                summary[f"eval/reward/{space}/{key}"] = sum(
                    getattr(sample.scores, space)[key] for sample in samples
                ) / len(samples)
    if samples[0].scores.diagnostics is not None:
        from ..rewards.geneval import summarize_diagnostics

        diagnostics = {key: torch.tensor([s.scores.diagnostics[key] for s in samples])
                       for key in samples[0].scores.diagnostics}
        summary.update({f"eval/diagnostics/{k}": v
                        for k, v in summarize_diagnostics(diagnostics).items()})
    _atomic_json(directory / "summary.json", summary)
    return summary


def run(c, *, resume=None):
    validate_config(c)
    if c.runtime.distributed == "ddp":
        from .image_ddp import run as run_ddp

        return run_ddp(c, resume=resume)
    if torch.distributed.is_initialized():
        raise ConfigError("single-device image runner must not enter an existing process group")
    data = PromptDataset.load(c.data.manifest, expected_sha256=c.data.sha256, task="t2i")
    validation = (
        PromptDataset.load(
            c.evaluation.manifest, expected_sha256=c.evaluation.manifest_sha256, task="t2i"
        )
        if c.evaluation.enabled
        else None
    )
    root = Path(c.runtime.output_dir)
    validate_reward_prompts(c.reward, [row.text for row in data.rows],
                            metadata=[row.metadata for row in data.rows])
    if validation is not None:
        validate_reward_prompts(c.reward, [row.text for row in validation.rows],
                                metadata=[row.metadata for row in validation.rows])
    if resume is None:
        root.mkdir(parents=True, exist_ok=False)
    elif not root.is_dir():
        raise ConfigError("resume requires the original output directory")
    import fcntl

    with (root / ".owner.lock").open("a") as lock, ExitStack() as resources:
        resources.enter_context(tf32_policy(c.training.allow_tf32))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if resume:
            _check_run_identity(c, root)
        torch.manual_seed(c.seed)
        random.seed(c.seed)
        np.random.seed(c.seed)
        parts = load_components(c)
        for scorer in parts.scorer.modules():
            close = getattr(scorer, "close", None)
            if callable(close):
                resources.callback(close)
        model = parts.adapter.trainable_module
        device = next(model.parameters()).device
        stream = PromptStream(data, seed=c.seed, shuffle=c.data.shuffle)
        generator = torch.Generator(device=device).manual_seed(c.seed)
        nft = NFTState(model, c.algorithm.options) if c.algorithm.name == "diffusionnft" else None
        grpo = (
            GRPOState(model, c.algorithm.options) if c.algorithm.name in GRPO_ALGORITHMS else None
        )
        algorithm_state = nft or grpo
        updates_per_rollout = (
            c.algorithm.options["updates_per_rollout"]
            if grpo is not None
            else c.algorithm.options.get("updates_per_rollout", 1)
            if nft is not None
            else c.algorithm.options.get("updates_per_rollout", 2)
        )
        stateful = {
            "prompts": stream,
            **(algorithm_state.stateful() if algorithm_state is not None else {}),
        }
        scaler = _scaler(c, device)
        if scaler is not None:
            stateful["grad_scaler"] = scaler
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=c.training.learning_rate,
            betas=c.training.betas,
            eps=c.training.epsilon,
            weight_decay=c.training.weight_decay,
        )
        progress = Progress()
        if resume:
            progress = load_checkpoint(
                resume,
                model=model,
                optimizer=optimizer,
                config_sha256=c.fingerprint(),
                stateful=stateful,
                generators={"sampling": generator},
            )
            if progress.optimizer_steps % updates_per_rollout:
                raise ConfigError("resume requires a complete rollout update boundary")
            if (root / "metrics.jsonl").exists():
                for line in (root / "metrics.jsonl").read_text().splitlines():
                    if json.loads(line).get("train/optimizer_step", 0) > progress.optimizer_steps:
                        raise ConfigError("metrics newer than checkpoint; refuse history splicing")
            _restore_cost(c, root, progress)
            if (
                progress.optimizer_steps == c.training.max_updates
                and (root / "status.json").exists()
            ):
                # A completed no-op resume must not replace final costs with the
                # checkpoint's pre-publication snapshot.
                return
        else:
            _atomic_json(root / "resolved-config.json", c.to_dict())
            _atomic_json(
                root / "run.json",
                {
                    "kind": f"{c.model.name}_native_development",
                    "config_sha256": c.fingerprint(),
                    "parameters": parameter_inventory(model),
                    "device": str(device),
                    "data_sha256": data.sha256,
                },
            )
        clock = CostClock(progress, gpu_count=int(device.type == "cuda"))
        if validation and not resume:
            with algorithm_state.evaluation() if algorithm_state is not None else nullcontext():
                evaluate(c, parts, validation, 0, root)
        p, k = c.sampling.prompts_per_rollout, c.sampling.samples_per_prompt
        groups = torch.arange(p).repeat_interleave(k)
        sigmas = _schedule(c, c.sampling.steps)
        selected = tuple(c.sampling.options.get("train_timesteps", ()))
        empty_rollouts = 0
        with open_metrics(c, root) as log:
            while progress.optimizer_steps < c.training.max_updates:
                with clock.training_phase():
                    progress.attempts += 1
                    rows = stream.take(p)
                    predict = _predict(c, parts, [row.text for row in rows], k)
                    noise_dtype = (
                        _precision(c)
                        if (algorithm_state is not None or c.model.name == "flux") and c.training.precision != "fp32"
                        else torch.float32
                    )
                    shared_noise = c.algorithm.name == "dancegrpo" and c.sampling.options["same_latent"]
                    noise = torch.randn(
                        _noise_shape(c, p if shared_noise else p * k),
                        generator=generator,
                        device=device,
                        dtype=noise_dtype,
                    )
                    if shared_noise:
                        noise = noise.repeat_interleave(k, dim=0)
                    if grpo is not None and c.algorithm.name == "flow_grpo" and c.model.name == "sd3":
                        # The full SD3 pipeline promotes prepared initial latents;
                        # the Fast pipeline promotes after its first transition.
                        noise = noise.float()
                    prompts = [row.text for row in rows for _ in range(k)]
                    if nft is not None:
                        from ..training.nft_sampling import collect_nft_endpoints

                        with nft.old.use():
                            endpoints = collect_nft_endpoints(
                                predict.rollout if c.model.name == "flux" else predict,
                                noise,
                                sigmas,
                                solver=c.sampling.options["solver"],
                                microbatch=c.sampling.microbatch,
                                quantize_model_time=c.model.name != "flux",
                            )
                    elif grpo is not None:
                        from ..training.grpo_sampling import collect_grpo_rollout

                        options = {
                            key: value
                            for key, value in c.sampling.options.items()
                            if key not in {"shift", "same_latent"}
                        }
                        options["pool"] = tuple(options["pool"])
                        cache = collect_grpo_rollout(
                            predict.rollout if c.model.name == "flux" else predict,
                            noise,
                            sigmas,
                            microbatch=c.sampling.microbatch,
                            generator=generator,
                            flux_precision=c.model.name == "flux",
                            **options,
                        )
                        endpoints = cache.endpoints
                    else:
                        cache = collect_rollout(
                            predict.rollout if c.model.name == "flux" else predict,
                            noise,
                            sigmas,
                            selected=selected,
                            microbatch=c.sampling.microbatch,
                            preserve_state_dtype=c.model.name == "flux",
                        )
                        endpoints = cache.endpoints
                    result = _score(c, parts, endpoints, prompts,
                                    metadata=[row.metadata for row in rows for _ in range(k)])
                    algorithm_metrics = {}
                    if nft is not None:
                        advantages = prompt_advantages(result.scores.weighted, prompts)
                        objective = prepare_nft_objective(
                            c, nft, predict, endpoints, sigmas, advantages, generator, device
                        )
                        windows = nft_windows(c, objective)
                    elif grpo is not None:
                        objective, windows, algorithm_metrics = prepare_grpo_update(
                            c, cache, predict, result.scores.weighted, prompts, generator, device
                        )
                        if not windows:
                            empty_rollouts += 1
                            record = {
                                **clock.metrics(),
                                **algorithm_metrics,
                                "system/skipped_zero_advantage": 1,
                                **_reward_metrics(result.scores, "train"),
                            }
                            log.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                            log.flush()
                            _publish_cost(c, root, clock)
                            if empty_rollouts >= c.algorithm.options["max_empty_rollouts"]:
                                raise RuntimeError(
                                    "configured consecutive zero-advantage rollout limit reached; no optimizer/EMA updates applied"
                                )
                            continue
                        empty_rollouts = 0
                    else:
                        guidance_scores, algorithm_metrics = _art_guidance_scores(
                            c, result, prompts
                        )
                        objective, windows = _art_update(
                            c,
                            cache,
                            predict,
                            result,
                            groups,
                            p,
                            k,
                            selected,
                            generator,
                            device,
                            guidance_scores=guidance_scores,
                        )
                    for items in windows:
                        if grpo is not None:
                            grpo.before_update(progress.optimizer_steps)
                        metrics = accumulated_step(
                            model,
                            optimizer,
                            items,
                            objective,
                            microbatch=c.training.microbatch,
                            denominator=len(items),
                            max_grad_norm=c.training.grad_clip,
                            scaler=scaler,
                        )
                        progress.optimizer_steps += 1
                        if nft is not None:
                            metrics.update(nft.after_update(progress.optimizer_steps,
                                refresh_old=progress.optimizer_steps % updates_per_rollout == 0))
                        elif grpo is not None:
                            grpo.after_update(progress.optimizer_steps)
                        metrics.update(algorithm_metrics)
                        if progress.optimizer_steps % updates_per_rollout == 0:
                            progress.completed_rollouts += 1
                        metrics.update(clock.metrics())
                        metrics.update(_reward_metrics(result.scores, "train"))
                        log.write(json.dumps(metrics, sort_keys=True, allow_nan=False) + "\n")
                        log.flush()
                step = progress.optimizer_steps
                try:
                    if validation and c.evaluation.due(step, c.training.max_updates):
                        with (
                            algorithm_state.evaluation()
                            if algorithm_state is not None
                            else nullcontext()
                        ):
                            evaluate(c, parts, validation, step, root)
                    if c.runtime.checkpoint_due(step, c.training.max_updates):
                        clock.tick()
                        save_checkpoint(
                            root / "checkpoints",
                            model=model,
                            optimizer=optimizer,
                            progress=progress,
                            config_sha256=c.fingerprint(),
                            stateful=stateful,
                            generators={"sampling": generator},
                        )
                finally:
                    _publish_cost(c, root, clock)
        _publish_cost(c, root, clock)
        clock.tick()
        _atomic_json(
            root / "status.json",
            {
                "status": "completed",
                "kind": f"{c.model.name}_native_development",
                "config_sha256": c.fingerprint(),
                **progress.state_dict(),
            },
        )


def _art_update(
    c, cache, predict, result, groups, p, k, selected, generator, device, *, guidance_scores=None
):
    stats = art_scalar_covariance(
        cache.endpoints,
        result.scores.weighted if guidance_scores is None else guidance_scores,
        groups,
        p,
        **c.algorithm.options["covariance"],
    )
    directions = stats["group_directions"][groups]
    if c.model.name == "flux" and c.training.precision == "bf16":
        # Historical FLUX casts group directions to the rollout latent dtype
        # before the loss's second normalization and corrective-mask test.
        directions = directions.bfloat16()
    objective = ARTObjective(
        cache,
        predict,
        directions,
        stats["group_guidance_magnitudes"][groups],
        torch.full((p * k,), c.algorithm.options["restore_zero_signal"], dtype=torch.bool),
        device,
        options=c.algorithm.options["loss"],
        reconstruct_from_mean=c.model.name == "flux",
    )
    # The one-window setting is the explicit restoration ablation.
    order = torch.randperm(p * k, generator=generator, device=device).cpu().tolist()
    windows = tuple(
        tuple((order[row], slot) for row, slot in window)
        for window in disjoint_windows(p * k, len(selected),
                                       updates=c.algorithm.options.get("updates_per_rollout", 2))
    )
    return objective, windows
