"""Native Hunyuan T2V training, scoring, evaluation and full local resume.

Uses cached native conditioning and the public FastVideo transformer ABI.
Shared algorithm dispatch also serves the full-shard runner.
"""

import json
import math
import random
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from ..algorithms import art_loss, art_scalar_covariance, clipped_grpo_loss
from ..config import ConfigError
from ..data import PromptDataset, PromptStream
from ..evaluation import SampleRecord
from ..evaluation.video import encode_video
from ..models.finetune import FineTuneConfig, parameter_inventory
from ..models.hunyuan import HunyuanVideoConditioning
from ..models.hunyuan_assets import cached_conditioning, load_adapter, load_reward
from ..rewards.scalar import weighted_score
from ..sampling import shifted_schedule
from ..training import video_grpo, video_nft, video_pairwise, video_refl
from ..training.checkpoint import _atomic_json, load_checkpoint, save_checkpoint
from ..training.objectives import ARTObjective, disjoint_windows
from ..training.resume import prepare_resume_history, validate_resume_request
from ..training.rollout import collect_rollout
from ..training.state import CostClock, Progress
from ..training.update import accumulated_step
from ..training.video_grpo import ALGORITHMS as VIDEO_GRPO
from ..wandb_export import open_metrics
from .sd3 import (
    _check_run_identity,
    _publish_cost,
    _publish_evaluation,
    _restore_cost,
    _reward_metrics,
)


def validate_config(c, *, model_name="hunyuan_public", task="t2v", extra_model_keys=()):
    if c.training.allow_tf32 is not None:
        raise ConfigError("native video uses model.options.allow_tf32, not training.allow_tf32")
    if (c.task != task or c.model.name != model_name or c.algorithm.name not in ("diffusionart", "diffusionnft", "refl", *VIDEO_GRPO)
            or c.reward.name != "video_reward" or c.runtime.distributed not in ("none", "fsdp")
            or c.training.precision != "bf16"):
        raise ConfigError("native Hunyuan integrates ART/NFT/ReFL/Fast-SDE/Dance T2V with VideoReward and BF16 autocast")
    if not c.data or not c.model.checkpoint or not c.model.revision or not c.reward.checkpoint or not c.reward.revision:
        raise ConfigError("explicit model/reward identities and hashed prompt data are required")
    if set(c.model.options) != {"native_root", "finetune", "gradient_checkpointing", "ffmpeg", "ffprobe", "allow_tf32", *extra_model_keys}:
        raise ConfigError("Hunyuan requires native_root, finetune, gradient_checkpointing, ffmpeg, ffprobe, allow_tf32")
    if set(c.reward.options) != {"backbone"} or set(c.reward.weights) != {"VQ", "MQ", "TA"}:
        raise ConfigError("VideoReward requires backbone and explicit VQ/MQ/TA weights")
    if c.algorithm.name == "diffusionart":
        allowed = {"covariance", "loss", "restore_zero_signal"}
        if video_pairwise.enabled(c):
            allowed.add("pairwise")
        if set(c.algorithm.options) != allowed:
            raise ConfigError("ART requires covariance, loss and restore_zero_signal")
        if type(c.algorithm.options["restore_zero_signal"]) is not bool:
            raise ConfigError("restore_zero_signal must be boolean")
        if video_pairwise.enabled(c):
            video_pairwise.validate_options(c)
        else:
            if c.algorithm.options["restore_zero_signal"]:
                raise ConfigError("scalar ART requires restore_zero_signal=false")
            if (not isinstance(c.algorithm.options["covariance"], dict)
                    or c.algorithm.options["covariance"].get("zero_signal_mode", "exact") != "exact"):
                raise ConfigError("scalar ART requires exact zero-signal protection")
    elif c.algorithm.name == "diffusionnft":
        video_nft.validate_options(c)
    elif c.algorithm.name == "refl":
        video_refl.validate_options(c)
    else:
        if set(c.algorithm.options) != {"loss", "advantage_epsilon"}:
            raise ConfigError("video GRPO requires loss and advantage_epsilon; no EMA or reference regularizer")
        for value in (c.algorithm.options["advantage_epsilon"], c.sampling.options.get("noise_level")):
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ConfigError("positive finite advantage_epsilon and noise_level are required")
        if c.sampling.options["noise_level"] > 1:
            raise ConfigError("video noise_level must not exceed one")
        if set(c.algorithm.options["loss"]) != {"ratio_clip", "advantage_clip"}:
            raise ConfigError("video GRPO loss requires explicit ratio_clip and advantage_clip")
    sampling_keys = {"shift", "train_timestep_count", "train_timestep_pool"}
    if c.algorithm.name in VIDEO_GRPO:
        sampling_keys.add("noise_level")
    if set(c.sampling.options) != sampling_keys:
        raise ConfigError(f"Hunyuan sampling requires {sorted(sampling_keys)}")
    if any(type(c.model.options[key]) is not bool for key in ("gradient_checkpointing", "allow_tf32")):
        raise ConfigError("gradient_checkpointing and allow_tf32 must be boolean")
    if c.sampling.guidance_scale != 1:
        raise ConfigError("public Hunyuan uses embedded guidance 6018 and no external CFG")
    if c.runtime.distributed == "fsdp" and c.model.options["finetune"].get("mode") != "full":
        raise ConfigError("native Hunyuan FSDP requires full-weight training")
    updates = updates_per_rollout(c)
    if ((c.algorithm.name != "refl" and c.sampling.samples_per_prompt < 2)
            or c.sampling.prompts_per_rollout * c.sampling.samples_per_prompt % updates):
        raise ConfigError("video updates require complete K>=2 groups and rows divisible by update windows")
    if c.sampling.frames < 5 or (c.sampling.frames - 1) % 4:
        raise ConfigError("Hunyuan frames must have form 4n+1")
    for value in (c.sampling.height, c.sampling.width, c.evaluation.height or c.sampling.height,
                  c.evaluation.width or c.sampling.width):
        if value % 16:
            raise ConfigError("Hunyuan dimensions must divide by 16; no implicit codec resizing")
    pool, count = c.sampling.options["train_timestep_pool"], c.sampling.options["train_timestep_count"]
    stop = c.sampling.steps if c.algorithm.name in ("diffusionnft", "refl") else c.sampling.steps - 1
    if (not isinstance(pool, list) or not pool or pool != sorted(set(pool))
            or any(type(i) is not int or not 0 <= i < stop for i in pool)
            or type(count) is not int or not 1 <= count <= len(pool)):
        raise ConfigError("invalid training timestep pool/count")
    if c.evaluation.fps != c.sampling.fps:
        raise ConfigError("train and eval video FPS must be identical")
    if c.evaluation.enabled and (not c.evaluation.manifest or not c.evaluation.manifest_sha256):
        raise ConfigError("evaluation requires its own hashed manifest")
    if any(value % updates for value in (c.training.max_updates, c.evaluation.every_updates,
            c.runtime.checkpoint_every, *c.evaluation.at_updates, *c.runtime.checkpoint_at_updates)):
        raise ConfigError("budget, eval and checkpoint intervals must align with rollout update boundaries")
    if not c.evaluation.retain_all:
        raise ConfigError("native video evaluation retains all scored videos")
    try:
        FineTuneConfig(**c.model.options["finetune"])
        shifted_schedule(c.sampling.steps, c.sampling.options["shift"])
        dummy = torch.ones(2, 1)
        if c.algorithm.name == "diffusionart":
            art_scalar_covariance(dummy, torch.tensor([0., 1.]), torch.zeros(2, dtype=torch.long), 1,
                              **c.algorithm.options["covariance"])
            art_loss(dummy, dummy, torch.ones(2), dummy, torch.ones(2), state=dummy,
                 ode_velocity_coefficient=-torch.ones(2), current_ode_mean=dummy,
                 old_ode_mean=dummy, **c.algorithm.options["loss"])
            if c.algorithm.options["loss"].get("reference_mse_beta", 0):
                raise ValueError("reference regularization is not configured for this video runner")
        elif c.algorithm.name in VIDEO_GRPO:
            clipped_grpo_loss(torch.ones(2), torch.ones(2), torch.ones(2), **c.algorithm.options["loss"])
    except (ValueError, TypeError) as error:
        raise ConfigError(f"invalid Hunyuan algorithm settings: {error}") from error


@dataclass
class Components:
    adapter: object
    scorer: object


def load_datasets(c):
    data = PromptDataset.load(c.data.manifest, expected_sha256=c.data.sha256, task="t2v")
    validation = PromptDataset.load(c.evaluation.manifest, expected_sha256=c.evaluation.manifest_sha256,
                                    task="t2v") if c.evaluation.enabled else None
    return data, validation


def updates_per_rollout(c):
    return 1 if c.algorithm.name in ("diffusionnft", "refl") else 2


def load_components(c, device):
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = c.model.options["allow_tf32"]
    parts = Components(load_adapter(c, device), load_reward(c))
    if c.algorithm.name == "refl":
        from ..training.refl_vjp import configure_hunyuan_vae, configure_video_reward

        configure_hunyuan_vae(parts.adapter.vae)
        configure_video_reward(parts.scorer)
    return parts


def _shape(c, rows, *, evaluation=False):
    h = (c.evaluation.height or c.sampling.height) if evaluation else c.sampling.height
    w = (c.evaluation.width or c.sampling.width) if evaluation else c.sampling.width
    return rows, 16, (c.sampling.frames - 1) // 4 + 1, h // 8, w // 8


@torch.no_grad()
def evaluation_endpoint(predict, noise, sigmas):
    """Native public evaluation: BF16 recurrence, final FP32 predicted x0."""
    state = noise
    rows = torch.arange(len(noise), device=noise.device)
    for step in range(len(sigmas) - 1):
        sigma = sigmas[step].to(noise.device)
        velocity = predict(state, sigma.expand(len(noise)), rows).float()
        endpoint = state.float() - sigma * velocity
        state = (state.float() + (sigmas[step + 1] - sigmas[step]).to(noise.device) * velocity).to(torch.bfloat16)
        if not torch.isfinite(endpoint).all():
            raise ValueError("nonfinite Hunyuan evaluation endpoint")
    return endpoint.detach().cpu()


def _predict(c, parts, rows, manifest, repeats, device, *, attempt=0, rank=0, evaluation=False):
    embed, mask = cached_conditioning(rows, manifest)
    embed, mask = embed.repeat_interleave(repeats, 0), mask.repeat_interleave(repeats, 0)
    def predict(state, times, indices):
        indices = indices.cpu()
        condition = HunyuanVideoConditioning(embed[indices].to(device), mask[indices].to(device))
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=c.training.precision == "bf16"):
            return parts.adapter.predict_velocity(state, times, condition)
    return predict


def _rollout_inputs(c, attempt, rank=0):
    generator = torch.Generator(device="cpu").manual_seed((c.seed + 1_000_003 * attempt + 10_007 * rank) % (2**63 - 1))
    rows = c.sampling.prompts_per_rollout * c.sampling.samples_per_prompt
    noise = torch.randn(_shape(c, rows), generator=generator,
                        dtype=torch.bfloat16 if c.training.precision == "bf16" else torch.float32)
    pool = torch.tensor(c.sampling.options["train_timestep_pool"])
    selected = torch.stack([pool[torch.randperm(len(pool), generator=generator)[:c.sampling.options["train_timestep_count"]]] for _ in range(rows)])
    return noise, selected


def training_inputs(c, attempt, rank=0):
    if c.algorithm.name in VIDEO_GRPO:
        return video_grpo.inputs(c, _shape(c, c.sampling.prompts_per_rollout * c.sampling.samples_per_prompt),
                                 attempt=attempt, rank=rank)
    noise, selected = _rollout_inputs(c, attempt, rank)
    return noise, selected, None


def collect_training(c, parts, predict, plan, sigmas, device):
    parts.adapter.transformer.eval()
    if c.algorithm.name in VIDEO_GRPO:
        return video_grpo.collect(c, predict, plan, sigmas, device)
    if c.algorithm.name == "diffusionnft":
        return video_nft.collect(predict, plan, sigmas, microbatch=c.sampling.microbatch, device=device)
    noise, selected, _ = plan
    return collect_rollout(predict, noise.to(device), sigmas, selected=selected[0].tolist(),
                           row_selected=selected, microbatch=c.sampling.microbatch)


def prepare_objective(c, parts, cache, predict, scores, global_scores, device):
    parts.adapter.transformer.train(c.algorithm.name != "flow_grpo_fast")
    if c.algorithm.name in VIDEO_GRPO:
        return video_grpo.prepare_objective(c, cache, predict, scores, global_scores, device)
    k = c.sampling.samples_per_prompt
    count = len(cache.endpoints)
    groups = torch.arange(count) // k
    covariance = art_scalar_covariance(cache.endpoints, scores.weighted, groups, count // k,
        **{"zero_signal_mode": "exact", **c.algorithm.options["covariance"]})
    anchors = (~covariance["group_active"][groups] if c.algorithm.options["restore_zero_signal"]
               else torch.zeros(count, dtype=torch.bool))
    objective = ARTObjective(cache, predict, covariance["group_directions"][groups],
        covariance["group_guidance_magnitudes"][groups], anchors, device,
        options=c.algorithm.options["loss"], reconstruct_old_from_mean=True)
    return objective, disjoint_windows(count, len(cache.selected))


@torch.no_grad()
def score_videos(c, parts, endpoints, prompts, directory, device, *, evaluation=False, decode_bf16=True):
    directory.mkdir(parents=True, exist_ok=False)
    paths = []
    vae = parts.adapter.vae
    try:
        vae.to(device).eval()
        for index, endpoint in enumerate(endpoints):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=decode_bf16 and c.training.precision == "bf16"):
                raw = parts.adapter.to_vae_latents(endpoint[None].to(device))
                decoded = vae.decode(raw, return_dict=False)[0].clamp(-1, 1)
            expected = _shape(c, 1, evaluation=evaluation)
            if decoded.shape != (1, 3, c.sampling.frames, expected[-2] * 8, expected[-1] * 8):
                raise ValueError("VAE output differs from configured video geometry")
            paths.append(encode_video(decoded[0].permute(1, 0, 2, 3), directory / f"{index:06d}.mp4",
                fps=c.sampling.fps, ffmpeg=c.model.options["ffmpeg"], ffprobe=c.model.options["ffprobe"]))
            del raw, decoded
    finally:
        vae.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()
    try:
        parts.scorer.to(device).eval()
        scores = parts.scorer.score_paths(paths, prompts)
    finally:
        parts.scorer.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return paths, scores


@torch.no_grad()
def evaluate(c, parts, dataset, step, root, device, *, backend=None):
    backend = sys.modules[__name__] if backend is None else backend
    parts.adapter.transformer.eval()
    directory = root / "evaluation" / f"step-{step:09d}" / uuid.uuid4().hex
    directory.mkdir(parents=True)
    samples = []
    steps = c.evaluation.sampling_steps or c.sampling.steps
    sigmas = shifted_schedule(steps, c.evaluation.shift or c.sampling.options["shift"])
    for index, row in enumerate(dataset.rows):
        seed = (c.seed if c.evaluation.seed is None else c.evaluation.seed) + index
        noise = torch.randn(backend._shape(c, 1, evaluation=True), generator=torch.Generator().manual_seed(seed),
                            dtype=torch.float32 if c.task == "i2v" else torch.bfloat16)
        predict = backend._predict(c, parts, [row], c.evaluation.manifest, 1, device, evaluation=True)
        endpoints = backend.evaluation_endpoint(predict, noise.to(device), sigmas)
        paths, scores = backend.score_videos(c, parts, endpoints, [row.text], directory / f"sample-{index:06d}", device, evaluation=True)
        score = weighted_score({key: float(value[0]) for key, value in scores.raw.items()}, scores.weights,
                               space=scores.space, normalization=scores.normalization)
        samples.append(SampleRecord.from_file(media_root=root, path=paths[0].relative_to(root).as_posix(),
            sample_id=row.sample_id, prompt=row.text, seed=seed, optimizer_step=step,
            framecount=c.sampling.frames, fps=c.sampling.fps, scores=score))
    return publish_evaluation(c, dataset, step, root, directory, samples)


publish_evaluation = _publish_evaluation


def score_training_videos(backend, c, parts, endpoints, prompts, root, device):
    """MP4 transport is temporary; only evaluation media is retained."""
    with TemporaryDirectory(prefix=".train-media-", dir=root) as directory:
        _, scores = backend.score_videos(c, parts, endpoints, prompts, Path(directory) / "samples", device)
    return scores


def run(c, *, resume=None, backend=None, resume_rewind=False, evaluate_only=False):
    validate_resume_request(resume, resume_rewind, evaluate_only)
    import fcntl

    from ..training.distributed import launch_identity

    backend = sys.modules[__name__] if backend is None else backend
    backend.validate_config(c)
    if c.runtime.distributed == "fsdp":
        from .hunyuan_fsdp import run as run_fsdp

        return run_fsdp(c, resume=resume, backend=backend,
                        resume_rewind=resume_rewind, evaluate_only=evaluate_only)
    launch_identity("none")
    if torch.distributed.is_initialized():
        raise ConfigError("single-device video runner must not enter an existing process group")
    device = torch.device(c.runtime.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ConfigError("CUDA unavailable")
    data, validation = backend.load_datasets(c)
    root = Path(c.runtime.output_dir)
    if resume is None:
        root.mkdir(parents=True, exist_ok=False)
    elif not root.is_dir():
        raise ConfigError("resume requires original output directory")
    with (root / ".owner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if resume:
            _check_run_identity(c, root)
        torch.manual_seed(c.seed)
        random.seed(c.seed)
        np.random.seed(c.seed)
        parts = backend.load_components(c, device)
        model = parts.adapter.trainable_module
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=c.training.learning_rate,
                                       betas=c.training.betas, eps=c.training.epsilon, weight_decay=c.training.weight_decay)
        stream = PromptStream(data, seed=c.seed, shuffle=c.data.shuffle)
        nft = video_nft.VideoNFTState(model, c.algorithm.options) if c.algorithm.name == "diffusionnft" else None
        stateful = {"prompts": stream, **(nft.stateful() if nft else {})}
        checkpoint = dict(model=model, optimizer=optimizer, config_sha256=c.fingerprint(), stateful=stateful)
        progress = load_checkpoint(resume, **checkpoint) if resume else Progress()
        if resume:
            if progress.optimizer_steps % updates_per_rollout(c) or progress.optimizer_steps > c.training.max_updates:
                raise ConfigError("resume must be at an in-budget completed rollout boundary")
            if evaluate_only:
                if validation is None:
                    raise ConfigError("checkpoint evaluation requires a validation manifest")
                return evaluate(c, parts, validation, progress.optimizer_steps, root, device, backend=backend)
            prepare_resume_history(c, root, progress, receipt=resume, rewind=resume_rewind)
            _restore_cost(c, root, progress)
            if progress.optimizer_steps == c.training.max_updates and (root / "status.json").exists():
                return
        else:
            _atomic_json(root / "resolved-config.json", c.to_dict())
            _atomic_json(root / "run.json", {"kind": f"{c.model.name}_native_development", "config_sha256": c.fingerprint(),
                "parameters": parameter_inventory(model), "device": str(device), "data_sha256": data.sha256})
        clock = CostClock(progress, gpu_count=int(device.type == "cuda"))
        sigmas = shifted_schedule(c.sampling.steps, c.sampling.options["shift"])
        if validation and not resume:
            evaluate(c, parts, validation, 0, root, device, backend=backend)
        with open_metrics(c, root) as log:
            while progress.optimizer_steps < c.training.max_updates:
                with clock.training_phase():
                    progress.attempts += 1
                    rows = stream.take(c.sampling.prompts_per_rollout)
                    k = c.sampling.samples_per_prompt
                    prompts = [row.text for row in rows for _ in range(k)]
                    predict = backend._predict(c, parts, rows, c.data.manifest, k, device, attempt=progress.attempts)
                    plan = backend.training_inputs(c, progress.attempts)
                    if nft:
                        with nft.use_old():
                            cache = backend.collect_training(c, parts, predict, plan, sigmas, device)
                    else:
                        cache = backend.collect_training(c, parts, predict, plan, sigmas, device)
                    scores = score_training_videos(backend, c, parts, cache.endpoints, prompts, root, device)
                    extra_metrics = {}
                    if nft:
                        objective = video_nft.prepare_objective(c, cache, predict, scores, scores, device,
                                                               attempt=progress.attempts)
                        with nft.use_old():
                            video_nft.predict_old(objective, c.training.microbatch)
                        windows = (tuple(range(len(objective.states))),)
                    elif c.algorithm.name == "refl":
                        objective, windows, extra_metrics = video_refl.prepare_objective(
                            c, parts, cache, predict, prompts, device)
                    elif video_pairwise.enabled(c):
                        parts.adapter.transformer.train()
                        feedback = video_pairwise.prepare(c, cache, scores, attempt=progress.attempts)
                        objective, windows, extra_metrics = video_pairwise.finalize(
                            c, cache, predict, feedback, video_pairwise.activity(feedback, device), device)
                        if objective is None:
                            metrics = {**_reward_metrics(scores, "train"), **extra_metrics, **clock.metrics()}
                            log.write(json.dumps(metrics, allow_nan=False, sort_keys=True) + "\n")
                            log.flush()
                            _publish_cost(c, root, clock)
                            del feedback, cache
                            continue
                    else:
                        objective, windows = backend.prepare_objective(c, parts, cache, predict, scores, scores, device)
                        if c.algorithm.name == "diffusionart" and not objective.has_guidance:
                            metrics = {**_reward_metrics(scores, "train"), **clock.metrics(),
                                       "system/skipped_zero_guidance": 1}
                            log.write(json.dumps(metrics, allow_nan=False, sort_keys=True) + "\n")
                            log.flush()
                            _publish_cost(c, root, clock)
                            continue
                    for items in windows:
                        metrics = accumulated_step(model, optimizer, items, objective, microbatch=c.training.microbatch,
                                                   denominator=len(items), max_grad_norm=c.training.grad_clip,
                                                   require_nonzero_grad=c.algorithm.name == "refl")
                        progress.optimizer_steps += 1
                        if nft:
                            metrics.update(nft.after_update(progress.optimizer_steps))
                        if progress.optimizer_steps % updates_per_rollout(c) == 0:
                            progress.completed_rollouts += 1
                        metrics.update(_reward_metrics(scores, "train"))
                        metrics.update(extra_metrics)
                        metrics.update(clock.metrics())
                        log.write(json.dumps(metrics, allow_nan=False, sort_keys=True) + "\n")
                        log.flush()
                step = progress.optimizer_steps
                if validation and c.evaluation.due(step, c.training.max_updates):
                    evaluate(c, parts, validation, step, root, device, backend=backend)
                if c.runtime.checkpoint_due(step, c.training.max_updates):
                    clock.tick()
                    save_checkpoint(root / "checkpoints", progress=progress,
                                    keep_last=c.runtime.checkpoint_keep, **checkpoint)
                _publish_cost(c, root, clock)
        _publish_cost(c, root, clock)
        _atomic_json(root / "status.json", {"status": "completed", "kind": f"{c.model.name}_native_development",
                                           "config_sha256": c.fingerprint(), **progress.state_dict()})
