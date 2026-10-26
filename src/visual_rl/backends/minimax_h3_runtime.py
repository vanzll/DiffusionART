"""Native opt-in H3 training, staged residency and exact rollout-boundary resume.

This backend owns execution; scientific objectives stay in training.audio_video.
The explicit post-training recipe can retain a frozen pretrained shard shadow.
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import tempfile
import time
import uuid
import warnings
from contextlib import ExitStack, contextmanager
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from ..config import ConfigError
from ..data import PromptStream
from ..evaluation import SampleRecord
from ..evaluation.video import encode_video
from ..models.finetune import parameter_inventory
from ..rewards.audio_video import AVScores, validate_media
from ..rewards.scalar import weighted_score
from ..training.audio_video import AVNFTState, run_av_updates
from ..training.audio_video_objectives import (
    prepare_art_objective,
    prepare_fast_objective,
    prepare_nft_objective,
    prepare_routed_fast_objective,
    route_rewards,
    select_nft_fresh_times,
)
from ..training.audio_video_rollout import collect_av_rollout, select_fast_window
from ..training.audio_video_types import AVState
from ..training.checkpoint import _atomic_json, load_checkpoint, save_checkpoint
from ..training.distributed import distributed_session, launch_identity
from ..training.fsdp_checkpoint import load_fsdp_checkpoint, save_fsdp_checkpoint
from ..training.image_contracts import tf32_policy
from ..training.resume import prepare_resume_history, validate_resume_request
from ..training.state import CostClock, Progress
from ..training.timesteps import art_training_positions
from ..wandb_export import open_metrics
from . import minimax_h3 as backend
from .minimax_h3_execution import execution_policy
from .minimax_h3_observability import BatchTrace, event
from .minimax_h3_posttraining import (
    duration_config,
    duration_frames,
    evaluation_step_directory,
    fast_innovation_seed,
    rollout_config,
    selected_positions,
)
from .sd3 import _check_run_identity, _publish_evaluation, _restore_cost


class H3AssetIdentity:
    """Immutable model+scorer identity, independently checked at checkpoint load.

    Paths are part of the declared identity. A valid replacement manifest at
    the same configured pathname is still a different scientific execution.
    """
    def __init__(self, manifest):
        self._canonical = json.dumps(manifest, sort_keys=True, allow_nan=False, separators=(",", ":"))

    def state_dict(self):
        return {"schema_version": 1, "assets": json.loads(self._canonical)}

    def load_state_dict(self, state):
        if state != self.state_dict():
            raise ConfigError("H3 model/reward asset identity differs from checkpoint")


def sigma_schedules(boundaries, *, video_shift=12., audio_shift=3.):
    """Pinned native MiniMaxH3Scheduler: linspace(1,0,N), N-1 transitions."""
    if type(boundaries) is not int or boundaries < 2:
        raise ConfigError("native H3 sampling requires at least two sigma boundaries")
    base = torch.linspace(1., 0., boundaries, dtype=torch.float32)
    return tuple(shift * base / (1 + (shift - 1) * base) for shift in (video_shift, audio_shift))


def wrap_policy(model, c, device, *, block_types=None):
    """FSDP1 FP32 masters; isolate trainable LoRA leaves from frozen bases."""
    if block_types is None:
        from ..models.minimax_h3 import native_h3_block_types
        block_types = native_h3_block_types()
    if not block_types:
        raise ValueError("explicit H3 native block classes required")
    model.float()
    if hasattr(model, "gradient_checkpointing"):
        model.gradient_checkpointing = False
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
    lora = c.model.options["finetune"]["mode"] == "lora"
    # The fixed native forward queries these objects with get_parameter_dtype
    # BEFORE calling them. Parent flattening leaves original leaves without
    # registered Parameters, so each queried object must retain its own FSDP
    # flat Parameter. No upstream helper or global monkeypatch is changed.
    dtype_units = set()
    for module in model.modules():
        name = type(module).__name__
        if name == "MiniMaxH3Transformer3DModel":
            dtype_units.update(id(getattr(module, attribute)) for attribute in (
                "proj_in", "audio_proj_in", "context_embedder", "time_embedder", "proj_out"))
        elif name in {"MiniMaxH3AdaLayerNormModulation", "MiniMaxH3AdaLayerNormOut"}:
            dtype_units.add(id(module.linear))
    def auto_wrap(module, recurse, nonwrapped_numel):
        if recurse:
            return True
        leaf = lora and not list(module.children()) and any(p.requires_grad for p in module.parameters(recurse=False))
        return isinstance(module, tuple(block_types)) or leaf or id(module) in dtype_units
    model = FSDP(model, device_id=device, auto_wrap_policy=auto_wrap,
        sharding_strategy=ShardingStrategy.FULL_SHARD, use_orig_params=False,
        mixed_precision=MixedPrecision(param_dtype=torch.float32, reduce_dtype=torch.float32,
            buffer_dtype=torch.float32, cast_forward_inputs=False), limit_all_gathers=True)
    if c.model.options["gradient_checkpointing"]:
        apply_activation_checkpointing(model,
            checkpoint_wrapper_fn=partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
            check_fn=lambda module: isinstance(module, tuple(block_types)))
    return model


def release_policy(model):
    """Free no-grad FSDP all-gathers before decode; preserve rank-local shards."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    if isinstance(model, FSDP):
        from torch.distributed.fsdp._runtime_utils import _reshard
        for unit in FSDP.fsdp_modules(model):
            handle = unit._handle
            if handle is not None and hasattr(handle.flat_param, "_local_shard"):
                _reshard(unit, handle, True)


@contextmanager
def staged(module, device):
    """Frozen auxiliary components return to CPU even if local work fails."""
    module.to(device)
    try:
        yield module
    finally:
        module.to("cpu")
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()


def _idle_policy(parts, session):
    release_policy(parts.adapter.trainable_module)
    if not session.distributed:
        parts.adapter.trainable_module.to("cpu")


def _active_policy(parts, session):
    if not session.distributed:
        parts.adapter.trainable_module.to(session.device)


def prediction_callback(parts, rows, repeats, session):
    """Encode once per prompt, keep conditioning on CPU, select original rows."""
    _idle_policy(parts, session)
    # CPU is deliberate: 66.7GB frozen conditioner plus FP32 FSDP policy
    # shards/Adam moments cannot coexist on an 80GB training GPU.
    with staged(parts.conditioner.text_encoder, "cpu"):
        conditioning = parts.conditioner.encode([row.text for row in rows]).to("cpu")
    indices = torch.arange(len(rows)).repeat_interleave(repeats)
    conditioning = conditioning.select(indices)
    def predict(state, video_sigma, audio_sigma, original_rows):
        selected = conditioning.select(original_rows.cpu()).to(session.device)
        with torch.autocast(device_type=session.device.type, dtype=torch.bfloat16,
                            enabled=session.device.type == "cuda"):
            return parts.adapter.predict_velocity(state, video_sigma, audio_sigma, selected)
    return predict


def _noise(parts, c, rows, generator, *, evaluation=False):
    height = (c.evaluation.height or c.sampling.height) if evaluation else c.sampling.height
    width = (c.evaluation.width or c.sampling.width) if evaluation else c.sampling.width
    shapes = parts.adapter.latent_shapes(rows, height, width, c.sampling.frames, c.sampling.fps)
    return AVState(*(torch.randn(shape, generator=generator, dtype=torch.bfloat16) for shape in shapes))


def _innovations(noise, positions, generator):
    return {position: AVState(torch.randn(noise.video.shape, generator=generator),
                             torch.randn(noise.audio.shape, generator=generator)) for position in positions}


def collect(parts, c, predict, noise, sigmas, session, generator, *, rollout_index, evaluation=False):
    _active_policy(parts, session)
    options = {}
    if not evaluation and c.algorithm.options.get("post_training") is not None:
        post = c.algorithm.options["post_training"]
        selections = selected_positions(c.seed, rollout_index + 1,
            video_pool=post["video_pool"], audio_pool=post["audio_pool"])
        options.update(selected=tuple(sorted(set(selections["video"]) | set(selections["audio"]))),
                       modality_selected=selections)
    elif not evaluation and c.algorithm.name == "diffusionart" and c.training.timestep_selection != "configured":
        options["selected"] = art_training_positions(c.training, (), seed=c.seed, attempt=rollout_index + 1)
    if not evaluation and c.algorithm.name == "flow_grpo_fast":
        if c.algorithm.options.get("post_training") is not None:
            positions = options["selected"]
            # Initial noises retain the same generator stream as matched ART.
            # Extra CPS innovations have private, reproducible per-rank seeds.
            innovations_generator = torch.Generator().manual_seed(
                fast_innovation_seed(c.seed, rollout_index + 1, session.identity.rank))
            options["modality_stochastic"] = selections
        else:
            positions = select_fast_window(len(sigmas[0]) - 1, rollout_index=rollout_index,
                seed=c.sampling.options["window_seed"], size=c.sampling.options["window_size"],
                window_range=tuple(c.sampling.options["window_range"]))
            innovations_generator = generator
        options.update(selected=positions, stochastic=positions, innovations=_innovations(noise, positions, innovations_generator),
            noise_level=c.sampling.options["noise_level"], unselected_kind="cps_zero_noise")
    elif evaluation or c.algorithm.name == "diffusionnft":
        options["selected"] = ()
    return collect_av_rollout(predict, noise, *sigmas, microbatch=c.sampling.microbatch,
                              device=session.device, **options)


@torch.no_grad()
def score_endpoints(parts, endpoints, rows, repeats, session, *, save_directory=None, c=None):
    """Decode bounded rows with only VAEs resident; score canonical media."""
    _idle_policy(parts, session)
    video_text, audio_text = backend.reward_text(rows, repeats)
    scores, diagnostics, paths = {}, {}, []
    event("decode_reward_begin", samples=len(endpoints))
    for index in range(len(endpoints)):
        # Both native VAEs are needed for adapter.decode; the policy and text
        # encoder have already released their unsharded GPU residency.
        with staged(parts.adapter.vae, session.device), staged(parts.adapter.audio_vae, session.device):
            media = parts.adapter.decode(endpoints.select(slice(index, index + 1)).to(session.device))
            media = type(media)(media.video.detach().cpu(), media.audio.detach().cpu(), media.fps, media.sample_rate)
        result = parts.reward.score(media, video_text[index:index + 1], audio_text[index:index + 1])
        for key, value in result.components.items():
            scores.setdefault(key, []).append(value.detach().cpu().reshape(-1))
        for key, value in result.diagnostics.items():
            diagnostics.setdefault(key, []).append(value.detach().cpu().reshape(-1))
        if save_directory is not None:
            paths.append(save_media(media, save_directory / f"sample-{index:06d}.mp4",
                ffmpeg=c.model.options["ffmpeg"], ffprobe=c.model.options["ffprobe"]))
        event("decode_reward_sample_complete", completed=index + 1, total=len(endpoints),
              sample_id=rows[index // repeats].sample_id, trajectory_index=index % repeats)
    return paths, AVScores({k: torch.cat(v) for k, v in scores.items()},
                           {k: torch.cat(v) for k, v in diagnostics.items()})


def save_media(media, target, *, ffmpeg="ffmpeg", ffprobe="ffprobe", timeout=120):
    """Preserve native float WAV and mux AAC preview; never trim or resample."""
    validate_media(media)
    if len(media) != 1:
        raise ValueError("one canonical AV row per preview")
    target = Path(target)
    if target.exists() or target.with_suffix(".wav").exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".av-media-", dir=target.parent) as temporary:
        temporary = Path(temporary)
        video = encode_video(media.video[0].permute(1, 0, 2, 3) * 2 - 1, temporary / "video.mp4",
                             fps=media.fps, ffmpeg=ffmpeg, ffprobe=ffprobe, timeout=timeout)
        raw = temporary / "audio.f32"
        raw.write_bytes(media.audio[0].float().transpose(0, 1).contiguous().numpy().tobytes())
        wav, muxed = temporary / "audio.wav", temporary / "preview.mp4"
        subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-f", "f32le", "-ar", str(media.sample_rate),
            "-ac", str(media.audio.shape[1]), "-i", str(raw), "-c:a", "pcm_f32le", str(wav)],
            check=True, capture_output=True, timeout=timeout)
        subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-i", str(video), "-i", str(wav),
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart", str(muxed)], check=True, capture_output=True, timeout=timeout)
        record = subprocess.run([ffprobe, "-v", "error", "-show_entries", "stream=codec_type,sample_rate,channels",
            "-of", "json", str(muxed)], check=True, capture_output=True, timeout=timeout)
        streams = json.loads(record.stdout)["streams"]
        audio = [s for s in streams if s["codec_type"] == "audio"]
        if len(audio) != 1 or int(audio[0]["sample_rate"]) != media.sample_rate or audio[0]["channels"] != media.audio.shape[1]:
            raise ValueError("AV preview changed native sample rate/channel layout")
        subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-xerror", "-i", str(muxed), "-f", "null", "-"],
            check=True, capture_output=True, timeout=timeout)
        os.link(wav, target.with_suffix(".wav"))
        os.link(muxed, target)
    _atomic_json(target.with_suffix(".media.json"), {"fps": media.fps, "sample_rate": media.sample_rate,
        "channels": media.audio.shape[1], "video_frames": media.video.shape[2], "audio_samples": media.audio.shape[-1],
        "audio_master": target.with_suffix(".wav").name, "audio_master_codec": "pcm_f32le",
        "preview_audio_codec": "aac", "resampled": False, "trimmed": False})
    return target


def global_scores(session, scores):
    """Gather transformed per-scorer scalars; normalize after complete groups."""
    session.agree({"components": sorted(scores.components), "diagnostics": sorted(scores.diagnostics),
                   "shapes": {k: list(v.shape) for k, v in scores.components.items()}})
    def gather(value):
        value = value.detach().float().to(session.device).contiguous()
        if not session.distributed:
            return value.cpu()
        received = [torch.empty_like(value) for _ in range(session.identity.world_size)]
        dist.all_gather(received, value)
        return torch.cat(received).cpu()
    return AVScores({key: gather(value) for key, value in sorted(scores.components.items())},
                    {key: gather(value) for key, value in sorted(scores.diagnostics.items())})


def _metrics(scores, *, prefix="train"):
    values = {f"{prefix}/reward/{name}": float(value.float().mean()) for name, value in scores.components.items()}
    values.update({f"{prefix}/reward_std/{name}": float(value.float().std(unbiased=False)) for name, value in scores.components.items()})
    values.update({f"{prefix}/diagnostics/{name}": float(value.float().mean()) for name, value in scores.diagnostics.items()})
    return values


def _cost(session, clock):
    values = clock.metrics()
    cuda = session.device.type == "cuda"
    peaks = ([torch.cuda.max_memory_allocated(session.device),
              torch.cuda.max_memory_reserved(session.device)] if cuda else [0, 0])
    times = torch.tensor([values["cost/elapsed_seconds"], values["cost/training_seconds"],
                          *peaks], dtype=torch.float64, device=session.device)
    if session.distributed:
        dist.all_reduce(times, op=dist.ReduceOp.MAX)
    elapsed, training, allocated, reserved = times.cpu().tolist()
    count = session.identity.world_size if session.device.type == "cuda" else 0
    values.update({"cost/elapsed_seconds": elapsed, "cost/training_seconds": training,
        "cost/elapsed_gpu_hours": elapsed * count / 3600, "cost/training_gpu_hours": training * count / 3600})
    if cuda:
        values.update({"system/peak_memory_allocated_bytes": allocated,
                       "system/peak_memory_reserved_bytes": reserved})
    return values


def _publish_cost(session, c, root, clock):
    metrics = _cost(session, clock)
    progress = {**clock.progress.state_dict(), "elapsed_seconds": metrics["cost/elapsed_seconds"],
                "training_seconds": metrics["cost/training_seconds"]}
    session.primary("AV cost publication", lambda: _atomic_json(root / "cost.json", {"config_sha256": c.fingerprint(), "progress": progress}))
    return progress


def _prepare_objective(c, parts, cache, predict, local_scores, population, groups, session, generator, nft,
                       reference=None):
    post = c.algorithm.options.get("post_training")
    if post is not None:
        local = route_rewards(local_scores.components, groups=groups, weights=c.reward.weights, prompt_group_std=True)
        global_routes = None
    else:
        local = route_rewards(local_scores.components)
        global_routes = route_rewards(population.components)
    reference_predictions = None
    if reference is not None:
        from .minimax_h3_reference import prepare_reference_predictions
        _active_policy(parts, session)
        # No live backward graph exists here. Complete all frozen-policy
        # collective forwards, then restore current shards before U2.
        with reference.use_reference(session):
            reference_predictions = prepare_reference_predictions(
                cache, predict, session.device, c.training.microbatch)
    if c.algorithm.name == "diffusionart":
        return session.local("AV ART complete-group objective", lambda: prepare_art_objective(cache, predict,
            local["video"], local["audio"], groups, session.device, covariance_options=c.algorithm.options["covariance"],
            loss_options=c.algorithm.options["loss"], reference_predictions=reference_predictions))
    if c.algorithm.name == "flow_grpo_fast" and post is not None:
        return session.local("AV routed Fast complete-group objective", lambda: prepare_routed_fast_objective(
            cache, predict, local["video"], local["audio"], groups, session.device,
            reference_predictions=reference_predictions, **c.algorithm.options["loss"]))
    if nft is not None:
        def prepare():
            times = select_nft_fresh_times(cache.video_sigmas, len(cache.endpoints), generator=generator)
            shapes = [tuple((len(cache.endpoints), times.shape[1], *getattr(cache.endpoints, name).shape[1:]))
                      for name in ("video", "audio")]
            noise = AVState(*(torch.randn(shape, generator=generator) for shape in shapes))
            return times, noise
        times, noise = session.local("AV NFT fresh-noise preparation", prepare)
        _active_policy(parts, session)
        with nft.use_old(session):
            # Contains actual FSDP forwards. Never enclose this in session.local
            # error voting; a model collective failure is terminal.
            return prepare_nft_objective(cache.endpoints, predict, predict, local["video"], local["audio"], groups,
                times, noise, session.device, microbatch=c.training.microbatch,
                global_video_rewards=global_routes["video"], global_audio_rewards=global_routes["audio"])
    return session.local("AV Fast complete-group objective", lambda: prepare_fast_objective(cache, predict,
        local["shared"], groups, session.device, global_shared_rewards=global_routes["shared"],
        **c.algorithm.options["loss"], advantage_epsilon=c.algorithm.options["advantage_epsilon"]))


def evaluate(session, c, parts, dataset, step, root):
    """Evaluate each duration independently, preserving prompt/seed identities."""
    if "duration_frames" not in (c.algorithm.options.get("post_training") or {}):
        return _evaluate_duration(session, c, parts, dataset, step, root)
    results = {}
    for frames in duration_frames(c):
        workload = duration_config(c, frames)
        complete = session.primary("AV duration evaluation receipt", lambda: _completed_duration_evaluation(
            workload, dataset, step, root))
        flag = torch.tensor(int(bool(complete)), dtype=torch.int32, device=session.device)
        if session.distributed:
            dist.broadcast(flag, src=0)
        if not bool(flag.item()):
            results[frames] = _evaluate_duration(session, workload, parts, dataset, step, root)
    return results


def _evaluate_duration(session, c, parts, dataset, step, root):
    """Equal FSDP forward counts, including dummy rows on final empty shards."""
    model = parts.adapter.trainable_module
    event("evaluation_begin", optimizer_steps=step, prompts=len(dataset.rows))
    model.eval()
    token = torch.tensor(list(uuid.uuid4().bytes) if session.identity.rank == 0 else [0] * 16,
                         dtype=torch.uint8, device=session.device)
    if session.distributed:
        dist.broadcast(token, src=0)
    directory = root / "evaluation" / evaluation_step_directory(c, step) / bytes(token.cpu().tolist()).hex()
    session.primary("AV evaluation directory", lambda: directory.mkdir(parents=True))
    rank, world = session.identity.rank, session.identity.world_size
    records, indices = [], []
    sigmas = sigma_schedules(c.evaluation.sampling_steps or c.sampling.steps,
        video_shift=c.sampling.options["video_shift"], audio_shift=c.sampling.options["audio_shift"])
    for offset in range(0, len(dataset.rows), world):
        index = offset + rank
        valid = index < len(dataset.rows)
        row = dataset.rows[index if valid else 0]
        seed = (c.seed if c.evaluation.seed is None else c.evaluation.seed) + (index if valid else 0)
        def prepare():
            predict = prediction_callback(parts, [row], 1, session)
            generator = torch.Generator().manual_seed(seed)
            return predict, generator, _noise(parts, c, 1, generator, evaluation=True)
        predict, generator, noise = session.local("AV evaluation conditioning", prepare)
        cache = collect(parts, c, predict, noise, sigmas, session, generator, rollout_index=0, evaluation=True)
        def score():
            if not valid:
                _idle_policy(parts, session)
                return None
            paths, scores = score_endpoints(parts, cache.endpoints, [row], 1, session,
                save_directory=directory / f"row-{index:06d}", c=c)
            value = weighted_score({key: float(value[0]) for key, value in scores.components.items()}, c.reward.weights,
                space="raw", diagnostics={key: float(value[0]) for key, value in scores.diagnostics.items()})
            return SampleRecord.from_file(media_root=root, path=paths[0].relative_to(root).as_posix(),
                sample_id=row.sample_id, prompt=row.text, seed=seed, optimizer_step=step,
                framecount=c.sampling.frames, fps=c.sampling.fps, scores=value).to_dict()
        record = session.local("AV evaluation decode and score", score)
        if valid:
            indices.append(index)
            records.append(record)
    session.local("AV evaluation shard", lambda: _atomic_json(directory / f"rank-{rank:05d}.json", {"indices": indices, "samples": records}))
    def publish():
        ordered = {}
        for worker in range(world):
            shard = json.loads((directory / f"rank-{worker:05d}.json").read_text())
            expected = list(range(worker, len(dataset.rows), world))
            if shard["indices"] != expected or len(shard["samples"]) != len(expected):
                raise ValueError("AV evaluation shard sample identities differ")
            for index, record in zip(expected, shard["samples"]):
                sample = SampleRecord.from_dict(record)
                if sample.sample_id != dataset.rows[index].sample_id or sample.prompt != dataset.rows[index].text:
                    raise ValueError("AV evaluation prompt identity differs")
                ordered[index] = sample
        samples = [ordered[i] for i in range(len(dataset.rows))]
        mixed = "duration_frames" in (c.algorithm.options.get("post_training") or {})
        namespace = f"eval/frames{c.sampling.frames}" if mixed else "eval"
        result = _publish_evaluation(c, dataset, step, root, directory, samples, metric_prefix=namespace)
        if c.algorithm.options.get("post_training", {}).get("eval_gallery", False):
            from ..evaluation.audio_video_gallery import publish_av_gallery
            try:
                result.update(publish_av_gallery(root, directory, samples, step, metric_namespace=namespace))
            except Exception as exc:
                warnings.warn(f"H3 gallery publication failed ({type(exc).__name__}); canonical eval is saved",
                              RuntimeWarning)
        return result
    result = session.primary("AV evaluation manifest", publish)
    model.train()
    event("evaluation_complete", optimizer_steps=step)
    return result


def _completed_evaluation(c, dataset, step, root):
    if "duration_frames" not in (c.algorithm.options.get("post_training") or {}):
        return _completed_duration_evaluation(c, dataset, step, root)
    return all(_completed_duration_evaluation(duration_config(c, frames), dataset, step, root)
               for frames in duration_frames(c))


def _completed_duration_evaluation(c, dataset, step, root):
    """Recognize a complete boundary publication before resuming past its eval."""
    from ..evaluation import EvaluationManifest, audit_manifest
    from ..evaluation.audio_video_gallery import read_media_manifest
    expected = tuple(row.sample_id for row in dataset.rows)
    base_seed = c.seed if c.evaluation.seed is None else c.evaluation.seed
    for directory in sorted((root / "evaluation" / evaluation_step_directory(c, step)).glob("*")):
        try:
            path = directory / "manifest.json"
            if path.stat().st_size > 4 * 1024 * 1024:
                continue
            manifest = EvaluationManifest.from_json(path.read_text())
            if (manifest.optimizer_step != step or set(manifest.expected_sample_ids) != set(expected)
                    or not (directory / "summary.json").is_file()
                    or not audit_manifest(manifest, media_root=root).complete):
                continue
            samples = {sample.sample_id: sample for sample in manifest.samples}
            if set(samples) != set(expected):
                continue
            if any(samples[row.sample_id].prompt != row.text
                   or samples[row.sample_id].seed != base_seed + index
                   or samples[row.sample_id].framecount != c.sampling.frames
                   or samples[row.sample_id].fps != c.sampling.fps
                   or set(samples[row.sample_id].scores.raw) != set(c.reward.weights)
                   for index, row in enumerate(dataset.rows)):
                continue
            if not (directory / "gallery.html").is_file():
                continue
            read_media_manifest(root, directory / "wandb-media.json")
            return True
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return False


@execution_policy()
def run(c, *, resume=None, resume_rewind=False, evaluate_only=False, backend_api=None):
    event("execution_policy", deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
          cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"])
    api = backend if backend_api is None else backend_api
    validate_resume_request(resume, resume_rewind, evaluate_only)
    api.validate_config(c)
    identity = launch_identity(c.runtime.distributed)
    prompts = identity.local_prompt_count(c.sampling.prompts_per_rollout)
    if c.runtime.distributed == "fsdp" and identity.world_size < 2:
        raise ConfigError("H3 full-shard checkpoint/shadow requires two or more ranks")
    timeout = float(os.environ.get("VISUAL_RL_COLLECTIVE_TIMEOUT_SECONDS", "300"))
    with distributed_session(c.runtime.distributed, device=c.runtime.device, timeout_seconds=timeout) as session:
        session.require_same_digest(c.fingerprint())
        session.agree({"resume": None if resume is None else str(Path(resume).absolute()),
                       "rewind": resume_rewind, "evaluate_only": evaluate_only})
        return _run(session, c, prompts, resume, api, resume_rewind=resume_rewind, evaluate_only=evaluate_only)


def _run(session, c, prompts, resume, api, *, resume_rewind, evaluate_only):
    rank, world = session.identity.rank, session.identity.world_size
    k = c.sampling.samples_per_prompt
    root = Path(c.runtime.output_dir)
    data, validation = session.local("AV hashed datasets", lambda: api.load_datasets(c))
    with ExitStack() as resources:
        resources.enter_context(tf32_policy(c.training.allow_tf32))
        def acquire():
            import fcntl
            if resume is None:
                root.mkdir(parents=True, exist_ok=False)
            elif not root.is_dir():
                raise ConfigError("AV resume requires the original output directory")
            lock = resources.enter_context((root / ".owner.lock").open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if resume:
                _check_run_identity(c, root)
        session.primary("AV output ownership", acquire)
        torch.manual_seed(c.seed)
        random.seed(c.seed)
        np.random.seed(c.seed)
        event("cpu_component_loading_begin", config_sha256=c.fingerprint())
        parts = session.local("AV CPU component loading", lambda: api.load_components(c, session.device))
        event("cpu_component_loading_complete")
        if hasattr(parts.reward, "close"):
            resources.callback(parts.reward.close)
        asset_identity = session.local("AV canonical asset identity", lambda: H3AssetIdentity(parts.manifest))
        session.agree(asset_identity.state_dict())
        if resume:
            def check_assets():
                frozen = json.loads((root / "run.json").read_text()).get("asset_manifest")
                if frozen != asset_identity.state_dict()["assets"]:
                    raise ConfigError("H3 model/reward asset identity differs from original run")
            session.local("AV resume asset identity", check_assets)
        model = parts.adapter.trainable_module
        # Loading on CPU is an explicit backend contract, before FSDP/optimizer.
        if any(p.device.type != "cpu" for p in model.parameters()):
            raise ConfigError("H3 components must load policy on CPU before runtime placement")
        inventory = session.local("AV parameter inventory", lambda: parameter_inventory(model))
        session.agree(inventory)
        trace = BatchTrace(os.environ.get("H3_TRACE_BATCH_SHAPES") == "1")
        trace.install(model)
        resources.callback(trace.close)
        if session.distributed:
            event("full_shard_wrap_begin", world_size=world)
            model = wrap_policy(model, c, session.device)
            event("full_shard_wrap_complete", world_size=world)
        else:
            if c.model.options["gradient_checkpointing"] and hasattr(model, "enable_gradient_checkpointing"):
                model.enable_gradient_checkpointing()
            # Optimizer.load_state_dict places moments on the live parameter
            # device. Place policy first so resumed CUDA moments are not CPU.
            model.to(session.device)
        parts.adapter.set_trainable_module(model)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=c.training.learning_rate,
            betas=c.training.betas, eps=c.training.epsilon, weight_decay=c.training.weight_decay)
        torch.manual_seed(c.seed + rank)
        random.seed(c.seed + rank)
        np.random.seed((c.seed + rank) % 2**32)
        generator = torch.Generator().manual_seed((c.seed + 10_007 * rank) % (2**63 - 1))
        stream = PromptStream(data, seed=c.seed, shuffle=c.data.shuffle)
        nft = session.local("AV NFT old shadow", lambda: AVNFTState(model)) if c.algorithm.name == "diffusionnft" else None
        reference = None
        if c.algorithm.options.get("loss", {}).get("reference_mse_beta", 0) > 0:
            from .minimax_h3_reference import ReferenceState
            reference = session.local("AV immutable pretrained reference", lambda: ReferenceState(model))
        stateful = {"h3_asset_identity": asset_identity, "prompts": stream,
                    **(reference.stateful() if reference is not None else {}),
                    **(nft.stateful() if nft is not None else {})}
        checkpoint = dict(model=model, optimizer=optimizer, config_sha256=c.fingerprint(), stateful=stateful,
                          generators={"av_sampling": generator})
        load = load_fsdp_checkpoint if session.distributed else load_checkpoint
        save = save_fsdp_checkpoint if session.distributed else save_checkpoint
        progress = load(resume, **checkpoint) if resume else Progress()
        if resume and session.distributed and os.environ.get("H3_RESTORE_AUDIT") == "1":
            from .minimax_h3_checkpoint_audit import audit_restored_checkpoint
            audit_restored_checkpoint(session, resume, model, optimizer, stateful,
                                      {"av_sampling": generator}, progress, root)
        event("optimizer_ready", optimizer_steps=progress.optimizer_steps, resumed=resume is not None)
        if resume:
            def validate_boundary():
                expected = api.updates_per_rollout(c)
                if progress.optimizer_steps % expected or progress.optimizer_steps > c.training.max_updates:
                    raise ConfigError("H3 resume requires an in-budget complete rollout boundary")
                if nft is not None and nft.successful_updates != progress.optimizer_steps:
                    raise ConfigError("NFT refresh counter and global optimizer step differ")
            session.local("AV checkpoint boundary", validate_boundary)
            if evaluate_only:
                if validation is None:
                    raise ConfigError("checkpoint evaluation requires a validation manifest")
                if c.algorithm.options.get("post_training") is not None:
                    session.primary("AV eval-only export", lambda: resources.enter_context(open_metrics(c, root)))
                return evaluate(session, c, parts, validation, progress.optimizer_steps, root)
            session.primary("AV resume history", lambda: prepare_resume_history(c, root, progress, receipt=resume, rewind=resume_rewind))
            session.local("AV cost lineage", lambda: _restore_cost(c, root, progress))
            if progress.optimizer_steps == c.training.max_updates and c.algorithm.options.get("post_training") is None:
                return
        else:
            def publish_run():
                _atomic_json(root / "resolved-config.json", c.to_dict())
                _atomic_json(root / "run.json", {"kind": "minimax_h3_native_development", "config_sha256": c.fingerprint(),
                    "parameters": inventory, "asset_manifest": asset_identity.state_dict()["assets"], "device": str(session.device),
                    "data_sha256": data.sha256, "dp_world_size": world, "local_prompts": prompts, "samples_per_prompt": k,
                    "actual_output_dir": str(root.resolve()), "strategy": c.runtime.distributed,
                    "ordering": "global-prompt-stream/prompt-half-K-u2-v1", "pretrained_reference": reference is not None})
            session.primary("AV run identity", publish_run)
        clock = CostClock(progress, gpu_count=world if session.device.type == "cuda" else 0)
        log = session.primary("AV metrics open", lambda: resources.enter_context(open_metrics(c, root)))
        if (resume and validation is not None and c.algorithm.options.get("post_training") is not None
                and c.evaluation.due(progress.optimizer_steps, c.training.max_updates,
                                     completed_rollouts=progress.completed_rollouts)):
            complete = session.primary("AV restored boundary evaluation", lambda: _completed_evaluation(
                c, validation, progress.optimizer_steps, root))
            flag = torch.tensor(int(bool(complete)), dtype=torch.int32, device=session.device)
            if session.distributed:
                dist.broadcast(flag, src=0)
            if not bool(flag.item()):
                evaluate(session, c, parts, validation, progress.optimizer_steps, root)
        sigmas = sigma_schedules(c.sampling.steps, video_shift=c.sampling.options["video_shift"], audio_shift=c.sampling.options["audio_shift"])
        if validation is not None and resume is None:
            evaluate(session, c, parts, validation, 0, root)
        while progress.optimizer_steps < c.training.max_updates:
            model.train()
            if session.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(session.device)
            with clock.training_phase():
                progress.attempts += 1
                workload = rollout_config(c, progress.completed_rollouts)
                session.agree({"rollout_frames": workload.sampling.frames,
                               "completed_rollouts": progress.completed_rollouts})
                phase = time.monotonic()
                def prepare():
                    rows = stream.take(c.sampling.prompts_per_rollout, rank=rank, world_size=world)
                    predict = prediction_callback(parts, rows, k, session)
                    noise = _noise(parts, workload, prompts * k, generator)
                    return rows, predict, noise
                rows, predict, noise = session.local("AV rollout conditioning", prepare)
                event("conditioning_complete", rollout_attempt=progress.attempts, optimizer_steps=progress.optimizer_steps)
                if nft is not None:
                    with nft.use_old(session):
                        cache = collect(parts, c, predict, noise, sigmas, session, generator, rollout_index=progress.attempts - 1)
                else:
                    cache = collect(parts, c, predict, noise, sigmas, session, generator, rollout_index=progress.attempts - 1)
                rollout_seconds = time.monotonic() - phase
                event("rollout_complete", optimizer_steps=progress.optimizer_steps, seconds=rollout_seconds)
                phase = time.monotonic()
                _, scores = session.local("AV decode and five rewards", lambda: score_endpoints(parts, cache.endpoints, rows, k, session))
                population = global_scores(session, scores)
                groups = torch.arange(prompts).repeat_interleave(k)
                objective = _prepare_objective(c, parts, cache, predict, scores, population, groups, session, generator, nft,
                                               reference=reference)
                if os.environ.get("H3_INPUT_AUDIT") == "1":
                    from .minimax_h3_input_audit import audit_inputs
                    session.local("H3 pre-update input audit", lambda: audit_inputs(
                        session, objective, rows, generator, progress, root))
                reward_seconds = time.monotonic() - phase
                event("five_rewards_complete", optimizer_steps=progress.optimizer_steps, seconds=reward_seconds)
                if nft is not None:
                    train_video, train_audio = objective.video_sigmas, objective.audio_sigmas
                    # Full fresh pool is permuted, never subsampled in H3 U1.
                    source_positions = tuple(range(len(cache.video_sigmas) - 1))
                else:
                    source_positions = cache.selected
                    train_video = cache.video_sigmas[list(source_positions)]
                    train_audio = cache.audio_sigmas[list(source_positions)]
                sigma_metrics = {
                    "av/training_video_sigma_min": float(train_video.min()),
                    "av/training_video_sigma_max": float(train_video.max()),
                    "av/training_audio_sigma_min": float(train_audio.min()),
                    "av/training_audio_sigma_max": float(train_audio.max()),
                    "av/training_source_position_min": min(source_positions),
                    "av/training_source_position_max": max(source_positions),
                    "av/nft_fresh_time_floor_quantized": int(nft is not None),
                }
                if cache.video_selected is not None:
                    for name, schedule in (("video", cache.video_sigmas), ("audio", cache.audio_sigmas)):
                        selected = getattr(cache, name + "_selected")
                        sigma_metrics[f"av/training_{name}_sigma_min"] = float(schedule[list(selected)].min())
                        sigma_metrics[f"av/training_{name}_sigma_max"] = float(schedule[list(selected)].max())
                        sigma_metrics[f"av/training_{name}_positions"] = len(selected)
                        for position in range(4):
                            sigma_metrics[f"av/training_{name}_position_{position}_selected"] = int(position in selected)
                _active_policy(parts, session)
                phase = time.monotonic()
                metrics = run_av_updates(model, optimizer, objective, groups,
                    algorithm={"diffusionart": "art", "diffusionnft": "nft", "flow_grpo_fast": "flow_grpo_fast"}[c.algorithm.name],
                    microbatch=c.training.microbatch, max_grad_norm=c.training.grad_clip, state=nft)
                update_seconds = time.monotonic() - phase
                actual = int(metrics["train/optimizer_steps_per_rollout"])
                progress.optimizer_steps += actual
                if actual:
                    progress.completed_rollouts += 1
                event("updates_complete", optimizer_steps=progress.optimizer_steps,
                      completed_rollouts=progress.completed_rollouts, successful_window_updates=actual,
                      seconds=update_seconds)
                trace.close()
                metrics.update(_metrics(population))
                metrics.update(sigma_metrics)
                metrics.update({"av/rollout_frames": workload.sampling.frames,
                                "av/rollout_duration_seconds": workload.sampling.frames / workload.sampling.fps})
                if reference is not None:
                    metrics["train/reference_mse"] = .5 * (metrics["av/video_reference_mse"] + metrics["av/audio_reference_mse"])
                    metrics["train/reference_mse_weighted"] = c.algorithm.options["loss"]["reference_mse_beta"] * metrics["train/reference_mse"]
                metrics.update({"time/rollout_seconds": rollout_seconds, "time/decode_reward_prepare_seconds": reward_seconds,
                    "time/update_seconds": update_seconds, "av/transitions": len(sigmas[0]) - 1,
                    "av/training_positions": (len(objective.video_sigmas) / len(cache.endpoints) if nft is not None else len(cache.selected))})
                metrics.update(_cost(session, clock))
                if "duration_frames" in (c.algorithm.options.get("post_training") or {}):
                    metrics.update({f"train/frames{workload.sampling.frames}/{key.removeprefix('train/')}": value
                                    for key, value in tuple(metrics.items())
                                    if key.startswith(("train/reward/", "train/diagnostics/", "train/reference_mse"))
                                    or key in ("train/loss", "train/grad_norm/before_clip")})
                def publish_metrics():
                    log.write(json.dumps(metrics, sort_keys=True, allow_nan=False) + "\n")
                    log.flush()
                session.primary("AV metrics publication", publish_metrics)
            step = progress.optimizer_steps
            checkpoint_due = actual and c.runtime.checkpoint_due(step, c.training.max_updates)
            eval_due = actual and validation is not None and c.evaluation.due(
                step, c.training.max_updates, completed_rollouts=progress.completed_rollouts)
            post = c.algorithm.options.get("post_training") is not None
            if eval_due and not post:
                evaluate(session, c, parts, validation, step, root)
            if checkpoint_due:
                event("checkpoint_begin", optimizer_steps=step)
                clock.tick()
                from .minimax_h3_checkpoint_admission import checkpoint_admission
                with checkpoint_admission(session, root, step):
                    save(root / "checkpoints", progress=progress, keep_last=c.runtime.checkpoint_keep, **checkpoint)
                event("checkpoint_complete", optimizer_steps=step)
            if eval_due and post:
                evaluate(session, c, parts, validation, step, root)
            _publish_cost(session, c, root, clock)
        final = _publish_cost(session, c, root, clock)
        session.primary("AV completion", lambda: _atomic_json(root / "status.json", {"status": "completed",
            "kind": "minimax_h3_native_development", "config_sha256": c.fingerprint(), **final}))
        return final
