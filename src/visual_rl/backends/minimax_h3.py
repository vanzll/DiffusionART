"""Opt-in MiniMax-H3 T2AV capability, configuration and component boundary."""

import math
import re
from dataclasses import dataclass
from pathlib import Path

from ..config import ConfigError

MODEL_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
DIFFUSERS_REVISION = "73faca368c0da77645ea58836aa8780d2d8c5353"
COMPONENTS = {"video_align", "hpsv3", "audiobox", "clap", "desync"}
SHARED_WEIGHTS = {"video_align": 1., "hpsv3": 1.5, "audiobox": .5, "clap": 1., "desync": 1.}


def updates_per_rollout(c):
    return 1 if c.algorithm.name == "diffusionnft" else 2


def _finite(value, label, *, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (float, int))
            or not math.isfinite(value) or value < 0 or (positive and value == 0)):
        raise ConfigError(f"{label} must be finite and {'positive' if positive else 'nonnegative'}")


def validate_config(c):
    """Network-free validation; no optional model or reward packages imported."""
    if (c.task != "t2av" or c.model.name != "minimax_h3"
            or c.algorithm.name not in {"diffusionart", "diffusionnft", "flow_grpo_fast"}
            or c.reward.name != "audio_video" or c.runtime.distributed not in {"none", "fsdp"}
            or c.training.precision != "bf16"):
        raise ConfigError("H3 supports T2AV ART/NFT/Fast, BF16, none/FSDP and five AV rewards")
    if not c.data or not c.model.checkpoint or not re.fullmatch(r"[0-9a-f]{40}", c.model.revision or ""):
        raise ConfigError("H3 requires a fixed model revision and hashed prompt manifest")
    if c.model.revision != MODEL_REVISION:
        raise ConfigError("this H3 ABI is validated only against the pinned model revision")
    mo = c.model.options
    required_model_options = {"finetune", "gradient_checkpointing", "local_files_only", "ffmpeg", "ffprobe", "conditioning_device"}
    if set(mo) not in (required_model_options, required_model_options | {"context_parallel_size"}):
        raise ConfigError("H3 model options require finetune, checkpointing, offline flag, ffmpeg/ffprobe")
    degree = mo.get("context_parallel_size", 1)
    if type(degree) is not int or degree not in (1, 2):
        raise ConfigError("H3 context_parallel_size must be 1 or 2")
    if degree == 2 and (c.runtime.distributed != "fsdp" or c.algorithm.name != "diffusionart"
                        or mo["finetune"]["mode"] != "full"):
        raise ConfigError("H3 CP2 requires full ART and FSDP")
    if degree == 2 and c.evaluation.enabled:
        raise ConfigError("H3 CP2 supports training only; evaluation ownership is not implemented")
    from ..models.finetune import FineTuneConfig
    try:
        FineTuneConfig(**mo["finetune"])
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"invalid H3 fine-tuning configuration: {exc}") from exc
    if any(type(mo[k]) is not bool for k in ["gradient_checkpointing", "local_files_only"]):
        raise ConfigError("H3 checkpointing/offline options must be boolean")
    if not mo["local_files_only"]:
        raise ConfigError("prepare H3 assets before workers; training component loading is offline")
    if mo["conditioning_device"] != "cpu":
        raise ConfigError("first H3 integration keeps its 66.7GB conditioner on CPU")
    post = c.algorithm.options.get("post_training")
    from .minimax_h3_posttraining import validate_post_training
    if post is not None:
        validate_post_training(post, c.sampling.steps)
        if "duration_frames" in post and c.sampling.frames != post["duration_frames"][0]:
            raise ConfigError("H3 sampling.frames must match the first duration bucket")
        if (c.algorithm.name not in {"diffusionart", "flow_grpo_fast"} or c.training.timestep_selection != "configured"
                or c.sampling.samples_per_prompt != 8):
            raise ConfigError("H3 post-training requires ART/routed Fast, K8, explicit dual selection and six reward group std")
    weights = {**SHARED_WEIGHTS, "imagebind": .5} if post is not None else SHARED_WEIGHTS
    if set(c.reward.weights) != set(weights) or dict(c.reward.weights) != weights:
        raise ConfigError("H3 shared reward weights must explicitly be 1/1.5/.5/1/1; DeSync counted once")
    if set(c.reward.options) != {"manifest"} or not isinstance(c.reward.options["manifest"], str):
        raise ConfigError("H3 rewards require an explicit pinned scorer manifest")
    if c.reward.normalization != "raw":
        raise ConfigError("AV scorer transforms are explicit; no extra top-level normalization")
    if c.sampling.samples_per_prompt < 2 or c.sampling.samples_per_prompt % 2:
        raise ConfigError("H3 recipes use even K>=2 for group statistics and prompt-balanced windows")
    if c.training.max_updates % updates_per_rollout(c):
        raise ConfigError("H3 training budget must end at a completed rollout boundary")
    if c.sampling.guidance_scale != 1:
        raise ConfigError("this H3 checkpoint is CFG-distilled; true CFG must equal one")
    if c.sampling.fps != 24 or c.sampling.frames % 17 != 5:
        raise ConfigError("fixed H3 VAE requires 24 FPS and frames=17n+5")
    if any(v % 16 for v in [c.sampling.height, c.sampling.width]):
        raise ConfigError("H3 spatial dimensions must divide by 16")
    so = c.sampling.options
    allowed = {"video_shift", "audio_shift", "noise_level", "window_size", "window_range", "window_seed"}
    if set(so) != allowed or so["video_shift"] != 12 or so["audio_shift"] != 3:
        raise ConfigError("H3 requires explicit dual schedules (video12/audio3) and CPS window options")
    if so["window_size"] != 3 or so["window_range"] != [0, 8] or so["window_seed"] != 42:
        raise ConfigError("H3 Fast reference window is length3, [0,8), seed42")
    if so["noise_level"] != .8:
        raise ConfigError("H3 Fast reference CPS noise level is .8")
    if c.sampling.steps < 3 or (c.algorithm.name == "flow_grpo_fast" and c.sampling.steps < 4):
        raise ConfigError("Fast requires at least three native transitions")
    options = c.algorithm.options
    if c.algorithm.name == "diffusionart":
        allowed = {"covariance", "loss", "exclude_final_step"} | ({"post_training"} if post is not None else set())
        if set(options) != allowed or options["exclude_final_step"] is not True:
            raise ConfigError("H3 ART requires covariance/loss and explicit final-transition exclusion")
        beta = options["loss"].get("reference_mse_beta", 0.)
        _finite(beta, "reference_mse_beta")
        if any(k.startswith("reference") and k != "reference_mse_beta" for k in options["loss"]):
            raise ConfigError("H3 reference regularization accepts only reference_mse_beta")
        import inspect

        from ..algorithms import art_loss, art_scalar_covariance
        for kind, kernel, excluded in [("loss", art_loss, {"state", "ode_velocity_coefficient", "current_ode_mean", "old_ode_mean", "restore_anchor_mask", "reference_prediction"}),
                                        ("covariance", art_scalar_covariance, {"latents", "rewards", "groups"})]:
            allowed_keys = {k for k, v in inspect.signature(kernel).parameters.items()
                            if v.kind == inspect.Parameter.KEYWORD_ONLY} - excluded
            if not isinstance(options[kind], dict) or set(options[kind]) - allowed_keys:
                raise ConfigError(f"unsupported H3 ART {kind} option")
    elif c.algorithm.name == "diffusionnft":
        if set(options) != {"old_policy", "loss", "advantage_epsilon", "fresh_time"}:
            raise ConfigError("H3 NFT requires explicit slow EMA, mirror loss, advantage epsilon and fresh time")
        if options["old_policy"] != {"mode": "slow-ema", "rate": .00125, "maximum": .6}:
            raise ConfigError("H3 NFT uses the declared repository slow EMA, not upstream delayed EMA")
        if options["loss"] != {"nft_beta": 1., "adv_clip_max": 5., "adv_mode": "all", "eps": 1e-5}:
            raise ConfigError("H3 NFT requires beta1 normalized-x0 mirror loss and explicit residual floor")
        if options["advantage_epsilon"] != 1e-4 or options["fresh_time"] != "shared_video_grid":
            raise ConfigError("H3 NFT uses routed group-center/global-std+1e-4 and shared video-grid fresh sigma")
    else:
        if post is not None:
            if (set(options) != {"loss", "post_training", "ratio_mode"}
                    or options["ratio_mode"] != "per_modality"
                    or set(options["loss"]) != {"ratio_clip", "advantage_clip", "reference_mse_beta"}
                    or options["loss"]["ratio_clip"] != 1e-4 or options["loss"]["advantage_clip"] != 5.):
                raise ConfigError("H3 routed Fast requires explicit per-modality ratio/clip and reference MSE")
            _finite(options["loss"]["reference_mse_beta"], "reference_mse_beta")
        elif options != {"loss": {"ratio_clip": 1e-4, "advantage_clip": 5.}, "advantage_epsilon": 1e-4}:
            raise ConfigError("H3 Fast requires its CPS PPO clip1e-4/advclip5 and explicit advantage epsilon")
    if c.evaluation.enabled and (not c.evaluation.manifest or not c.evaluation.manifest_sha256):
        raise ConfigError("H3 evaluation requires an explicitly hashed manifest")
    for n in (*c.evaluation.at_updates, *c.runtime.checkpoint_at_updates):
        if n % updates_per_rollout(c):
            raise ConfigError("H3 evaluation/checkpoint events must be rollout boundaries")
    for period in [c.evaluation.every_updates, c.runtime.checkpoint_every]:
        if period % updates_per_rollout(c):
            raise ConfigError("H3 periodic artifacts must be rollout boundaries")
    if c.evaluation.every_rollouts and post is None:
        raise ConfigError("H3 rollout eval cadence requires the explicit post-training recipe")
    if c.evaluation.enabled:
        if c.evaluation.fps != 24 or c.evaluation.reward_fps != 24 or c.evaluation.shift is not None:
            raise ConfigError("H3 evaluation preserves 24fps and explicit paired native shifts")
        if any(v is not None and v % 16 for v in [c.evaluation.height, c.evaluation.width]):
            raise ConfigError("H3 evaluation geometry must divide by16")


def load_datasets(c):
    from ..data import PromptDataset
    data = PromptDataset.load(c.data.manifest, expected_sha256=c.data.sha256, task="t2av")
    validation = (PromptDataset.load(c.evaluation.manifest,
        expected_sha256=c.evaluation.manifest_sha256, task="t2av") if c.evaluation.enabled else None)
    for dataset in [data, validation]:
        if dataset is None:
            continue
        for row in dataset.rows:
            text = (row.metadata or {}).get("reward_text")
            if (not isinstance(text, dict) or set(text) != {"video", "audio"}
                    or any(not isinstance(v, str) or not v.strip() for v in text.values())):
                raise ConfigError("T2AV prompt metadata requires explicit reward_text.video and reward_text.audio")
    if c.algorithm.options.get("post_training") is not None and validation is not None:
        if len(validation.rows) != 32:
            raise ConfigError("the H3 post-training evaluation protocol requires exactly 32 fixed prompts")
        train_ids = {row.sample_id for row in data.rows}
        train_texts = {row.text.strip() for row in data.rows}
        if any(row.sample_id in train_ids or row.text.strip() in train_texts for row in validation.rows):
            raise ConfigError("H3 post-training evaluation prompts must be held out from training")
    return data, validation


def reward_text(rows, repeats=1):
    video, audio = [], []
    for row in rows:
        text = row.metadata["reward_text"]
        video.extend([text["video"]] * repeats)
        audio.extend([text["audio"]] * repeats)
    return video, audio


def load_reward(c, device):
    import hashlib
    import json
    import os

    from ..models.prepare_minimax_h3 import validate_reward_provenance
    from ..rewards.audio_video import AudioVideoReward
    path = Path(c.reward.options["manifest"])
    if path.stat().st_size > 1024 * 1024:
        raise ConfigError("scorer asset manifest exceeds 1 MiB")
    manifest_bytes = path.read_bytes()
    record = json.loads(manifest_bytes)
    if set(record) != {"schema_version", "paths", "sha256", "provenance"} or record["schema_version"] != 1:
        raise ConfigError("invalid H3 five-scorer asset manifest")
    if (not isinstance(record["paths"], dict)
            or any(not isinstance(v, str) or not v.strip() for v in record["paths"].values())
            or not isinstance(record["provenance"], dict)
            or set(record["provenance"]) != set(record["paths"])):
        raise ConfigError("every scorer asset requires explicit source provenance")
    for name, source in record["provenance"].items():
        try:
            validate_reward_provenance(source, name=name)
        except ValueError as error:
            raise ConfigError("scorer provenance needs repo, fixed revision and license") from error
    paths = {key: os.path.abspath(path.parent / value) for key, value in record["paths"].items()}
    expected_six = "imagebind" in c.reward.weights
    if expected_six != ("imagebind" in paths):
        raise ConfigError("H3 reward asset manifest must match the configured five/six scorer set")
    duration_options = ({"full_duration": True}
                        if c.algorithm.options.get("post_training", {}).get("full_duration_rewards", False) else {})
    reward = AudioVideoReward.from_paths(paths, device=device, expected_sha256=record["sha256"], **duration_options)
    reward.asset_identity = {"manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                             "paths": paths, "sha256": record["sha256"],
                             "provenance": record["provenance"]}
    return reward


@dataclass
class Components:
    adapter: object
    conditioner: object
    reward: object
    manifest: dict


def load_components(c, device):
    import torch

    from ..models.finetune import FineTuneConfig
    from ..models.minimax_h3_assets import load_minimax_h3
    # Sharding must precede large model CUDA placement. The native loader owns
    # frozen encoder/decoder residency; runtime moves only the component in use.
    adapter, conditioner, manifest = load_minimax_h3(c.model.checkpoint, revision=c.model.revision,
        device="cpu",
        dtype=torch.bfloat16, local_files_only=True,
        finetune=FineTuneConfig(**c.model.options["finetune"]))
    reward = load_reward(c, device)
    manifest = {**manifest, "reward": reward.asset_identity}
    return Components(adapter, conditioner, reward, manifest)


def run(c, *, resume=None, resume_rewind=False, evaluate_only=False):
    from .minimax_h3_runtime import run as run_native
    validate_config(c)
    return run_native(c, resume=resume, resume_rewind=resume_rewind, evaluate_only=evaluate_only)
