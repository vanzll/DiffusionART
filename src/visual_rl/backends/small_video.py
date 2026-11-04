"""Optional native small T2V recipes; old video/image entrypoints are untouched."""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch

from ..config import ConfigError
from ..data import PromptDataset
from ..models.finetune import FineTuneConfig, prepare_policy
from ..models.ltx_video import LTX_VIDEO_LORA_TARGETS, ltx_video_latent_shape
from ..models.ltx_video_sampling import LTX_VIDEO_NATIVE_SIGMAS, LTX_VIDEO_WEIGHT_REVISION
from ..models.wan_t2v import WAN_T2V_NATIVE_LORA_TARGETS, WAN_T2V_NATIVE_REVISION, WanT2VConfig
from ..models.wan_t2v_schedule import native_wan_schedule
from ..training.small_video_replay import NATIVE_AFFINE_ART_CONTRACT

MODELS = {"wan_t2v_1_3b", "ltx_video_2b_distilled"}


def profile(name):
    if name == "wan_t2v_1_3b":
        return dict(
            height=480,
            width=832,
            frames=81,
            fps=16.0,
            steps=50,
            guidance_scale=6.0,
            solver="unipc_order2_bh2_predict_x0",
            shift=8.0,
            revision=WAN_T2V_NATIVE_REVISION,
            targets=WAN_T2V_NATIVE_LORA_TARGETS,
        )
    if name == "ltx_video_2b_distilled":
        return dict(
            height=704,
            width=1216,
            frames=153,
            fps=30.0,
            steps=8,
            guidance_scale=1.0,
            solver="from_checkpoint_rf_stochastic",
            shift=None,
            revision=LTX_VIDEO_WEIGHT_REVISION,
            targets=LTX_VIDEO_LORA_TARGETS,
        )
    raise ConfigError("unknown small T2V model")


def validate_config(c):
    p = profile(c.model.name)
    if (
        c.task != "t2v"
        or c.algorithm.name != "diffusionart"
        or c.reward.name != "video_reward"
        or c.training.precision != "bf16"
        or c.runtime.distributed not in ("none", "ddp", "fsdp")
    ):
        raise ConfigError("small native T2V supports BF16 ART with raw VideoReward and DP only")
    for key in ("height", "width", "frames", "fps", "steps", "guidance_scale"):
        if getattr(c.sampling, key) != p[key]:
            raise ConfigError(f"native small-video profile freezes sampling.{key}={p[key]}")
    if (
        c.sampling.prompts_per_rollout != 16
        or c.sampling.samples_per_prompt != 8
        or c.training.max_updates != 10
        or c.seed != 42
    ):
        raise ConfigError(
            "acceptance contract requires seed42, P16/K8, ten cumulative actual steps"
        )
    if c.sampling.microbatch not in (1, 2, 4, 8) or c.training.microbatch not in (1, 2, 4, 8):
        raise ConfigError(
            "physical microbatches must be explicit 1/2/4/8; P/K/windows never shrink"
        )
    required = {
        "native_root",
        "asset_root",
        "asset_receipt",
        "finetune",
        "gradient_checkpointing",
        "allow_tf32",
        "ffmpeg",
        "ffprobe",
        "replay_contract",
    }
    if set(c.model.options) != required:
        raise ConfigError(f"small video model options must be explicit: {sorted(required)}")
    if c.model.revision != p["revision"] or not c.model.checkpoint:
        raise ConfigError("require the exact native model revision and local snapshot")
    mode = FineTuneConfig(**c.model.options["finetune"])
    if mode.mode == "lora" and (
        mode.rank != 32 or mode.alpha != 32 or tuple(mode.target_modules) != tuple(p["targets"])
    ):
        raise ConfigError(
            "native attention LoRA freezes rank32/alpha32/dropout0 and verified targets"
        )
    expected_lr = 1e-4 if mode.mode == "lora" else 1e-5
    if (
        c.training.learning_rate != expected_lr
        or c.training.grad_clip != 1.0
        or c.training.optimizer != "adamw"
        or c.training.betas != (0.9, 0.999)
        or c.training.epsilon != 1e-8
        or c.training.weight_decay != 0.0001
    ):
        raise ConfigError("freeze canonical AdamW values, clip1 and mode-specific learning rate")
    if c.reward.normalization != "raw" or c.reward.weights != {"VQ": 0.5, "MQ": 0.5, "TA": 0.0}:
        raise ConfigError("VideoReward requires raw VQ/MQ equal weights, TA monitoring only")
    if (
        c.reward.revision != "4f26600130683e6f1de9f5d463887f28e8ef995c"
        or set(c.reward.options) != {"python", "native_root", "backbone"}
        or not c.reward.checkpoint
    ):
        raise ConfigError(
            "explicit immutable VideoReward checkpoint and canonical native source required"
        )
    if (
        not c.data
        or not c.data.sha256
        or not c.evaluation.manifest_sha256
        or not c.evaluation.manifest
        or c.evaluation.reward_fps != 4.0
        or c.evaluation.fps != p["fps"]
    ):
        raise ConfigError(
            "require hashed disjoint prompt manifests, native video FPS and canonical reward4fps"
        )
    if (
        c.evaluation.every_updates != 0
        or c.evaluation.every_rollouts != 0
        or c.evaluation.at_updates != (10,)
        or c.runtime.checkpoint_keep != 2
        or c.runtime.checkpoint_at_updates != (4, 10)
        or c.runtime.checkpoint_every != 0
    ):
        raise ConfigError(
            "ten-step acceptance uses paired base/post8 and checkpoint4/final10, keeping two"
        )
    if c.training.timestep_selection not in ("configured", "high_noise"):
        raise ConfigError(
            "Original uses configured all-valid steps; Flash uses high_noise first4/rand2"
        )
    expected_sampling_options = {"solver": p["solver"], "shift": p["shift"]}
    if c.sampling.options != expected_sampling_options:
        raise ConfigError("native solver and shift/stochastic profile cannot be substituted")
    options = c.algorithm.options
    if (
        set(options) != {"restore_zero_signal", "covariance", "loss"}
        or options["restore_zero_signal"] is not True
    ):
        raise ConfigError("freeze canonical video restoring anchors, covariance and loss options")
    if options["covariance"] != {
        "reduction": "unbiased",
        "direction_mode": "historical",
        "scale_mode": "unit_normalized_covariance",
        "guidance_gain": 1.0,
        "zero_signal_mode": "exact",
    }:
        raise ConfigError(
            "covariance must use complete-K native endpoint groups and frozen normalized target"
        )
    if options["loss"] != {
        "loss_space": "velocity_linear_sigma",
        "direction_normalization": "rms",
        "mask_space": "velocity",
        "velocity_mask_metric": "scheduler_reference_metric",
        "corrective_mask_enabled": True,
        "q2_reference_sde_noise_level": 0.8,
        "q2_reference_kl_threshold": 1e-5,
    }:
        raise ConfigError(
            "native replay preserves canonical ART weights/restoration/Q2-reference mask; KL0"
        )
    if c.model.options["replay_contract"] not in (
        "pending_scientific_decision",
        NATIVE_AFFINE_ART_CONTRACT,
    ):
        raise ConfigError("unknown native replay science contract")
    if (
        c.model.options["gradient_checkpointing"] is not True
        or c.model.options["allow_tf32"] is not False
        or c.training.allow_tf32 not in (None, False)
    ):
        raise ConfigError("gradient checkpointing and disabled TF32 are explicit")


def require_training_contract(c):
    validate_config(c)
    if c.model.options["replay_contract"] != NATIVE_AFFINE_ART_CONTRACT:
        raise ConfigError(
            "native ART replay decision pending: conditional affine normalized-mean regression "
            "and canonical Q2 REFERENCE mask require a scientific decision before training"
        )


def resolved_protocol(c):
    validate_config(c)
    p = profile(c.model.name)
    sigmas = (
        native_wan_schedule().sigmas.tolist()
        if c.model.name == "wan_t2v_1_3b"
        else list(LTX_VIDEO_NATIVE_SIGMAS)
    )
    eligible = list(range(len(sigmas) - 1))
    return {
        "model": c.model.name,
        "model_revision": c.model.revision,
        "shape": {key: p[key] for key in ("height", "width", "frames", "fps")},
        "solver": p["solver"],
        "sigmas": sigmas,
        "cfg": p["guidance_scale"],
        "eligible_positions": eligible,
        "terminal_policy": "train_last_nonzero_input_sigma; exclude_sigma0_endpoint",
        "training_selection": "first4_without_replacement2_shared_private_rng"
        if c.training.timestep_selection == "high_noise"
        else "all_eligible",
        "flash_pool_sigmas": sigmas[:4],
        "replay_contract": c.model.options["replay_contract"],
        "cache_branch": "guided_CFG6_native_UniPC_frozen_history"
        if c.model.name == "wan_t2v_1_3b"
        else "CFG1_STG0_native_stochastic_mean_actual_draws",
        "prompts_per_rollout": 16,
        "samples_per_prompt": 8,
        "window_samples": 64,
        "updates_per_rollout": 2,
        "cumulative_actual_optimizer_step_limit": 10,
        "paired_base_post_count": 8,
        "seed": 42,
        "cp": 1,
        "target_parallel_mesh": {
            "nodes": 1,
            "gpus_per_node": 8,
            "dp": 8,
            "cp": 1,
            "strategy": c.runtime.distributed,
        },
        "physical_rollout_microbatch": c.sampling.microbatch,
        "physical_update_microbatch": c.training.microbatch,
        "optimizer": {
            "name": c.training.optimizer,
            "learning_rate": c.training.learning_rate,
            "betas": list(c.training.betas),
            "epsilon": c.training.epsilon,
            "weight_decay": c.training.weight_decay,
            "grad_clip": c.training.grad_clip,
        },
        "finetune": c.model.options["finetune"],
        "reward_weights": c.reward.weights,
        "reward_normalization": "independent_complete_K8_population_std_plus1e-4_then_equal_mean_once",
        "reward_revision": c.reward.revision,
        "data_sha256": c.data.sha256,
        "scorer_placement": "fresh_canonical_reward_env_process_same_rank_GPU_after_decode_before_backward",
        "paired_eval_sha256": c.evaluation.manifest_sha256,
        "initial_seed_rule": "42+rank+1000003*zero_based_rollout; dedicated_device_generator",
        "transition_rng_rule": "global_torch_per_rank_seed42+rank_checkpointed; LTX_native_packed_draws",
        "eval_seed_rule": "42+fixed_manifest_index; isolated_initial_and_global_transition_decode_RNG",
        "mask_metric": "canonical_Q2_reference_not_native_transition_KL",
        "EMA": False,
        "KL": 0,
    }


def load_datasets(c):
    import unicodedata

    train = PromptDataset.load(c.data.manifest, expected_sha256=c.data.sha256, task="t2v")
    evaluation = PromptDataset.load(
        c.evaluation.manifest, expected_sha256=c.evaluation.manifest_sha256, task="t2v"
    )

    def norm(value):
        return " ".join(unicodedata.normalize("NFKC", value).casefold().split())

    if len(evaluation.rows) != 8 or {norm(r.text) for r in train.rows} & {
        norm(r.text) for r in evaluation.rows
    }:
        raise ConfigError(
            "acceptance requires eight fixed held-out prompts with no normalized train overlap"
        )
    return train, evaluation


def latent_shape(c, rows):
    if c.model.name == "wan_t2v_1_3b":
        return (
            rows,
            *WanT2VConfig().latent_shape(
                height=c.sampling.height, width=c.sampling.width, frames=c.sampling.frames
            ),
        )
    return ltx_video_latent_shape(rows, c.sampling.height, c.sampling.width, c.sampling.frames)


def run(c, *, resume=None, resume_rewind=False, evaluate_only=False):
    require_training_contract(
        c
    )  # Before initializing CUDA, loading weights or acquiring any runtime resources.
    if resume_rewind:
        raise ConfigError(
            "rewinding actual optimizer history would violate the cumulative ten-step budget"
        )
    from .small_video_runner import run_native

    return run_native(c, resume=resume, evaluate_only=evaluate_only)


@dataclass
class Components:
    adapter: object
    text_encoder: object
    tokenizer: object
    scorer: object
    scheduler_factory: object
    block_types: tuple
    native_code_revision: str
    asset_receipt: dict
    data_revision: str = "9d947fb510d67a8e5334ec1d8c7d8927f7658c08"


def _verified_assets(c):
    from ..small_video_assets import load_manifest
    from ..small_video_assets import verify_bound_receipt as verify_receipt

    kind = "wan" if c.model.name == "wan_t2v_1_3b" else "ltx"
    receipt = verify_receipt(
        load_manifest(),
        c.model.options["asset_root"],
        c.model.options["asset_receipt"],
        profile=kind,
    )
    assets = {key: Path(value).resolve() for key, value in receipt["assets"].items()}
    expected = assets[kind]
    if kind == "ltx":
        from ..models.ltx_video_sampling import LTX_VIDEO_CHECKPOINT

        expected = expected / LTX_VIDEO_CHECKPOINT
    sources = {item["id"]: Path(item["path"]).resolve() for item in receipt["sources"]}
    if (
        Path(c.model.checkpoint).resolve() != expected
        or Path(c.reward.checkpoint).resolve() != assets["reward"]
        or Path(c.reward.options["backbone"]).resolve() != assets["reward_backbone"]
        or Path(c.model.options["native_root"]).resolve() != sources[kind]
        or Path(c.reward.options["native_root"]).resolve() != sources["reward"]
        or Path(c.data.manifest).resolve() != assets["prompts"] / "t2v/train.jsonl"
        or Path(c.evaluation.manifest).resolve() != Path(receipt["data"]["path"]).resolve()
    ):
        raise ValueError(
            "configured model/reward/source/data paths differ from verified asset bindings"
        )
    return receipt, assets


def load_components(c, device):
    """Load all optional native dependencies only after receipt verification."""
    receipt, assets = _verified_assets(c)
    if c.model.name == "wan_t2v_1_3b":
        from ..models.wan_t2v import WAN_T2V_SOURCE_REVISION, WanT2VAdapter

        adapter = WanT2VAdapter.from_pretrained(
            c.model.checkpoint,
            native_root=c.model.options["native_root"],
            revision=c.model.revision,
            asset_preflight=lambda path, revision: None,
        )  # Full receipt checked above once per rank.
        from wan.modules.model import WanAttentionBlock
        from wan.modules.t5 import T5EncoderModel
        from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

        encoder = T5EncoderModel(
            512,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
            checkpoint_path=str(assets["wan"] / "models_t5_umt5-xxl-enc-bf16.pth"),
            tokenizer_path=str(assets["wan"] / "google/umt5-xxl"),
        )
        tokenizer, factory, blocks, revision = (
            None,
            FlowUniPCMultistepScheduler,
            (WanAttentionBlock,),
            WAN_T2V_SOURCE_REVISION,
        )
    else:
        from ..models.ltx_video import load_ltx_video_components
        from ..models.ltx_video_sampling import LTX_VIDEO_CODE_REVISION

        loaded = load_ltx_video_components(
            c.model.checkpoint,
            assets["ltx_text"],
            native_root=c.model.options["native_root"],
            device="cpu",
        )
        adapter, encoder, tokenizer = loaded.adapter, loaded.text_encoder, loaded.tokenizer
        from ltx_video.models.transformers.attention import BasicTransformerBlock

        factory, blocks, revision = None, (BasicTransformerBlock,), LTX_VIDEO_CODE_REVISION
    adapter.transformer = prepare_policy(
        adapter.transformer, FineTuneConfig(**c.model.options["finetune"])
    )
    adapter.vae.requires_grad_(False).eval()
    from ..rewards.small_video_process import PhasedVideoReward

    scorer = PhasedVideoReward(
        python=c.reward.options["python"],
        native_root=c.reward.options["native_root"],
        checkpoint=c.reward.checkpoint,
        backbone=c.reward.options["backbone"],
        revision=c.reward.revision,
        video_fps=c.sampling.fps,
        reward_fps=4.0,
        ffprobe=c.model.options["ffprobe"],
        scratch=Path(c.runtime.output_dir) / "reward-phases",
    )
    return Components(adapter, encoder, tokenizer, scorer, factory, blocks, revision, receipt)


@torch.no_grad()
def make_predict(c, parts, prompts, repeats, device):
    """Encode each prompt once, then reuse detached CPU text across native replay."""
    from ..models.ltx_video import LTXVideoConditioning, encode_ltx_video_prompts
    from ..models.wan_t2v import WanT2VConditioning

    count = len(prompts) * repeats
    if c.model.name == "wan_t2v_1_3b":
        from wan.configs.wan_t2v_1_3B import t2v_1_3B

        parts.text_encoder.model.to(device)
        try:
            values = parts.text_encoder(list(prompts) + [t2v_1_3B.sample_neg_prompt], device)

            def pad(value):
                embed = torch.nn.functional.pad(value.cpu(), (0, 0, 0, 512 - len(value)))
                return embed, torch.arange(512) < len(value)

            positive = [pad(value) for value in values[:-1]]
            embed = torch.stack([x[0] for x in positive]).repeat_interleave(repeats, 0)
            mask = torch.stack([x[1] for x in positive]).repeat_interleave(repeats, 0)
            neg, negmask = pad(values[-1])
            negative, negative_mask = (
                neg[None].expand(count, -1, -1),
                negmask[None].expand(count, -1),
            )
        finally:
            parts.text_encoder.model.to("cpu")
        timesteps = native_wan_schedule().timesteps

        def conditioning(rows, index):
            rows = rows.cpu()
            return WanT2VConditioning(
                prompt_embeds=embed[rows].to(device),
                prompt_attention_mask=mask[rows].to(device),
                negative_prompt_embeds=negative[rows].to(device),
                negative_prompt_attention_mask=negative_mask[rows].to(device),
                timesteps=timesteps[index].expand(len(rows)).to(device),
            )
    else:
        parts.text_encoder.to(device)
        try:
            text = encode_ltx_video_prompts(
                prompts,
                parts.tokenizer,
                parts.text_encoder,
                device="cpu",
                frame_rate=c.sampling.fps,
            )
        finally:
            parts.text_encoder.to("cpu")
        embed, mask = (
            text.prompt_embeds.repeat_interleave(repeats, 0),
            text.attention_mask.repeat_interleave(repeats, 0),
        )

        def conditioning(rows, index):
            rows = rows.cpu()
            return LTXVideoConditioning(
                embed[rows].to(device), mask[rows].to(device), c.sampling.fps
            )

    if device.type == "cuda":
        torch.cuda.empty_cache()

    def predict(state, sigma, rows, index):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=True):
            return parts.adapter.predict_velocity(state, sigma, conditioning(rows, index))

    return predict


@torch.no_grad()
def score_videos(c, parts, endpoints, prompts, directory, device, *, decode_seeds=None):
    """Decode and score sequential phases on the SAME rank GPU; no external scorer GPU."""
    from ..evaluation.video import encode_video

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    paths = []
    parts.adapter.vae.to(device).eval()
    try:
        for index, endpoint in enumerate(endpoints):
            latent = endpoint[None].to(device)
            if c.model.name == "ltx_video_2b_distilled":
                # Independent paired decode seed is explicit and repeatable for base/post.
                generator = (
                    torch.Generator(device=device).manual_seed(decode_seeds[index])
                    if decode_seeds is not None
                    else None
                )
                decoded = parts.adapter.decode(latent, generator=generator)
            else:
                decoded = parts.adapter.decode(latent)
            expected = (1, 3, c.sampling.frames, c.sampling.height, c.sampling.width)
            if decoded.shape != expected or not torch.isfinite(decoded).all():
                raise ValueError("native video decode geometry or finiteness differs")
            paths.append(
                encode_video(
                    decoded[0].clamp(-1, 1).permute(1, 0, 2, 3),
                    directory / f"{index:06d}.mp4",
                    fps=c.sampling.fps,
                    ffmpeg=c.model.options["ffmpeg"],
                    ffprobe=c.model.options["ffprobe"],
                )
            )
    finally:
        parts.adapter.vae.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()
    parts.scorer.to(device).eval()
    try:
        scores = parts.scorer.score_paths(paths, prompts)
    finally:
        parts.scorer.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return paths, scores


@torch.no_grad()
def initial_noise(c, rows, device, seed):
    """Native initial draw order; dedicated generator does not consume transition RNG."""
    shape = latent_shape(c, rows)
    generator = torch.Generator(device=device).manual_seed(seed)
    if c.model.name == "ltx_video_2b_distilled":
        from ..models.ltx_video_sampling import unpack_ltx_latents

        b, channels, frames, height, width = shape
        packed = torch.randn(
            (b, frames * height * width, channels),
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        return unpack_ltx_latents(packed, shape)
    return torch.randn(shape, device=device, dtype=torch.float32, generator=generator)


def _tensor_hash(value):
    import hashlib

    return hashlib.sha256(
        value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def _completed_evaluation(directory, c, dataset, step):
    import json

    path = directory / "COMPLETE.json"
    if not path.exists():
        return False
    metadata = json.loads(path.read_text())
    if (
        metadata.get("config_sha256") != c.fingerprint()
        or metadata.get("data_sha256") != dataset.sha256
        or metadata.get("actual_optimizer_steps") != step
    ):
        raise ValueError("completed evaluation identity differs")
    for record in metadata["files"]:
        from ..data import file_sha256

        if file_sha256(directory / record["path"]) != record["sha256"]:
            raise ValueError("completed evaluation file changed")
    return True


@torch.no_grad()
def evaluate(session, c, parts, dataset, step, root, *, label=None):
    """General held-out evaluation; acceptance calls use exactly the same eight rows.

    Per-item RNG isolation pairs initial, native transition and decode draws
    across base/post. Dummy ranks keep FSDP collective forward counts identical.
    Evaluation never advances training/selection RNG, data cursor or Adam.
    """
    import json
    import uuid

    from ..data import file_sha256
    from ..training.checkpoint import _atomic_json
    from ..training.small_video_replay import collect_native_video

    root = Path(root)
    destination = root / "evaluation" / (label or f"step-{step:09d}")
    complete = session.local(
        "evaluation completion check", lambda: _completed_evaluation(destination, c, dataset, step)
    )
    session.agree(complete)
    if complete:
        if step == 10 and label is None:
            session.primary("paired evaluation summary", lambda: _publish_paired(root))
        return destination
    # An incomplete attempt is retained, never mistaken for a complete evaluation.
    token = str(uuid.uuid4()) if session.identity.rank == 0 else ""
    if session.distributed:
        import torch.distributed as dist

        data = torch.tensor(
            list(bytes.fromhex(token.replace("-", ""))) if token else [0] * 16,
            device=session.device,
            dtype=torch.uint8,
        )
        dist.broadcast(data, 0)
        token = bytes(data.cpu().tolist()).hex()
    attempt = destination / f"attempt-{token}"
    session.primary(
        "evaluation attempt directory", lambda: attempt.mkdir(parents=True, exist_ok=False)
    )
    rank, world = session.identity.rank, session.identity.world_size
    parts.adapter.transformer.eval()
    records = []
    devices = (
        [session.device.index if session.device.index is not None else torch.cuda.current_device()]
        if session.device.type == "cuda"
        else []
    )
    for offset in range(0, len(dataset.rows), world):
        index = offset + rank
        valid, actual = index < len(dataset.rows), min(index, len(dataset.rows) - 1)
        row = dataset.rows[actual]
        seed = (c.evaluation.seed if c.evaluation.seed is not None else c.seed) + actual
        with evaluation_rng(seed, session.device, devices=devices):
            predict = session.local(
                "evaluation text", lambda: make_predict(c, parts, [row.text], 1, session.device)
            )
            noise = session.local(
                "evaluation initial noise", lambda: initial_noise(c, 1, session.device, seed)
            )
            cache = collect_native_video(
                predict,
                noise,
                model=c.model.name,
                selected=(0,),
                microbatch=1,
                scheduler_factory=parts.scheduler_factory,
            )
            decode_rng = (
                torch.cuda.get_rng_state(session.device) if devices else torch.get_rng_state()
            )

            def score(cache=cache, noise=noise):
                if not valid:
                    return None
                paths, scores = score_videos(
                    c,
                    parts,
                    cache.endpoints,
                    [row.text],
                    attempt / f"sample-{actual:06d}",
                    session.device,
                )
                raw = {key: float(value[0]) for key, value in scores.raw.items()}
                if set(raw) != {"VQ", "MQ", "TA"} or not all(
                    torch.isfinite(torch.tensor(v)) for v in raw.values()
                ):
                    raise ValueError("evaluation requires all finite raw VQ/MQ/TA scores")
                return {
                    "index": actual,
                    "sample_id": row.sample_id,
                    "prompt": row.text,
                    "seed": seed,
                    "actual_optimizer_steps": step,
                    "initial_noise_sha256": _tensor_hash(noise),
                    "transition_innovations_sha256": _tensor_hash(cache.innovations)
                    if cache.innovations is not None
                    else None,
                    "decode_rng_sha256": _tensor_hash(decode_rng),
                    "raw": raw,
                    "raw_VQ_MQ_equal_mean": (raw["VQ"] + raw["MQ"]) / 2,
                    "video": paths[0].relative_to(destination).as_posix(),
                    "video_sha256": file_sha256(paths[0]),
                }

            value = session.local("evaluation decode reward", score)
            if valid:
                records.append(value)
            del score, cache, noise, predict
    session.local(
        "evaluation shard",
        lambda: _atomic_json(attempt / f"rank-{rank:05d}.json", {"records": records}),
    )

    def publish():
        ordered = []
        for worker in range(world):
            rows = json.loads((attempt / f"rank-{worker:05d}.json").read_text())["records"]
            if [item["index"] for item in rows] != list(range(worker, len(dataset.rows), world)):
                raise ValueError("evaluation sample membership differs")
            ordered.extend(rows)
        ordered.sort(key=lambda item: item["index"])
        metrics = {
            key: sum(item["raw"][key] for item in ordered) / len(ordered)
            for key in ("VQ", "MQ", "TA")
        }
        metrics["raw_VQ_MQ_equal_mean"] = (metrics["VQ"] + metrics["MQ"]) / 2
        manifest = destination / "samples.json"
        _atomic_json(
            manifest,
            {
                "records": ordered,
                "raw_means": metrics,
                "count": len(ordered),
                "sampling": resolved_protocol(c),
                "data_sha256": dataset.sha256,
            },
        )
        files = [{"path": "samples.json", "sha256": file_sha256(manifest)}]
        files += [{"path": item["video"], "sha256": item["video_sha256"]} for item in ordered]
        evaluation_metrics = {
            "eval/optimizer_step": step,
            **{f"eval/raw_{key}": value for key, value in metrics.items()},
        }
        with (root / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(evaluation_metrics, allow_nan=False, sort_keys=True) + "\n")
            stream.flush()
        _atomic_json(
            destination / "COMPLETE.json",
            {
                "config_sha256": c.fingerprint(),
                "data_sha256": dataset.sha256,
                "actual_optimizer_steps": step,
                "files": files,
            },
        )

    session.primary("evaluation complete publication", publish)
    if step == 10 and label is None:
        session.primary("paired evaluation summary", lambda: _publish_paired(root))
    return destination


def export_final(session, c, parts, state, receipt, root):
    """Named policy export; full resumable training remains in its final checkpoint."""
    import json
    import os
    import tempfile

    from torch.distributed.fsdp import FullStateDictConfig, StateDictType
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.nn.parallel import DistributedDataParallel as DDP

    from ..data import file_sha256
    from ..training.checkpoint import _atomic_json

    model = parts.adapter.transformer
    final = Path(root) / "final"

    def already():
        marker = final / "COMPLETE.json"
        if not marker.is_file():
            return False
        metadata = json.loads(marker.read_text())
        if (
            metadata.get("identity_sha256") != state.identity.sha256
            or metadata.get("actual_optimizer_steps") != 10
        ):
            raise ValueError("existing final export identity differs")
        for row in metadata["files"]:
            if file_sha256(final / row["path"]) != row["sha256"]:
                raise ValueError("existing final export bytes differ")
        return True

    done = session.local("final export completion", already)
    session.agree(done)
    if done:
        return final
    if state.budget.completed != 10:
        raise ValueError("named final is only published at ten actual optimizer steps")
    if isinstance(model, FSDP):
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            weights = model.state_dict()  # Collective: never hide inside session.primary/local.
        unwrapped = model.module
    else:
        unwrapped = model.module if isinstance(model, DDP) else model
        weights = None

    def publish():
        if final.exists():
            raise ValueError(
                "incomplete existing final export retained; choose a new owned attempt after investigation"
            )
        stage = Path(tempfile.mkdtemp(prefix=".final-pending-", dir=root))
        # Preserve a failed stage for diagnosis; no cleanup of unrelated assets.
        if c.model.options["finetune"]["mode"] == "lora":
            unwrapped.save_pretrained(stage / "adapter", safe_serialization=True)
        else:
            policy = (
                weights
                if weights is not None
                else {k: v.detach().cpu() for k, v in unwrapped.state_dict().items()}
            )
            torch.save(policy, stage / "policy.pt")
        _atomic_json(
            stage / "identity.json",
            {
                "identity": state.identity.state_dict(),
                "protocol": resolved_protocol(c),
                "resumable_checkpoint_receipt": str(receipt),
                "format": "peft_adapter"
                if c.model.options["finetune"]["mode"] == "lora"
                else "native_transformer_state_dict",
                "note": "requires the immutable native base/assets; full Adam/RNG continuation uses the checkpoint",
            },
        )
        files = [
            {"path": path.relative_to(stage).as_posix(), "sha256": file_sha256(path)}
            for path in stage.rglob("*")
            if path.is_file()
        ]
        for item in files:
            with (stage / item["path"]).open("rb") as stream:
                os.fsync(stream.fileno())
        _atomic_json(
            stage / "COMPLETE.json",
            {
                "identity_sha256": state.identity.sha256,
                "actual_optimizer_steps": 10,
                "files": files,
            },
        )
        os.replace(stage, final)
        fd = os.open(Path(root), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    session.primary("named final policy export", publish)
    return final


@contextmanager
def evaluation_rng(seed, device, *, devices=None):
    """Restore Python/NumPy/Torch streams; seed only this rank's CUDA generator."""
    import random

    import numpy as np

    before_python, before_numpy = random.getstate(), np.random.get_state()
    devices = (
        ([] if device.type != "cuda" else [device.index or torch.cuda.current_device()])
        if devices is None
        else devices
    )
    try:
        with torch.random.fork_rng(devices=devices):
            torch.default_generator.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed % 2**32)
            yield
    finally:
        random.setstate(before_python)
        np.random.set_state(before_numpy)


def paired_evaluation_change(base_path, post_path):
    """Paired raw-score uncertainty, without selecting best clips or claiming learning."""
    import json

    import numpy as np

    base, post = [json.loads(Path(path).read_text())["records"] for path in (base_path, post_path)]
    if len(base) != len(post) or not base:
        raise ValueError("paired evaluation populations differ")
    fields = (
        "index",
        "sample_id",
        "prompt",
        "seed",
        "initial_noise_sha256",
        "transition_innovations_sha256",
        "decode_rng_sha256",
    )
    for before, after in zip(base, post):
        if any(before[key] != after[key] for key in fields):
            raise ValueError("paired prompt/seed/native noise identity differs")
    generator = np.random.default_rng(42)  # Independent analysis RNG; training streams unchanged.
    indices = generator.integers(0, len(base), size=(10000, len(base)))
    result = {
        "count": len(base),
        "interpretation": "short-run sanity evidence; not long-term stability or generalization proof",
    }
    for key in ("VQ", "MQ", "TA", "VQ_MQ_equal_mean"):

        def values(rows):
            return np.array(
                [
                    (
                        (row["raw"]["VQ"] + row["raw"]["MQ"]) / 2
                        if key == "VQ_MQ_equal_mean"
                        else row["raw"][key]
                    )
                    for row in rows
                ]
            )

        before, after = values(base), values(post)
        delta = after - before
        low, high = np.quantile(delta[indices].mean(1), [0.025, 0.975])
        result[key] = {
            "base_mean": float(before.mean()),
            "post_mean": float(after.mean()),
            "paired_mean_delta": float(delta.mean()),
            "paired_bootstrap_95_percent_interval": [float(low), float(high)],
            "paired_deltas": delta.tolist(),
        }
    return result


def _publish_paired(root):
    from ..training.checkpoint import _atomic_json

    root = Path(root)
    base = root / "evaluation/step-000000000/samples.json"
    post = root / "evaluation/step-000000010/samples.json"
    if not base.is_file() or not post.is_file():
        raise ValueError("final paired evaluation requires complete base and post results")
    summary = paired_evaluation_change(base, post)
    _atomic_json(root / "paired-evaluation.json", summary)
    return summary
