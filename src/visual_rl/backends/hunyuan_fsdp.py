"""Full-shard Hunyuan training with collective-safe sampling/evaluation."""

import json
import os
import random
import uuid
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from ..config import ConfigError
from ..data import PromptStream
from ..evaluation import SampleRecord
from ..models.finetune import parameter_inventory
from ..rewards.scalar import weighted_score
from ..sampling import shifted_schedule
from ..training import video_nft, video_pairwise, video_refl
from ..training.checkpoint import _atomic_json
from ..training.distributed import distributed_session, launch_identity
from ..training.fsdp_checkpoint import load_fsdp_checkpoint, save_fsdp_checkpoint
from ..training.resume import prepare_resume_history, validate_resume_request
from ..training.sharding import wrap_video_policy
from ..training.state import CostClock, Progress
from ..training.update import accumulated_step
from ..wandb_export import open_metrics
from . import hunyuan
from .image_ddp import _cost, _global_scores, _publish_cost
from .sd3 import _check_run_identity, _restore_cost, _reward_metrics


def wrap_policy(c, parts, session):
    if c.model.name == "wan_i2v_distilled":
        from diffusers.models.transformers.transformer_wan import WanTransformerBlock

        return wrap_video_policy(parts.adapter.trainable_module, device=session.device,
            block_types=(WanTransformerBlock,), activation_checkpointing=c.model.options["gradient_checkpointing"])
    from fastvideo.models.hunyuan_hf.modeling_hunyuan import (
        HunyuanVideoSingleTransformerBlock,
        HunyuanVideoTransformerBlock,
    )

    return wrap_video_policy(parts.adapter.trainable_module, device=session.device,
        block_types=(HunyuanVideoSingleTransformerBlock, HunyuanVideoTransformerBlock),
        activation_checkpointing=c.model.options["gradient_checkpointing"])


@torch.no_grad()
def evaluate(session, c, parts, dataset, step, root, *, backend=hunyuan):
    """Even forward counts, including a dummy trajectory for a last empty shard.

    FSDP inference retains all-gather collectives; it must not unwrap the module
    or let an empty rank leave early. Dummy trajectories are never scored/saved.
    """
    parts.adapter.transformer.eval()
    rank, world = session.identity.rank, session.identity.world_size
    token = torch.tensor(list(uuid.uuid4().bytes) if rank == 0 else [0] * 16,
                         dtype=torch.uint8, device=session.device)
    dist.broadcast(token, src=0)
    directory = root / "evaluation" / f"step-{step:09d}" / bytes(token.cpu().tolist()).hex()
    session.primary("evaluation directory", lambda: directory.mkdir(parents=True))
    samples, indices = [], []
    steps = c.evaluation.sampling_steps or c.sampling.steps
    sigmas = shifted_schedule(steps, c.evaluation.shift or c.sampling.options["shift"])
    for offset in range(0, len(dataset.rows), world):
        index = offset + rank
        valid = index < len(dataset.rows)
        actual = index if valid else 0
        row = dataset.rows[actual]
        seed = (c.seed if c.evaluation.seed is None else c.evaluation.seed) + actual

        def prepare():
            predict = backend._predict(c, parts, [row], c.evaluation.manifest, 1, session.device, evaluation=True)
            noise = torch.randn(backend._shape(c, 1, evaluation=True),
                                generator=torch.Generator().manual_seed(seed),
                                dtype=torch.float32 if c.task == "i2v" else torch.bfloat16)
            return predict, noise.to(session.device)

        predict, noise = session.local("evaluation conditioning", prepare)
        endpoints = backend.evaluation_endpoint(predict, noise, sigmas)

        def score():
            if not valid:
                return
            paths, scores = backend.score_videos(c, parts, endpoints, [row.text],
                directory / f"sample-{index:06d}", session.device, evaluation=True)
            value = weighted_score({key: float(tensor[0]) for key, tensor in scores.raw.items()},
                                   scores.weights, space=scores.space, normalization=scores.normalization)
            samples.append(SampleRecord.from_file(media_root=root, path=paths[0].relative_to(root).as_posix(),
                sample_id=row.sample_id, prompt=row.text, seed=seed, optimizer_step=step,
                framecount=c.sampling.frames, fps=c.sampling.fps, scores=value))
            indices.append(index)

        session.local("evaluation decode/scoring", score)
    session.local("evaluation shard publication", lambda: _atomic_json(directory / f"rank-{rank:05d}.json",
        {"indices": indices, "samples": [sample.to_dict() for sample in samples]}))

    def publish():
        ordered = {}
        for worker in range(world):
            payload = json.loads((directory / f"rank-{worker:05d}.json").read_text())
            expected = list(range(worker, len(dataset.rows), world))
            if payload["indices"] != expected or len(payload["samples"]) != len(expected):
                raise ValueError("evaluation shard population differs")
            for index, value in zip(expected, payload["samples"]):
                sample = SampleRecord.from_dict(value)
                row = dataset.rows[index]
                seed = (c.seed if c.evaluation.seed is None else c.evaluation.seed) + index
                if (sample.sample_id != row.sample_id or sample.prompt != row.text
                        or sample.seed != seed or sample.optimizer_step != step):
                    raise ValueError("evaluation shard identity differs")
                ordered[index] = sample
        hunyuan.publish_evaluation(c, dataset, step, root, directory,
                                  [ordered[i] for i in range(len(dataset.rows))])

    session.primary("evaluation manifest", publish)


def run(c, *, resume=None, backend=hunyuan, resume_rewind=False, evaluate_only=False):
    validate_resume_request(resume, resume_rewind, evaluate_only)
    backend.validate_config(c)
    identity = launch_identity("fsdp")
    if identity.world_size < 2:
        raise ConfigError("full-shard video training requires at least two ranks; use distributed=none for one device")
    p = identity.local_prompt_count(c.sampling.prompts_per_rollout)
    k = c.sampling.samples_per_prompt
    if p * k % hunyuan.updates_per_rollout(c):
        raise ConfigError("local row count must divide by the number of update windows")
    timeout = float(os.environ.get("VISUAL_RL_COLLECTIVE_TIMEOUT_SECONDS", "300"))
    with distributed_session("fsdp", device=c.runtime.device, timeout_seconds=timeout) as session:
        session.require_same_digest(c.fingerprint())
        session.agree({"resume": None if resume is None else str(Path(resume).absolute()),
                       "rewind": resume_rewind, "evaluate_only": evaluate_only})
        _run(session, c, p, k, resume, backend=backend,
             resume_rewind=resume_rewind, evaluate_only=evaluate_only)


def _run(session, c, p, k, resume, *, backend=hunyuan, resume_rewind=False, evaluate_only=False):
    rank, world = session.identity.rank, session.identity.world_size
    root = Path(c.runtime.output_dir)

    data, validation = session.local("data validation", lambda: backend.load_datasets(c))
    with ExitStack() as resources:
        def acquire():
            import fcntl

            if resume is None:
                root.mkdir(parents=True, exist_ok=False)
            elif not root.is_dir():
                raise ConfigError("resume requires original output directory")
            lock = resources.enter_context((root / ".owner.lock").open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if resume:
                _check_run_identity(c, root)

        session.primary("output ownership", acquire)
        torch.manual_seed(c.seed)
        random.seed(c.seed)
        np.random.seed(c.seed)
        parts = session.local("component loading", lambda: backend.load_components(c, session.device))
        inventory = session.local("parameter inventory", lambda: parameter_inventory(parts.adapter.trainable_module))
        session.agree(inventory)
        # Constructor and all model forwards contain collectives. Never wrap
        # them in a local-error all_reduce that could mismatch a peer's forward.
        model = wrap_policy(c, parts, session)
        parts.adapter.transformer = model
        optimizer = torch.optim.AdamW(model.parameters(), lr=c.training.learning_rate,
            betas=c.training.betas, eps=c.training.epsilon, weight_decay=c.training.weight_decay)
        torch.manual_seed(c.seed + rank)
        random.seed(c.seed + rank)
        np.random.seed((c.seed + rank) % 2**32)
        stream = PromptStream(data, seed=c.seed, shuffle=c.data.shuffle)
        nft = (session.local("NFT shadow creation", lambda: video_nft.VideoNFTState(model, c.algorithm.options))
               if c.algorithm.name == "diffusionnft" else None)
        stateful = {"prompts": stream, **(nft.stateful() if nft else {})}
        checkpoint = dict(model=model, optimizer=optimizer, config_sha256=c.fingerprint(), stateful=stateful)
        progress = load_fsdp_checkpoint(resume, **checkpoint) if resume else Progress()
        if resume:
            def validate_resume():
                if progress.optimizer_steps % hunyuan.updates_per_rollout(c) or progress.optimizer_steps > c.training.max_updates:
                    raise ConfigError("resume requires an in-budget completed rollout boundary")
            session.local("resume boundary", validate_resume)
            if evaluate_only:
                if validation is None:
                    raise ConfigError("checkpoint evaluation requires a validation manifest")
                return evaluate(session, c, parts, validation, progress.optimizer_steps, root, backend=backend)
            session.primary("resume history rollback", lambda: prepare_resume_history(
                c, root, progress, receipt=resume, rewind=resume_rewind))
            session.local("resume cost", lambda: _restore_cost(c, root, progress))
            done = session.local("resume completion", lambda:
                progress.optimizer_steps == c.training.max_updates and (root / "status.json").exists())
            session.agree(done)
            if done:
                return
        else:
            def publish_run():
                _atomic_json(root / "resolved-config.json", c.to_dict())
                _atomic_json(root / "run.json", {"kind": f"{c.model.name}_native_development",
                    "config_sha256": c.fingerprint(), "parameters": inventory, "device": str(session.device),
                    "data_sha256": data.sha256, "dp_world_size": world, "local_prompts": p,
                    "samples_per_prompt": k, "strategy": "full_shard_fp32_masters",
                    "ordering": "global-prompt-stream/rank-local-seed-and-disjoint-windows-v1"})

            session.primary("run identity", publish_run)
        clock = CostClock(progress, gpu_count=world if session.device.type == "cuda" else 0)
        local_config = replace(c, sampling=replace(c.sampling, prompts_per_rollout=p))
        sigmas = shifted_schedule(c.sampling.steps, c.sampling.options["shift"])
        log = session.primary("metrics open", lambda: resources.enter_context(open_metrics(c, root)))
        if validation and not resume:
            evaluate(session, c, parts, validation, 0, root, backend=backend)
        while progress.optimizer_steps < c.training.max_updates:
            with clock.training_phase():
                progress.attempts += 1

                def prepare_rollout():
                    rows = stream.take(c.sampling.prompts_per_rollout)[rank * p:(rank + 1) * p]
                    predict = backend._predict(c, parts, rows, c.data.manifest, k, session.device,
                                               attempt=progress.attempts, rank=rank)
                    plan = backend.training_inputs(local_config, progress.attempts, rank)
                    return rows, predict, plan

                rows, predict, plan = session.local("rollout conditioning", prepare_rollout)
                if nft:
                    with nft.use_old(session):
                        cache = backend.collect_training(c, parts, predict, plan, sigmas, session.device)
                else:
                    cache = backend.collect_training(c, parts, predict, plan, sigmas, session.device)
                scores = session.local("rollout decode/scoring", lambda: hunyuan.score_training_videos(
                    backend, c, parts, cache.endpoints, [row.text for row in rows for _ in range(k)], root, session.device))
                global_scores = _global_scores(session, scores)

                extra_metrics = {}
                if nft:
                    loss_fn = session.local("NFT state preparation", lambda: video_nft.prepare_objective(
                        c, cache, predict, scores, global_scores, session.device, attempt=progress.attempts, rank=rank))
                    with nft.use_old(session):
                        video_nft.predict_old(loss_fn, c.training.microbatch)
                    windows = (tuple(range(len(loss_fn.states))),)
                elif c.algorithm.name == "refl":
                    loss_fn, windows, extra_metrics = session.local("ReFL local VAE/reward VJP", lambda:
                        video_refl.prepare_objective(c, parts, cache, predict, [row.text for row in rows], session.device))
                    keys = sorted(extra_metrics)
                    session.agree(keys)
                    values = torch.tensor([extra_metrics[key] for key in keys], dtype=torch.float64, device=session.device)
                    dist.all_reduce(values)
                    extra_metrics = dict(zip(keys, (values / world).cpu().tolist()))
                elif video_pairwise.enabled(c):
                    parts.adapter.transformer.train()
                    feedback = session.local("pairwise frozen feedback", lambda: video_pairwise.prepare(
                        c, cache, scores, attempt=progress.attempts, group_offset=rank * p))
                    flags = session.local("pairwise local activity", lambda: video_pairwise.activity(feedback, session.device))
                    dist.all_reduce(flags, op=dist.ReduceOp.MAX)
                    loss_fn, windows, extra_metrics = session.local("pairwise objective", lambda: video_pairwise.finalize(
                        c, cache, predict, feedback, flags, session.device))
                    keys = sorted(extra_metrics)
                    values = torch.tensor([extra_metrics[key] for key in keys], dtype=torch.float64, device=session.device)
                    dist.all_reduce(values)
                    extra_metrics = dict(zip(keys, (values / world).cpu().tolist()))
                    if loss_fn is None:
                        metrics = {**_reward_metrics(global_scores, "train"), **extra_metrics, **_cost(session, clock)}

                        def write_skip():
                            log.write(json.dumps(metrics, allow_nan=False, sort_keys=True) + "\n")
                            log.flush()

                        session.primary("inactive pairwise publication", write_skip)
                        _publish_cost(session, c, root, clock)
                        del feedback, cache
                        continue
                else:
                    loss_fn, windows = session.local("objective preparation", lambda: backend.prepare_objective(
                        c, parts, cache, predict, scores, global_scores, session.device))
                    if c.algorithm.name == "diffusionart":
                        active = session.local("scalar ART activity", lambda: torch.tensor(
                            int(loss_fn.has_guidance), dtype=torch.int32, device=session.device))
                        dist.all_reduce(active, op=dist.ReduceOp.MAX)
                        if not active.item():
                            metrics = {**_reward_metrics(global_scores, "train"), **_cost(session, clock),
                                       "system/skipped_zero_guidance": 1}

                            def write_inactive():
                                log.write(json.dumps(metrics, allow_nan=False, sort_keys=True) + "\n")
                                log.flush()

                            session.primary("inactive scalar ART publication", write_inactive)
                            _publish_cost(session, c, root, clock)
                            continue
                for items in windows:
                    metrics = accumulated_step(model, optimizer, items, loss_fn, microbatch=c.training.microbatch,
                                               denominator=len(items), max_grad_norm=c.training.grad_clip,
                                               require_nonzero_grad=c.algorithm.name == "refl")
                    progress.optimizer_steps += 1
                    if nft:
                        metrics.update(session.local("NFT shadow refresh", lambda: nft.after_update(progress.optimizer_steps)))
                    if progress.optimizer_steps % hunyuan.updates_per_rollout(c) == 0:
                        progress.completed_rollouts += 1
                    metrics.update(_reward_metrics(global_scores, "train"))
                    metrics.update(extra_metrics)
                    metrics.update(_cost(session, clock))

                    def write_log():
                        log.write(json.dumps(metrics, sort_keys=True, allow_nan=False) + "\n")
                        log.flush()

                    session.primary("metrics publication", write_log)
            step = progress.optimizer_steps
            if validation and c.evaluation.due(step, c.training.max_updates):
                evaluate(session, c, parts, validation, step, root, backend=backend)
            if c.runtime.checkpoint_due(step, c.training.max_updates):
                clock.tick()
                save_fsdp_checkpoint(root / "checkpoints", progress=progress,
                                     keep_last=c.runtime.checkpoint_keep, **checkpoint)
            _publish_cost(session, c, root, clock)
        state = _publish_cost(session, c, root, clock)
        session.primary("completion", lambda: _atomic_json(root / "status.json", {
            "status": "completed", "kind": f"{c.model.name}_native_development",
            "config_sha256": c.fingerprint(), **state,
        }))
