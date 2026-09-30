"""Native replicated data-parallel image training with explicit global batches.

Shares scientific kernels/loaders/evaluation with the single-device runner.
This is a development backend, not a frozen historical sampler reproduction.
"""

import json
import os
import random
import uuid
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from ..config import ConfigError
from ..data import PromptDataset, PromptStream
from ..evaluation import SampleRecord
from ..models.finetune import parameter_inventory
from ..rewards.image_factory import validate_reward_prompts
from ..rewards.scalar import weighted_score
from ..training.checkpoint import _atomic_json
from ..training.distributed import distributed_session, launch_identity
from ..training.distributed_checkpoint import load_ddp_checkpoint, save_ddp_checkpoint
from ..training.grpo import ALGORITHMS as GRPO_ALGORITHMS
from ..training.grpo import GRPOState, dance_advantages, prepare_grpo_update
from ..training.grpo_sampling import collect_grpo_rollout
from ..training.image_contracts import prompt_advantages, tf32_policy
from ..training.nft import NFTState, nft_windows, prepare_nft_objective
from ..training.nft_sampling import collect_nft_endpoints
from ..training.rollout import collect_rollout
from ..training.state import CostClock, Progress
from ..training.update import accumulated_step
from ..wandb_export import open_metrics


@contextmanager
def _inference(adapter, wrapped):
    # Inference has no model collectives and can run unequal evaluation shards.
    adapter.transformer = wrapped.module
    try:
        yield
    finally:
        adapter.transformer = wrapped


def _global_scores(session, scores):
    def contract():
        values = {**{f"raw/{k}": v for k, v in scores.raw.items()},
                  **{f"diag/{k}": v for k, v in (scores.diagnostics or {}).items()}}
        return {"weights": dict(scores.weights), "space": scores.space,
                "normalization": None if scores.normalization is None else {
                    key: asdict(value) for key, value in scores.normalization.items()
                },
                "values": {key: [list(value.shape), str(value.dtype)] for key, value in values.items()}}

    session.agree(session.local("reward contract", contract))

    def gather(value):
        value = value.detach().to(session.device).contiguous()
        received = [torch.empty_like(value) for _ in range(session.identity.world_size)]
        dist.all_gather(received, value)
        return torch.cat(received).cpu()

    raw = {key: gather(scores.raw[key]) for key in sorted(scores.raw)}
    diagnostics = None if scores.diagnostics is None else {
        key: gather(scores.diagnostics[key]) for key in sorted(scores.diagnostics)
    }
    return weighted_score(raw, scores.weights, space=scores.space,
                          normalization=scores.normalization, diagnostics=diagnostics)


def _cost(session, clock):
    metrics = clock.metrics()
    times = torch.tensor([metrics["cost/elapsed_seconds"], metrics["cost/training_seconds"]],
                         dtype=torch.float64, device=session.device)
    dist.all_reduce(times, op=dist.ReduceOp.MAX)
    elapsed, training = times.cpu().tolist()
    gpu_count = session.identity.world_size if session.device.type == "cuda" else 0
    metrics.update({"cost/elapsed_seconds": elapsed, "cost/training_seconds": training,
                    "cost/elapsed_gpu_hours": elapsed * gpu_count / 3600,
                    "cost/training_gpu_hours": training * gpu_count / 3600,
                    "cost/gpu_count": gpu_count, "system/dp_world_size": session.identity.world_size})
    return metrics


def _publish_cost(session, c, root, clock):
    metrics = _cost(session, clock)
    progress = {**clock.progress.state_dict(), "elapsed_seconds": metrics["cost/elapsed_seconds"],
                "training_seconds": metrics["cost/training_seconds"]}
    session.primary("cost publication", lambda: _atomic_json(root / "cost.json", {
        "config_sha256": c.fingerprint(), "progress": progress,
    }))
    return progress


def _evaluate(session, c, parts, wrapped, dataset, step, root, nft):
    from . import sd3

    token = torch.tensor(list(uuid.uuid4().bytes) if session.identity.rank == 0 else [0] * 16,
                         device=session.device, dtype=torch.uint8)
    dist.broadcast(token, src=0)
    directory = root / "evaluation" / f"step-{step:09d}" / bytes(token.cpu().tolist()).hex()
    session.primary("evaluation directory", lambda: directory.mkdir(parents=True))
    indices = tuple(range(session.identity.rank, len(dataset.rows), session.identity.world_size))

    def evaluate_local():
        with _inference(parts.adapter, wrapped), nft.evaluation() if nft else nullcontext():
            samples = sd3.evaluate(c, parts, dataset, step, root, indices=indices,
                                   directory=directory, publish=False)
        _atomic_json(directory / f"rank-{session.identity.rank:05d}.json", {
            "samples": [sample.to_dict() for sample in samples], "indices": list(indices),
        })

    session.local("evaluation generation/scoring", evaluate_local)

    def publish():
        indexed = {}
        for rank in range(session.identity.world_size):
            payload = json.loads((directory / f"rank-{rank:05d}.json").read_text())
            expected = list(range(rank, len(dataset.rows), session.identity.world_size))
            if payload["indices"] != expected or len(payload["samples"]) != len(expected):
                raise ValueError("evaluation shard population differs")
            for index, value in zip(expected, payload["samples"]):
                sample = SampleRecord.from_dict(value)
                row = dataset.rows[index]
                if (sample.sample_id != row.sample_id or sample.prompt != row.text
                        or sample.seed != (c.seed if c.evaluation.seed is None else c.evaluation.seed) + index
                        or sample.optimizer_step != step):
                    raise ValueError("evaluation shard identity differs")
                indexed[index] = sample
        sd3._publish_evaluation(c, dataset, step, root, directory,
                               [indexed[index] for index in range(len(dataset.rows))])

    session.primary("evaluation manifest", publish)


def run(c, *, resume=None):
    identity = launch_identity("ddp")
    p = identity.local_prompt_count(c.sampling.prompts_per_rollout)
    k = c.sampling.samples_per_prompt
    if c.algorithm.name == "diffusionart" and p * k % c.algorithm.options.get("updates_per_rollout", 2):
        raise ConfigError("DDP ART requires local rows divisible by its disjoint windows")
    if c.algorithm.name == "diffusionnft" and p * k % c.algorithm.options.get("updates_per_rollout", 1):
        raise ConfigError("DDP NFT requires local rows divisible by its disjoint windows")
    if c.algorithm.name in GRPO_ALGORITHMS:
        if p * k % c.algorithm.options["training_batches"]:
            raise ConfigError("DDP GRPO logical training batches must divide local rows")
        if p * k % c.sampling.options["logical_batch_size"]:
            raise ConfigError("DDP GRPO logical sampling batch must divide local rows")
    timeout = float(os.environ.get("VISUAL_RL_COLLECTIVE_TIMEOUT_SECONDS", "300"))
    with distributed_session("ddp", device=c.runtime.device, timeout_seconds=timeout) as session:
        session.require_same_digest(c.fingerprint())
        session.agree({"resume": None if resume is None else str(Path(resume).absolute())})
        _run(session, c, p, k, resume)


def _run(session, c, p, k, resume):
    from . import sd3

    rank, world = session.identity.rank, session.identity.world_size
    root = Path(c.runtime.output_dir)

    def datasets():
        data = PromptDataset.load(c.data.manifest, expected_sha256=c.data.sha256, task="t2i")
        validation = PromptDataset.load(c.evaluation.manifest, expected_sha256=c.evaluation.manifest_sha256,
                                        task="t2i") if c.evaluation.enabled else None
        for dataset in (data, validation):
            if dataset is not None:
                validate_reward_prompts(c.reward, [r.text for r in dataset.rows], metadata=[r.metadata for r in dataset.rows])
        return data, validation

    data, validation = session.local("data validation", datasets)
    with ExitStack() as resources:
        def acquire():
            import fcntl

            if resume is None:
                root.mkdir(parents=True, exist_ok=False)
            elif not root.is_dir():
                raise ConfigError("resume requires the original output directory")
            lock = resources.enter_context((root / ".owner.lock").open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if resume:
                sd3._check_run_identity(c, root)

        session.primary("output ownership", acquire)
        resources.enter_context(tf32_policy(c.training.allow_tf32))
        torch.manual_seed(c.seed)
        random.seed(c.seed)
        np.random.seed(c.seed)

        def components():
            parts = sd3.load_components(c)
            for scorer in parts.scorer.modules():
                close = getattr(scorer, "close", None)
                if callable(close):
                    resources.callback(close)
            return parts

        parts = session.local("component loading", components)
        raw = parts.adapter.trainable_module
        wrapped = DistributedDataParallel(raw, broadcast_buffers=False,
                                           device_ids=[session.device.index] if session.device.type == "cuda" else None)
        parts.adapter.transformer = wrapped
        optimizer = torch.optim.AdamW([parameter for parameter in raw.parameters() if parameter.requires_grad],
                                     lr=c.training.learning_rate, betas=c.training.betas,
                                     eps=c.training.epsilon, weight_decay=c.training.weight_decay)
        # Parameters/scorers initialize identically; subsequent data-parallel
        # stochastic work is rank-local and fully checkpointed.
        torch.manual_seed(c.seed + rank)
        random.seed(c.seed + rank)
        np.random.seed((c.seed + rank) % 2**32)
        generator = torch.Generator(device=session.device).manual_seed(c.seed + rank)
        stream = PromptStream(data, seed=c.seed, shuffle=c.data.shuffle)
        nft = session.local("algorithm state", lambda: NFTState(raw, c.algorithm.options)
                            if c.algorithm.name == "diffusionnft" else None)
        grpo = session.local("GRPO state", lambda: GRPOState(raw, c.algorithm.options)
                             if c.algorithm.name in GRPO_ALGORITHMS else None)
        algorithm_state = nft or grpo
        updates = (c.algorithm.options["updates_per_rollout"] if grpo else
                   c.algorithm.options.get("updates_per_rollout", 1) if nft else
                   c.algorithm.options.get("updates_per_rollout", 2))
        scaler = sd3._scaler(c, session.device)
        replicated = algorithm_state.stateful() if algorithm_state else {}
        if scaler is not None:
            replicated["grad_scaler"] = scaler
        checkpoint_args = dict(model=wrapped, optimizer=optimizer, config_sha256=c.fingerprint(),
                               stateful={"prompts": stream}, replicated_stateful=replicated,
                               generators={"sampling": generator})
        progress = Progress()
        if resume:
            progress = load_ddp_checkpoint(resume, **checkpoint_args)

            def check_resume():
                if progress.optimizer_steps % updates or progress.optimizer_steps > c.training.max_updates:
                    raise ConfigError("resume requires an in-budget completed rollout boundary")
                if (root / "metrics.jsonl").exists():
                    for line in (root / "metrics.jsonl").read_text().splitlines():
                        if json.loads(line).get("train/optimizer_step", 0) > progress.optimizer_steps:
                            raise ConfigError("metrics newer than checkpoint; refuse history splicing")
                sd3._restore_cost(c, root, progress)

            session.local("resume history", check_resume)
            done = session.local("resume completion", lambda: progress.optimizer_steps == c.training.max_updates
                                 and (root / "status.json").exists())
            session.agree(done)
            if done:
                return
        else:
            def publish_run():
                _atomic_json(root / "resolved-config.json", c.to_dict())
                _atomic_json(root / "run.json", {"kind": f"{c.model.name}_native_development",
                             "config_sha256": c.fingerprint(), "parameters": parameter_inventory(raw),
                             "device": str(session.device), "data_sha256": data.sha256,
                             "dp_world_size": world, "local_prompts": p, "samples_per_prompt": k,
                             "ordering": "global-prompt-stream/rank-local-seed-and-disjoint-windows-v1"})
            session.primary("run identity", publish_run)
        clock = CostClock(progress, gpu_count=world if session.device.type == "cuda" else 0)
        log = session.primary("metrics open", lambda: resources.enter_context(open_metrics(c, root)))
        if validation and not resume:
            _evaluate(session, c, parts, wrapped, validation, 0, root, algorithm_state)
        sigmas = sd3._schedule(c, c.sampling.steps)
        selected = tuple(c.sampling.options.get("train_timesteps", ()))
        empty_rollouts = 0
        while progress.optimizer_steps < c.training.max_updates:
            with clock.training_phase():
                progress.attempts += 1

                def rollout():
                    global_rows = stream.take(c.sampling.prompts_per_rollout)
                    rows = global_rows[rank * p:(rank + 1) * p]
                    with _inference(parts.adapter, wrapped):
                        predict = sd3._predict(c, parts, [row.text for row in rows], k)
                        dtype = sd3._precision(c) if (algorithm_state or c.model.name == "flux") and c.training.precision != "fp32" else torch.float32
                        shared_noise = c.algorithm.name == "dancegrpo" and c.sampling.options["same_latent"]
                        noise = torch.randn(sd3._noise_shape(c, p if shared_noise else p * k), generator=generator, device=session.device, dtype=dtype)
                        if shared_noise:
                            noise = noise.repeat_interleave(k, dim=0)
                        if grpo and c.algorithm.name == "flow_grpo" and c.model.name == "sd3":
                            noise = noise.float()
                        if nft:
                            with nft.old.use():
                                endpoints = collect_nft_endpoints(predict.rollout if c.model.name == "flux" else predict,
                                    noise, sigmas, solver=c.sampling.options["solver"], microbatch=c.sampling.microbatch,
                                    quantize_model_time=c.model.name != "flux")
                            cache = None
                        elif grpo:
                            options = {key: value for key, value in c.sampling.options.items() if key not in {"shift", "same_latent"}}
                            options["pool"] = tuple(options["pool"])
                            cache = collect_grpo_rollout(predict.rollout if c.model.name == "flux" else predict,
                                noise, sigmas, microbatch=c.sampling.microbatch, generator=generator,
                                flux_precision=c.model.name == "flux", **options)
                            endpoints = cache.endpoints
                        else:
                            cache = collect_rollout(predict.rollout if c.model.name == "flux" else predict, noise, sigmas,
                                                    selected=selected, microbatch=c.sampling.microbatch,
                                                    preserve_state_dtype=c.model.name == "flux")
                            endpoints = cache.endpoints
                        result = sd3._score(c, parts, endpoints, [row.text for row in rows for _ in range(k)],
                                            metadata=[row.metadata for row in rows for _ in range(k)])
                    return global_rows, predict, endpoints, cache, result

                global_rows, predict, endpoints, cache, result = session.local("rollout/scoring", rollout)
                scores = _global_scores(session, result.scores)
                prompts = [row.text for row in global_rows for _ in range(k)]

                def objective():
                    span = slice(rank * p * k, (rank + 1) * p * k)
                    with _inference(parts.adapter, wrapped):
                        if nft:
                            advantages = prompt_advantages(scores.weighted, prompts)[span]
                            obj = prepare_nft_objective(c, nft, predict, endpoints, sigmas, advantages, generator, session.device)
                            return obj, nft_windows(c, obj), {}
                        if grpo:
                            advantages = (dance_advantages(c, scores.weighted, prompts)
                                          if c.algorithm.name == "dancegrpo" else
                                          prompt_advantages(scores.weighted, prompts))
                            target = None
                            if c.algorithm.options["zero_advantage_mode"] == "filter_pad":
                                counts = (advantages != 0).reshape(world, p * k).sum(1)
                                maximum = int(counts.max())
                                batches = c.algorithm.options["training_batches"]
                                target = ((maximum + batches - 1) // batches) * batches
                            obj, windows, _ = prepare_grpo_update(c, cache, predict, scores.weighted[span],
                                prompts[span], generator, session.device, advantages=advantages[span], target_count=target)
                            metrics = {"grpo/nonzero_rows": int((advantages != 0).sum()),
                                       "grpo/retained_rows": (target if target is not None else p * k) * world}
                            return obj, windows, metrics
                        guidance, metrics = sd3._art_guidance_scores(c, SimpleNamespace(scores=scores), prompts)
                        obj, windows = sd3._art_update(c, cache, predict, result, torch.arange(p).repeat_interleave(k),
                                                      p, k, selected, generator, session.device, guidance_scores=guidance[span])
                        return obj, windows, metrics

                objective_fn, windows, algorithm_metrics = session.local("objective preparation", objective)
                if not windows:
                    empty_rollouts += 1
                    metrics = {**algorithm_metrics, **sd3._reward_metrics(scores, "train"),
                               **_cost(session, clock), "system/skipped_zero_advantage": 1}
                    def log_skip():
                        log.write(json.dumps(metrics, sort_keys=True, allow_nan=False) + "\n")
                        log.flush()
                    session.primary("skip publication", log_skip)
                    _publish_cost(session, c, root, clock)
                    if empty_rollouts >= c.algorithm.options["max_empty_rollouts"]:
                        raise RuntimeError("configured consecutive zero-advantage rollout limit reached; no optimizer/EMA updates applied")
                    continue
                empty_rollouts = 0
                for items in windows:
                    if grpo:
                        session.local("GRPO pre-update EMA", lambda: grpo.before_update(progress.optimizer_steps))
                    metrics = accumulated_step(wrapped, optimizer, items, objective_fn,
                                               microbatch=c.training.microbatch, denominator=len(items),
                                               max_grad_norm=c.training.grad_clip, scaler=scaler)
                    progress.optimizer_steps += 1
                    if nft:
                        metrics.update(session.local("policy shadow update", lambda: nft.after_update(
                            progress.optimizer_steps, refresh_old=progress.optimizer_steps % updates == 0)))
                    elif grpo:
                        session.local("GRPO post-update EMA", lambda: grpo.after_update(progress.optimizer_steps))
                    if progress.optimizer_steps % updates == 0:
                        progress.completed_rollouts += 1
                    metrics.update(algorithm_metrics)
                    metrics.update(sd3._reward_metrics(scores, "train"))
                    metrics.update(_cost(session, clock))
                    def write_log():
                        log.write(json.dumps(metrics, sort_keys=True, allow_nan=False) + "\n")
                        log.flush()
                    session.primary("metrics publication", write_log)
            step = progress.optimizer_steps
            if validation and c.evaluation.due(step, c.training.max_updates):
                _evaluate(session, c, parts, wrapped, validation, step, root, algorithm_state)
            if c.runtime.checkpoint_due(step, c.training.max_updates):
                clock.tick()
                save_ddp_checkpoint(root / "checkpoints", progress=progress, **checkpoint_args)
            _publish_cost(session, c, root, clock)
        state = _publish_cost(session, c, root, clock)
        session.primary("completion", lambda: _atomic_json(root / "status.json", {
            "status": "completed", "kind": f"{c.model.name}_native_development",
            "config_sha256": c.fingerprint(), **state,
        }))
