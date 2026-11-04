"""Bounded native small-video ART runner; resource ownership stays with launchers."""

import fcntl
import json
import os
import random
import re
import time
import traceback
import unicodedata
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from ..algorithms import art_scalar_covariance
from ..config import ConfigError
from ..data import PromptDataset, PromptStream
from ..models.finetune import parameter_inventory
from ..training import checkpoint as local_checkpoint
from ..training.distributed import distributed_session
from ..training.distributed_checkpoint import load_ddp_checkpoint, save_ddp_checkpoint
from ..training.fsdp_checkpoint import load_fsdp_checkpoint, save_fsdp_checkpoint
from ..training.sharding import wrap_video_policy
from ..training.small_video_checkpoint import (
    SmallVideoIdentity,
    SmallVideoResumeState,
    _hash,
    _policy_identity,
)
from ..training.small_video_protocol import group_video_reward, group_windows
from ..training.small_video_replay import (
    NativeARTObjective,
    collect_native_video,
    native_window_items,
)
from ..training.state import CostClock, Progress
from ..training.update import accumulated_step
from ..wandb_export import execution_source, open_metrics


def _amp(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def validate_extra_evaluation(dataset, *, evaluate_only, label=None, training=None):
    """Read-only held-out identity checks; safe before any CUDA initialization."""
    if dataset is not None and not evaluate_only:
        raise ConfigError("an extra evaluation dataset requires evaluation-only mode")
    if label is not None and (
        not isinstance(label, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", label)
    ):
        raise ConfigError("evaluation label must be one safe relative path component")
    if dataset is None:
        return
    if (
        not isinstance(dataset, PromptDataset)
        or not dataset.rows
        or not isinstance(dataset.sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", dataset.sha256)
    ):
        raise ConfigError("extra evaluation requires a nonempty hashed PromptDataset")
    if training is not None:

        def normalized(text):
            return " ".join(unicodedata.normalize("NFKC", text).casefold().split())

        if {normalized(row.text) for row in dataset.rows} & {
            normalized(row.text) for row in training.rows
        }:
            raise ConfigError("extra evaluation contains normalized training prompt overlap")


def _event(root, rank, stage, progress, **values):
    with (root / f"events-rank-{rank:05d}.jsonl").open("a") as stream:
        stream.write(
            json.dumps(
                {
                    "stage": stage,
                    "time": time.time(),
                    "actual_optimizer_steps": progress.optimizer_steps,
                    **values,
                },
                allow_nan=False,
            )
            + "\n"
        )
        stream.flush()
    selected = {
        key: values[key]
        for key in ("native_position", "microbatch_start", "microbatch_stop", "window")
        if key in values
    }
    print(
        "SMALL_VIDEO_PROGRESS "
        + json.dumps(
            {
                "rank": rank,
                "stage": stage,
                "actual_optimizer_steps": progress.optimizer_steps,
                **selected,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _average(session, metrics):
    keys = sorted(metrics)
    session.agree(keys)
    values = torch.tensor(
        [metrics[key] for key in keys], dtype=torch.float64, device=session.device
    )
    if session.distributed:
        dist.all_reduce(values)
    return dict(zip(keys, (values / session.identity.world_size).cpu().tolist()))


def _checkpoint(c, model, optimizer, state, progress, root, *, resume=None):
    strategy = c.runtime.distributed
    arguments = dict(
        model=model,
        optimizer=optimizer,
        config_sha256=c.fingerprint(),
        stateful={"small_video": state},
    )
    if strategy == "none":
        save, load = local_checkpoint.save_checkpoint, local_checkpoint.load_checkpoint
    elif strategy == "ddp":
        save, load = save_ddp_checkpoint, load_ddp_checkpoint
    else:
        save, load = save_fsdp_checkpoint, load_fsdp_checkpoint
    if resume is not None:
        restored = load(resume, **arguments)
        state.validate_progress(restored)
        return restored
    state.validate_progress(progress)
    return save(root, progress=progress, keep_last=2, **arguments)



@contextmanager
def _timed_optimizer(optimizer, device):
    """Actual Adam/commit interval; no model/RNG/counter changes."""
    timing = {"start": None, "end": None}
    def before(optimizer, args, kwargs):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timing["start"] = time.monotonic()
    def after(optimizer, args, kwargs):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timing["end"] = time.monotonic()
    handles = [optimizer.register_step_pre_hook(before), optimizer.register_step_post_hook(after)]
    try:
        yield timing
    finally:
        for handle in handles:
            handle.remove()

def train_loop(
    session,
    c,
    parts,
    state,
    progress,
    validation,
    root,
    *,
    backend,
    clock,
    log,
    save,
    pause_after=None,
    resume_receipt=None,
):
    """Two windows from every complete localK8 group; no zero-signal skip path."""
    rank, world = session.identity.rank, session.identity.world_size
    local_p = 16 // world
    groups = torch.arange(local_p).repeat_interleave(8)
    state.validate_progress(progress)
    receipt = resume_receipt
    if progress.optimizer_steps == 10:
        backend.evaluate(session, c, parts, validation, 10, root)
        backend.export_final(session, c, parts, state, receipt, root)
        return {
            "status": "completed",
            "progress": progress.state_dict(),
            "checkpoint": str(receipt),
        }
    if pause_after == progress.optimizer_steps:
        return {"status": "paused", "progress": progress.state_dict(), "checkpoint": str(receipt)}
    while state.budget.completed < 10:
        with clock.training_phase():
            stage_start = time.monotonic()
            if session.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(session.device)
            rollout = state.budget.completed // 2
            progress.attempts += 1
            selected = state.selection.choose(
                backend.resolved_protocol(c)["eligible_positions"],
                flash=c.training.timestep_selection == "high_noise",
            )
            session.agree({"rollout": rollout, "selected": selected})
            prompts = session.local(
                "prompt cursor",
                lambda: state.data_cursor.take(16)[rank * local_p : (rank + 1) * local_p],
            )
            predict = session.local(
                "native text conditioning",
                lambda: backend.make_predict(
                    c, parts, [row.text for row in prompts], 8, session.device
                ),
            )
            parts.adapter.transformer.eval()
            _event(root, rank, "rollout_start", progress, rollout=rollout, selected=list(selected),
                   prompt_sample_ids=[row.sample_id for row in prompts],
                   global_prompt_groups=list(range(rank*local_p, (rank+1)*local_p)))
            noise = backend.initial_noise(
                c, local_p * 8, session.device, c.seed + rank + 1_000_003 * rollout
            )
            with _amp(session.device):
                cache = collect_native_video(
                    predict,
                    noise,
                    model=c.model.name,
                    selected=selected,
                    microbatch=c.sampling.microbatch,
                    scheduler_factory=parts.scheduler_factory,
                    progress=lambda **values: _event(
                        root, rank, "native_transition_complete", progress, **values
                    ),
                )
            _event(
                root,
                rank,
                "rollout_complete",
                progress,
                history_audit=cache.history_audit_max_error,
                elapsed_seconds=time.monotonic() - stage_start,
            )
            rollout_seconds = time.monotonic() - stage_start
            reward_start = time.monotonic()
            _event(root, rank, "decode_reward_start", progress)
            _, scores = session.local(
                "native decode/reward",
                lambda: backend.score_videos(
                    c,
                    parts,
                    cache.endpoints,
                    [row.text for row in prompts for _ in range(8)],
                    root / "rollouts" / f"rollout-{rollout:03d}-rank-{rank:05d}",
                    session.device,
                ),
            )
            scorer_metrics = dict(getattr(getattr(parts, "scorer", None), "last_metrics", {}))
            feedback = session.local(
                "complete K8 reward/covariance", lambda: group_video_reward(scores.raw, groups)
            )
            covariance = session.local(
                "frozen endpoint covariance",
                lambda: art_scalar_covariance(
                    cache.endpoints,
                    feedback["target"],
                    groups,
                    local_p,
                    **c.algorithm.options["covariance"],
                ),
            )
            active = covariance["group_active"][groups]
            anchors = (
                ~active if c.algorithm.options["restore_zero_signal"] else torch.zeros_like(active)
            )
            objective = NativeARTObjective(
                cache,
                predict,
                covariance["group_directions"][groups],
                covariance["group_guidance_magnitudes"][groups],
                anchors,
                session.device,
                contract=c.model.options["replay_contract"],
                options=c.algorithm.options["loss"],
            )
            _event(
                root,
                rank,
                "reward_complete",
                progress,
                raw_equal_mean=feedback["metrics"]["reward/raw/VQ_MQ_equal_mean"],
            )
            reward_seconds = time.monotonic() - reward_start
            parts.adapter.transformer.train()
            for window, rows in enumerate(group_windows(groups)):
                state.budget.begin_window(
                    rollout=rollout,
                    window=window,
                    rows=rows.tolist(),
                    groups=groups,
                    selected_positions=selected,
                )
                items = native_window_items(rows, len(selected))
                objective.observations.clear()
                _event(root, rank, "backward_start", progress, window=window, items=len(items))

                def replay_loss(batch):
                    losses = objective(batch)
                    _event(
                        root,
                        rank,
                        "replay_forward_complete",
                        progress,
                        window=window,
                        native_position=selected[batch[0][1]],
                        physical_video_batch=len(batch),
                    )
                    return losses

                window_start = time.monotonic()
                with _timed_optimizer(state.optimizer, session.device) as timing:
                    with _amp(session.device):
                        metrics = accumulated_step(
                            parts.adapter.transformer,
                            state.optimizer,
                            items,
                            replay_loss,
                            microbatch=c.training.microbatch,
                            denominator=len(items),
                            max_grad_norm=c.training.grad_clip,
                            require_nonzero_grad=False,
                        )
                window_end = time.monotonic()
                if timing["start"] is None or timing["end"] is None:
                    raise RuntimeError("window did not execute one actual optimizer call")
                progress.optimizer_steps = state.budget.completed
                if window == 1:
                    progress.completed_rollouts += 1
                session.agree({"actual_optimizer_steps": progress.optimizer_steps})
                diagnostics = {
                    "train/masked_fraction": sum(x["masked"] for x in objective.observations)
                    / len(items),
                    "train/kept_fraction": sum(x["kept"] for x in objective.observations)
                    / len(items),
                    "train/active_row_fraction": float(active.float().mean()),
                    "train/anchor_row_fraction": float(anchors.float().mean()),
                    "train/history_audit_max_error": cache.history_audit_max_error,
                    "cost/rollout_and_text_seconds": rollout_seconds,
                    "cost/decode_reward_covariance_seconds": reward_seconds,
                    "cost/update_window_seconds": window_end-window_start,
                    "cost/forward_backward_clip_pending_commit_seconds": timing["start"]-window_start,
                    "cost/optimizer_and_commit_seconds": timing["end"]-timing["start"],
                    "cost/update_finite_checks_seconds": window_end-timing["end"],
                    "memory/peak_allocated_bytes": (
                        torch.cuda.max_memory_allocated(session.device)
                        if session.device.type == "cuda"
                        else 0
                    ),
                    "memory/peak_reserved_bytes": (
                        torch.cuda.max_memory_reserved(session.device)
                        if session.device.type == "cuda"
                        else 0
                    ),
                }
                raw_aliases = {
                    "train/raw_VQ": feedback["metrics"]["reward/raw/VQ_mean"],
                    "train/raw_MQ": feedback["metrics"]["reward/raw/MQ_mean"],
                    "train/raw_VQ_MQ_equal_mean": feedback["metrics"][
                        "reward/raw/VQ_MQ_equal_mean"
                    ],
                }
                metrics = _average(
                    session,
                    {
                        **metrics,
                        **feedback["metrics"],
                        **raw_aliases,
                        **diagnostics,
                        **scorer_metrics,
                    },
                )
                metrics.update(clock.metrics())
                metrics.update(
                    {
                        "sampling/selected_positions": list(selected),
                        "sampling/selected_sigmas": cache.sigmas[list(selected)].tolist(),
                        "train/global_loss_denominator": len(items) * world,
                        "train/window": window,
                    }
                )
                session.primary(
                    "metrics publication",
                    lambda: (
                        log.write(json.dumps(metrics, allow_nan=False, sort_keys=True) + "\n"),
                        log.flush(),
                    ),
                )
                _event(
                    root,
                    rank,
                    "optimizer_complete",
                    progress,
                    window=window,
                    local_peak_allocated_bytes=diagnostics["memory/peak_allocated_bytes"],
                    local_peak_reserved_bytes=diagnostics["memory/peak_reserved_bytes"],
                    local_scorer_phase=scorer_metrics,
                    **metrics,
                )
            del objective, cache, noise, feedback, covariance
        if progress.optimizer_steps in (4, 10):
            receipt = save(progress)
            _event(root, rank, "checkpoint_complete", progress, receipt=str(receipt))
        if pause_after == progress.optimizer_steps:
            return {
                "status": "paused",
                "progress": progress.state_dict(),
                "checkpoint": str(receipt),
            }
    backend.evaluate(session, c, parts, validation, 10, root)
    backend.export_final(session, c, parts, state, receipt, root)
    return {"status": "completed", "progress": progress.state_dict(), "checkpoint": str(receipt)}


def _wrap(c, parts, session):
    policy = parts.adapter.trainable_module
    if c.runtime.distributed == "fsdp":
        return wrap_video_policy(
            policy,
            device=session.device,
            block_types=parts.block_types,
            activation_checkpointing=True,
        )
    policy = session.local("unsharded policy device placement", lambda: policy.to(session.device))
    if c.model.options["gradient_checkpointing"]:
        from functools import partial

        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointImpl,
            apply_activation_checkpointing,
            checkpoint_wrapper,
        )

        apply_activation_checkpointing(
            policy,
            checkpoint_wrapper_fn=partial(
                checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
            ),
            check_fn=lambda block: isinstance(block, tuple(parts.block_types)),
        )
    if c.runtime.distributed == "ddp":
        return DistributedDataParallel(
            policy,
            device_ids=[session.device.index] if session.device.type == "cuda" else None,
            broadcast_buffers=False,
        )
    return policy


def _run(
    session,
    c,
    *,
    resume,
    evaluate_only,
    backend,
    pause_after,
    evaluation_dataset=None,
    evaluation_label=None,
):
    rank, world = session.identity.rank, session.identity.world_size
    if 16 % world or (session.device.type == "cuda" and
                      (world != 8 or session.identity.local_world_size != 8)):
        raise ConfigError("GPU acceptance requires one eight-rank node with complete K8 groups")
    mode = c.model.options["finetune"]["mode"]
    if c.runtime.distributed != "none" and c.runtime.distributed != (
        "ddp" if mode == "lora" else "fsdp"
    ):
        raise ConfigError("native small-video uses LoRA DDP or full-weight FSDP")
    root = Path(c.runtime.output_dir)
    with ExitStack() as resources:

        def acquire():
            root.mkdir(parents=True, exist_ok=resume is not None)
            lock = resources.enter_context((root / ".owner.lock").open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if resume is not None:
                from .sd3 import _check_run_identity

                _check_run_identity(c, root)

        session.primary("output ownership", acquire)
        data, validation = session.local("prompt provenance", lambda: backend.load_datasets(c))
        session.local(
            "extra evaluation held-out provenance",
            lambda: validate_extra_evaluation(
                evaluation_dataset,
                evaluate_only=evaluate_only,
                label=evaluation_label,
                training=data,
            ),
        )
        if session.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(c.seed)
        random.seed(c.seed)
        np.random.seed(c.seed)
        parts = session.local(
            "native component loading", lambda: backend.load_components(c, session.device)
        )
        inventory = parameter_inventory(parts.adapter.trainable_module)
        policy_identity = _policy_identity(parts.adapter.trainable_module)
        session.agree(policy_identity)
        model = _wrap(c, parts, session)  # Collective wrappers must not be inside session.local.
        parts.adapter.transformer = model
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=c.training.learning_rate,
            betas=c.training.betas,
            eps=c.training.epsilon,
            weight_decay=c.training.weight_decay,
        )
        protocol = backend.resolved_protocol(c)
        identity = SmallVideoIdentity(
            c.model.name,
            mode,
            "flash" if c.training.timestep_selection == "high_noise" else "original",
            parts.native_code_revision,
            c.model.revision,
            c.reward.revision,
            parts.data_revision,
            data.sha256,
            _hash(protocol),
            _hash({"manifest": validation.sha256, "seed": c.evaluation.seed, "count": 8}),
            c.fingerprint(),
        )
        session.agree(identity.state_dict())
        stream = PromptStream(
            data,
            seed=c.seed if c.data.sampler_seed is None else c.data.sampler_seed,
            shuffle=c.data.shuffle,
            order_shards=c.data.order_shards,
            drop_last=c.data.drop_last,
        )
        state = SmallVideoResumeState(
            run_root=root / "step-journals" / f"rank-{rank:05d}",
            identity=identity,
            model=model,
            optimizer=optimizer,
            data_cursor=stream,
            rank=rank,
            world_size=world,
            distributed=session.distributed,
            policy_identity=policy_identity,
            read_only=evaluate_only,
        )
        resources.callback(state.close)
        progress = (
            _checkpoint(c, model, optimizer, state, None, root, resume=resume)
            if resume
            else Progress()
        )
        if evaluate_only:
            dataset = validation if evaluation_dataset is None else evaluation_dataset
            label = evaluation_label
            if evaluation_dataset is not None and label is None:
                label = f"external-{dataset.sha256[:16]}-step-{progress.optimizer_steps:09d}"
            return backend.evaluate(
                session, c, parts, dataset, progress.optimizer_steps, root, label=label
            )
        if not resume:

            def publish():
                local_checkpoint._atomic_json(root / "resolved-config.json", c.to_dict())
                local_checkpoint._atomic_json(
                    root / "resolved-protocol.json", {**protocol, "identity": identity.state_dict()}
                )
                local_checkpoint._atomic_json(
                    root / "run.json",
                    {
                        "kind": f"{c.model.name}_native_development",
                        "config_sha256": c.fingerprint(),
                        "execution_source": execution_source(),
                        "identity": identity.state_dict(),
                        "parameters": inventory,
                        "dp_world_size": world,
                        "local_prompts": 16 // world,
                        "samples_per_prompt": 8,
                        "strategy": c.runtime.distributed,
                        "cp": 1,
                        "protocol_sha256": identity.protocol_sha256,
                    },
                )

            session.primary("resolved config/protocol/run identity", publish)
            backend.evaluate(session, c, parts, validation, 0, root)
        # Evaluation owns fork_rng; rank-local generation streams start identically
        # across resume because checkpoint restoration, not reseeding, owns continuation.
        if not resume:
            torch.manual_seed(c.seed + rank)
            random.seed(c.seed + rank)
            np.random.seed(c.seed + rank)
        clock = CostClock(progress, world if session.device.type == "cuda" else 0)
        log = session.primary(
            "metrics open", lambda: resources.enter_context(open_metrics(c, root))
        )
        return train_loop(
            session,
            c,
            parts,
            state,
            progress,
            validation,
            root,
            backend=backend,
            clock=clock,
            log=log,
            pause_after=pause_after,
            resume_receipt=resume,
            save=lambda current: _checkpoint(c, model, optimizer, state, current, root),
        )


def run_native(
    c,
    *,
    resume=None,
    evaluate_only=False,
    evaluation_dataset=None,
    evaluation_label=None,
    backend=None,
):
    if backend is None:
        from . import small_video as backend
    backend.require_training_contract(c)
    validate_extra_evaluation(
        evaluation_dataset, evaluate_only=evaluate_only, label=evaluation_label
    )
    pause = os.environ.get("SMALL_VIDEO_PAUSE_AFTER_UPDATE")
    if pause not in (None, "4"):
        raise ConfigError("operational pause is supported only at complete checkpoint update4")
    if resume is None and evaluate_only:
        raise ConfigError("evaluation-only requires an explicit complete checkpoint")
    timeout = float(os.environ.get("VISUAL_RL_COLLECTIVE_TIMEOUT_SECONDS", "300"))
    with distributed_session(
        c.runtime.distributed, device=c.runtime.device, timeout_seconds=timeout
    ) as session:
        session.require_same_digest(c.execution_fingerprint())
        session.agree(
            {
                "resume": str(resume) if resume else None,
                "evaluate_only": evaluate_only,
                "pause_after": pause,
                "extra_evaluation_sha256": evaluation_dataset.sha256
                if evaluation_dataset
                else None,
                "evaluation_label": evaluation_label,
            }
        )
        try:
            result = _run(
                session,
                c,
                resume=resume,
                evaluate_only=evaluate_only,
                backend=backend,
                pause_after=4 if pause else None,
                evaluation_dataset=evaluation_dataset,
                evaluation_label=evaluation_label,
            )
            if not evaluate_only:
                session.primary(
                    "completion publication",
                    lambda: local_checkpoint._atomic_json(
                        Path(c.runtime.output_dir) / "status.json", result
                    ),
                )
            return result
        except Exception:
            # A model peer can be inside a collective. Never enter a new collective
            # to report failure; preserve this rank's traceback for the supervisor.
            root = Path(c.runtime.output_dir)
            if root.is_dir():
                (root / f"failure-rank-{session.identity.rank:05d}.txt").write_text(
                    traceback.format_exc()
                )
            raise
