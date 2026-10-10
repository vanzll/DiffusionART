"""Fixed-denominator accumulation for unsharded, DDP and full-shard FSDP models."""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Callable, Sequence

import torch
from torch.distributed.fsdp import FullyShardedDataParallel, ShardingStrategy
from torch.nn.parallel import DistributedDataParallel


class TerminalStepError(RuntimeError):
    """Unsafe to retry or coordinate another collective on this process group."""


def _distributed():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _all_ok(ok: bool, device: torch.device, stage: str, group=None):
    flag = torch.tensor(int(ok), dtype=torch.int32, device=device)
    if _distributed():
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN, group=group)
    if not flag.item():
        raise RuntimeError(f"at least one rank failed {stage}; optimizer was not advanced")


def _check_scaler(scaler, max_scale_retries, device, group):
    _all_ok(
        isinstance(scaler, torch.amp.GradScaler) and scaler.is_enabled(),
        device, "enabled GradScaler validation", group,
    )
    _all_ok(
        type(max_scale_retries) is int and 0 <= max_scale_retries <= 16,
        device, "max_scale_retries validation (integer in [0, 16])", group,
    )
    state = scaler.state_dict()
    _all_ok(
        math.isfinite(state["scale"]) and state["scale"] > 0,
        device, "positive finite scaler scale validation", group,
    )
    if _distributed():
        # Include growth settings and the retry budget so ranks cannot diverge
        # after a shared overflow or successful update. Use the DDP subgroup.
        values = torch.tensor([
            state["scale"], state["growth_factor"], state["backoff_factor"],
            state["growth_interval"], state["_growth_tracker"], max_scale_retries,
        ], dtype=torch.float64, device=device)
        low, high = values.clone(), values.clone()
        torch.distributed.all_reduce(low, op=torch.distributed.ReduceOp.MIN, group=group)
        torch.distributed.all_reduce(high, op=torch.distributed.ReduceOp.MAX, group=group)
        if not torch.equal(low, high):
            raise RuntimeError("ranks require identical scaler state and max_scale_retries at entry")


def accumulated_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    items: Sequence,
    loss_fn: Callable[[Sequence], torch.Tensor],
    *,
    microbatch: int,
    denominator: float,
    max_grad_norm: float,
    validate_batch: Callable[[Sequence], None] | None = None,
    require_nonzero_grad: bool = False,
    scaler: torch.amp.GradScaler | None = None,
    max_scale_retries: int = 16,
) -> dict[str, float]:
    """Backpropagate per-item losses, dividing by the ORIGINAL window weight.

    ``denominator`` is local; DDP sums it globally and corrects DDP's gradient
    averaging (also used by full-shard FSDP). Masked items remain in this denominator. If rows are padded,
    callers set their losses to zero and explicitly exclude padding weight.
    Callers own autocast and must return FP32/FP64 losses. An enabled GradScaler
    is supported for unsharded/DDP models only. FSDP reduces every microbatch and clips
    using its global sharded norm; it never retains full gradients with no_sync.
    ``validate_batch`` is local-only preparation, coordinated BEFORE model
    collectives. Arbitrary model exceptions are terminal, never sent through
    another collective. The caller must use a finite process-group timeout and
    a rank supervisor (such as torchrun) that tears down peers on rank failure.
    Each rank's loss_fn must invoke the same ordered model-collective sequence;
    it cannot skip a model forward conditionally for masked or padding rows.

    Scaling must be enabled uniformly across DDP ranks with identical scaler
    state. Gradient overflow replays the SAME window, at most max_scale_retries
    times (0..16), restoring torch CPU and the model device's CUDA RNG before
    each retry. Scaled models must reside on one device. loss_fn and
    validate_batch must be replay-safe: diagnostics may repeat, but parameters,
    buffers, and other mutable training state must not change. Other RNGs and
    explicit generators are not restored. Only successful attempts contribute
    returned losses; callers must update EMA only after this function returns.
    Scaled-only telemetry reports overflow_retries, the successful loss_scale,
    and next_loss_scale after the scaler growth update, under the train/ prefix.
    """
    if scaler is not None:
        optimizer.zero_grad(set_to_none=True)
    if type(microbatch) is not int or microbatch < 1:
        raise ValueError("microbatch must be a positive integer")
    if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive and finite")
    if type(require_nonzero_grad) is not bool:
        raise ValueError("require_nonzero_grad must be boolean")
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("no trainable parameters")
    device = parameters[0].device
    ddp = isinstance(model, DistributedDataParallel)
    fsdp = isinstance(model, FullyShardedDataParallel)
    group = model.process_group if ddp or fsdp else None
    if scaler is not None and any(
        isinstance(child, FullyShardedDataParallel) for child in model.modules()
    ):
        raise ValueError("FSDP with a GradScaler is not supported")
    if _distributed() and not (ddp or fsdp):
        raise ValueError("distributed accumulation requires a DDP or FSDP model")
    if fsdp and (model.sharding_strategy not in (ShardingStrategy.FULL_SHARD, ShardingStrategy.NO_SHARD)
                 or model.cpu_offload.offload_params):
        raise ValueError("FSDP accumulation requires full sharding without CPU parameter offload")
    if ddp and any(isinstance(child, FullyShardedDataParallel) for child in model.modules()):
        raise ValueError("mixed DDP/FSDP wrapping is not supported")
    _all_ok(
        bool(items) and math.isfinite(denominator) and denominator >= 0, device, "window validation", group
    )
    counts = torch.tensor([math.ceil(len(items) / microbatch)] * 2, device=device)
    total = torch.tensor(float(denominator), dtype=torch.float64, device=device)
    world = torch.distributed.get_world_size(group) if _distributed() else 1
    if _distributed():
        torch.distributed.all_reduce(counts[:1], op=torch.distributed.ReduceOp.MIN, group=group)
        torch.distributed.all_reduce(counts[1:], op=torch.distributed.ReduceOp.MAX, group=group)
        torch.distributed.all_reduce(total, group=group)
    if counts[0] != counts[1] or total.item() <= 0:
        raise ValueError("ranks need equal microbatch counts and a positive global denominator")
    scale = world / total.item()
    optimizer.zero_grad(set_to_none=True)
    try:
        if scaler is not None:
            _check_scaler(scaler, max_scale_retries, device, group)
            _all_ok(all(p.device == device for p in parameters), device,
                    "single-device scaled model validation", group)
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
        retries = 0
        while True:
            loss_sum = 0.0
            for start in range(0, len(items), microbatch):
                batch = items[start : start + microbatch]
                final = start + microbatch >= len(items)
                preparation_error = None
                if validate_batch is not None:
                    try:
                        validate_batch(batch)
                    except Exception as exc:
                        preparation_error = exc
                try:
                    _all_ok(preparation_error is None, device, "input preparation", group)
                except RuntimeError as exc:
                    raise exc from preparation_error
                sync = model.no_sync() if ddp and not final else nullcontext()
                with sync:
                    try:
                        losses = loss_fn(batch)
                    except Exception as exc:
                        # A peer may already be inside DDP's buffer broadcast or
                        # reducer. Another collective here could deadlock it.
                        raise TerminalStepError(
                            "model/loss execution failed; terminate peer ranks, do not retry this process group"
                        ) from exc
                    valid = (
                        isinstance(losses, torch.Tensor)
                        and losses.shape == (len(batch),)
                        and losses.dtype in (torch.float32, torch.float64)
                        and losses.requires_grad
                        and bool(torch.isfinite(losses).all())
                    )
                    _all_ok(valid, device, "loss validation", group)
                    loss_sum += float(losses.detach().double().sum())
                    try:
                        loss = losses.sum() * scale
                        (loss if scaler is None else scaler.scale(loss)).backward()
                    except Exception as exc:
                        raise TerminalStepError(
                            "backward failed; terminate peer ranks, do not retry this process group"
                        ) from exc
            if scaler is not None:
                scaler.unscale_(optimizer)
            finite = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in parameters)
            if scaler is None:
                _all_ok(finite, device, "gradient validation", group)
            else:
                flag = torch.tensor(int(finite), dtype=torch.int32, device=device)
                if _distributed():
                    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN, group=group)
                if not flag.item():
                    # Explicit update clears unscale_'s per-optimizer state even
                    # on ranks whose local gradients were finite. The public
                    # checkpoint API resets growth after a coordinated overflow.
                    scaler.update(new_scale=scaler.get_scale() * scaler.get_backoff_factor())
                    state = scaler.state_dict()
                    state["_growth_tracker"] = 0
                    scaler.load_state_dict(state)
                    if retries >= max_scale_retries:
                        raise RuntimeError(
                            f"scaled gradient overflow exhausted {max_scale_retries} retries; "
                            "optimizer was not advanced"
                        )
                    if not math.isfinite(scaler.get_scale()) or scaler.get_scale() <= 0:
                        raise RuntimeError("scaled gradient overflow produced an invalid loss scale")
                    retries += 1
                    optimizer.zero_grad(set_to_none=True)
                    torch.set_rng_state(cpu_rng)
                    if cuda_rng is not None:
                        torch.cuda.set_rng_state(cuda_rng, device)
                    continue
            before = (model.clip_grad_norm_(max_grad_norm) if fsdp else
                      torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm, error_if_nonfinite=True))
            if fsdp:
                _all_ok(bool(torch.isfinite(before)), device, "gradient norm validation", group)
            if require_nonzero_grad:
                _all_ok(bool(before > 0), device, "nonzero policy gradient", group)
            if scaler is None:
                optimizer.step()
            else:
                loss_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
            break
    finally:
        optimizer.zero_grad(set_to_none=True)
    # Optimizer exceptions or nonfinite post-step state are terminal: do not
    # attempt a partial rollback or relabel them as skipped media rollouts.
    if not all(bool(torch.isfinite(p).all()) for p in parameters):
        raise RuntimeError("optimizer produced nonfinite parameters; state is not resumable here")
    loss_value = torch.tensor(loss_sum, device=device, dtype=torch.float64)
    if _distributed():
        torch.distributed.all_reduce(loss_value, group=group)
    metrics = {
        "train/loss": loss_value.item() / total.item(),
        "train/grad_norm/before_clip": float(before),
        "train/grad_clip_triggered": float(before > max_grad_norm),
    }
    if scaler is not None:
        metrics.update({
            "train/overflow_retries": float(retries),
            "train/loss_scale": loss_scale,
            "train/next_loss_scale": scaler.get_scale(),
        })
    return metrics
