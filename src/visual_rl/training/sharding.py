"""Full-weight video FSDP policy matching the native FP32-master recipe."""

from functools import partial

import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


def wrap_video_policy(model, *, device, block_types, activation_checkpointing):
    """Wrap before constructing AdamW. BF16 forward autocast belongs to caller.

    Every rank loads the same pinned weights. CPU device support is for small
    collective integration tests, not a large-model training recommendation.
    """
    if not block_types or not all(isinstance(kind, type) for kind in block_types):
        raise ValueError("explicit native transformer block classes required")
    if any(not p.requires_grad or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("video FSDP requires full-weight FP32 parameters")
    if hasattr(model, "gradient_checkpointing"):
        model.gradient_checkpointing = False
    wrapped = FSDP(
        model, device_id=device, sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls=set(block_types)),
        mixed_precision=MixedPrecision(param_dtype=torch.float32, reduce_dtype=torch.float32,
                                       buffer_dtype=torch.float32, cast_forward_inputs=False),
        use_orig_params=False, limit_all_gathers=True,
    )
    if activation_checkpointing:
        apply_activation_checkpointing(
            wrapped,
            checkpoint_wrapper_fn=partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
            check_fn=lambda module: isinstance(module, tuple(block_types)),
        )
    return wrapped
