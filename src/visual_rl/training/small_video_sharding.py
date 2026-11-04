"""LTX native BF16 compute with FP32 full-weight optimizer masters.

The existing video wrapper deliberately keeps FP32 compute parameters and
relies on global AMP. LTX's pinned native pipeline instead uses BF16 weights
without global AMP. Keep this policy local to the new LTX backend.
"""

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


def wrap_ltx_native_full_policy(model, *, device, block_types, activation_checkpointing):
    """Use native BF16 parameters during forward, retaining FP32 Adam state.

    Do not cast all forward arguments: physical times and position coordinates
    must retain their native FP32 precision. The adapter casts only the latent
    and text tensors to BF16 and disables the caller's outer autocast context.
    FSDP owns parameter casting and checkpoint recomputation, rather than a
    temporary functional parameter substitution.
    """
    if not block_types or not all(isinstance(kind, type) for kind in block_types):
        raise ValueError("explicit native LTX block classes required")
    parameters = list(model.parameters())
    if not parameters or any(not p.requires_grad or p.dtype != torch.float32 for p in parameters):
        raise ValueError("native LTX full FSDP requires materialized FP32 trainable masters")
    if any(p.is_meta for p in parameters):
        raise ValueError("native LTX full FSDP requires materialized FP32 trainable masters")
    if hasattr(model, "gradient_checkpointing"):
        model.gradient_checkpointing = False
    wrapped = FSDP(
        model,
        device_id=device,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=partial(
            transformer_auto_wrap_policy, transformer_layer_cls=set(block_types)
        ),
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,
            cast_forward_inputs=False,
            cast_root_forward_inputs=False,
        ),
        use_orig_params=False,
        limit_all_gathers=True,
    )
    if activation_checkpointing:
        apply_activation_checkpointing(
            wrapped,
            checkpoint_wrapper_fn=partial(
                checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
            ),
            check_fn=lambda module: isinstance(module, tuple(block_types)),
        )
    return wrapped
