"""FSDP1 video policies with homogeneous flat parameter groups."""

from functools import partial

import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp._runtime_utils import _reshard
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


def reshard_local_parameters(model):
    # A no-grad root forward can retain a gathered copy. Invalidate it before
    # modifying local shards so the next forward observes the restored policy.
    for module in FSDP.fsdp_modules(model):
        handle = module._handle
        if handle is not None and hasattr(handle.flat_param, "_local_shard"):
            _reshard(module, handle, True)


def wrap_video_policy(model, *, device, block_types, activation_checkpointing, lora=False):
    """Wrap before constructing AdamW. BF16 forward autocast belongs to caller.

    Every rank loads the same pinned weights. CPU device support is for small
    collective integration tests, not a large-model training recommendation.
    """
    if not block_types or not all(isinstance(kind, type) for kind in block_types):
        raise ValueError("explicit native transformer block classes required")
    if lora:
        # PEFT stores each A/B projection in a callable Linear leaf. Wrap each
        # separately so native block handles retain only frozen BF16 tensors.
        leaves = set()
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and any(
                key in name.split(".") for key in ("lora_A", "lora_B")
            ):
                leaves.add(module)
        trainable = {p for p in model.parameters() if p.requires_grad}
        frozen = {p for p in model.parameters() if not p.requires_grad}
        if (not trainable or not frozen or trainable != {p for m in leaves for p in m.parameters()}
                or any(p.dtype != torch.float32 for p in trainable)
                or any(p.dtype != torch.bfloat16 for p in frozen)):
            raise ValueError("LoRA FSDP requires FP32 PEFT A/B projections and a frozen BF16 base")

        def policy(module, recurse, nonwrapped_numel):
            return recurse or module in leaves or isinstance(module, tuple(block_types))

        precision = MixedPrecision()  # Preserve each handle's dtype under forward autocast.
    elif any(not p.requires_grad or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("video FSDP requires full-weight FP32 parameters")
    else:
        policy = partial(transformer_auto_wrap_policy, transformer_layer_cls=set(block_types))
        precision = MixedPrecision(param_dtype=torch.float32, reduce_dtype=torch.float32,
                                   buffer_dtype=torch.float32, cast_forward_inputs=False)
    if hasattr(model, "gradient_checkpointing"):
        model.gradient_checkpointing = False
    wrapped = FSDP(
        model, device_id=device, sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=policy, mixed_precision=precision,
        use_orig_params=False, limit_all_gathers=True,
    )
    if activation_checkpointing:
        apply_activation_checkpointing(
            wrapped,
            checkpoint_wrapper_fn=partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
            check_fn=lambda module: isinstance(module, tuple(block_types)),
        )
    if lora:
        wrapped._visual_rl_lora_shards = True
    return wrapped
