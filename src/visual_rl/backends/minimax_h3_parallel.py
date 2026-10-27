"""H3-only packed-sequence CP with explicit logical data-parallel ownership.

The output gather has SUM backward: every CP peer evaluates the same full loss.
FSDP averages over all physical ranks, canceling that duplication exactly once.
K/V gathers sum contributions from disjoint query shards; they add no loss scale.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist

from ..config import ConfigError


class _GatherSequence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, group, full_length):
        size, rank = dist.get_world_size(group), dist.get_rank(group)
        chunk = (full_length + size - 1) // size
        lengths = [max(0, min(chunk, full_length - i * chunk)) for i in range(size)]
        if tensor.shape[1] != lengths[rank]:
            raise ValueError("H3 CP local sequence disagrees with global layout")
        ctx.group, ctx.start, ctx.length = group, rank * chunk, lengths[rank]
        padded = tensor.new_zeros((tensor.shape[0], chunk, *tensor.shape[2:]))
        padded[:, :tensor.shape[1]].copy_(tensor)
        buffers = [torch.empty_like(padded) for _ in range(size)]
        dist.all_gather(buffers, padded.contiguous(), group=group)
        return torch.cat([value[:, :length] for value, length in zip(buffers, lengths)], dim=1)

    @staticmethod
    def backward(ctx, gradient):
        # CP query shards contribute to common K/V; output heads receive the
        # duplicated complete-loss gradient. Both require an actual SUM here.
        total = gradient.contiguous().clone()
        dist.all_reduce(total, op=dist.ReduceOp.SUM, group=ctx.group)
        return total.narrow(1, ctx.start, ctx.length).contiguous(), None, None


@dataclass
class H3Parallel:
    degree: int
    physical_rank: int
    physical_world: int
    cp_group: object = None
    dp_group: object = None

    @property
    def dp_rank(self):
        return self.physical_rank // self.degree

    @property
    def dp_world(self):
        return self.physical_world // self.degree

    @property
    def cp_rank(self):
        return self.physical_rank % self.degree

    def local_prompt_count(self, total):
        if total < self.dp_world or total % self.dp_world:
            raise ConfigError("H3 prompts must divide logical DP ranks, not physical CP workers")
        return total // self.dp_world

    @classmethod
    def from_config(cls, config, identity):
        degree = config.model.options.get("context_parallel_size", 1)
        if type(degree) is not int or degree not in (1, 2) or identity.world_size % degree:
            raise ConfigError("H3 context_parallel_size requires degree1/2 dividing physical world")
        if degree > 1 and (config.runtime.distributed != "fsdp" or config.algorithm.name != "diffusionart"
                           or config.model.options["finetune"]["mode"] != "full"):
            raise ConfigError("H3 CP2 currently requires full ART with physical-world FSDP")
        return cls(degree, identity.rank, identity.world_size)

    def initialize(self):
        if self.degree == 1:
            return self
        for position in range(self.degree):
            ranks = list(range(position, self.physical_world, self.degree))
            group = dist.new_group(ranks)
            if self.physical_rank in ranks:
                self.dp_group = group
        for start in range(0, self.physical_world, self.degree):
            ranks = list(range(start, start + self.degree))
            group = dist.new_group(ranks)
            if self.physical_rank in ranks:
                self.cp_group = group
        return self

    def bounds(self, full_length):
        chunk = (full_length + self.degree - 1) // self.degree
        start = self.cp_rank * chunk
        return start, max(0, min(chunk, full_length - start))

    def split(self, tensor, full_length):
        start, length = self.bounds(full_length)
        return tensor.narrow(1, start, length).contiguous()

    def gather(self, tensor, full_length):
        if self.degree == 1:
            return tensor
        return _GatherSequence.apply(tensor, self.cp_group, full_length)

    def identity(self):
        return {"physical_world_size": self.physical_world, "dp_world_size": self.dp_world,
                "context_parallel_size": self.degree, "gradient_rule": "CP-SUM/FSDP-physical-average"}


def enable_context_parallel(model, parallel):
    """Only mutate the selected H3 instance; keep parameter/state names intact."""
    if parallel.degree == 1:
        return
    from ..models.minimax_h3_batch import MiniMaxH3Transformer3DModel
    if not isinstance(model, MiniMaxH3Transformer3DModel):
        raise TypeError("H3 CP requires the batched native adapter")
    model._h3_context_parallel = parallel
    for module in model.transformer_blocks:
        module.attn.processor._h3_context_parallel = parallel

