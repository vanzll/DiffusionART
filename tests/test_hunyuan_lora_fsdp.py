"""Two-rank PEFT projection wrapping, adapter-only recovery and NFT EMA."""

import time
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_distributed_checkpoint import assert_tree_equal

from visual_rl.models.finetune import FineTuneConfig, prepare_policy
from visual_rl.training.fsdp_checkpoint import (
    DistributedCheckpointError,
    load_fsdp_checkpoint,
    save_fsdp_checkpoint,
)
from visual_rl.training.sharding import wrap_video_policy
from visual_rl.training.state import Progress
from visual_rl.training.update import accumulated_step
from visual_rl.training.video_nft import VideoNFTState


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(5, 5)

    def forward(self, x):
        return self.proj(x).tanh()


def _components():
    torch.manual_seed(12)
    base = torch.nn.Sequential(Block(), Block(), torch.nn.Linear(5, 2)).bfloat16()
    policy = prepare_policy(base, FineTuneConfig("lora", rank=3, alpha=3, target_modules=["proj"]))
    model = wrap_video_policy(policy, device=torch.device("cpu"), block_types=(Block,),
                              activation_checkpointing=True, lora=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable) == 4 and all(p.dtype == torch.float32 for p in trainable)
    frozen = [p for p in model.parameters() if not p.requires_grad]
    assert frozen and all(p.dtype == torch.bfloat16 for p in frozen)
    optimizer = torch.optim.AdamW(trainable, lr=.1)
    nft = VideoNFTState(model, {"old_policy": {"mode": "slow-ema", "rate": .25, "maximum": .9}})
    return model, optimizer, nft


def _step(model, optimizer, nft, step):
    x = torch.randn(4, 5)
    def loss(items):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = model(x[list(items)])
        return output.float().square().mean(1)

    metrics = accumulated_step(model, optimizer, range(4), loss, microbatch=2, denominator=4,
                               max_grad_norm=1, require_nonzero_grad=True)
    nft.after_update(step)
    return metrics


@torch.no_grad()
def _predict(model):
    with torch.autocast("cpu", dtype=torch.bfloat16):
        return model(torch.ones(2, 5)).clone()


def _checkpoint_args(model, optimizer, nft):
    return dict(model=model, optimizer=optimizer, stateful=nft.stateful(), config_sha256="a" * 64)


def _worker(rank, root):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=2, init_method=f"file://{root}/rdzv",
                            timeout=timedelta(seconds=30))
    try:
        model, optimizer, nft = _components()
        torch.manual_seed(100 + rank)
        for step in (1, 2):
            _step(model, optimizer, nft, step)
        current = _predict(model)
        with nft.use_old():
            previous = _predict(model)
        assert not torch.equal(current, previous)
        assert torch.equal(current, _predict(model))
        progress = Progress(optimizer_steps=2)
        receipt = save_fsdp_checkpoint(root / "checkpoints", progress=progress,
                                       **_checkpoint_args(model, optimizer, nft))
        payload = torch.load(receipt.parent / f"rank-{rank:05d}.pt", weights_only=True)
        names = {name for name, p in model.named_parameters() if p.requires_grad}
        assert names == set(payload["model"]["trainable"]) == set(nft.old.shadow)
        assert set(payload["model"]) == {"trainable", "frozen_sha256"}
        assert sum(p.numel() for p in nft.old.shadow.values()) == 4 * 8  # ceil(15/2) per leaf.
        expected = _step(model, optimizer, nft, 3)

        recovered, adam, old = _components()
        args = _checkpoint_args(recovered, adam, old)
        assert load_fsdp_checkpoint(receipt, **args) == progress
        assert torch.equal(_predict(recovered), current)
        with old.use_old():
            assert torch.equal(_predict(recovered), previous)
        assert _step(recovered, adam, old, 3) == expected
        assert_tree_equal(dict(recovered.named_parameters()), dict(model.named_parameters()))
        assert_tree_equal(adam.state_dict(), optimizer.state_dict())
        assert_tree_equal(old.old.state_dict(), nft.old.state_dict())

        with torch.no_grad():
            if rank == 0:
                next(p for p in recovered.parameters() if not p.requires_grad).add_(1)
        with pytest.raises(DistributedCheckpointError, match="model validation"):
            load_fsdp_checkpoint(receipt, **args)
    finally:
        dist.destroy_process_group()


def test_sharded_lora_nft_exact_resume(tmp_path):
    pytest.importorskip("peft")
    context = mp.spawn(_worker, args=(tmp_path,), nprocs=2, join=False)
    deadline = time.monotonic() + 100
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("LoRA FSDP regression exceeded 100 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
