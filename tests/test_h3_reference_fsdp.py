"""Real two-rank local-shard reference save/restore and collective inference."""
import time
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from visual_rl.backends.minimax_h3_reference import ReferenceState
from visual_rl.training.fsdp_checkpoint import load_fsdp_checkpoint, save_fsdp_checkpoint
from visual_rl.training.fsdp_shadow import FSDPPolicyShadow
from visual_rl.training.state import Progress


def make_model():
    return FSDP(torch.nn.Linear(4, 4, bias=False), device_id=torch.device('cpu'), use_orig_params=False)


def worker(rank, root):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', rank=rank, world_size=2,
        init_method=f'file://{root}/rendezvous', timeout=timedelta(seconds=25))
    try:
        torch.manual_seed(21)
        model = make_model()
        x = torch.tensor([[.1, .2, .3, .4]])
        with torch.no_grad():
            initial = model(x).clone()
        reference = ReferenceState(model)
        assert isinstance(reference._shadow, FSDPPolicyShadow)
        assert sum(p.numel() for p in reference._shadow.shadow.values()) == 8
        assert all(p.device.type == 'cpu' for p in reference._shadow.shadow.values())
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        optimizer.zero_grad(set_to_none=True)
        model(x).square().mean().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            current = model(x).clone()
        assert not torch.equal(current, initial)
        with reference.use_reference():
            torch.testing.assert_close(model(x), initial, rtol=0, atol=0)
        with torch.no_grad():
            torch.testing.assert_close(model(x), current, rtol=0, atol=0)
        receipt = save_fsdp_checkpoint(root + '/checkpoints', model=model,
            optimizer=optimizer, progress=Progress(optimizer_steps=2), config_sha256='b' * 64,
            stateful=reference.stateful())
        # A changed model seed cannot redefine the pretrained reference on resume:
        # stateful restoration must recover the saved immutable initial shards.
        torch.manual_seed(32)
        resumed = make_model()
        restored_reference = ReferenceState(resumed)
        resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=.01)
        restored = load_fsdp_checkpoint(receipt, model=resumed, optimizer=resumed_optimizer,
            config_sha256='b' * 64, stateful=restored_reference.stateful())
        assert restored.optimizer_steps == 2
        with restored_reference.use_reference():
            torch.testing.assert_close(resumed(x), initial, rtol=0, atol=0)
        with torch.no_grad():
            torch.testing.assert_close(resumed(x), current, rtol=0, atol=0)
        # Continue training, then prove the same frozen initial target survives.
        resumed(x).square().mean().backward()
        resumed_optimizer.step()
        resumed_optimizer.zero_grad(set_to_none=True)
        with restored_reference.use_reference():
            torch.testing.assert_close(resumed(x), initial, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_two_rank_frozen_reference_checkpoint(tmp_path):
    context = mp.spawn(worker, args=(str(tmp_path),), nprocs=2, join=False)
    deadline = time.monotonic() + 75
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail('H3 frozen-reference regression exceeded75s')
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
