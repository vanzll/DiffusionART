import json
import time
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from visual_rl.training.sharding import wrap_video_policy
from visual_rl.training.update import accumulated_step


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(3, 3)

    def forward(self, value):
        return self.proj(value).tanh()


def network():
    return torch.nn.Sequential(Block(), Block(), torch.nn.Linear(3, 2))


def fsdp_worker(rank, init_file, output, checkpointing):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2,
                            timeout=timedelta(seconds=20))
    try:
        torch.manual_seed(91)
        reference = network()
        model = network()
        model.load_state_dict(reference.state_dict())
        wrapped = wrap_video_policy(model, device=torch.device("cpu"), block_types=(Block,),
                                    activation_checkpointing=checkpointing)
        def no_sync():
            raise AssertionError("FSDP must reduce-scatter every microbatch")
        wrapped.no_sync = no_sync
        optimizer = torch.optim.AdamW(wrapped.parameters(), lr=.001)
        control = torch.optim.AdamW(reference.parameters(), lr=.001)
        x, y = torch.randn(7, 3), torch.randn(7, 2)
        indices = [0, 1, 2, 3] if rank == 0 else [4, 5, 6, 0]
        for _ in range(2):
            control.zero_grad(set_to_none=True)
            full_loss = (reference(x) - y).square().mean(1).mean()
            full_loss.backward()
            expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), .1)
            control.step()
            def loss(items):
                result = (wrapped(x[[indices[i] for i in items]]) - y[[indices[i] for i in items]]).square().mean(1)
                return result * result.new_tensor([0 if rank == 1 and i == 3 else 1 for i in items])
            metrics = accumulated_step(wrapped, optimizer, list(range(4)), loss, microbatch=2,
                denominator=4 if rank == 0 else 3, max_grad_norm=.1)
            assert metrics["train/loss"] == pytest.approx(full_loss.item(), abs=2e-6)
            assert metrics["train/grad_norm/before_clip"] == pytest.approx(expected_norm.item(), abs=2e-6)
            with FSDP.summon_full_params(wrapped):
                for expected, actual in zip(reference.parameters(), wrapped.parameters()):
                    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
        if rank == 0:
            from pathlib import Path
            Path(output).write_text(json.dumps({"updates": 2, "checkpointing": checkpointing}))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("checkpointing", [False, True])
def test_fsdp_global_gradient_and_per_microbatch_reduction(tmp_path, checkpointing):
    context = mp.spawn(fsdp_worker, args=(str(tmp_path / "rendezvous"), str(tmp_path / "result.json"), checkpointing),
                       nprocs=2, join=False)
    deadline = time.monotonic() + 60
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("FSDP update test exceeded 60 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    assert json.loads((tmp_path / "result.json").read_text())["updates"] == 2
