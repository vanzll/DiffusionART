"""Actual two-rank Gloo evidence for global raw std and NFT U1 scaling."""
import time
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from visual_rl.training.audio_video import AVNFTState, run_av_updates
from visual_rl.training.audio_video_objectives import prepare_nft_objective
from visual_rl.training.audio_video_types import AVState


class DualLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.video = torch.nn.Parameter(torch.tensor(.1))
        self.audio = torch.nn.Parameter(torch.tensor(.2))

    def forward(self, state, sv, sa, rows):
        return AVState(state.video * self.video, state.audio * self.audio)


def case(model, state, rank=None, global_rewards=None):
    g = torch.Generator().manual_seed(7351)
    endpoint = AVState(torch.randn(8, 2, 3, generator=g), torch.randn(8, 3, generator=g))
    noise = AVState(torch.randn(8, 2, 2, 3, generator=g), torch.randn(8, 2, 3, generator=g))
    groups = torch.arange(8) // 4
    rewards = torch.tensor([0., 1., 2., 4., 100., 101., 103., 106.])
    rows = torch.arange(8) if rank is None else torch.arange(rank * 4, rank * 4 + 4)
    def old_predict(*args):
        with state.use_old():
            return state.old.model(*args)
    return prepare_nft_objective(endpoint.select(rows), model, old_predict, rewards[rows], rewards.flip(0)[rows],
        groups[rows], torch.tensor([.8, .4]), noise.select(rows), "cpu", microbatch=3,
        global_video_rewards=None if global_rewards is None else global_rewards[0],
        global_audio_rewards=None if global_rewards is None else global_rewards[1]), groups[rows]


def worker(rank, root):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{root}/rendezvous", rank=rank, world_size=2,
                            timeout=timedelta(seconds=15))
    try:
        model = DDP(DualLinear())
        state = AVNFTState(model.module)
        rewards = torch.tensor([0., 1., 2., 4., 100., 101., 103., 106.])
        populations = []
        for population in (rewards, rewards.flip(0)):
            gathered = [torch.empty(4) for _ in range(2)]
            dist.all_gather(gathered, population[rank * 4:rank * 4 + 4].contiguous())
            populations.append(torch.cat(gathered))
        obj, groups = case(model, state, rank, populations)
        optimizer = torch.optim.SGD(model.parameters(), lr=.001)
        steps = []
        optimizer.register_step_post_hook(lambda *args: steps.append(1))
        metrics = run_av_updates(model, optimizer, obj, groups, algorithm="nft", microbatch=3,
            max_grad_norm=10, state=state)
        torch.save({"weights": model.module.state_dict(), "metrics": metrics, "steps": len(steps),
                    "refreshes": state.successful_updates}, Path(root) / f"rank-{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_two_rank_nft_matches_full_batch_manual_mean_and_global_std(tmp_path):
    model = DualLinear()
    state = AVNFTState(model)
    obj, _ = case(model, state)
    optimizer = torch.optim.SGD(model.parameters(), lr=.001)
    # Independent reduction: no AV step owner or accumulated_step can repeat
    # a denominator bug in this full-batch PyTorch reference.
    loss = obj(obj.items).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
    optimizer.step()
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
    expected_modality = dict(obj.diagnostics)
    ctx = mp.spawn(worker, args=(str(tmp_path),), nprocs=2, join=False)
    deadline = time.monotonic() + 45
    try:
        while not ctx.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("AV two-rank step exceeded 45 seconds")
    finally:
        for process in ctx.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    for rank in range(2):
        result = torch.load(tmp_path / f"rank-{rank}.pt", weights_only=True)
        assert result["steps"] == result["refreshes"] == 1
        for name, value in expected.items():
            torch.testing.assert_close(result["weights"][name], value, rtol=1e-5, atol=1e-7)
        metrics = result["metrics"]
        assert metrics["train/loss"] == pytest.approx(loss.item(), rel=1e-6)
        for name, value in expected_modality.items():
            assert metrics[name] == pytest.approx(value, rel=1e-6)
        assert metrics["train/loss"] == pytest.approx(.5 * (metrics["av/video_loss"] + metrics["av/audio_loss"]), rel=1e-6)
