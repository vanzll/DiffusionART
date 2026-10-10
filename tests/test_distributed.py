import json
import time
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from visual_rl.training.update import accumulated_step


def worker(rank, init_file, output, invalid):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    try:
        torch.manual_seed(44)
        model = torch.nn.Linear(3, 2).double()
        model = torch.nn.parallel.DistributedDataParallel(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        all_x, all_y = torch.randn(7, 3).double(), torch.randn(7, 2).double()
        indices = [0, 1, 2, 3] if rank == 0 else [4, 5, 6, 0]

        def loss(chunk):
            rows = [indices[i] for i in chunk]
            value = (model(all_x[rows]) - all_y[rows]).square().mean(1)
            weights = value.new_tensor([0.0 if rank == 1 and i == 3 else 1.0 for i in chunk])
            if invalid and rank == 1 and 3 in chunk:
                value = value * float("nan")
            return value * weights

        try:
            metrics = accumulated_step(
                model,
                optimizer,
                list(range(4)),
                loss,
                microbatch=2,
                denominator=4 if rank == 0 else 3,
                max_grad_norm=1.0,
            )
        except RuntimeError as exc:
            if not invalid or "loss validation" not in str(exc):
                raise
            assert not optimizer.state
            assert all(p.grad is None for p in model.parameters())
        else:
            assert not invalid
            if rank == 0:
                torch.save({"model": model.module.state_dict(), "metrics": metrics}, output)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("invalid", [False, True])
def test_real_two_rank_accumulation_and_coordinated_loss_failure(tmp_path, invalid):
    ctx = mp.spawn(
        worker,
        args=(str(tmp_path / "rendezvous"), str(tmp_path / "result.pt"), invalid),
        nprocs=2,
        join=False,
    )
    deadline = time.monotonic() + 60
    try:
        while not ctx.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("distributed regression exceeded 60s")
    finally:
        for process in ctx.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    if invalid:
        return
    torch.manual_seed(44)
    model = torch.nn.Linear(3, 2).double()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    x, y = torch.randn(7, 3).double(), torch.randn(7, 2).double()
    metrics = accumulated_step(
        model,
        optimizer,
        list(range(7)),
        lambda rows: (model(x[rows]) - y[rows]).square().mean(1),
        microbatch=7,
        denominator=7,
        max_grad_norm=1.0,
    )
    distributed = torch.load(tmp_path / "result.pt", weights_only=True)
    assert metrics == pytest.approx(distributed["metrics"], rel=1e-12, abs=1e-12)
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, distributed["model"][key], atol=1e-14, rtol=1e-12)


def failure_worker(rank, init_file, output, phase):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=4),
    )
    reductions = [0]
    at_forward = []
    original = dist.all_reduce

    def counted(*args, **kwargs):
        reductions[0] += 1
        return original(*args, **kwargs)

    dist.all_reduce = counted
    try:
        model = torch.nn.Linear(1, 1)
        model.register_buffer("broadcast_buffer", torch.ones(1))
        model = torch.nn.parallel.DistributedDataParallel(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

        def prepare(rows):
            if phase == "preparation" and rank == 0:
                raise ValueError("bad local input")

        def loss(rows):
            at_forward.append(reductions[0])
            if phase == "forward" and rank == 0:
                raise ValueError("before DDP buffer broadcast")
            return model(torch.ones(len(rows), 1)).square().mean(1)

        try:
            accumulated_step(
                model,
                optimizer,
                [0],
                loss,
                microbatch=1,
                denominator=1,
                max_grad_norm=1,
                validate_batch=prepare,
            )
        except RuntimeError as exc:
            assert not optimizer.state
            if phase == "forward":
                assert "terminate peer ranks" in str(exc)
                assert reductions[0] == at_forward[0], (
                    "must not enter error collective after forward exception"
                )
            else:
                assert "input preparation" in str(exc)
                assert not at_forward
            from pathlib import Path

            Path(output + str(rank)).write_text(json.dumps({"failed": True, "updates": 0}))
        else:
            raise AssertionError("both ranks must fail without updating")
    finally:
        dist.all_reduce = original
        dist.destroy_process_group()


@pytest.mark.parametrize("phase", ["preparation", "forward"])
def test_pre_forward_rank_error_with_ddp_buffer_broadcast(tmp_path, phase):
    ctx = mp.spawn(
        failure_worker,
        args=(str(tmp_path / "rendezvous"), str(tmp_path / "result"), phase),
        nprocs=2,
        join=False,
    )
    deadline = time.monotonic() + 30
    try:
        while not ctx.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("rank failure teardown exceeded 30s")
    finally:
        for process in ctx.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    for rank in range(2):
        assert json.loads((tmp_path / f"result{rank}").read_text()) == {
            "failed": True,
            "updates": 0,
        }
