import json
import os
import time
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from visual_rl.config import ConfigError
from visual_rl.training.distributed import distributed_session, launch_identity
from visual_rl.training.update import accumulated_step


def environment(rank=0, world=2):
    return dict(RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE=str(world))


def test_identity_and_global_prompt_budget():
    assert launch_identity("none", {}).world_size == 1
    identity = launch_identity("ddp", environment(1))
    assert identity.rank == identity.local_rank == 1
    assert identity.local_prompt_count(32) == 16
    for budget in (0, 1, 3, True, 2.5):
        with pytest.raises(ConfigError, match="prompt groups"):
            identity.local_prompt_count(budget)
    assert launch_identity("none", environment(0, 1)).world_size == 1
    multinode = dict(RANK="9", WORLD_SIZE="16", LOCAL_RANK="1", LOCAL_WORLD_SIZE="8")
    assert launch_identity("ddp", multinode).rank == 9


@pytest.mark.parametrize("strategy,env", [
    ("ddp", {}), ("fsdp", {}), ("none", environment()),
    ("none", {"RANK": "0"}), ("ddp", {"WORLD_SIZE": "2"}),
    ("ddp", {**environment(), "WORLD_SIZE": "no"}),
    ("ddp", {**environment(), "RANK": "2"}),
    ("ddp", {**environment(), "LOCAL_WORLD_SIZE": "0"}),
    ("ddp", {**environment(), "LOCAL_RANK": "1"}),
    ("ddp", {**environment(), "WORLD_SIZE": "3"}),
])
def test_bad_launcher_identity(strategy, env):
    with pytest.raises(ConfigError):
        launch_identity(strategy, env)


def clear_env(monkeypatch):
    for key in environment():
        monkeypatch.delenv(key, raising=False)


def test_single_device_session_owns_no_group(monkeypatch):
    clear_env(monkeypatch)
    with distributed_session("none", device="cpu", timeout_seconds=5) as session:
        assert session.device == torch.device("cpu")
        assert not session.distributed
        assert not dist.is_initialized()
        session.require_same_digest("a" * 64)
        with pytest.raises(ConfigError):
            session.require_same_digest("x" * 64)
    assert not dist.is_initialized()


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True, "5", None])
def test_timeout_is_explicit_and_finite(monkeypatch, timeout):
    clear_env(monkeypatch)
    with pytest.raises(ConfigError, match="timeout"):
        with distributed_session("none", device="cpu", timeout_seconds=timeout):
            pytest.fail("entered invalid session")


def test_existing_group_never_adopted(monkeypatch):
    clear_env(monkeypatch)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "destroy_process_group", lambda: pytest.fail("destroyed caller group"))
    with pytest.raises(ConfigError, match="existing process group"):
        with distributed_session("none", device="cpu", timeout_seconds=5):
            pytest.fail("adopted group")


def test_cuda_mapping_precedes_nccl_initialization_and_cleanup_on_failure(monkeypatch):
    for key, value in environment(1).items():
        monkeypatch.setenv(key, value)
    events = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "set_device", lambda rank: events.append(("device", rank)))
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(dist, "is_nccl_available", lambda: True)
    monkeypatch.setattr(dist, "init_process_group", lambda **kw: events.append(("init", kw)))
    monkeypatch.setattr(dist, "destroy_process_group", lambda: events.append(("destroy",)))
    with pytest.raises(RuntimeError, match="local training error"):
        with distributed_session("ddp", device="cuda", timeout_seconds=7) as session:
            assert session.device == torch.device("cuda", 1)
            raise RuntimeError("local training error")
    assert events[0] == ("device", 1)
    assert events[1][0] == "init"
    assert events[1][1]["backend"] == "nccl"
    assert events[1][1]["rank"] == 1
    assert events[1][1]["timeout"].total_seconds() == 7
    assert events[2] == ("destroy",)


def test_insufficient_visible_devices_fails_before_rendezvous(monkeypatch):
    for key, value in environment().items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(dist, "init_process_group", lambda **kw: pytest.fail("entered rendezvous"))
    with pytest.raises(ConfigError, match="visible CUDA"):
        with distributed_session("ddp", device="cuda", timeout_seconds=7):
            pytest.fail("entered session")


def runtime_worker(rank, root, mode):
    torch.set_num_threads(1)
    os.environ.update(environment(rank))
    with distributed_session(
        "ddp", device="cpu", timeout_seconds=8, init_method=f"file://{root}/rendezvous"
    ) as session:
        if mode in {"mismatch", "invalid"}:
            digest = "a" * 64 if rank == 0 else ("b" * 64 if mode == "mismatch" else "invalid")
            with pytest.raises(ConfigError, match="identity"):
                session.require_same_digest(digest)
        else:
            session.require_same_digest("a" * 64)
            # Independent DDP groups must not exchange denominators, metrics,
            # validation flags or microbatch counts through the default group.
            groups = [dist.new_group([r], backend="gloo") for r in range(2)]
            torch.manual_seed(21 + rank)
            model = torch.nn.Linear(2, 1).double()
            initial = {key: value.clone() for key, value in model.state_dict().items()}
            wrapped = torch.nn.parallel.DistributedDataParallel(model, process_group=groups[rank])
            optimizer = torch.optim.AdamW(wrapped.parameters(), lr=0.001)
            x = torch.randn(rank + 2, 2).double()
            y = torch.randn(rank + 2, 1).double()
            metrics = accumulated_step(
                wrapped, optimizer, list(range(len(x))),
                lambda rows: (wrapped(x[rows]) - y[rows]).square().mean(1),
                microbatch=1, denominator=len(x), max_grad_norm=1,
            )
            # Independent reference uses a plain PyTorch update, no helper that
            # could repeat the same process-group selection bug.
            reference = torch.nn.Linear(2, 1).double()
            reference.load_state_dict(initial)
            reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.001)
            loss = (reference(x) - y).square().mean()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 1)
            reference_optimizer.step()
            assert metrics["train/loss"] == pytest.approx(loss.item(), abs=1e-12)
            assert metrics["train/grad_norm/before_clip"] == pytest.approx(norm.item(), abs=1e-12)
            for key, tensor in reference.state_dict().items():
                torch.testing.assert_close(tensor, model.state_dict()[key], atol=1e-14, rtol=1e-12)
            dist.destroy_process_group(groups[rank])
    assert not dist.is_initialized()
    Path(root, f"rank-{rank}.json").write_text(json.dumps({"clean": True}))


@pytest.mark.parametrize("mode", ["subgroups", "mismatch", "invalid"])
def test_real_two_rank_lifecycle_and_subgroup_updates(tmp_path, mode):
    ctx = mp.spawn(runtime_worker, args=(str(tmp_path), mode), nprocs=2, join=False)
    deadline = time.monotonic() + 45
    try:
        while not ctx.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("distributed lifecycle exceeded 45 seconds")
    finally:
        for process in ctx.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    for rank in range(2):
        assert json.loads((tmp_path / f"rank-{rank}.json").read_text()) == {"clean": True}


def test_cli_rejects_independent_workers_before_backend(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace

    from visual_rl import train

    for key, value in environment().items():
        monkeypatch.setenv(key, value)
    config = SimpleNamespace(runtime=SimpleNamespace(distributed="none"))
    monkeypatch.setattr(train, "load_config", lambda *args: config)
    monkeypatch.setattr(train, "get_backend", lambda *args: SimpleNamespace(
        load=lambda: pytest.fail("must reject before loading backend")
    ))
    assert train.main(["--config", "unused.yaml"]) == 2
    assert "multiple torchrun workers" in capsys.readouterr().err


def subgroup_worker(rank, root, invalid):
    torch.set_num_threads(1)
    os.environ.update(environment(rank, 4))
    with distributed_session(
        "ddp", device="cpu", timeout_seconds=10, init_method=f"file://{root}/rendezvous"
    ):
        groups = [dist.new_group(ranks, backend="gloo") for ranks in ([0, 1], [2, 3])]
        group_index, member = divmod(rank, 2)
        torch.manual_seed(51 + group_index)
        model = torch.nn.Linear(2, 1).double()
        initial = {key: value.clone() for key, value in model.state_dict().items()}
        wrapped = torch.nn.parallel.DistributedDataParallel(model, process_group=groups[group_index])
        optimizer = torch.optim.AdamW(wrapped.parameters(), lr=0.001)
        x, y = torch.randn(7, 2).double(), torch.randn(7, 1).double()
        indices = [0, 1, 2, 3] if member == 0 else [4, 5, 6, 0]

        def loss(rows):
            values = (wrapped(x[[indices[i] for i in rows]]) - y[[indices[i] for i in rows]]).square().mean(1)
            weights = values.new_tensor([0 if member == 1 and i == 3 else 1 for i in rows])
            if invalid and rank == 0:
                values = values * float("nan")
            return values * weights

        if invalid and group_index == 0:
            with pytest.raises(RuntimeError, match="loss validation"):
                accumulated_step(wrapped, optimizer, list(range(4)), loss,
                                 microbatch=2, denominator=4 if member == 0 else 3, max_grad_norm=1)
            assert not optimizer.state
            for key, tensor in initial.items():
                assert torch.equal(tensor, model.state_dict()[key])
        else:
            metrics = accumulated_step(wrapped, optimizer, list(range(4)), loss,
                                       microbatch=2, denominator=4 if member == 0 else 3, max_grad_norm=1)
            reference = torch.nn.Linear(2, 1).double()
            reference.load_state_dict(initial)
            reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.001)
            reference_loss = (reference(x) - y).square().mean()
            reference_loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 1)
            reference_optimizer.step()
            assert metrics["train/loss"] == pytest.approx(reference_loss.item(), abs=1e-12)
            assert metrics["train/grad_norm/before_clip"] == pytest.approx(norm.item(), abs=1e-12)
            for key, tensor in reference.state_dict().items():
                torch.testing.assert_close(tensor, model.state_dict()[key], atol=1e-14, rtol=1e-12)
        dist.destroy_process_group(groups[group_index])
    Path(root, f"rank-{rank}.json").write_text(json.dumps({"clean": True}))


@pytest.mark.parametrize("invalid", [False, True])
def test_two_independent_multimember_ddp_groups(tmp_path, invalid):
    ctx = mp.spawn(subgroup_worker, args=(str(tmp_path), invalid), nprocs=4, join=False)
    deadline = time.monotonic() + 60
    try:
        while not ctx.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("multi-member subgroup regression exceeded 60s")
    finally:
        for process in ctx.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    for rank in range(4):
        assert json.loads((tmp_path / f"rank-{rank}.json").read_text()) == {"clean": True}


def missing_peer_worker(root):
    torch.set_num_threads(1)
    os.environ.update(environment())
    with pytest.raises(RuntimeError):
        with distributed_session(
            "ddp", device="cpu", timeout_seconds=1, init_method=f"file://{root}/rendezvous"
        ):
            raise AssertionError("rendezvous succeeded without peer")
    assert not dist.is_initialized()
    Path(root, "timed-out").write_text("yes")


def test_missing_peer_rendezvous_has_a_finite_timeout(tmp_path):
    process = mp.get_context("spawn").Process(target=missing_peer_worker, args=(str(tmp_path),))
    process.start()
    try:
        process.join(timeout=20)
        assert not process.is_alive(), "missing-peer rendezvous exceeded 20s"
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)
    assert (tmp_path / "timed-out").read_text() == "yes"
