import copy
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from visual_rl.training import distributed_checkpoint as checkpoint
from visual_rl.training.checkpoint import restore_rng, rng_state
from visual_rl.training.distributed import distributed_session
from visual_rl.training.state import PolicyShadow, Progress
from visual_rl.training.update import accumulated_step


class Cursor:
    def __init__(self, value):
        self.value = value

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, state):
        self.value = state["value"]


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    else:
        assert left == right


class VersionedLinear(torch.nn.Linear):
    _version = 2

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination._metadata[prefix[:-1]]["owner"] = dist.get_rank()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, *args):
        self.loaded_metadata = dict(local_metadata)
        if local_metadata.get("version", 1) < 2:
            state_dict[prefix + "weight"] = state_dict[prefix + "weight"] / 2
        super()._load_from_state_dict(state_dict, prefix, local_metadata, *args)


def make_model(group=None):
    module = VersionedLinear(3, 1).double()
    module.register_buffer("local_counter", torch.zeros(1))
    model = torch.nn.parallel.DistributedDataParallel(module, broadcast_buffers=False, process_group=group)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    return model, optimizer, PolicyShadow(module, trainable_only=True)


def update(model, optimizer, shadow, generator, cursor, rank):
    x = torch.randn(3, 3).double() + torch.rand(3, 3, generator=generator).double()
    y = torch.full((3, 1), random.random() + np.random.rand() + cursor.value / 100, dtype=torch.float64)
    accumulated_step(model, optimizer, [0, 1, 2],
                     lambda rows: (model(x[rows]) - y[rows]).square().mean(1),
                     microbatch=2, denominator=3, max_grad_norm=1)
    shadow.update(0.6)
    model.module.local_counter.add_(rank + 1)
    cursor.value += 3


def worker(rank, root, mode):
    torch.set_num_threads(1)
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE="2")
    with distributed_session("ddp", device="cpu", timeout_seconds=15,
                             init_method=f"file://{root}/rendezvous"):
        torch.manual_seed(35)
        model, optimizer, shadow = make_model()
        cursor, generator = Cursor(rank * 100), torch.Generator().manual_seed(40 + rank)
        torch.manual_seed(60 + rank)
        random.seed(50 + rank)
        np.random.seed(70 + rank)
        for _ in range(2):
            update(model, optimizer, shadow, generator, cursor, rank)
        progress = Progress(attempts=1, completed_rollouts=1, optimizer_steps=2,
                            elapsed_seconds=10 + rank, training_seconds=8 + rank)
        kwargs = dict(model=model, optimizer=optimizer, config_sha256="a" * 64,
                      stateful={"cursor": cursor}, replicated_stateful={"old": shadow},
                      generators={"sampling": generator})
        output = Path(root, "checkpoints")
        if mode == "bad_names":
            if rank == 1:
                kwargs["stateful"][1] = cursor
            with pytest.raises(checkpoint.DistributedCheckpointError, match="preflight"):
                checkpoint.save_ddp_checkpoint(output, progress=progress, **kwargs)
            assert not output.exists()
            return
        if mode == "step_mismatch":
            progress.optimizer_steps += rank
            with pytest.raises(checkpoint.DistributedCheckpointError, match="identities differ"):
                checkpoint.save_ddp_checkpoint(output, progress=progress, **kwargs)
            assert not output.exists()
            return
        receipt = checkpoint.save_ddp_checkpoint(output, progress=progress, **kwargs)
        if mode == "apply_failure":
            if rank == 1:
                def reject(state):
                    raise ValueError("component load failed")
                cursor.load_state_dict = reject
            with pytest.raises(checkpoint.DistributedCheckpointError, match="terminal on error"):
                checkpoint.load_ddp_checkpoint(receipt, **kwargs)
            return
        if mode == "write_failure":
            original = checkpoint._write
            latest = (output / "latest.json").read_bytes()
            if rank == 1:
                def fail(*args):
                    raise OSError("injected storage failure")
                checkpoint._write = fail
            try:
                with pytest.raises(checkpoint.DistributedCheckpointError, match="serialization"):
                    checkpoint.save_ddp_checkpoint(output, progress=progress, **kwargs)
            finally:
                checkpoint._write = original
            assert (output / "latest.json").read_bytes() == latest
            assert len(list(output.glob("*/manifest.json"))) == 1
            assert checkpoint.load_ddp_checkpoint(receipt, **kwargs) == progress
            return
        if mode in {"corrupt_shared", "corrupt_rank", "missing_rank", "world_mismatch", "component_mismatch"}:
            before = copy.deepcopy(model.module.state_dict())
            before_optimizer = copy.deepcopy(optimizer.state_dict())
            if rank == 0:
                if mode == "corrupt_shared":
                    (receipt.parent / "shared.pt").write_bytes(b"bad")
                elif mode == "corrupt_rank":
                    (receipt.parent / "rank-00001.pt").write_bytes(b"bad")
                elif mode == "missing_rank":
                    (receipt.parent / "rank-00001.pt").unlink()
                elif mode == "world_mismatch":
                    manifest = json.loads(receipt.read_text())
                    manifest["world_size"] = 3
                    receipt.write_text(json.dumps(manifest))
            if mode == "component_mismatch":
                kwargs["stateful"] = {"other": cursor}
            dist.barrier()
            with pytest.raises(checkpoint.DistributedCheckpointError, match="integrity validation"):
                checkpoint.load_ddp_checkpoint(receipt, **kwargs)
            assert_tree_equal(before, model.module.state_dict())
            assert_tree_equal(before_optimizer, optimizer.state_dict())
            return
        assert mode == "resume"
        for _ in range(2):
            update(model, optimizer, shadow, generator, cursor, rank)
        expected = copy.deepcopy((model.module.state_dict(), optimizer.state_dict(), shadow.state_dict(), cursor.state_dict()))
        expected_random = (random.random(), np.random.rand(), torch.rand(2), torch.rand(2, generator=generator))
        model2, optimizer2, shadow2 = make_model()
        cursor2, generator2 = Cursor(-1), torch.Generator().manual_seed(999)
        loaded = checkpoint.load_ddp_checkpoint(
            receipt, model=model2, optimizer=optimizer2, config_sha256="a" * 64,
            stateful={"cursor": cursor2}, replicated_stateful={"old": shadow2},
            generators={"sampling": generator2},
        )
        assert loaded == progress
        assert model2.module.loaded_metadata == {"version": 2, "owner": rank}
        assert model2.module.local_counter.item() == 2 * (rank + 1)
        for _ in range(2):
            update(model2, optimizer2, shadow2, generator2, cursor2, rank)
        assert_tree_equal(expected, (model2.module.state_dict(), optimizer2.state_dict(), shadow2.state_dict(), cursor2.state_dict()))
        assert_tree_equal(expected_random, (random.random(), np.random.rand(), torch.rand(2), torch.rand(2, generator=generator2)))
        if rank == 0:
            manifest = json.loads(receipt.read_text())
            assert len(manifest["ranks"]) == 2
            for record in manifest["ranks"]:
                payload = torch.load(receipt.parent / record["file"], weights_only=True)
                assert "optimizer" not in payload and "model" not in payload
                assert set(payload["model_local"]) == {"local_counter"}


@pytest.mark.parametrize("mode", ["resume", "step_mismatch", "bad_names", "write_failure", "corrupt_shared", "corrupt_rank",
                                  "missing_rank", "world_mismatch", "component_mismatch", "apply_failure"])
def test_real_ddp_checkpoint(tmp_path, mode):
    ctx = mp.spawn(worker, args=(str(tmp_path), mode), nprocs=2, join=False)
    deadline = time.monotonic() + 60
    try:
        while not ctx.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("checkpoint regression exceeded 60s")
    finally:
        for process in ctx.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)


def test_rank_local_cuda_rng_never_touches_other_devices(monkeypatch):
    events = []
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: pytest.fail("touched every GPU"))
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda device: events.append(("get", device)) or torch.ones(4, dtype=torch.uint8))
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda state: pytest.fail("restored every GPU"))
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda state, device: events.append(("set", device)))
    state = rng_state(cuda_device="cuda:3")
    restore_rng(state, cuda_device="cuda:3")
    assert events == [("get", torch.device("cuda:3")), ("set", torch.device("cuda:3"))]
    with pytest.raises(ValueError, match="explicit device"):
        restore_rng(state)
    with pytest.raises(ValueError, match="device type"):
        restore_rng(state, cuda_device="cpu")


def test_rank_local_cpu_rng_does_not_initialize_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: pytest.fail("touched CUDA"))
    state = rng_state(cuda_device="cpu")
    assert state["local_cuda_rng"] is None and not state["torch_cuda"]
    restore_rng(state, cuda_device="cpu")


@pytest.mark.parametrize("mixed", [False, True])
def test_explicit_rng_device_rejects_global_or_mixed_payload_before_changes(monkeypatch, mixed):
    state = rng_state(cuda_device="cpu")
    if not mixed:
        del state["local_cuda_rng"]
    state["torch_cuda"] = [torch.ones(4, dtype=torch.uint8)]
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda state: pytest.fail("changed every GPU"))
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda *args: pytest.fail("changed local GPU"))
    monkeypatch.setattr(random, "setstate", lambda state: pytest.fail("changed Python RNG"))
    with pytest.raises(ValueError, match="global/mixed"):
        restore_rng(state, cuda_device="cpu")


def subgroup_checkpoint_worker(rank, root):
    torch.set_num_threads(1)
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE="2")
    with distributed_session("ddp", device="cpu", timeout_seconds=15,
                             init_method=f"file://{root}/rendezvous"):
        groups = [dist.new_group([member], backend="gloo") for member in range(2)]
        model, optimizer, shadow = make_model(groups[rank])
        cursor, generator = Cursor(100 * rank), torch.Generator().manual_seed(rank)
        update(model, optimizer, shadow, generator, cursor, rank)
        progress = Progress(attempts=1, completed_rollouts=1, optimizer_steps=1)
        kwargs = dict(model=model, optimizer=optimizer, config_sha256=str(rank) * 64,
                      stateful={"cursor": cursor}, replicated_stateful={"old": shadow},
                      generators={"sampling": generator})
        receipt = checkpoint.save_ddp_checkpoint(Path(root, f"group-{rank}"), progress=progress, **kwargs)
        before = copy.deepcopy(model.module.state_dict())
        with torch.no_grad():
            model.module.weight.fill_(0)
        assert checkpoint.load_ddp_checkpoint(receipt, **kwargs) == progress
        assert_tree_equal(before, model.module.state_dict())
        assert model.module.loaded_metadata["owner"] == rank
        assert json.loads(receipt.read_text())["world_size"] == 1
        dist.destroy_process_group(groups[rank])


def test_checkpoint_with_nonzero_global_rank_as_group_leader(tmp_path):
    ctx = mp.spawn(subgroup_checkpoint_worker, args=(str(tmp_path),), nprocs=2, join=False)
    deadline = time.monotonic() + 45
    try:
        while not ctx.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("subgroup checkpoint regression exceeded 45s")
    finally:
        for process in ctx.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
