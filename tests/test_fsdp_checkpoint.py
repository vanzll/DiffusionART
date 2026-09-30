import copy
import json
import random
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed._shard.api import load_with_process_group
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy

from visual_rl.training import fsdp_checkpoint as checkpoint
from visual_rl.training.checkpoint import rng_state
from visual_rl.training.state import Progress


class Cursor:
    def __init__(self, value):
        self.value = value

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, state):
        self.value = state["value"]


class Block(torch.nn.Linear):
    _version = 3

    def __init__(self, padding=False):
        super().__init__(3, 3, bias=not padding)
        self.register_buffer("counter", torch.zeros(1))
        self.register_buffer("transient", torch.zeros(1), persistent=False)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination._metadata[prefix[:-1]]["owner"] = dist.get_rank()

    def _load_from_state_dict(self, state, prefix, metadata, *args):
        self.loaded_metadata = dict(metadata)
        super()._load_from_state_dict(state, prefix, metadata, *args)

    def forward(self, x):
        return super().forward(x) + self.counter / 100 + self.transient / 100


class Scalar(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x * self.scale


def make_model(mixed=False, nested=True, reverse=False, padding=False, scalar=False):
    kwargs = dict(device_id=torch.device("cpu"), use_orig_params=False,
                  sharding_strategy=ShardingStrategy.FULL_SHARD)
    if mixed:
        kwargs["mixed_precision"] = MixedPrecision(param_dtype=torch.bfloat16,
                                                    buffer_dtype=torch.bfloat16)
    first, second = Block(padding), Block(padding)
    module = torch.nn.Sequential(FSDP(first, **kwargs) if nested else first,
                                 torch.nn.Tanh(), second)
    if scalar:
        module = Scalar()
    model = FSDP(module, **kwargs)
    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(list(reversed(parameters)) if reverse else parameters, lr=0.003)
    return model, optimizer


def update(model, optimizer, generator, cursor, rank):
    x = torch.randn(3, 3) + torch.rand(3, 3, generator=generator)
    y = random.random() + np.random.rand() + cursor.value / 100
    (model(x).float() - y).square().mean().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    for buffer in model.buffers():
        buffer.add_(rank + 1)
    cursor.value += 3


def equal(left, right):
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right


def snapshot(model, optimizer, cursor, generator):
    return copy.deepcopy((list(model.parameters()), list(model.buffers()), optimizer.state_dict(),
                          cursor.state_dict(), rng_state(cuda_device="cpu"), generator.get_state()))


def worker(rank, root, mode):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=2,
                            init_method=f"file://{root}/rendezvous", timeout=timedelta(seconds=20))
    try:
        torch.manual_seed(35)
        model, optimizer = make_model(mixed=mode == "mixed", padding=mode == "padding",
                                      scalar=mode == "scalar")
        cursor, generator = Cursor(rank * 100), torch.Generator().manual_seed(40 + rank)
        torch.manual_seed(60 + rank)
        random.seed(50 + rank)
        np.random.seed(70 + rank)
        for _ in range(2):
            update(model, optimizer, generator, cursor, rank)
        progress = Progress(attempts=1, completed_rollouts=1, optimizer_steps=2,
                            elapsed_seconds=10 + rank, training_seconds=8 + rank)
        kwargs = dict(model=model, optimizer=optimizer, config_sha256="a" * 64,
                      stateful={"cursor": cursor}, generators={"sampling": generator})
        output = Path(root, "checkpoints")
        if mode in {"bad_names", "gradients", "step_mismatch"}:
            if rank == 1:
                if mode == "bad_names":
                    kwargs["stateful"][1] = cursor
                elif mode == "gradients":
                    next(model.parameters()).grad = torch.ones_like(next(model.parameters()))
                else:
                    progress.optimizer_steps += 1
            with pytest.raises(checkpoint.DistributedCheckpointError):
                checkpoint.save_fsdp_checkpoint(output, progress=progress, **kwargs)
            assert not output.exists()
            return
        before = snapshot(model, optimizer, cursor, generator)
        receipt = checkpoint.save_fsdp_checkpoint(output, progress=progress, **kwargs)
        equal(before, snapshot(model, optimizer, cursor, generator))
        with load_with_process_group(model.process_group), torch.serialization.safe_globals(checkpoint._SHARD_TYPES):
            payload = torch.load(receipt.parent / f"rank-{rank:05d}.pt", weights_only=True)
        shards = [value for value in payload["model"].values() if isinstance(value, ShardedTensor)]
        assert len(shards) == (1 if mode == "scalar" else 2)
        assert sum(s.tensor.numel() for st in shards for s in st.local_shards()) == (
            1 - rank if mode == "scalar" else 10 - 2 * rank if mode == "padding" else 12)
        assert sum(st.size().numel() for st in shards) == (
            1 if mode == "scalar" else 18 if mode == "padding" else 24)
        assert sum(s["exp_avg"].numel() for s in payload["optimizer"]["state"].values()) == (
            1 if mode == "scalar" else 10 if mode == "padding" else 12)
        dist.barrier()

        if mode in {"resume", "mixed", "padding", "scalar"}:
            for _ in range(2):
                update(model, optimizer, generator, cursor, rank)
            expected = snapshot(model, optimizer, cursor, generator)
            expected_draws = (random.random(), np.random.rand(), torch.rand(3), torch.rand(3, generator=generator))
            model2, optimizer2 = make_model(mixed=mode == "mixed", padding=mode == "padding",
                                            scalar=mode == "scalar")
            cursor2, generator2 = Cursor(-1), torch.Generator().manual_seed(999)
            assert not optimizer2.state
            restored = checkpoint.load_fsdp_checkpoint(
                receipt, model=model2, optimizer=optimizer2, config_sha256="a" * 64,
                stateful={"cursor": cursor2}, generators={"sampling": generator2})
            assert restored == progress
            for module in model2.modules():
                if isinstance(module, Block):
                    assert module.loaded_metadata == {"version": 3, "owner": rank}
            for _ in range(2):
                update(model2, optimizer2, generator2, cursor2, rank)
            equal(expected, snapshot(model2, optimizer2, cursor2, generator2))
            equal(expected_draws, (random.random(), np.random.rand(), torch.rand(3),
                                   torch.rand(3, generator=generator2)))
        elif mode == "extraction_failure":
            original = checkpoint._local_state

            def fail(model):
                raise ValueError("native extraction failed")

            checkpoint._local_state = fail
            try:
                # Collective-phase errors propagate directly, without entering
                # _Group.local's additional (potentially mismatched) error vote.
                with pytest.raises(ValueError, match="native extraction failed"):
                    checkpoint.save_fsdp_checkpoint(output, progress=progress, **kwargs)
                with pytest.raises(ValueError, match="native extraction failed"):
                    checkpoint.load_fsdp_checkpoint(receipt, **kwargs)
            finally:
                checkpoint._local_state = original
        elif mode == "write_failure":
            latest = (output / "latest.json").read_bytes()
            original = checkpoint._write
            if rank == 1:
                def fail(*args):
                    raise OSError("injected rank-local storage failure")
                checkpoint._write = fail
            try:
                with pytest.raises(checkpoint.DistributedCheckpointError, match="serialization"):
                    checkpoint.save_fsdp_checkpoint(output, progress=progress, **kwargs)
            finally:
                checkpoint._write = original
            assert (output / "latest.json").read_bytes() == latest
            assert len(list(output.glob("*/manifest.json"))) == 1
            assert checkpoint.load_fsdp_checkpoint(receipt, **kwargs) == progress
        elif mode == "apply_failure":
            if rank == 1:
                def reject(state):
                    raise ValueError("injected component load failure")
                cursor.load_state_dict = reject
            with pytest.raises(checkpoint.DistributedCheckpointError, match="terminal on error"):
                checkpoint.load_fsdp_checkpoint(receipt, **kwargs)
        else:
            update(model, optimizer, generator, cursor, rank)
            if mode == "corrupt" and rank == 0:
                (receipt.parent / "rank-00001.pt").write_bytes(b"corrupt")
            if mode == "world" and rank == 0:
                manifest = json.loads(receipt.read_text())
                manifest["layout"]["world_size"] = 3
                receipt.write_text(json.dumps(manifest))
            if mode == "config":
                kwargs["config_sha256"] = "b" * 64
            if mode == "wrapping":
                model, optimizer = make_model(nested=False)
                kwargs.update(model=model, optimizer=optimizer)
            if mode == "optimizer_order":
                model, optimizer = make_model(reverse=True)
                kwargs.update(model=model, optimizer=optimizer)
            if mode == "components":
                kwargs["stateful"] = {}
            dist.barrier()
            before = snapshot(model, optimizer, cursor, generator)
            with pytest.raises(checkpoint.DistributedCheckpointError, match="integrity validation"):
                checkpoint.load_fsdp_checkpoint(receipt, **kwargs)
            equal(before, snapshot(model, optimizer, cursor, generator))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("mode", ["resume", "mixed", "padding", "scalar", "write_failure", "corrupt", "world",
                                  "config", "wrapping", "optimizer_order", "components",
                                  "bad_names", "gradients", "step_mismatch", "apply_failure",
                                  "extraction_failure"])
def test_real_fsdp_checkpoint(tmp_path, mode):
    ctx = mp.spawn(worker, args=(str(tmp_path), mode), nprocs=2, join=False)
    deadline = time.monotonic() + 75
    try:
        while not ctx.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("FSDP checkpoint regression exceeded 75s")
    finally:
        for process in ctx.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
