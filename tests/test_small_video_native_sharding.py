"""Real two-rank CPU FSDP precision/resume checks; no pretrained GPU claim."""

import copy
import json
import time
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from visual_rl.training.small_video_sharding import wrap_ltx_native_full_policy


class NativePrecisionBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(8, 8)

    def forward(self, x, coordinates, physical_time):
        assert x.dtype == torch.bfloat16
        assert coordinates.dtype == physical_time.dtype == torch.float32
        assert coordinates[0, 0].item() == 32767.25
        assert not torch.is_autocast_enabled("cpu")
        assert self.projection.weight.dtype == torch.bfloat16
        # Native FP32 normalization is intentional, independent of BF16 GEMMs.
        value = torch.nn.functional.layer_norm(x.float(), (8,)).to(x.dtype)
        output = self.projection(value)
        return output + (coordinates * 1e-6 + physical_time[:, None] * 0.01).to(output)


class NativePrecisionPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([NativePrecisionBlock(), NativePrecisionBlock()])
        # Native LTX has root patch/time/output parameters outside its blocks.
        self.root_scale = torch.nn.Parameter(torch.ones(1))
        self.register_buffer("native_buffer", torch.ones(1))

    def forward(self, x, coordinates, physical_time):
        assert self.native_buffer.dtype == torch.bfloat16
        for block in self.blocks:
            x = block(x, coordinates, physical_time)
        return x * self.root_scale


def _native_precision_worker(rank, directory, checkpointing):
    from pathlib import Path

    torch.set_num_threads(1)
    root = Path(directory)
    dist.init_process_group(
        "gloo", init_method=f"file://{root / 'rendezvous'}", rank=rank, world_size=2
    )
    try:
        torch.manual_seed(42)
        model = NativePrecisionPolicy()
        reference = copy.deepcopy(model).to(torch.bfloat16)

        def wrap(policy):
            if checkpointing:
                # Exercise the actual backend dispatch, not only the helper.
                from visual_rl.backends.small_video_runner import _wrap

                return _wrap(
                    SimpleNamespace(
                        model=SimpleNamespace(name="ltx_video_2b_distilled"),
                        runtime=SimpleNamespace(distributed="fsdp"),
                    ),
                    SimpleNamespace(
                        adapter=SimpleNamespace(trainable_module=policy),
                        block_types=(NativePrecisionBlock,),
                    ),
                    SimpleNamespace(device=torch.device("cpu")),
                )
            return wrap_ltx_native_full_policy(
                policy,
                device=torch.device("cpu"),
                block_types=(NativePrecisionBlock,),
                activation_checkpointing=checkpointing,
            )

        wrapped = wrap(model)
        optimizer = torch.optim.AdamW(wrapped.parameters(), lr=1e-5)
        torch.manual_seed(100 + rank)
        x = torch.randn(2, 8, dtype=torch.bfloat16)
        coordinates = torch.full((2, 8), 32767.25)
        physical_time = torch.tensor([0.99375, 0.725], dtype=torch.float32)
        target = torch.randn(2, 8)
        with torch.no_grad():
            expected = reference(x, coordinates, physical_time)
            actual = wrapped(x, coordinates, physical_time)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        masters = tuple(wrapped.parameters())
        assert masters and all(p.dtype == torch.float32 for p in masters)

        def step(policy, adam):
            adam.zero_grad(set_to_none=True)
            prediction = policy(x, coordinates, physical_time)
            assert prediction.dtype == torch.bfloat16
            loss = (prediction.float() - target).square().mean()
            loss.backward()
            parameters = list(policy.parameters())
            assert all(p.dtype == torch.float32 for p in parameters)
            assert all(p.grad is not None and p.grad.dtype == torch.float32 for p in parameters)
            assert all(torch.isfinite(p.grad).all() for p in parameters)
            assert any(p.grad.count_nonzero().item() for p in parameters)
            norm = policy.clip_grad_norm_(1.0)
            assert torch.isfinite(norm)
            adam.step()
            for value in adam.state.values():
                for key in ("exp_avg", "exp_avg_sq"):
                    assert value[key].dtype == torch.float32

        def full_state(policy, adam):
            with FSDP.state_dict_type(
                policy,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
            ):
                return copy.deepcopy(
                    {"model": policy.state_dict(), "adam": FSDP.optim_state_dict(policy, adam)}
                )

        step(wrapped, optimizer)
        saved = full_state(wrapped, optimizer)
        for name, value in saved["model"].items():
            if "native_buffer" not in name:
                assert value.dtype == torch.float32
        path = root / f"checkpoint-rank{rank}.pt"
        torch.save(saved, path)
        step(wrapped, optimizer)
        continuous = full_state(wrapped, optimizer)

        fresh = wrap(NativePrecisionPolicy())
        fresh_optimizer = torch.optim.AdamW(fresh.parameters(), lr=1e-5)
        saved = torch.load(path, weights_only=True)
        with FSDP.state_dict_type(
            fresh,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
        ):
            fresh.load_state_dict(saved["model"], strict=True)
            fresh_optimizer.load_state_dict(
                FSDP.optim_state_dict_to_load(fresh, fresh_optimizer, saved["adam"])
            )
        step(fresh, fresh_optimizer)
        resumed = full_state(fresh, fresh_optimizer)

        def exact(left, right):
            if isinstance(left, torch.Tensor):
                torch.testing.assert_close(left, right, atol=0, rtol=0)
            elif isinstance(left, dict):
                assert left.keys() == right.keys()
                for key in left:
                    exact(left[key], right[key])
            elif isinstance(left, (list, tuple)):
                assert len(left) == len(right)
                for a, b in zip(left, right):
                    exact(a, b)
            else:
                assert left == right

        exact(continuous, resumed)
        (root / f"rank{rank}-passed.json").write_text(
            json.dumps({"passed": True, "checkpointing": checkpointing})
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("checkpointing", [False, True])
def test_native_ltx_full_compute_and_optimizer_resume_two_ranks(tmp_path, checkpointing):
    context = mp.spawn(
        _native_precision_worker, args=(str(tmp_path), checkpointing), nprocs=2, join=False
    )
    deadline = time.monotonic() + 50
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("native LTX CPU FSDP precision/resume exceeded 50 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    for rank in range(2):
        assert json.loads((tmp_path / f"rank{rank}-passed.json").read_text())["passed"]


def test_native_ltx_full_refuses_non_fp32_masters():
    with pytest.raises(ValueError, match="FP32"):
        wrap_ltx_native_full_policy(
            NativePrecisionPolicy().to(torch.bfloat16),
            device=torch.device("cpu"),
            block_types=(NativePrecisionBlock,),
            activation_checkpointing=False,
        )
