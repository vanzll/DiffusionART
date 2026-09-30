import copy
import time
from contextlib import contextmanager
from datetime import timedelta
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from visual_rl.training.fsdp_checkpoint import load_fsdp_checkpoint, save_fsdp_checkpoint
from visual_rl.training.fsdp_shadow import FSDPPolicyShadow
from visual_rl.training.state import Progress


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Linear(3, 3, bias=False)
        self.last = torch.nn.Linear(3, 1, bias=False)
        self.register_buffer("offset", torch.tensor([0.125]))

    def forward(self, x):
        return self.last(self.first(x).tanh()) + self.offset


def make_model():
    kwargs = dict(device_id=torch.device("cpu"), use_orig_params=False)
    model = Policy()
    model.first = FSDP(model.first, **kwargs)
    return FSDP(model, **kwargs)


@contextmanager
def local_only():
    def reject(*args, **kwargs):
        raise AssertionError("shadow must not invoke collectives or gather full parameters")

    with patch.multiple(dist, all_gather=reject, all_gather_into_tensor=reject,
                        all_gather_object=reject, all_reduce=reject, broadcast=reject,
                        barrier=reject), patch.object(FSDP, "summon_full_params", reject):
        yield


def equal_parameters(left, right):
    assert left.keys() == right.keys()
    for name in left:
        torch.testing.assert_close(left[name], right[name], rtol=0, atol=0)


def worker(rank, root):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=2,
                            init_method=f"file://{root}/rendezvous", timeout=timedelta(seconds=20))
    try:
        torch.manual_seed(91)
        model = make_model()
        x = torch.tensor([[1., 2., 3.]])
        with torch.no_grad():
            initial = model(x).clone()
        # Prove that this exercises the root's retained no-grad gathered cache.
        assert model._handle.flat_param._full_param_padded.untyped_storage().nbytes() > 0
        with local_only():
            old = FSDPPolicyShadow(model)
            initial_state = old.state_dict()
            assert sum(p.numel() for p in old.shadow.values()) == 7  # Padded 12 / 2.
            assert all(p.device.type == "cpu" for p in old.shadow.values())
            assert model._handle.flat_param._full_param_padded.untyped_storage().nbytes() == 0
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(0.5)
        with torch.no_grad():
            current = model(x).clone()
        assert not torch.allclose(initial, current)
        buffers = {name: b.clone() for name, b in model.named_buffers()}

        for fail in (False, True):
            context = old.use()
            with local_only():
                context.__enter__()
            try:
                torch.testing.assert_close(model(x), initial)
                with local_only():
                    with pytest.raises(RuntimeError, match="active"):
                        with old.use():
                            pass
                    with pytest.raises(RuntimeError, match="active"):
                        FSDPPolicyShadow(model)
                    for operation in (lambda: old.update(0), old.state_dict,
                                      lambda: old.load_state_dict(initial_state)):
                        with pytest.raises(RuntimeError, match="active"):
                            operation()
                if fail:
                    raise ValueError("inference failure")
            except ValueError as error:
                with local_only():
                    assert not context.__exit__(type(error), error, error.__traceback__)
            else:
                with local_only():
                    context.__exit__(None, None, None)
            assert not old.active
            with torch.no_grad():
                torch.testing.assert_close(model(x), current)
        equal_parameters(buffers, dict(model.named_buffers()))

        with local_only():
            old.update(0.25)
            expected = {name: p * 0.25 + dict(model.named_parameters())[name].detach() * 0.75
                        for name, p in initial_state["parameters"].items()}
            for name, value in old.shadow.items():
                torch.testing.assert_close(value, expected[name])
            ema_state = old.state_dict()
            expected = copy.deepcopy(ema_state["parameters"])
            for decay in (-0.1, 1, float("nan"), float("inf")):
                with pytest.raises(ValueError, match="decay"):
                    old.update(decay)
            # Validation is atomic and rejects equal-numel changes, not just sizes.
            for kind in ("name", "shape", "dtype", "rank", "topology", "layout", "nonfinite"):
                invalid = copy.deepcopy(ema_state)
                name = next(iter(invalid["parameters"]))
                if kind == "name":
                    invalid["parameters"]["wrong"] = invalid["parameters"].pop(name)
                elif kind == "shape":
                    invalid["parameters"][name] = invalid["parameters"][name].reshape(1, -1)
                elif kind == "dtype":
                    invalid["parameters"][name] = invalid["parameters"][name].double()
                elif kind == "rank":
                    invalid["rank"] = 1 - rank
                elif kind == "topology":
                    invalid["layout"]["world_size"] = 3
                elif kind == "layout":
                    invalid["layout"]["units"][0]["parameters"][0][0] = "wrong"
                else:
                    invalid["parameters"][name].fill_(float("nan"))
                with pytest.raises(ValueError):
                    old.load_state_dict(invalid)
                equal_parameters(old.state_dict()["parameters"], expected)
            old.update(0)
            equal_parameters(old.shadow, dict(model.named_parameters()))
            old.load_state_dict(ema_state)
            ema_state["parameters"][next(iter(expected))].zero_()
            equal_parameters(old.shadow, expected)
            p = next(model.parameters())
            p.grad = torch.ones_like(p)
            with pytest.raises(RuntimeError, match="gradients"):
                with old.use():
                    pass
            p.grad = None

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        progress = Progress(optimizer_steps=2)
        receipt = save_fsdp_checkpoint(root + "/checkpoints", model=model, optimizer=optimizer,
                                       progress=progress, config_sha256="a" * 64,
                                       stateful={"old_policy": old})
        model2 = make_model()
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=0.001)
        with local_only():
            restored = FSDPPolicyShadow(model2)
        assert load_fsdp_checkpoint(receipt, model=model2, optimizer=optimizer2,
                                    config_sha256="a" * 64,
                                    stateful={"old_policy": restored}) == progress
        equal_parameters(restored.shadow, expected)
        with torch.no_grad():
            torch.testing.assert_close(model2(x), current)
        with old.use():
            ema_output = model(x).clone()
        with restored.use():
            torch.testing.assert_close(model2(x), ema_output)
        with torch.no_grad():
            torch.testing.assert_close(model2(x), current)
    finally:
        dist.destroy_process_group()


def test_two_process_fsdp_shadow(tmp_path):
    context = mp.spawn(worker, args=(str(tmp_path),), nprocs=2, join=False)
    deadline = time.monotonic() + 75
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("FSDP shadow regression exceeded 75s")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
