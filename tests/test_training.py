import copy
import json
import random

import numpy as np
import pytest
import torch

from visual_rl.training.checkpoint import load_checkpoint, save_checkpoint
from visual_rl.training.state import CostClock, PolicyShadow, Progress
from visual_rl.training.update import accumulated_step


def fixture():
    torch.manual_seed(5)
    model = torch.nn.Linear(3, 2).double()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    data = torch.randn(7, 3, dtype=torch.float64)
    target = torch.randn(7, 2, dtype=torch.float64)
    return model, optimizer, data, target


def step(model, optimizer, data, target, microbatch, mask=None):
    def loss(indices):
        result = (model(data[indices]) - target[indices]).square().mean(1)
        return result if mask is None else result * mask[indices]

    return accumulated_step(
        model,
        optimizer,
        list(range(len(data))),
        loss,
        microbatch=microbatch,
        denominator=len(data),
        max_grad_norm=1.0,
    )


@pytest.mark.parametrize("microbatch", [1, 2, 4, 7, 8])
def test_microbatch_does_not_change_masked_update(microbatch):
    a, oa, data, target = fixture()
    b, ob, _, _ = fixture()
    mask = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0], dtype=torch.float64)
    for _ in range(3):
        ma = step(a, oa, data, target, 7, mask)
        mb = step(b, ob, data, target, microbatch, mask)
        assert ma == pytest.approx(mb, rel=1e-12, abs=1e-12)
    for pa, pb in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(pa, pb, atol=1e-14, rtol=1e-12)


def test_invalid_loss_never_advances_optimizer():
    model, optimizer, data, target = fixture()
    original = copy.deepcopy(model.state_dict())

    def invalid(indices):
        return model(data[indices]).mean(1) * float("nan")

    with pytest.raises(RuntimeError, match="loss validation"):
        accumulated_step(
            model, optimizer, list(range(7)), invalid, microbatch=2, denominator=7, max_grad_norm=1
        )
    assert not optimizer.state
    for key, value in model.state_dict().items():
        assert torch.equal(value, original[key])
    assert all(p.grad is None for p in model.parameters())


def test_shadow_restores_on_error_and_copies_buffers():
    model = torch.nn.BatchNorm1d(3)
    shadow = PolicyShadow(model)
    with torch.no_grad():
        model.weight.add_(1)
        model.num_batches_tracked.add_(4)
    shadow.update(0.5)
    assert torch.equal(shadow.shadow["weight"], torch.full((3,), 1.5))
    assert shadow.shadow["num_batches_tracked"].item() == 4
    current = copy.deepcopy(model.state_dict())
    with pytest.raises(RuntimeError, match="example"):
        with shadow.use():
            assert torch.equal(model.weight, torch.full((3,), 1.5))
            raise RuntimeError("example")
    for key, value in model.state_dict().items():
        assert torch.equal(value, current[key])


def test_shadow_preserves_shared_parameter_aliases():
    model = torch.nn.Module()
    model.a = torch.nn.Linear(1, 1, bias=False)
    model.b = model.a
    with torch.no_grad():
        model.a.weight.fill_(1)
    shadow = PolicyShadow(model)
    with torch.no_grad():
        model.a.weight.fill_(3)
    shadow.update(0.5)
    with shadow.use():
        assert model.a.weight.item() == model.b.weight.item() == 2
    assert model.a.weight.item() == 3
    corrupt = shadow.state_dict()
    corrupt["b.weight"].fill_(4)
    with pytest.raises(ValueError, match="aliases"):
        shadow.load_state_dict(corrupt)


def test_checkpoint_reproduces_updates_shadows_and_rng(tmp_path):
    model, optimizer, data, target = fixture()
    shadow = PolicyShadow(model)
    generator = torch.Generator().manual_seed(32)
    for _ in range(3):
        step(model, optimizer, data, target, 2)
        shadow.update(0.6)
    progress = Progress(attempts=2, completed_rollouts=1, optimizer_steps=3)
    receipt = save_checkpoint(
        tmp_path,
        model=model,
        optimizer=optimizer,
        progress=progress,
        config_sha256="a" * 64,
        stateful={"old": shadow},
        generators={"sampling": generator},
    )
    expected_random = (
        random.random(),
        np.random.rand(),
        torch.rand(3),
        torch.rand(3, generator=generator),
    )
    for _ in range(2):
        step(model, optimizer, data, target, 2)
        shadow.update(0.6)
    expected = copy.deepcopy(model.state_dict())
    expected_shadow = shadow.state_dict()
    model2, optimizer2, _, _ = fixture()
    shadow2 = PolicyShadow(model2)
    generator2 = torch.Generator().manual_seed(9)
    loaded = load_checkpoint(
        receipt,
        model=model2,
        optimizer=optimizer2,
        config_sha256="a" * 64,
        stateful={"old": shadow2},
        generators={"sampling": generator2},
    )
    assert loaded == progress
    assert random.random() == expected_random[0]
    assert np.random.rand() == expected_random[1]
    assert torch.equal(torch.rand(3), expected_random[2])
    assert torch.equal(torch.rand(3, generator=generator2), expected_random[3])
    for _ in range(2):
        step(model2, optimizer2, data, target, 2)
        shadow2.update(0.6)
    for key, value in model2.state_dict().items():
        assert torch.equal(value, expected[key])
        assert torch.equal(shadow2.shadow[key], expected_shadow[key])


def test_corrupt_checkpoint_rejected_before_load(tmp_path):
    model, optimizer, _, _ = fixture()
    receipt = save_checkpoint(
        tmp_path, model=model, optimizer=optimizer, progress=Progress(), config_sha256="b" * 64
    )
    metadata = json.loads(receipt.read_text())
    (tmp_path / metadata["file"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        load_checkpoint(receipt, model=model, optimizer=optimizer, config_sha256="b" * 64)


def test_cost_counts_failures_and_excludes_eval_from_training():
    now = [0.0]
    progress = Progress()
    clock = CostClock(progress, 32, clock=lambda: now[0])
    with clock.training_phase():
        progress.attempts += 1
        now[0] = 10
    now[0] = 15  # Evaluation is outside the training phase.
    m = clock.metrics()
    assert m["train/optimizer_step"] == 0
    assert m["cost/elapsed_gpu_hours"] == 15 * 32 / 3600
    assert m["cost/training_gpu_hours"] == 10 * 32 / 3600
