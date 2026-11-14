"""Real CPU Adam/RNG/selection continuation; no assets or GPU calls."""

import copy
import json
import random
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

from visual_rl.data import Prompt, PromptDataset, PromptStream
from visual_rl.models.finetune import FineTuneConfig, prepare_policy
from visual_rl.training import checkpoint
from visual_rl.training.small_video_checkpoint import (
    SmallVideoIdentity,
    SmallVideoResumeState,
    export_small_video_final,
    load_small_video_checkpoint,
    save_small_video_checkpoint,
)
from visual_rl.training.small_video_protocol import RolloutSelection, group_windows
from visual_rl.training.state import Progress


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(4, 4)
        self.out = nn.Linear(4, 2)

    def forward(self, x):
        return self.out(torch.tanh(self.q(x)))


def identity(mode="full"):
    return SmallVideoIdentity(
        model_name="wan_t2v_1_3b",
        finetune_mode=mode,
        timestep_mode="flash",
        native_code_revision="a" * 40,
        model_revision="b" * 40,
        scorer_revision="c" * 40,
        data_revision="d" * 40,
        data_sha256="e" * 64,
        protocol_sha256="f" * 64,
        paired_eval_sha256="1" * 64,
        config_sha256="2" * 64,
    )


def cursor():
    data = PromptDataset(tuple(Prompt(str(i), f"prompt {i}") for i in range(48)), "e" * 64)
    return PromptStream(data, seed=71, shuffle=True)


def seeded(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def setup(root, *, mode="full", seed=42, read_only=False):
    seeded(seed)
    base = TinyPolicy()
    base_state = copy.deepcopy(base.state_dict())
    if mode == "lora":
        model = prepare_policy(
            base, FineTuneConfig("lora", rank=32, alpha=32, target_modules=("q", "out"))
        )
    else:
        model = prepare_policy(base, FineTuneConfig("full"))
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=0.003, betas=(0.9, 0.999), eps=1e-8
    )
    state = SmallVideoResumeState(
        run_root=root,
        identity=identity(mode),
        model=model,
        optimizer=optimizer,
        data_cursor=cursor(),
        selection=RolloutSelection(seed=271),
        read_only=read_only,
    )
    generator = torch.Generator().manual_seed(180)
    return model, optimizer, state, generator, base_state


def update_until(model, optimizer, state, generator, progress, target):
    groups = torch.arange(16).repeat_interleave(8)
    choices = []
    while state.budget.completed < target:
        rollout = state.budget.completed // 2
        selected = state.selection.choose(tuple(range(8)), flash=True)
        choices.append(selected)
        prompts = state.data_cursor.take(16)
        offset = sum(int(p.sample_id) for p in prompts) / 1000
        for window, rows in enumerate(group_windows(groups)):
            state.budget.begin_window(
                rollout=rollout,
                window=window,
                rows=rows.tolist(),
                groups=groups,
                selected_positions=selected,
            )
            x = torch.randn(64, 4) + torch.randn(64, 4, generator=generator) * 0.1
            x = x + random.random() * 0.1 + float(np.random.random()) * 0.1 + offset
            loss = (model(x) - torch.randn(64, 2)).square().mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            progress.optimizer_steps = state.budget.completed
        progress.attempts += 1
        progress.completed_rollouts += 1
    return choices


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("mode", ["full", "lora"])
def test_true_adam_continuous10_equals_fresh_load4plus6_rng_and_budget(tmp_path, mode):
    if mode == "lora":
        pytest.importorskip("peft")
    model, optimizer, state, generator, _ = setup(tmp_path / "continuous", mode=mode)
    progress = Progress()
    continuous_choices = update_until(model, optimizer, state, generator, progress, 10)
    expected = {
        "model": copy.deepcopy(model.state_dict()),
        "adam": copy.deepcopy(optimizer.state_dict()),
        "selection": copy.deepcopy(state.selection.state_dict()),
        "data": copy.deepcopy(state.data_cursor.state_dict()),
        "memberships": copy.deepcopy(state.budget.memberships),
        "generation_rng": generator.get_state().clone(),
        "global_rng": checkpoint.rng_state(),
    }
    state.close()
    model, optimizer, state, generator, _ = setup(tmp_path / "resumed", mode=mode)
    progress = Progress()
    before = update_until(model, optimizer, state, generator, progress, 4)
    receipt = save_small_video_checkpoint(
        state.root,
        model=model,
        progress=progress,
        state=state,
        generators={"generation": generator},
    )
    state.close()
    fresh, fresh_optimizer, resumed, fresh_generator, _ = setup(
        tmp_path / "resumed", mode=mode, seed=999
    )
    restored = load_small_video_checkpoint(
        receipt, model=fresh, state=resumed, generators={"generation": fresh_generator}
    )
    assert restored.optimizer_steps == 4 and restored.completed_rollouts == 2
    assert resumed.budget.completed == 4 and resumed.budget.limit == 10
    after = update_until(fresh, fresh_optimizer, resumed, fresh_generator, restored, 10)
    assert before + after == continuous_choices
    assert_tree_equal(fresh.state_dict(), expected["model"])
    assert_tree_equal(fresh_optimizer.state_dict(), expected["adam"])
    assert_tree_equal(resumed.selection.state_dict(), expected["selection"])
    assert resumed.data_cursor.state_dict() == expected["data"]
    assert resumed.budget.memberships == expected["memberships"]
    assert torch.equal(fresh_generator.get_state(), expected["generation_rng"])
    assert_tree_equal(checkpoint.rng_state(), expected["global_rng"])
    assert restored.optimizer_steps == 10 and restored.completed_rollouts == 5
    assert len(resumed.budget.memberships) == 10
    with pytest.raises(RuntimeError):
        fresh_optimizer.step()
    with pytest.raises(RuntimeError):
        resumed.budget.begin_window(
            rollout=5,
            window=0,
            rows=list(range(64)),
            groups=torch.arange(16).repeat_interleave(8),
            selected_positions=[0, 2],
        )
    assert resumed.budget.completed == 10
    resumed.close()


def test_completed_rollout_only_and_progress_counter_cannot_reset(tmp_path):
    model, optimizer, state, generator, _ = setup(tmp_path)
    progress = Progress()
    update_until(model, optimizer, state, generator, progress, 2)
    with pytest.raises(ValueError, match="actual cumulative"):
        save_small_video_checkpoint(tmp_path, model=model, progress=Progress(), state=state)
    groups = torch.arange(16).repeat_interleave(8)
    state.selection.choose(tuple(range(8)), flash=True)
    state.budget.begin_window(
        rollout=1,
        window=0,
        rows=group_windows(groups)[0].tolist(),
        groups=groups,
        selected_positions=[0, 1],
    )
    with pytest.raises(ValueError, match="both frozen-target"):
        save_small_video_checkpoint(tmp_path, model=model, progress=progress, state=state)
    state.close()


def test_retains_last2_complete_checkpoints_and_preserves_unregistered_evidence(tmp_path):
    model, optimizer, state, generator, _ = setup(tmp_path)
    unrelated = tmp_path / "other-experiment.pt"
    unrelated.write_bytes(b"unrelated evidence")
    progress, receipts = Progress(), []
    for step in (2, 4, 6, 8, 10):
        update_until(model, optimizer, state, generator, progress, step)
        receipts.append(
            save_small_video_checkpoint(tmp_path, model=model, progress=progress, state=state)
        )
    assert [p.exists() for p in receipts] == [False, False, False, True, True]
    assert unrelated.read_bytes() == b"unrelated evidence"
    assert len(list(tmp_path.glob("step-*.pt"))) == 2
    assert json.loads((tmp_path / "latest.json").read_text())["receipt"] == receipts[-1].name
    state.close()


def test_failed_write_keeps_previous_receipt_and_latest_then_retry_same_boundary(
    tmp_path, monkeypatch
):
    model, optimizer, state, generator, _ = setup(tmp_path)
    progress = Progress()
    update_until(model, optimizer, state, generator, progress, 2)
    first = save_small_video_checkpoint(tmp_path, model=model, progress=progress, state=state)
    first_record, latest = first.read_bytes(), (tmp_path / "latest.json").read_bytes()
    payload = tmp_path / json.loads(first_record)["file"]
    first_payload = payload.read_bytes()
    update_until(model, optimizer, state, generator, progress, 4)
    original = torch.save

    def fail(*args, **kwargs):
        raise OSError("injected write failure")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="injected"):
        save_small_video_checkpoint(tmp_path, model=model, progress=progress, state=state)
    assert first.read_bytes() == first_record and payload.read_bytes() == first_payload
    assert (tmp_path / "latest.json").read_bytes() == latest
    monkeypatch.setattr(torch, "save", original)
    current = save_small_video_checkpoint(tmp_path, model=model, progress=progress, state=state)
    assert json.loads(current.read_text())["optimizer_step"] == 4
    state.close()


@pytest.mark.parametrize(
    "problem", ["old_checkpoint", "pending_step", "paired_identity", "selection"]
)
def test_resume_rejects_rollback_uncertain_step_or_changed_identity(tmp_path, problem):
    model, optimizer, state, generator, _ = setup(tmp_path)
    progress = Progress()
    update_until(model, optimizer, state, generator, progress, 4)
    receipt = save_small_video_checkpoint(tmp_path, model=model, progress=progress, state=state)
    if problem == "old_checkpoint":
        update_until(model, optimizer, state, generator, progress, 6)
    elif problem == "pending_step":
        value = json.loads(state.ledger.read_text())
        value["pending"] = 5
        checkpoint._atomic_json(state.ledger, value)
    elif problem == "selection":
        payload = tmp_path / json.loads(receipt.read_text())["file"]
        data = torch.load(payload, weights_only=True)
        data["stateful"]["small_video"]["selection"]["rollouts"] = 0
        torch.save(data, payload)
        record = json.loads(receipt.read_text())
        record.update(sha256=checkpoint._digest(payload), bytes=payload.stat().st_size)
        checkpoint._atomic_json(receipt, record)
    state.close()
    fresh, _, resumed, _, _ = setup(tmp_path, seed=999)
    if problem == "paired_identity":
        resumed.identity = replace(resumed.identity, paired_eval_sha256="9" * 64)
    with pytest.raises(ValueError):
        load_small_video_checkpoint(receipt, model=fresh, state=resumed)
    resumed.close()


def test_frozen_identity_rejects_budget_or_eval_expansion():
    with pytest.raises(ValueError, match="remain10"):
        replace(identity(), step_limit=20)
    with pytest.raises(ValueError, match="same8"):
        replace(identity(), paired_eval_count=32)


@pytest.mark.parametrize("mode", ["full", "lora"])
def test_named_final_export_atomic_identity_and_lora_disable_parity(tmp_path, mode):
    peft = pytest.importorskip("peft") if mode == "lora" else None
    model, optimizer, state, generator, base_state = setup(tmp_path / "checkpoints", mode=mode)
    progress = Progress()
    update_until(model, optimizer, state, generator, progress, 10)
    receipt = save_small_video_checkpoint(state.root, model=model, progress=progress, state=state)
    exported = export_small_video_final(
        tmp_path / "exports", model=model, state=state, checkpoint_receipt=receipt
    )
    manifest = json.loads((exported / "complete.json").read_text())
    assert manifest["complete"] and manifest["actual_global_optimizer_steps"] == 10
    assert manifest["identity"] == identity(mode).state_dict()
    assert manifest["completed_rollouts"] == 5 and manifest["files"]
    x = torch.randn(5, 4)
    base = TinyPolicy()
    if mode == "lora":
        base.load_state_dict(base_state)
        loaded = peft.PeftModel.from_pretrained(base, exported / "adapter")
        torch.testing.assert_close(loaded(x), model(x), atol=0, rtol=0)
        with loaded.disable_adapter():
            expected = TinyPolicy()
            expected.load_state_dict(base_state)
            torch.testing.assert_close(loaded(x), expected(x), atol=0, rtol=0)
    else:
        base.load_state_dict(torch.load(exported / "policy.pt", weights_only=True))
        torch.testing.assert_close(base(x), model(x), atol=0, rtol=0)
    with pytest.raises(FileExistsError):
        export_small_video_final(
            tmp_path / "exports", model=model, state=state, checkpoint_receipt=receipt
        )
    assert receipt.exists()
    state.close()


def test_failed_final_export_has_no_complete_target_and_preserves_checkpoint(tmp_path, monkeypatch):
    model, optimizer, state, generator, _ = setup(tmp_path / "checkpoints")
    progress = Progress()
    update_until(model, optimizer, state, generator, progress, 10)
    receipt = save_small_video_checkpoint(state.root, model=model, progress=progress, state=state)
    exports = tmp_path / "exports"
    exports.mkdir()
    evidence = exports / "unrelated.txt"
    evidence.write_text("preserve")

    def fail(*args, **kwargs):
        raise OSError("injected export failure")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="injected export"):
        export_small_video_final(exports, model=model, state=state, checkpoint_receipt=receipt)
    assert not (exports / "final").exists()
    assert not list(exports.glob(".final.pending-*"))
    assert evidence.read_text() == "preserve" and receipt.exists()
    state.close()


def test_read_only_old_checkpoint_restores_adam_rng_without_rewinding_journal(tmp_path):
    model, optimizer, state, generator, _ = setup(tmp_path)
    progress = Progress()
    update_until(model, optimizer, state, generator, progress, 4)
    receipt4 = save_small_video_checkpoint(
        tmp_path, model=model, progress=progress, state=state, generators={"generation": generator}
    )
    expected_adam = copy.deepcopy(optimizer.state_dict())
    expected_rng = checkpoint.rng_state()
    expected_cursor = copy.deepcopy(state.data_cursor.state_dict())
    expected_model = copy.deepcopy(model.state_dict())
    update_until(model, optimizer, state, generator, progress, 10)
    save_small_video_checkpoint(
        tmp_path, model=model, progress=progress, state=state, generators={"generation": generator}
    )
    journal = state.ledger.read_bytes()
    state.close()
    fresh, fresh_optimizer, evaluation, fresh_generator, _ = setup(
        tmp_path, seed=999, read_only=True
    )
    restored = load_small_video_checkpoint(
        receipt4, model=fresh, state=evaluation, generators={"generation": fresh_generator}
    )
    assert restored.optimizer_steps == evaluation.budget.completed == 4
    assert evaluation.data_cursor.state_dict() == expected_cursor
    assert evaluation.selection.rollouts == 2
    assert_tree_equal(fresh_optimizer.state_dict(), expected_adam)
    assert_tree_equal(fresh.state_dict(), expected_model)
    assert_tree_equal(checkpoint.rng_state(), expected_rng)
    assert evaluation.ledger.read_bytes() == journal
    groups = torch.arange(16).repeat_interleave(8)
    evaluation.budget.begin_window(
        rollout=2,
        window=0,
        rows=group_windows(groups)[0].tolist(),
        groups=groups,
        selected_positions=[0, 2],
    )
    with pytest.raises(RuntimeError, match="read-only"):
        fresh_optimizer.step()
    assert evaluation.ledger.read_bytes() == journal and evaluation.budget.completed == 4
    assert_tree_equal(fresh.state_dict(), expected_model)
    with pytest.raises(ValueError):
        evaluation.state_dict()
    evaluation.close()
