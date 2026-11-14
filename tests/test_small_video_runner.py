"""CPU native stochastic sampler and actual optimizer loop with tiny components."""

import copy
import io
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from visual_rl.backends import small_video_runner as runner
from visual_rl.config import ConfigError
from visual_rl.data import Prompt, PromptDataset, PromptStream
from visual_rl.training import checkpoint
from visual_rl.training.distributed import DistributedSession, RankInfo
from visual_rl.training.small_video_checkpoint import SmallVideoIdentity, SmallVideoResumeState
from visual_rl.training.small_video_protocol import group_video_reward
from visual_rl.training.small_video_replay import NATIVE_AFFINE_ART_CONTRACT
from visual_rl.training.state import CostClock, Progress


class TinyVelocity(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.15))
        self.calls = []

    def forward(self, state):
        self.calls.append((len(state), self.training, torch.is_grad_enabled()))
        return state * self.weight


class Components:
    def __init__(self):
        self.adapter = SimpleNamespace(transformer=TinyVelocity())
        self.scheduler_factory = None


class Backend:
    def __init__(self, *, zero_signal=False, invalid=False):
        self.zero_signal, self.invalid = zero_signal, invalid
        self.scored = []
        self.evaluated, self.exported = [], []
        self.selections, self.initial_seeds = [], []
        self.raw_metrics = []

    def resolved_protocol(self, c):
        return {"eligible_positions": list(range(8))}

    def make_predict(self, c, parts, prompts, repeats, device):
        assert len(prompts) == 16 and repeats == 8
        condition = torch.tensor([float(text.split()[-1]) for text in prompts]).repeat_interleave(8)

        def predict(state, sigma, rows, index):
            assert len(rows) == len(state)
            return (
                parts.adapter.transformer(state) + condition[rows].reshape(-1, 1, 1, 1, 1) * 0.001
            )

        return predict

    def initial_noise(self, c, rows, device, seed):
        self.initial_seeds.append(seed)
        return torch.randn(rows, 2, 1, 1, 1, generator=torch.Generator().manual_seed(seed)).to(
            device
        )

    def score_videos(self, c, parts, endpoints, prompts, directory, device):
        assert len(prompts) == len(endpoints) == 128
        self.scored.append(endpoints.clone())
        raw = {
            "VQ": endpoints[:, 0].flatten(1).mean(1),
            "MQ": endpoints.square().flatten(1).mean(1),
            "TA": endpoints.abs().flatten(1).mean(1),
        }
        if self.zero_signal:
            raw = {key: torch.ones(128) for key in raw}
        if self.invalid:
            raw["MQ"][0] = float("nan")
        feedback = group_video_reward(raw, torch.arange(16).repeat_interleave(8))
        self.raw_metrics.append(feedback["metrics"])
        return [], SimpleNamespace(raw=raw)

    def evaluate(self, session, c, parts, validation, step, root):
        self.evaluated.append(step)

    def export_final(self, session, c, parts, state, receipt, root):
        assert state.budget.completed == 10 and receipt is not None
        self.exported.append(str(receipt))


def configuration(*, flash=True):
    loss = {
        "loss_space": "velocity_linear_sigma",
        "direction_normalization": "rms",
        "mask_space": "velocity",
        "velocity_mask_metric": "scheduler_reference_metric",
        "corrective_mask_enabled": True,
        "q2_reference_sde_noise_level": 0.8,
        "q2_reference_kl_threshold": 1e-5,
    }
    covariance = {
        "reduction": "unbiased",
        "direction_mode": "historical",
        "scale_mode": "unit_normalized_covariance",
        "guidance_gain": 1.0,
        "zero_signal_mode": "exact",
    }
    return SimpleNamespace(
        seed=42,
        model=SimpleNamespace(
            name="ltx_video_2b_distilled", options={"replay_contract": NATIVE_AFFINE_ART_CONTRACT}
        ),
        algorithm=SimpleNamespace(
            options={"covariance": covariance, "loss": loss, "restore_zero_signal": True}
        ),
        sampling=SimpleNamespace(microbatch=8),
        training=SimpleNamespace(
            microbatch=8, grad_clip=1.0, timestep_selection="high_noise" if flash else "configured"
        ),
    )


def setup(root, *, flash=True, seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    parts = Components()
    model = parts.adapter.transformer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    data = PromptDataset(tuple(Prompt(str(i), f"prompt {i}") for i in range(64)), "e" * 64)
    stream = PromptStream(data, seed=42, shuffle=True)
    identity = SmallVideoIdentity(
        "ltx_video_2b_distilled",
        "full",
        "flash" if flash else "original",
        "a" * 40,
        "b" * 40,
        "c" * 40,
        "d" * 40,
        "e" * 64,
        "f" * 64,
        "1" * 64,
        "2" * 64,
    )
    state = SmallVideoResumeState(
        run_root=root, identity=identity, model=model, optimizer=optimizer, data_cursor=stream
    )
    session = DistributedSession(RankInfo(), torch.device("cpu"), False)
    return parts, state, session


def execute(
    root,
    *,
    flash=True,
    zero_signal=False,
    invalid=False,
    pause=None,
    resume=None,
    seed=42,
    backend=None,
):
    parts, state, session = setup(root, flash=flash, seed=seed)
    model = parts.adapter.transformer
    arguments = dict(
        model=model,
        optimizer=state.optimizer,
        config_sha256=state.identity.config_sha256,
        stateful={"small_video": state},
    )
    progress = checkpoint.load_checkpoint(resume, **arguments) if resume else Progress()
    state.validate_progress(progress)
    backend = Backend(zero_signal=zero_signal, invalid=invalid) if backend is None else backend
    log = io.StringIO()
    receipts = []

    def save(current):
        state.validate_progress(current)
        receipt = checkpoint.save_checkpoint(root, progress=current, keep_last=2, **arguments)
        receipts.append(receipt)
        return receipt

    try:
        result = runner.train_loop(
            session,
            configuration(flash=flash),
            parts,
            state,
            progress,
            None,
            root,
            backend=backend,
            clock=CostClock(progress, 0),
            log=log,
            save=save,
            pause_after=pause,
            resume_receipt=resume,
        )
        return (
            result,
            parts,
            state.state_dict(),
            backend,
            receipts,
            log.getvalue(),
            copy.deepcopy(state.optimizer.state_dict()),
        )
    finally:
        state.close()


@pytest.mark.parametrize("flash", [True, False])
def test_native_loop_real_two_window_counts_complete_groups_and_denominator(
    tmp_path, monkeypatch, flash
):
    calls = []
    objectives = []
    actual = runner.accumulated_step
    actual_objective = runner.NativeARTObjective

    def record_objective(*args, **kwargs):
        value = actual_objective(*args, **kwargs)
        objectives.append(value)
        return value

    def tracked(model, optimizer, items, loss_fn, **kwargs):
        # accumulated_step accepts any callable. The runner's progress wrapper
        # intentionally hides implementation-specific attributes such as cache;
        # observe the real native objective at its construction boundary instead.
        objective = objectives[-1]
        calls.append(
            (
                len(items),
                kwargs["denominator"],
                kwargs["microbatch"],
                list(objective.cache.selected),
                objective.cache.states.shape[1],
            )
        )
        metrics = actual(model, optimizer, items, loss_fn, **kwargs)
        assert sum(row["items"] for row in objective.observations) == len(items)
        assert {row["position"] for row in objective.observations} == set(objective.cache.selected)
        assert all(row["items"] == kwargs["microbatch"] for row in objective.observations)
        return metrics

    monkeypatch.setattr(runner, "NativeARTObjective", record_objective)
    monkeypatch.setattr(runner, "accumulated_step", tracked)
    result, parts, state, backend, receipts, logs, _ = execute(tmp_path, flash=flash)
    selected_count = 2 if flash else 8
    assert result["status"] == "completed" and result["progress"]["optimizer_steps"] == 10
    assert result["progress"]["completed_rollouts"] == 5
    assert len(backend.scored) == 5 and backend.evaluated == [10] and len(backend.exported) == 1
    assert len(calls) == 10 and all(
        x[:3] == (64 * selected_count, 64 * selected_count, 8) for x in calls
    )
    assert all(x[4] == 8 for x in calls)  # Flash preserves every native trajectory position.
    assert len(receipts) == 2 and [
        json.loads(p.read_text())["optimizer_step"] for p in receipts
    ] == [4, 10]
    assert len(state["budget"]["memberships"]) == 10
    assert len(objectives) == 5
    assert len(state["budget"]["memberships"][0]["rows"]) == 64
    assert all(batch == 8 for batch, _, _ in parts.adapter.transformer.calls)
    metrics = [json.loads(row) for row in logs.splitlines()]
    assert len(metrics) == 10
    assert all(row["train/raw_VQ"] == row["reward/raw/VQ_mean"] for row in metrics)
    assert all(row["train/raw_MQ"] == row["reward/raw/MQ_mean"] for row in metrics)
    assert all(
        row["train/raw_VQ_MQ_equal_mean"] == row["reward/raw/VQ_MQ_equal_mean"] for row in metrics
    )
    assert [row["train/optimizer_step"] for row in metrics] == list(range(1, 11))
    assert all(row["train/global_loss_denominator"] == 64 * selected_count for row in metrics)
    for rollout in range(5):
        assert calls[rollout * 2][3] == calls[rollout * 2 + 1][3]
        assert abs(metrics[rollout * 2]["reward/normalized/target_mean"]) < 2e-6
        assert metrics[rollout * 2]["reward/raw/VQ_MQ_equal_mean"] == pytest.approx(
            backend.raw_metrics[rollout]["reward/raw/VQ_MQ_equal_mean"]
        )
    assert backend.initial_seeds == [42 + 1_000_003 * i for i in range(5)]


def test_zero_signal_still_has_ten_actual_updates_and_restoring_anchors(tmp_path):
    result, _, state, backend, _, logs, _ = execute(tmp_path, zero_signal=True)
    assert result["progress"]["optimizer_steps"] == 10 and len(backend.scored) == 5
    assert state["budget"]["completed"] == 10
    metrics = [json.loads(row) for row in logs.splitlines()]
    assert all(row["train/active_row_fraction"] == 0 for row in metrics)
    assert all(row["train/anchor_row_fraction"] == 1 for row in metrics)


def test_pause4_fresh_resume6_matches_continuous_parameters_adam_selection_and_cursor(tmp_path):
    _, baseline, full_state, _, _, _, full_adam = execute(tmp_path / "continuous")
    expected_rng = checkpoint.rng_state()
    paused, _, _, backend, receipts, _, _ = execute(tmp_path / "resumed", pause=4)
    assert paused["status"] == "paused" and len(backend.scored) == 2
    assert backend.evaluated == [] and backend.exported == []
    result, resumed, resumed_state, backend, _, _, resumed_adam = execute(
        tmp_path / "resumed", resume=receipts[-1], seed=999, backend=backend
    )
    assert result["status"] == "completed" and len(backend.scored) == 5
    assert result["progress"]["optimizer_steps"] == 10
    assert torch.equal(baseline.adapter.transformer.weight, resumed.adapter.transformer.weight)
    assert full_state["budget"] == resumed_state["budget"]
    assert full_state["data_cursor"] == resumed_state["data_cursor"]
    assert torch.equal(
        full_state["selection"]["generator"], resumed_state["selection"]["generator"]
    )
    for parameter, record in full_adam["state"].items():
        for key, value in record.items():
            other = resumed_adam["state"][parameter][key]
            assert torch.equal(value, other) if isinstance(value, torch.Tensor) else value == other
    actual_rng = checkpoint.rng_state()
    assert torch.equal(expected_rng["torch_cpu"], actual_rng["torch_cpu"])
    assert (
        expected_rng["python"] == actual_rng["python"]
        and expected_rng["numpy"] == actual_rng["numpy"]
    )


def test_resume_at10_finalizes_without_any_sixth_rollout_or_extra_optimizer(tmp_path):
    _, _, _, backend, receipts, _, _ = execute(tmp_path)
    count = len(backend.scored)
    result, parts, state, backend, new_receipts, _, _ = execute(
        tmp_path, resume=receipts[-1], seed=999, backend=backend
    )
    assert result["progress"]["optimizer_steps"] == 10 and state["budget"]["completed"] == 10
    assert len(backend.scored) == count == 5 and not parts.adapter.transformer.calls
    assert new_receipts == []


def test_nonfinite_reward_fails_before_first_optimizer_and_preserves_journal(tmp_path):
    with pytest.raises(ValueError, match="nonfinite"):
        execute(tmp_path, invalid=True)
    ledger = json.loads((tmp_path / "small-video-step-ledger.json").read_text())
    assert ledger["completed"] == 0 and ledger["pending"] is None
    assert not list(tmp_path.glob("step-*.pt"))


def test_scientific_gate_precedes_distributed_session_or_cuda(monkeypatch):
    def reject(c):
        raise ConfigError("scientific decision pending")

    def forbidden(*args, **kwargs):
        pytest.fail("session/device initialization occurred before science gate")

    monkeypatch.setattr(runner, "distributed_session", forbidden)
    with pytest.raises(ConfigError, match="scientific"):
        runner.run_native(
            SimpleNamespace(), backend=SimpleNamespace(require_training_contract=reject)
        )


@pytest.mark.parametrize(
    "evaluation_only,label",
    [(False, "heldout"), (True, "../escape"), (True, "/absolute"), (True, "nested/name")],
)
def test_extra_evaluation_argument_gate_precedes_session_or_cuda(
    monkeypatch, evaluation_only, label
):
    data = PromptDataset((Prompt("heldout-0", "a held-out scene"),), "3" * 64)

    def forbidden(*args, **kwargs):
        pytest.fail("device initialization occurred before extra-evaluation argument validation")

    monkeypatch.setattr(runner, "distributed_session", forbidden)
    with pytest.raises(ConfigError):
        runner.run_native(
            SimpleNamespace(),
            resume="complete.json",
            evaluate_only=evaluation_only,
            evaluation_dataset=data,
            evaluation_label=label,
            backend=SimpleNamespace(require_training_contract=lambda c: None),
        )


def test_extra_evaluation_rejects_normalized_training_overlap_with_verified_manifests(tmp_path):
    import hashlib

    def manifest(name, text):
        path = tmp_path / name
        path.write_text(json.dumps({"sample_id": name, "prompt": text}) + "\n")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return PromptDataset.load(path, task="t2v", expected_sha256=digest)

    train = manifest("train.jsonl", "  dance  NEXT to a tree ")
    evaluation = manifest("evaluation.jsonl", "ＤＡＮＣＥ next TO A TREE")
    assert train.sha256 != evaluation.sha256
    with pytest.raises(ConfigError, match="normalized training prompt overlap"):
        runner.validate_extra_evaluation(
            evaluation, evaluate_only=True, label="heldout", training=train
        )


def test_extra_evaluation_allows_general_population_without_mutating_training_cursor():
    train = PromptDataset(tuple(Prompt(str(i), f"train scene {i}") for i in range(16)), "4" * 64)
    evaluation = PromptDataset(
        tuple(Prompt(str(i), f"held-out scene {i}") for i in range(13)), "5" * 64
    )
    stream = PromptStream(train, seed=42, shuffle=True)
    before = stream.state_dict()
    runner.validate_extra_evaluation(
        evaluation, evaluate_only=True, label="heldout-13", training=train
    )
    assert len(evaluation.rows) == 13 and stream.state_dict() == before
    with pytest.raises(ConfigError, match="hashed PromptDataset"):
        runner.validate_extra_evaluation(PromptDataset(evaluation.rows, "main"), evaluate_only=True)


@pytest.mark.parametrize("world,local", [(8, 4), (16, 8), (4, 4), (64, 8)])
def test_gpu_acceptance_rejects_any_non_single_eight_gpu_topology_before_loading(
    world, local, tmp_path
):
    session = DistributedSession(
        RankInfo(world_size=world, local_world_size=local), torch.device("cuda", 0), True
    )
    with pytest.raises(ConfigError, match="eight-rank node"):
        runner._run(
            session,
            SimpleNamespace(),
            resume=None,
            evaluate_only=False,
            backend=SimpleNamespace(),
            pause_after=None,
        )
