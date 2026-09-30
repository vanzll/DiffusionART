import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from test_sd3_runner import config_file, payload, threads, tiny_parts  # noqa: F401

from visual_rl.backends import sd3
from visual_rl.config import ConfigError, load_config
from visual_rl.models.finetune import FineTuneConfig, prepare_policy
from visual_rl.training.nft import (
    NFTState,
    nft_advantages,
    nft_schedule,
    nft_windows,
    prepare_nft_objective,
)
from visual_rl.training.state import PolicyShadow


def nft_config(tmp_path):
    c = load_config(config_file(tmp_path))
    return replace(
        c,
        algorithm=replace(
            c.algorithm,
            name="diffusionnft",
            options={
                "loss": {"nft_beta": 1.0, "adv_clip_max": 5.0},
                "old_policy": {"rate": 0.001, "maximum": 0.5, "delay": 0},
                "evaluation_ema": {"enabled": True, "decay": 0.9},
            },
        ),
        sampling=replace(
            c.sampling,
            options={
                "shift": 1.0,
                "solver": "dpm2",
                "train_timestep_count": 2,
                "timestep_selection": "random_per_row",
            },
        ),
        training=replace(c.training, max_updates=3),
        evaluation=replace(c.evaluation, every_updates=1),
        runtime=replace(c.runtime, checkpoint_every=1),
    )


@pytest.mark.parametrize("precision,mode", [("fp32", "full"), ("bf16", "lora"), ("fp16", "full"), ("fp16", "lora")])
def test_nft_native_updates_shadow_resume_and_eval(tmp_path, monkeypatch, precision, mode):
    c = nft_config(tmp_path)
    finetune = (
        {"mode": "full"}
        if mode == "full"
        else {"mode": "lora", "rank": 2, "alpha": 4, "target_modules": ["attn.to_q", "attn.to_v"]}
    )
    c = replace(
        c,
        training=replace(c.training, precision=precision),
        model=replace(c.model, options={**c.model.options, "finetune": finetune}),
    )

    def parts(config):
        result = tiny_parts(config)
        result.adapter.transformer = prepare_policy(
            result.adapter.transformer, FineTuneConfig(**finetune)
        )
        result.adapter.eval()
        return result

    monkeypatch.setattr(sd3, "load_components", parts)
    sd3.run(c)
    root = Path(c.runtime.output_dir)
    expected = payload(root)
    rows = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
    assert [r["system/completed_rollouts"] for r in rows] == [1, 2, 3]
    assert [r["nft/old_policy_decay"] for r in rows] == [0.001, 0.002, 0.003]
    assert [r["nft/evaluation_ema_decay"] for r in rows] == pytest.approx([2 / 11, 3 / 12, 4 / 13])
    assert len(list(root.glob("evaluation/step-*/*/manifest.json"))) == 4
    assert set(expected["stateful"]) == {"prompts", "nft_old_policy", "nft_eval_ema"} | (
        {"grad_scaler"} if precision == "fp16" else set()
    )
    if mode == "lora":
        assert all("lora_" in key for key in expected["stateful"]["nft_old_policy"])
    interrupted = replace(c, runtime=replace(c.runtime, output_dir=str(tmp_path / "interrupted")))
    save = sd3.save_checkpoint

    def stop(*args, **kwargs):
        result = save(*args, **kwargs)
        if kwargs["progress"].optimizer_steps == 1:
            raise RuntimeError("interrupt NFT after first boundary")
        return result

    monkeypatch.setattr(sd3, "save_checkpoint", stop)
    with pytest.raises(RuntimeError, match="interrupt NFT"):
        sd3.run(interrupted)
    monkeypatch.setattr(sd3, "save_checkpoint", save)
    root = Path(interrupted.runtime.output_dir)
    sd3.run(interrupted, resume=root / "checkpoints/latest.json")
    actual = payload(root)
    if precision == "fp16":
        assert actual["stateful"]["grad_scaler"] == expected["stateful"]["grad_scaler"]
    for name, value in expected["model"].items():
        torch.testing.assert_close(actual["model"][name], value, rtol=0, atol=0)
    for state in ("nft_old_policy", "nft_eval_ema"):
        for name, value in expected["stateful"][state].items():
            torch.testing.assert_close(actual["stateful"][state][name], value, rtol=0, atol=0)
    assert actual["stateful"]["prompts"] == expected["stateful"]["prompts"]
    assert torch.equal(actual["generators"]["sampling"], expected["generators"]["sampling"])


def test_trainable_shadow_does_not_copy_frozen_base_or_buffers():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    model[0].requires_grad_(False)
    model.register_buffer("counter", torch.tensor(1))
    shadow = PolicyShadow(model, trainable_only=True)
    assert set(shadow.state_dict()) == {"1.weight", "1.bias"}
    original = model[1].weight.detach().clone()
    with torch.no_grad():
        model[1].weight.add_(1)
        model[0].weight.add_(2)
        model.counter.add_(3)
    frozen = model[0].weight.detach().clone()
    with pytest.raises(RuntimeError, match="injected"):
        with shadow.use():
            torch.testing.assert_close(model[1].weight, original)
            raise RuntimeError("injected")
    torch.testing.assert_close(model[1].weight, original + 1)
    torch.testing.assert_close(model[0].weight, frozen)
    assert model.counter.item() == 4


def test_separate_old_and_eval_ema_one_based_refresh(tmp_path):
    c = nft_config(tmp_path)
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1)
    state = NFTState(model, c.algorithm.options)
    with torch.no_grad():
        model.weight.fill_(2)
    metrics = state.after_update(1)
    with state.old.use():
        assert model.weight.item() == pytest.approx(1.999)
    with state.evaluation():
        assert model.weight.item() == pytest.approx(2 - 2 / 11)
    assert model.weight.item() == 2
    assert metrics["nft/old_policy_decay"] == 0.001


def test_nft_advantage_float64_global_population_duplicate_prompt():
    rewards = torch.tensor([1.0, 3.0, 9.0, 9.0, 5.0, 7.0])
    prompts = ["a", "a", "b", "b", "a", "a"]
    expected = torch.tensor([-3.0, -1.0, 0.0, 0.0, 1.0, 3.0]).double() / (
        rewards.double().std(unbiased=False) + 1e-4
    )
    torch.testing.assert_close(nft_advantages(rewards, prompts), expected.float(), rtol=0, atol=0)


def test_nft_preparation_independent_of_physical_microbatch(tmp_path):
    c = nft_config(tmp_path)
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.2)
    state = NFTState(model, c.algorithm.options)
    with torch.no_grad():
        model.weight.fill_(0.3)

    def predict(x, t, rows):
        return model(x) + t[:, None] * 0.1

    endpoints = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    schedule = nft_schedule(3, 1)
    advantages = torch.tensor([-1.0, 1.0, 0.0, 0.0])

    def prepare(mb):
        return prepare_nft_objective(
            replace(c, training=replace(c.training, microbatch=mb)),
            state,
            predict,
            endpoints,
            schedule,
            advantages,
            torch.Generator().manual_seed(12),
            torch.device("cpu"),
        )

    one, all_ = prepare(1), prepare(8)
    for key in ("states", "sigmas", "old_predictions", "endpoints", "row_indices", "advantages"):
        torch.testing.assert_close(getattr(one, key), getattr(all_, key), rtol=0, atol=0)
    torch.testing.assert_close(one.old_predictions, 0.2 * one.states + 0.1 * one.sigmas[:, None])
    assert not one.states.requires_grad and not one.old_predictions.requires_grad
    assert len(one.states) == 8  # Include zero-advantage rows in the denominator.
    assert model.weight.item() == pytest.approx(0.3)
    model.zero_grad(set_to_none=True)
    (one(list(range(8))).sum() / 8).backward()
    gradient = model.weight.grad.clone()
    model.zero_grad(set_to_none=True)
    for i in range(8):
        (one([i]).sum() / 8).backward()
    torch.testing.assert_close(model.weight.grad, gradient, rtol=1e-5, atol=1e-7)


def test_nft_two_disjoint_windows_keep_endpoint_rows_and_old_policy_frozen(tmp_path):
    c = nft_config(tmp_path)
    c = replace(c, algorithm=replace(c.algorithm, options={**c.algorithm.options, "updates_per_rollout": 2}))
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1)
    state = NFTState(model, c.algorithm.options)
    obj = prepare_nft_objective(c, state, lambda x, t, rows: model(x),
        torch.arange(4.0)[:, None], nft_schedule(3, 1), torch.arange(4.0),
        torch.Generator().manual_seed(17), torch.device("cpu"))
    windows = nft_windows(c, obj)
    assert list(windows[0] + windows[1]) == list(range(8))
    first = set(obj.row_indices[list(windows[0])].tolist())
    second = set(obj.row_indices[list(windows[1])].tolist())
    assert len(first) == len(second) == 2 and not first & second
    with torch.no_grad():
        model.weight.fill_(2)
    metrics = state.after_update(1, refresh_old=False)
    assert metrics["nft/old_policy_refreshed"] == 0
    with state.old.use():
        assert model.weight.item() == 1
    with state.evaluation():
        assert model.weight.item() == pytest.approx(2 - 2 / 11)
    with torch.no_grad():
        model.weight.fill_(3)
    metrics = state.after_update(2)
    assert metrics["nft/old_policy_refreshed"] == 1
    with state.old.use():
        assert model.weight.item() == pytest.approx(1 * .002 + 3 * .998)


@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_nft_two_update_exact_resume(tmp_path, monkeypatch, precision):
    from test_distributed_checkpoint import assert_tree_equal

    c = nft_config(tmp_path)
    c = replace(c,
        algorithm=replace(c.algorithm, options={**c.algorithm.options, "updates_per_rollout": 2}),
        training=replace(c.training, max_updates=4, precision=precision),
        evaluation=replace(c.evaluation, every_updates=2),
        runtime=replace(c.runtime, checkpoint_every=2))
    monkeypatch.setattr(sd3, "load_components", tiny_parts)
    sd3.validate_config(c)
    sd3.run(c)
    root = Path(c.runtime.output_dir)
    expected = payload(root)
    logs = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
    assert [row["system/completed_rollouts"] for row in logs] == [0, 1, 1, 2]
    assert [row["nft/old_policy_refreshed"] for row in logs] == [0, 1, 0, 1]
    interrupted = replace(c, runtime=replace(c.runtime, output_dir=str(tmp_path / "interrupted-two")))
    save = sd3.save_checkpoint

    def stop(*args, **kwargs):
        receipt = save(*args, **kwargs)
        if kwargs["progress"].optimizer_steps == 2:
            raise RuntimeError("interrupt two-update NFT")
        return receipt

    monkeypatch.setattr(sd3, "save_checkpoint", stop)
    with pytest.raises(RuntimeError, match="interrupt two-update"):
        sd3.run(interrupted)
    monkeypatch.setattr(sd3, "save_checkpoint", save)
    root = Path(interrupted.runtime.output_dir)
    sd3.run(interrupted, resume=root / "checkpoints/latest.json")
    actual = payload(root)
    for key in ("model", "optimizer", "stateful", "generators"):
        assert_tree_equal(expected[key], actual[key])


@pytest.mark.parametrize("overrides", [
    {"updates_per_rollout": 0}, {"updates_per_rollout": True},
    {"updates_per_rollout": 3}, {"unknown": 2},
])
def test_nft_rejects_invalid_window_options(tmp_path, overrides):
    c = nft_config(tmp_path)
    c = replace(c, algorithm=replace(c.algorithm, options={**c.algorithm.options, **overrides}))
    with pytest.raises(ConfigError):
        sd3.validate_config(c)


def test_nft_two_update_rejects_mid_window_artifacts(tmp_path):
    c = nft_config(tmp_path)
    c = replace(c,
        algorithm=replace(c.algorithm, options={**c.algorithm.options, "updates_per_rollout": 2}),
        training=replace(c.training, max_updates=4))
    with pytest.raises(ConfigError, match="complete rollout"):
        sd3.validate_config(c)


def test_nft_bf16_forward_keeps_fresh_noise_fp32(tmp_path, monkeypatch):
    c = nft_config(tmp_path)
    c = replace(c, training=replace(c.training, precision="bf16"))
    model = torch.nn.Linear(1, 1)
    state = NFTState(model, c.algorithm.options)
    original_randn = torch.randn
    dtypes = []

    def record(*args, **kwargs):
        dtypes.append(kwargs.get("dtype"))
        return original_randn(*args, **kwargs)

    monkeypatch.setattr(torch, "randn", record)

    def predict(x, t, rows):
        return model(x).bfloat16()

    objective = prepare_nft_objective(
        c,
        state,
        predict,
        torch.ones(4, 1),
        nft_schedule(3, 1),
        torch.zeros(4),
        torch.Generator().manual_seed(12),
        torch.device("cpu"),
    )
    assert dtypes == [torch.float32] * 8
    assert objective.states.dtype == torch.float32


@pytest.mark.parametrize(
    "options",
    [
        {"old_policy": {"rate": -1}},
        {"evaluation_ema": {"enabled": "yes", "decay": 0.9}},
        {"loss": {"reference_mse_beta": 0.1}},
    ],
)
def test_nft_invalid_options_rejected(tmp_path, options):
    c = nft_config(tmp_path)
    c = replace(c, algorithm=replace(c.algorithm, options={**c.algorithm.options, **options}))
    with pytest.raises(ConfigError):
        sd3.validate_config(c)
