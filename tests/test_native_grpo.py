import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from test_sd3_runner import config_file, payload, threads, tiny_parts  # noqa: F401

from visual_rl.backends import sd3
from visual_rl.config import ConfigError, load_config
from visual_rl.rewards.scalar import weighted_score
from visual_rl.training.grpo import GRPOState, prepare_grpo_update
from visual_rl.training.grpo_sampling import collect_grpo_rollout
from visual_rl.training.image_contracts import sd3_schedule
from visual_rl.training.rollout import collect_rollout, replay_transition


def grpo_config(tmp_path, algorithm="flow_grpo_fast"):
    c = load_config(config_file(tmp_path))
    loss = {"advantage_clip": 5.0, "ratio_clip": 1e-5}
    sampling = {
        "shift": 1.0,
        "transition_kind": "cps",
        "likelihood": "mse",
        "noise_level": 0.8,
        "selection": "contiguous",
        "pool": [0, 1],
        "count": 1,
        "logical_batch_size": 4,
        "unselected_noise": "ode",
    }
    if algorithm == "flow_grpo":
        sampling.update(
            transition_kind="sde",
            likelihood="gaussian",
            unselected_noise="all",
            selection="fixed",
            count=2,
        )
    if algorithm == "flow_dppo":
        loss = {"advantage_clip": 5.0, "kl_mask_threshold": 1e-5}
    if algorithm == "dancegrpo":
        sampling.update(transition_kind="dance", likelihood="gaussian", unselected_noise="all",
                        selection="random_per_row", pool=[0, 1, 2], count=2, same_latent=True,
                        endpoint_kind="last_x0")
    return replace(
        c,
        algorithm=replace(
            c.algorithm,
            name=algorithm,
            options={
                "loss": loss,
                "updates_per_rollout": 2,
                "training_batches": 4,
                "shuffle_rows": algorithm != "flow_grpo_fast",
                "zero_advantage_mode": "retain",
                "max_empty_rollouts": 2,
                "evaluation_ema": {"enabled": True, "decay": 0.9, "refresh": "logical_batch"},
                **({"advantage_mode": "group_sample_std"} if algorithm == "dancegrpo" else {}),
            },
        ),
        sampling=replace(c.sampling, samples_per_prompt=4, options=sampling),
    )


@pytest.mark.parametrize(
    "algorithm,precision",
    [
        ("flow_grpo", "fp32"),
        ("flow_grpo_fast", "fp32"),
        ("flow_dppo", "fp32"),
        ("flow_grpo_fast", "bf16"),
        ("flow_grpo", "fp16"),
        ("flow_grpo_fast", "fp16"),
        ("flow_dppo", "fp16"),
        ("dancegrpo", "fp32"),
        ("dancegrpo", "fp16"),
    ],
)
def test_native_grpo_train_and_exact_resume(tmp_path, monkeypatch, algorithm, precision):
    c = grpo_config(tmp_path, algorithm)
    c = replace(c, training=replace(c.training, precision=precision))
    monkeypatch.setattr(sd3, "load_components", tiny_parts)
    sd3.run(c)
    root = Path(c.runtime.output_dir)
    expected = payload(root)
    logs = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
    assert [r["train/optimizer_step"] for r in logs] == [1, 2, 3, 4]
    assert [r["system/completed_rollouts"] for r in logs] == [0, 1, 1, 2]
    assert all(r["grpo/retained_rows"] == 8 for r in logs)
    assert any(r["train/grad_norm/before_clip"] > 0 for r in logs)
    assert len(list(root.glob("evaluation/step-*/*/manifest.json"))) == 3
    assert set(expected["stateful"]) == {"prompts", "grpo_eval_ema"} | (
        {"grad_scaler"} if precision == "fp16" else set()
    )
    interrupted = replace(c, runtime=replace(c.runtime, output_dir=str(tmp_path / "interrupted")))
    save = sd3.save_checkpoint

    def stop(*args, **kwargs):
        result = save(*args, **kwargs)
        if kwargs["progress"].optimizer_steps == 2:
            raise RuntimeError("interrupt GRPO checkpoint")
        return result

    monkeypatch.setattr(sd3, "save_checkpoint", stop)
    with pytest.raises(RuntimeError, match="interrupt GRPO"):
        sd3.run(interrupted)
    monkeypatch.setattr(sd3, "save_checkpoint", save)
    root = Path(interrupted.runtime.output_dir)
    sd3.run(interrupted, resume=root / "checkpoints/latest.json")
    actual = payload(root)
    if precision == "fp16":
        assert actual["stateful"]["grad_scaler"] == expected["stateful"]["grad_scaler"]
    for key in expected["model"]:
        torch.testing.assert_close(actual["model"][key], expected["model"][key], rtol=0, atol=0)
    for key in expected["stateful"]["grpo_eval_ema"]:
        torch.testing.assert_close(
            actual["stateful"]["grpo_eval_ema"][key],
            expected["stateful"]["grpo_eval_ema"][key],
            rtol=0,
            atol=0,
        )
    assert torch.equal(actual["generators"]["sampling"], expected["generators"]["sampling"])


def test_grpo_ema_matches_logical_batch_refresh_not_physical_microbatch(tmp_path):
    c = grpo_config(tmp_path)
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    state = GRPOState(model, c.algorithm.options)
    expected = 1.0
    for step in range(2):
        state.before_update(step)
        decay = min((1 + step) / (10 + step), 0.9)
        expected = decay * expected + (1 - decay) * (step + 1)
        with torch.no_grad():
            model.weight.add_(1)
        state.after_update(step + 1)
        decay = min((2 + step) / (11 + step), 0.9)
        expected = decay * expected + (1 - decay) * (step + 2)
        assert state.ema.state_dict()["weight"].item() == pytest.approx(expected)


def test_bf16_model_recomputed_likelihood_matches_sampling_with_different_batches(tmp_path):
    c = grpo_config(tmp_path)
    c = replace(c, training=replace(c.training, precision="bf16"))
    torch.manual_seed(13)
    parts = tiny_parts(c)
    predict = sd3._predict(c, parts, ["2", "3"], 4)
    rng = torch.Generator().manual_seed(8)
    noise = torch.randn(sd3._noise_shape(c, 8), generator=rng, dtype=torch.bfloat16)
    cache = collect_grpo_rollout(
        predict,
        noise,
        sd3_schedule(3, 1),
        pool=(0, 1),
        count=2,
        selection="fixed",
        logical_batch_size=4,
        microbatch=2,
        transition_kind="cps",
        likelihood="mse",
        noise_level=0.8,
        unselected_noise="ode",
        generator=rng,
    )
    rows = torch.arange(8).repeat_interleave(2)
    slots = torch.arange(2).repeat(8)
    with torch.no_grad():
        for batch in (2, 4):
            for start in range(0, len(rows), batch):
                row, slot = rows[start : start + batch], slots[start : start + batch]
                velocity = predict(
                    cache.states[row, slot], cache.sigmas[cache.indices(row, slot)], row
                )
                replay = replay_transition(cache, velocity, row, slot)
                torch.testing.assert_close(
                    velocity.float(), cache.old_velocities[row, slot], rtol=0, atol=0
                )
                torch.testing.assert_close(
                    replay.log_prob, cache.old_log_probs[row, slot], rtol=0, atol=0
                )

        # A CPU BF16 singleton kernel can round differently from batched GEMM.
        # Isolate that effect with direct model calls, without changing times,
        # observations or old likelihoods to force an artificial ratio of one.
        row, slot = torch.tensor([6, 7]), torch.tensor([1, 1])
        times = cache.sigmas[cache.indices(row, slot)]
        pair = predict(cache.states[row, slot], times, row)
        single = predict(cache.states[row, slot][-1:], times[-1:], row[-1:])
        torch.testing.assert_close(pair.float(), cache.old_velocities[row, slot], rtol=0, atol=0)
        replay_pair = replay_transition(cache, pair, row, slot).log_prob[-1:]
        replay_single = replay_transition(cache, single, row[-1:], slot[-1:]).log_prob
        direct_delta = replay_single - replay_pair
        cached_delta = replay_single - cache.old_log_probs[row[-1:], slot[-1:]]
        torch.testing.assert_close(cached_delta, direct_delta, rtol=0, atol=0)
        # This fixed diagnostic's observed likelihood error is below its PPO
        # clip, not a guarantee for other models, devices or batch dimensions.
        assert direct_delta.abs().max() < c.algorithm.options["loss"]["ratio_clip"]


def test_grpo_filter_pad_retains_original_logical_batch_count(tmp_path):
    c = grpo_config(tmp_path)
    c = replace(
        c,
        algorithm=replace(
            c.algorithm, options={**c.algorithm.options, "zero_advantage_mode": "filter_pad"}
        ),
    )
    generator = torch.Generator().manual_seed(2)
    noise = torch.randn(8, 2, generator=generator)

    def predict(x, t, rows):
        return x * 0.1

    cache = collect_rollout(
        predict,
        noise,
        sd3_schedule(3, 1.0),
        selected=(0,),
        stochastic=(0,),
        innovations={0: noise},
        transition_kind="cps",
        likelihood="mse",
    )
    objective, windows, metrics = prepare_grpo_update(
        c,
        cache,
        predict,
        torch.tensor([1.0, 3.0, 2.0, 2.0, 5.0, 5.0, 5.0, 5.0]),
        ["a"] * 4 + ["b"] * 4,
        generator,
        torch.device("cpu"),
    )
    assert metrics == {"grpo/nonzero_rows": 2, "grpo/retained_rows": 4}
    assert len(windows) == 2 and all(len(window) == 2 for window in windows)
    retained = {row for window in windows for row, _ in window}
    assert {0, 1} <= retained
    assert sum(objective.advantages[row].item() == 0 for row in retained) == 2


def test_grpo_empty_rollouts_do_not_update_optimizer_or_ema(tmp_path, monkeypatch):
    c = grpo_config(tmp_path)
    c = replace(
        c,
        algorithm=replace(
            c.algorithm, options={**c.algorithm.options, "zero_advantage_mode": "filter_pad"}
        ),
        evaluation=replace(c.evaluation, every_updates=0, manifest=None, manifest_sha256=None),
    )
    monkeypatch.setattr(sd3, "load_components", tiny_parts)
    score = sd3._score

    def constant(*args, **kwargs):
        result = score(*args, **kwargs)
        return replace(
            result,
            scores=weighted_score(
                {"pickscore": torch.ones(len(result.pixels))}, {"pickscore": 1.0}, space="raw"
            ),
        )

    monkeypatch.setattr(sd3, "_score", constant)

    def forbidden(*args):
        raise AssertionError("empty rollout must not refresh EMA")

    monkeypatch.setattr(GRPOState, "before_update", forbidden)
    monkeypatch.setattr(GRPOState, "after_update", forbidden)
    with pytest.raises(RuntimeError, match="zero-advantage rollout limit"):
        sd3.run(c)
    root = Path(c.runtime.output_dir)
    logs = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
    assert [r["train/optimizer_step"] for r in logs] == [0, 0]
    assert [r["system/rollout_attempts"] for r in logs] == [1, 2]
    assert logs[1]["cost/training_seconds"] > logs[0]["cost/training_seconds"]
    assert not (root / "checkpoints").exists()


@pytest.mark.parametrize(
    "override",
    [
        {"sampling": {"options": {"pool": [0, 2], "selection": "contiguous"}}},
        {"sampling": {"options": {"likelihood": "gaussian"}}},
        {"sampling": {"options": {"pool": [2]}}},
        {"training": {"max_updates": 3}},
    ],
)
def test_grpo_invalid_settings(tmp_path, override):
    c = grpo_config(tmp_path)
    if "sampling" in override:
        c = replace(
            c,
            sampling=replace(
                c.sampling, options={**c.sampling.options, **override["sampling"]["options"]}
            ),
        )
    else:
        c = replace(c, training=replace(c.training, **override["training"]))
    with pytest.raises(ConfigError):
        sd3.validate_config(c)
