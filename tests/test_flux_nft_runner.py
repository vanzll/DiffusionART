"""Offline native FLUX NFT CLI integration with real tiny model/reward assets."""

import hashlib
import json
import math
from dataclasses import replace

import pytest
import torch
import yaml
from test_flux_runner import (  # noqa: F401 -- shared offline fixtures
    assert_evaluations,
    assert_tree_equal,
    local_flux_config,
    offline_cpu,
    precision,
    real_flux_pipeline,
    records,
)
from test_sd3_runner import payload

from visual_rl.backends import sd3
from visual_rl.config import ConfigError, load_config
from visual_rl.models import FluxAdapter
from visual_rl.models.flux import unpack_latents
from visual_rl.registry import get_backend
from visual_rl.train import main
from visual_rl.training import nft_sampling, objectives
from visual_rl.training.image import ConditionedImagePolicy


@pytest.fixture
def local_flux_nft_config(request, precision):  # noqa: F811 -- imported parametrized fixture
    path = request.getfixturevalue("local_flux_config")
    config = yaml.safe_load(path.read_text())
    config["name"] = "flux-nft-native-test"
    config["algorithm"] = {
        "name": "diffusionnft",
        "options": {
            "loss": {"nft_beta": 1.0, "adv_clip_max": 5.0},
            "old_policy": {"rate": 0.001, "maximum": 0.5, "delay": 0},
            "evaluation_ema": {"enabled": True, "decay": 0.9},
        },
    }
    config["sampling"]["options"] = {
        "solver": "euler",
        "train_timestep_count": 2,
        "timestep_selection": "random_per_row",
    }
    config["training"]["max_updates"] = 3
    config["evaluation"]["every_updates"] = 1
    config["runtime"]["checkpoint_every"] = 1
    path.write_text(yaml.safe_dump(config))
    return path


def assert_nft_checkpoint(root, config, step):
    receipt = json.loads((root / "checkpoints/latest.json").read_text())
    checkpoint = root / "checkpoints" / receipt["file"]
    assert receipt["optimizer_step"] == step
    assert receipt["bytes"] == checkpoint.stat().st_size
    assert receipt["sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    state = payload(root)
    assert set(state) == {"schema_version", "config_sha256", "model", "optimizer", "progress",
                          "rng", "stateful", "generators"}
    assert state["config_sha256"] == config.fingerprint()
    for name in ("optimizer_steps", "completed_rollouts", "attempts"):
        assert state["progress"][name] == step
    assert set(state["stateful"]) == {"prompts", "nft_old_policy", "nft_eval_ema"}
    assert set(state["generators"]) == {"sampling"}
    assert set(state["rng"]) == {"python", "numpy", "torch_cpu", "torch_cuda"}
    assert state["rng"]["torch_cuda"] == []
    for weights in (state["model"], state["stateful"]["nft_old_policy"],
                    state["stateful"]["nft_eval_ema"]):
        assert weights
        assert all(value.dtype == torch.float32 and torch.isfinite(value).all()
                   for value in weights.values())
    assert state["optimizer"]["state"] and state["optimizer"]["param_groups"]
    assert all(value["step"].item() == step for value in state["optimizer"]["state"].values())
    assert any(value["exp_avg"].count_nonzero() > 0
               for value in state["optimizer"]["state"].values())
    return state


def test_offline_native_flux_nft_cli_updates_shadows_eval_and_completed_resume(
        local_flux_nft_config, tmp_path, request):
    path = local_flux_nft_config
    config = load_config(path)
    assert "shift" not in config.sampling.options
    assert get_backend(config).name == "flux"
    assert main(["--config", str(path), "--validate"]) == 0
    root = tmp_path / "run"
    assert not root.exists()
    pipeline = request.getfixturevalue("real_flux_pipeline")
    initial = {name: value.detach().clone()
               for name, value in pipeline.transformer.named_parameters()}
    assert main(["--config", str(path)]) == 0
    status = json.loads((root / "status.json").read_text())
    assert status["kind"] == "flux_native_development" and status["status"] == "completed"
    assert status["optimizer_steps"] == status["completed_rollouts"] == 3
    logs = records(root)
    assert [row["train/optimizer_step"] for row in logs] == [1, 2, 3]
    assert [row["system/completed_rollouts"] for row in logs] == [1, 2, 3]
    assert [row["system/rollout_attempts"] for row in logs] == [1, 2, 3]
    assert [row["nft/old_policy_decay"] for row in logs] == [0.001, 0.002, 0.003]
    assert [row["nft/evaluation_ema_decay"] for row in logs] == pytest.approx(
        [2 / 11, 3 / 12, 4 / 13])
    assert any(row["train/grad_norm/before_clip"] > 0 for row in logs)
    for row in logs:
        assert all(math.isfinite(value) for value in row.values()
                   if isinstance(value, (int, float)))
        assert row["cost/elapsed_gpu_hours"] == row["cost/training_gpu_hours"] == 0
        assert row["train/grad_norm/before_clip"] >= 0
        assert row["train/reward"] == pytest.approx(row["train/reward/raw/pickscore"] / 26,
                                                  abs=1e-7)
    assert_evaluations(root, steps=(0, 1, 2, 3))
    state = assert_nft_checkpoint(root, config, 3)
    assert any(not torch.equal(initial[key], state["model"][key]) for key in initial)
    receipts = sorted(root.glob("checkpoints/step-*.json"))
    assert [json.loads(p.read_text())["optimizer_step"] for p in receipts] == [1, 2, 3]
    previous = {"nft_old_policy": initial, "nft_eval_ema": initial}
    for step, receipt_path in enumerate(receipts, 1):
        receipt = json.loads(receipt_path.read_text())
        checkpoint = root / "checkpoints" / receipt["file"]
        assert receipt["sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        saved = torch.load(checkpoint, weights_only=True)
        assert saved["progress"]["optimizer_steps"] == step
        assert saved["progress"]["completed_rollouts"] == step
        for name, decay in (("nft_old_policy", step * 0.001),
                            ("nft_eval_ema", (1 + step) / (10 + step))):
            shadow = saved["stateful"][name]
            assert shadow.keys() == initial.keys()
            for key, value in shadow.items():
                expected = previous[name][key] * decay + saved["model"][key] * (1 - decay)
                torch.testing.assert_close(value, expected, rtol=1e-6, atol=1e-7)
            previous[name] = shadow
    assert any(not torch.equal(previous["nft_old_policy"][key], previous["nft_eval_ema"][key])
               for key in initial)
    existing = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert main(["--config", str(path), "--resume", str(root / "checkpoints/latest.json")]) == 0
    assert {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()} == existing


def test_offline_native_flux_nft_interrupted_boundary_resume_is_exact(
        local_flux_nft_config, tmp_path, monkeypatch):
    path = local_flux_nft_config
    assert main(["--config", str(path)]) == 0
    expected_root = tmp_path / "run"
    expected = payload(expected_root)
    expected_evaluations = assert_evaluations(expected_root, steps=(0, 1, 2, 3))
    root = tmp_path / "interrupted"
    argv = ["--config", str(path), "--output-dir", str(root)]
    save = sd3.save_checkpoint

    def stop_after_boundary(*args, **kwargs):
        receipt = save(*args, **kwargs)
        if kwargs["progress"].optimizer_steps == 1:
            raise RuntimeError("injected shutdown after full FLUX NFT checkpoint")
        return receipt

    with monkeypatch.context() as patch:
        patch.setattr(sd3, "save_checkpoint", stop_after_boundary)
        with pytest.raises(RuntimeError, match="injected shutdown after full FLUX NFT checkpoint"):
            main(argv)
    assert not (root / "status.json").exists()
    config = load_config(path, {"runtime": {"output_dir": str(root)}})
    assert_nft_checkpoint(root, config, 1)
    assert_evaluations(root, steps=(0, 1))
    prefix = (root / "metrics.jsonl").read_bytes()
    assert [row["train/optimizer_step"] for row in records(root)] == [1]
    assert main([*argv, "--resume", str(root / "checkpoints/latest.json")]) == 0
    actual = assert_nft_checkpoint(root, config, 3)
    for key in ("model", "optimizer", "stateful", "generators", "rng"):
        assert_tree_equal(actual[key], expected[key])
    for key in ("optimizer_steps", "completed_rollouts", "attempts"):
        assert actual["progress"][key] == expected["progress"][key]
    assert (root / "metrics.jsonl").read_bytes().startswith(prefix)
    actual_logs, expected_logs = records(root), records(expected_root)
    assert [row["train/optimizer_step"] for row in actual_logs] == [1, 2, 3]
    for a, b in zip(actual_logs, expected_logs, strict=True):
        assert {k: v for k, v in a.items() if k.startswith(("train/", "nft/", "system/"))} == {
            k: v for k, v in b.items() if k.startswith(("train/", "nft/", "system/"))}
    assert assert_evaluations(root, steps=(0, 1, 2, 3)) == expected_evaluations


@pytest.mark.parametrize("invalid,match", [
    ({"solver": "dpm2"}, "(?i)(euler|solver|dpm2)"),
    ({"shift": 1.0}, "(?i)(shift|sampling)"),
])
def test_flux_nft_rejects_sd3_sampling_options(local_flux_nft_config, tmp_path, invalid, match):
    config = load_config(local_flux_nft_config)
    assert get_backend(config).name == "flux"
    config = replace(config, sampling=replace(
        config.sampling, options={**config.sampling.options, **invalid}))
    with pytest.raises(ConfigError, match=match):
        get_backend(config)
    assert not (tmp_path / "run").exists()


def test_flux_nft_real_forward_keeps_training_replay_fp32_and_rollout_precision(
        local_flux_nft_config, request, monkeypatch):
    # Canonical nft_flux_runtime.fresh_training_state/flux_prediction keep xt
    # FP32 even under BF16 autocast, unlike rollout's rounded latent inputs.
    phase = None
    seen = {"rollout": [], "old_replay": [], "train_replay": []}
    policy_predict = ConditionedImagePolicy._predict
    adapter_predict = FluxAdapter.predict_velocity

    def record_phase(self, state, sigma, rows, *, rollout):
        nonlocal phase
        phase = "rollout" if rollout else (
            "train_replay" if torch.is_grad_enabled() else "old_replay")
        try:
            return policy_predict(self, state, sigma, rows, rollout=rollout)
        finally:
            phase = None

    def record_inputs(self, latents, sigma, conditioning, *, timestep_dtype=None):
        assert phase in seen
        seen[phase].append((latents.detach().clone(), timestep_dtype))
        return adapter_predict(self, latents, sigma, conditioning, timestep_dtype=timestep_dtype)

    monkeypatch.setattr(ConditionedImagePolicy, "_predict", record_phase)
    monkeypatch.setattr(FluxAdapter, "predict_velocity", record_inputs)
    assert main(["--config", str(local_flux_nft_config)]) == 0
    rollout_dtype = torch.bfloat16 if request.getfixturevalue("precision") == "bf16" else torch.float32
    for name, observations in seen.items():
        assert observations, f"missing real FLUX forward coverage: {name}"
        expected_dtype = rollout_dtype if name == "rollout" else torch.float32
        assert all(latents.dtype == expected_dtype and time_dtype == expected_dtype
                   for latents, time_dtype in observations)
        assert all(torch.isfinite(latents).all() for latents, _ in observations)
    for name in ("old_replay", "train_replay"):
        assert any(not torch.equal(latents, latents.bfloat16().float())
                   for latents, _ in seen[name])
    assert_tree_equal(seen["train_replay"], seen["old_replay"])


def test_flux_nft_packed_fresh_noise_model_times_and_fp32_loss_inputs(
        local_flux_nft_config, monkeypatch):
    collect = nft_sampling.collect_nft_endpoints
    prepare = sd3.prepare_nft_objective
    loss = objectives.nft_loss
    counts = {"rollout": 0, "prepare": 0, "loss": 0}

    def checked_collect(predict, noise, sigmas, **kwargs):
        counts["rollout"] += 1
        assert kwargs["solver"] == "euler"
        assert kwargs["quantize_model_time"] is False
        assert torch.any(sigmas * 1000 != (sigmas * 1000).floor())
        times = []

        def checked_predict(state, sigma, rows):
            assert state.dtype == noise.dtype
            times.append(sigma.detach().clone())
            return predict(state, sigma, rows)

        result = collect(checked_predict, noise, sigmas, **kwargs)
        for index, sigma in enumerate(times):
            torch.testing.assert_close(sigma, sigmas[index % (len(sigmas) - 1)].expand_as(sigma),
                                       rtol=0, atol=0)
        assert times and result.dtype == torch.float32
        return result

    def checked_prepare(c, state, predict, endpoints, schedule, advantages, generator, device):
        counts["prepare"] += 1
        draws = []
        randn = torch.randn

        def record_noise(*args, **kwargs):
            result = randn(*args, **kwargs)
            draws.append(result.detach().clone())
            return result

        with monkeypatch.context() as patch:
            patch.setattr(torch, "randn", record_noise)
            objective = prepare(c, state, predict, endpoints, schedule, advantages, generator, device)
        assert objective.upcast_predictions is True
        assert len(draws) == len(endpoints) * c.sampling.options["train_timestep_count"]
        channels, height, width = endpoints.shape[1:]
        assert all(draw.shape == (1, height * width // 4, channels * 4)
                   and draw.dtype == torch.float32 for draw in draws)
        noise = unpack_latents(torch.cat(draws), height, width)
        times = objective.sigmas.reshape(-1, 1, 1, 1)
        x0 = endpoints[objective.row_indices].float()
        torch.testing.assert_close(objective.endpoints, x0, rtol=0, atol=0)
        torch.testing.assert_close(objective.states, (1 - times) * x0 + times * noise,
                                   rtol=0, atol=0)
        normalized = (schedule[:-1] * 1000.0) / 1000.0
        assert torch.isin(objective.sigmas, normalized).all()
        for row in range(len(endpoints)):
            selected = objective.sigmas[objective.row_indices == row]
            assert len(selected.unique()) == c.sampling.options["train_timestep_count"]
        return objective

    def checked_loss(prediction, old_prediction, *args, **kwargs):
        counts["loss"] += 1
        assert prediction.requires_grad and not old_prediction.requires_grad
        assert prediction.dtype == old_prediction.dtype == torch.float32
        return loss(prediction, old_prediction, *args, **kwargs)

    monkeypatch.setattr(nft_sampling, "collect_nft_endpoints", checked_collect)
    monkeypatch.setattr(sd3, "prepare_nft_objective", checked_prepare)
    monkeypatch.setattr(objectives, "nft_loss", checked_loss)
    assert main(["--config", str(local_flux_nft_config)]) == 0
    assert counts["rollout"] == counts["prepare"] == 3
    assert counts["loss"] > 0
