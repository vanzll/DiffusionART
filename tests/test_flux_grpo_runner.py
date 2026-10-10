"""Offline real-asset FLUX FlowGRPO/Fast/DPPO CLI, recovery, and precision wiring."""

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
import yaml
from test_flux_runner import (  # noqa: F401 -- shared offline asset fixtures
    assert_evaluations,
    assert_tree_equal,
    local_flux_config,
    offline_cpu,
    precision,
    real_flux_pipeline,
    records,
)
from test_native_grpo import grpo_config
from test_sd3_runner import payload

from visual_rl.backends import sd3
from visual_rl.config import ConfigError, load_config
from visual_rl.models import FluxAdapter
from visual_rl.registry import get_backend
from visual_rl.train import main
from visual_rl.training import grpo_sampling, objectives, rollout
from visual_rl.training.image import ConditionedImagePolicy


@pytest.fixture(params=["flow_grpo", "flow_grpo_fast", "flow_dppo"])
def algorithm(request):
    return request.param


@pytest.fixture
def local_flux_grpo_config(local_flux_config, tmp_path, algorithm):  # noqa: F811
    path = local_flux_config
    config = yaml.safe_load(path.read_text())
    recipe_dir = tmp_path / "grpo-recipe"
    recipe_dir.mkdir()
    recipe = grpo_config(recipe_dir, algorithm)
    config["name"] = f"flux-{algorithm}-native-test"
    config["algorithm"] = asdict(recipe.algorithm)
    config["algorithm"]["options"]["updates_per_rollout"] = 1
    config["sampling"]["samples_per_prompt"] = recipe.sampling.samples_per_prompt
    config["sampling"]["options"] = dict(recipe.sampling.options)
    config["sampling"]["options"].pop("shift")
    if algorithm == "flow_dppo":
        config["sampling"]["options"].update(transition_kind="sde", likelihood="gaussian",
                                            selection="fixed", count=2, noise_level=.7)
    config["training"]["max_updates"] = 2
    config["evaluation"]["every_updates"] = 1
    config["runtime"]["checkpoint_every"] = 1
    path.write_text(yaml.safe_dump(config))
    return path


def assert_grpo_checkpoint(root, config, step):
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
    assert set(state["stateful"]) == {"prompts", "grpo_eval_ema"}
    assert set(state["generators"]) == {"sampling"}
    assert set(state["rng"]) == {"python", "numpy", "torch_cpu", "torch_cuda"}
    assert state["rng"]["torch_cuda"] == []
    for weights in (state["model"], state["stateful"]["grpo_eval_ema"]):
        assert weights
        assert all(value.dtype == torch.float32 and torch.isfinite(value).all()
                   for value in weights.values())
    assert state["optimizer"]["state"] and state["optimizer"]["param_groups"]
    assert all(value["step"].item() == step for value in state["optimizer"]["state"].values())
    assert any(value["exp_avg"].count_nonzero() > 0
               for value in state["optimizer"]["state"].values())
    return state


def test_offline_flux_grpo_cli_two_updates_scoring_ema_eval_and_completed_resume(
        local_flux_grpo_config, tmp_path, request):
    path = local_flux_grpo_config
    config = load_config(path)
    assert config.model.name == "flux" and "shift" not in config.sampling.options
    assert get_backend(config).name == "flux"
    asset_scheduler = json.loads(
        (tmp_path / "flux-asset/scheduler/scheduler_config.json").read_text())
    assert all(asset_scheduler[key] == value
               for key, value in config.model.options["scheduler"].items())
    assert main(["--config", str(path), "--validate"]) == 0
    root = tmp_path / "run"
    assert not root.exists()
    initial = {name: value.detach().clone() for name, value in
               request.getfixturevalue("real_flux_pipeline").transformer.named_parameters()}
    assert main(["--config", str(path)]) == 0
    status = json.loads((root / "status.json").read_text())
    assert status["kind"] == "flux_native_development" and status["status"] == "completed"
    assert status["optimizer_steps"] == status["completed_rollouts"] == 2
    logs = records(root)
    for name in ("train/optimizer_step", "system/completed_rollouts", "system/rollout_attempts"):
        assert [row[name] for row in logs] == [1, 2]
    assert all(row["grpo/retained_rows"] == 8 for row in logs)
    assert any(row["train/grad_norm/before_clip"] > 0 for row in logs)
    for row in logs:
        assert all(math.isfinite(value) for value in row.values()
                   if isinstance(value, (int, float)))
        assert row["cost/elapsed_gpu_hours"] == row["cost/training_gpu_hours"] == 0
        assert row["train/grad_norm/before_clip"] >= 0
        assert row["train/reward"] == pytest.approx(
            row["train/reward/raw/pickscore"] / 26, abs=1e-7)
    assert_evaluations(root, steps=(0, 1, 2))
    state = assert_grpo_checkpoint(root, config, 2)
    assert any(not torch.equal(initial[key], state["model"][key]) for key in initial)
    receipts = sorted(root.glob("checkpoints/step-*.json"))
    assert [json.loads(p.read_text())["optimizer_step"] for p in receipts] == [1, 2]
    previous_model, previous_ema = initial, initial
    for step, receipt_path in enumerate(receipts, 1):
        receipt = json.loads(receipt_path.read_text())
        checkpoint = root / "checkpoints" / receipt["file"]
        assert receipt["sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        saved = torch.load(checkpoint, weights_only=True)
        shadow = saved["stateful"]["grpo_eval_ema"]
        assert shadow.keys() == initial.keys()
        # Three logical-batch refreshes precede the optimizer, one follows it.
        before_decay = min(step / (step + 9), 0.9)
        after_decay = min((step + 1) / (step + 10), 0.9)
        for key, value in shadow.items():
            expected = previous_ema[key]
            for _ in range(config.algorithm.options["training_batches"] - 1):
                expected = expected * before_decay + previous_model[key] * (1 - before_decay)
            expected = expected * after_decay + saved["model"][key] * (1 - after_decay)
            torch.testing.assert_close(value, expected, rtol=1e-6, atol=1e-7)
        previous_model, previous_ema = saved["model"], shadow
    assert any(not torch.equal(previous_model[key], previous_ema[key]) for key in initial)
    existing = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert main(["--config", str(path), "--resume", str(root / "checkpoints/latest.json")]) == 0
    assert {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()} == existing


def test_offline_flux_grpo_interrupted_boundary_resume_is_exact(
        local_flux_grpo_config, tmp_path, monkeypatch):
    path = local_flux_grpo_config
    assert main(["--config", str(path)]) == 0
    expected_root = tmp_path / "run"
    expected = payload(expected_root)
    expected_evaluations = assert_evaluations(expected_root, steps=(0, 1, 2))
    root = tmp_path / "interrupted"
    argv = ["--config", str(path), "--output-dir", str(root)]
    save = sd3.save_checkpoint

    def stop_after_boundary(*args, **kwargs):
        receipt = save(*args, **kwargs)
        if kwargs["progress"].optimizer_steps == 1:
            raise RuntimeError("injected shutdown after full FLUX GRPO checkpoint")
        return receipt

    with monkeypatch.context() as patch:
        patch.setattr(sd3, "save_checkpoint", stop_after_boundary)
        with pytest.raises(RuntimeError, match="injected shutdown after full FLUX GRPO checkpoint"):
            main(argv)
    assert not (root / "status.json").exists()
    config = load_config(path, {"runtime": {"output_dir": str(root)}})
    assert_grpo_checkpoint(root, config, 1)
    assert_evaluations(root, steps=(0, 1))
    prefix = (root / "metrics.jsonl").read_bytes()
    assert [row["train/optimizer_step"] for row in records(root)] == [1]
    assert main([*argv, "--resume", str(root / "checkpoints/latest.json")]) == 0
    actual = assert_grpo_checkpoint(root, config, 2)
    for key in ("model", "optimizer", "stateful", "generators", "rng"):
        assert_tree_equal(actual[key], expected[key])
    for key in ("optimizer_steps", "completed_rollouts", "attempts"):
        assert actual["progress"][key] == expected["progress"][key]
    assert (root / "metrics.jsonl").read_bytes().startswith(prefix)
    actual_logs, expected_logs = records(root), records(expected_root)
    assert [row["train/optimizer_step"] for row in actual_logs] == [1, 2]
    for a, b in zip(actual_logs, expected_logs, strict=True):
        assert {k: v for k, v in a.items() if k.startswith(("train/", "grpo/", "system/"))} == {
            k: v for k, v in b.items() if k.startswith(("train/", "grpo/", "system/"))}
    assert assert_evaluations(root, steps=(0, 1, 2)) == expected_evaluations


@pytest.mark.parametrize("invalid,match", [
    ({"pool": [0, 3]}, "pool"),
    ({"pool": [1, 0]}, "pool"),
    ({"pool": [0, 0]}, "pool"),
    ({"pool": [0, 2], "selection": "contiguous"}, "contiguous"),
    ({"transition_kind": "not-an-sde"}, "transition"),
    ({"transition_kind": "cps", "likelihood": "gaussian"}, "CPS"),
    ({"shift": 1.0}, "sampling"),
    (None, None),
])
def test_flux_grpo_cli_rejects_invalid_sampling_before_loading(
        local_flux_grpo_config, tmp_path, monkeypatch, capsys, algorithm, invalid, match):
    if invalid is None:
        invalid, match = (
            ({"transition_kind": "cps", "likelihood": "mse"}, "full SDE")
            if algorithm == "flow_grpo" else
            ({"likelihood": "mse"}, "requires SDE with Gaussian") if algorithm == "flow_dppo"
            else ({"pool": [2]}, "terminal zero-variance CPS")
        )
    path = local_flux_grpo_config
    config = yaml.safe_load(path.read_text())
    config["sampling"]["options"].update(invalid)
    path.write_text(yaml.safe_dump(config))

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid sampling must fail before model loading")

    monkeypatch.setattr(sd3, "load_components", forbidden)
    with pytest.raises(ConfigError, match=match):
        get_backend(load_config(path))
    assert main(["--config", str(path)]) == 2
    assert match in capsys.readouterr().err
    assert not (tmp_path / "run").exists()


def test_flux_grpo_real_forward_rollout_rounding_replay_inputs_and_times(
        local_flux_grpo_config, precision, algorithm, monkeypatch):  # noqa: F811
    diffusers = pytest.importorskip("diffusers")
    dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    phase = None
    seen = {"rollout": [], "replay": []}
    counts = {"logical_rollouts": 0, "replay": 0, "rounded": 0, "cps_zero_noise": 0,
              "evaluation_batches": 0, "vae": 0, "affine_rounded": 0}
    policy_predict = ConditionedImagePolicy._predict
    forward = diffusers.FluxTransformer2DModel.forward
    collect = grpo_sampling.collect_rollout
    transition = rollout.flow_transition
    replay = objectives.replay_transition
    evaluation_collect = sd3.collect_rollout
    decode = FluxAdapter.decode

    def record_phase(self, state, sigma, rows, *, rollout):
        nonlocal phase
        phase = ("rollout" if rollout else "replay", sigma.detach().clone())
        try:
            return policy_predict(self, state, sigma, rows, rollout=rollout)
        finally:
            phase = None

    def checked_forward(self, *args, **kwargs):
        assert phase is not None
        name, sigma = phase
        state, timestep = kwargs["hidden_states"], kwargs["timestep"]
        assert state.dtype == dtype and state.device.type == "cpu"
        expected_time_dtype = dtype if name == "rollout" else torch.float32
        assert timestep.dtype == expected_time_dtype
        expected_time = (sigma.float() * 1000).to(expected_time_dtype) / 1000
        torch.testing.assert_close(timestep, expected_time, rtol=0, atol=0)
        assert torch.isfinite(state).all()
        seen[name].append(timestep.detach().clone())
        return forward(self, *args, **kwargs)

    def checked_collect(predict, noise, sigmas, **kwargs):
        counts["logical_rollouts"] += 1
        assert noise.dtype == dtype and kwargs["preserve_state_dtype"] is True
        assert kwargs["transition_kind"] == ("cps" if algorithm == "flow_grpo_fast" else "sde")
        assert kwargs["likelihood"] == ("mse" if algorithm == "flow_grpo_fast" else "gaussian")
        assert kwargs["unselected_kind"] == ("ode" if algorithm != "flow_grpo_fast"
                                             else "cps_zero_noise")
        assert kwargs["stochastic"] == (tuple(range(len(sigmas) - 1))
                                         if algorithm == "flow_grpo" else kwargs["selected"])
        observations = []

        def checked_predict(state, sigma, rows):
            # Observe the collector too: an adapter cast must not hide an FP32 trajectory.
            assert state.dtype == dtype
            return predict(state, sigma, rows)

        def capture_transition(*args, **options):
            result = transition(*args, **options)
            assert result.sample.dtype == torch.float32
            observations.append((args[0].detach().clone(), result))
            if options["kind"] == "cps_zero_noise":
                counts["cps_zero_noise"] += 1
                assert options.get("innovation") is None
            return result

        cache = collect(checked_predict, noise, sigmas, transition_fn=capture_transition, **kwargs)
        steps = len(sigmas) - 1
        assert len(observations) == math.ceil(len(noise) / kwargs["microbatch"]) * steps
        for index, (state, result) in enumerate(observations):
            start = (index // steps) * kwargs["microbatch"]
            rows = slice(start, start + len(state))
            step = index % steps
            rounded = result.sample.to(dtype).float()
            if step < steps - 1:
                torch.testing.assert_close(observations[index + 1][0].float(), rounded,
                                           rtol=0, atol=0)
            else:
                torch.testing.assert_close(cache.endpoints[rows], rounded, rtol=0, atol=0)
            if step in cache.selected:
                slot = cache.selected.index(step)
                torch.testing.assert_close(cache.states[rows, slot], state.float(), rtol=0, atol=0)
                torch.testing.assert_close(cache.next_states[rows, slot], rounded, rtol=0, atol=0)
                # Historical old logp belongs to the FP32 sample, before latent rounding.
                torch.testing.assert_close(cache.old_log_probs[rows, slot], result.log_prob,
                                           rtol=0, atol=0)
                counts["rounded"] += int(not torch.equal(result.sample, rounded))
        return cache

    def checked_replay(cache, velocity, rows, slots):
        counts["replay"] += 1
        assert velocity.requires_grad

        def checked_observation(*args, **kwargs):
            observed = kwargs["observed"]
            torch.testing.assert_close(observed, cache.next_states[rows, slots], rtol=0, atol=0)
            assert observed.dtype == torch.float32
            if dtype == torch.bfloat16:
                assert torch.equal(observed, observed.bfloat16().float())
            return transition(*args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(rollout, "flow_transition", checked_observation)
            result = replay(cache, velocity, rows, slots)
        # Neither rounding nor different forward time precision guarantees ratio == 1.
        assert result.log_prob.requires_grad and torch.isfinite(result.log_prob).all()
        return result

    def checked_evaluation(*args, **kwargs):
        counts["evaluation_batches"] += 1
        assert kwargs["preserve_state_dtype"] is True
        expected_kind = "cps_zero_noise" if algorithm == "flow_grpo_fast" else "ode"
        assert kwargs["unselected_kind"] == expected_kind

        def checked_transition(*a, **options):
            assert options["kind"] == expected_kind
            assert options.get("innovation") is None
            return transition(*a, **options)

        return evaluation_collect(*args, transition_fn=checked_transition, **kwargs)

    def checked_decode(self, endpoints):
        assert self.config.vae_affine_dtype == dtype
        scale, shift = self.config.scaling_factor, self.config.shift_factor
        expected = (endpoints.to(dtype) / scale + shift).float()
        counts["affine_rounded"] += int(not torch.equal(
            expected, endpoints.float() / scale + shift))
        vae_decode = self.vae.decode

        def checked_vae(raw, *args, **kwargs):
            counts["vae"] += 1
            assert raw.dtype == torch.float32
            torch.testing.assert_close(raw, expected, rtol=0, atol=0)
            return vae_decode(raw, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(self.vae, "decode", checked_vae)
            return decode(self, endpoints)

    monkeypatch.setattr(ConditionedImagePolicy, "_predict", record_phase)
    monkeypatch.setattr(diffusers.FluxTransformer2DModel, "forward", checked_forward)
    monkeypatch.setattr(grpo_sampling, "collect_rollout", checked_collect)
    monkeypatch.setattr(objectives, "replay_transition", checked_replay)
    monkeypatch.setattr(sd3, "collect_rollout", checked_evaluation)
    monkeypatch.setattr(FluxAdapter, "decode", checked_decode)
    assert main(["--config", str(local_flux_grpo_config)]) == 0
    assert counts["logical_rollouts"] == 4 and counts["replay"] > 0
    assert counts["evaluation_batches"] == 6 and counts["vae"] == 28
    assert seen["rollout"] and seen["replay"]
    if dtype == torch.bfloat16:
        assert counts["rounded"] > 0
        assert counts["affine_rounded"] > 0
        assert any(not torch.equal(t, t.bfloat16().float()) for t in seen["replay"])
    else:
        assert counts["rounded"] == 0
        assert counts["affine_rounded"] == 0
    assert (counts["cps_zero_noise"] > 0) == (algorithm == "flow_grpo_fast")


@pytest.mark.parametrize("algorithm", ["flow_dppo"])
def test_flux_dppo_actual_objective_and_two_windows_without_ema(
        local_flux_grpo_config, tmp_path, monkeypatch):
    path = local_flux_grpo_config
    config = yaml.safe_load(path.read_text())
    config["algorithm"]["options"]["updates_per_rollout"] = 2
    config["algorithm"]["options"]["evaluation_ema"]["enabled"] = False
    config["training"]["max_updates"] = 4
    config["evaluation"]["every_updates"] = 2
    config["runtime"]["checkpoint_every"] = 2
    path.write_text(yaml.safe_dump(config))
    original = objectives.flowdppo_loss
    calls = []

    def checked_loss(logp, old_logp, mean, old_mean, std, advantage, **kwargs):
        result = original(logp, old_logp, mean, old_mean, std, advantage, **kwargs)
        # Independent transcription of flowdppo_loss.py at research 3ca2a36f.
        # std is the full transition std, not the pre-sqrt(-dt) noise coefficient.
        variance = std.float().reshape((-1,) + (1,) * (mean.ndim - 1)).square()
        kl = ((mean.float() - old_mean.float()).square() / (2 * variance)).flatten(1).mean(1)
        ratio = (logp - old_logp).exp()
        outward = ((ratio > 1) & (advantage > 0)) | ((ratio < 1) & (advantage < 0))
        keep = ~((kl >= kwargs["kl_mask_threshold"]) & outward)
        expected = torch.where(keep, -advantage * ratio, 0.)
        torch.testing.assert_close(result["kl"], kl.detach(), rtol=0, atol=0)
        torch.testing.assert_close(result["per_item_loss"], expected, rtol=0, atol=0)
        gradient, = torch.autograd.grad(result["per_item_loss"].sum(), logp, retain_graph=True)
        torch.testing.assert_close(gradient, torch.where(keep, -advantage * ratio, 0.), rtol=0, atol=0)
        assert advantage.abs().max() <= 5
        calls.append(len(logp))
        return result

    monkeypatch.setattr(objectives, "flowdppo_loss", checked_loss)
    assert main(["--config", str(path)]) == 0
    assert calls
    state = payload(tmp_path / "run")
    assert set(state["stateful"]) == {"prompts"}
    assert state["progress"]["completed_rollouts"] == 2
    assert state["progress"]["optimizer_steps"] == 4
    assert_evaluations(tmp_path / "run", steps=(0, 2, 4))


def test_flux_dppo_example_uses_sde_first_three_actual_kl_and_no_ema():
    directory = Path(__file__).resolve().parents[1] / "configs/examples"
    c = load_config(directory / "flux_dppo_pickscore.yaml")
    assert get_backend(c).name == "flux"
    assert c.algorithm.name == "flow_dppo"
    assert c.sampling.options["transition_kind"] == "sde"
    assert c.sampling.options["likelihood"] == "gaussian"
    assert c.sampling.options["unselected_noise"] == "ode"
    assert c.sampling.options["selection"] == "fixed"
    assert c.sampling.options["pool"] == [0, 1, 2]
    assert c.sampling.options["count"] == 3 and c.sampling.options["noise_level"] == .7
    assert c.algorithm.options["updates_per_rollout"] == 2
    assert c.algorithm.options["loss"] == {"kl_mask_threshold": 1e-5, "advantage_clip": 5.}
    assert c.algorithm.options["evaluation_ema"]["enabled"] is False
    ddp = load_config(directory / "flux_dppo_ddp.yaml")
    assert ddp.runtime.distributed == "ddp"
    assert ddp.algorithm == c.algorithm and ddp.sampling == c.sampling
