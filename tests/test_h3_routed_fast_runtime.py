"""Matched configuration, initial-noise isolation and exact routed CPS resume."""

import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_h3_mixed_reference_runtime import (
    api,
    checkpoint,
    equal_nested,
    metric_frames,
    mixed_config,
)
from test_h3_posttraining import Policy
from test_minimax_h3_runtime import config_factory

from visual_rl.backends import minimax_h3 as backend
from visual_rl.backends import minimax_h3_runtime as runtime
from visual_rl.backends.minimax_h3_posttraining import selected_positions
from visual_rl.config import load_config
from visual_rl.training.audio_video_types import AVState


def routed(config):
    return replace(config, algorithm=replace(config.algorithm, name="flow_grpo_fast", options={
        "ratio_mode": "per_modality",
        "post_training": copy.deepcopy(config.algorithm.options["post_training"]),
        "loss": {"ratio_clip": 1e-4, "advantage_clip": 5., "reference_mse_beta": 1e-3}}))


def test_matched_recipe_changes_only_algorithm_and_run_labels(tmp_path, monkeypatch):
    factory = config_factory.__wrapped__(tmp_path, monkeypatch)
    factory("art", "full")
    for key in ("H3_OUTPUT_DIR", "H3_EVAL_MANIFEST"):
        monkeypatch.setenv(key, str(tmp_path / key))
    monkeypatch.setenv("H3_EVAL_SHA256", "a" * 64)
    root = Path(__file__).resolve().parents[1] / "configs/examples"
    art = load_config(root / "minimax_h3_art_posttrain_full_32gpu.yaml")
    fast = load_config(root / "minimax_h3_fast_routed_posttrain_full_32gpu.yaml")
    backend.validate_config(fast)
    for name in ("model", "reward", "data", "sampling", "training", "evaluation", "seed"):
        assert getattr(art, name) == getattr(fast, name)
    assert art.algorithm.options["post_training"] == fast.algorithm.options["post_training"]
    assert art.algorithm.options["loss"]["reference_mse_beta"] == fast.algorithm.options["loss"]["reference_mse_beta"] == .001
    assert "covariance" not in fast.algorithm.options and "advantage_epsilon" not in fast.algorithm.options
    assert fast.sampling.microbatch == fast.training.microbatch == 8
    legacy = load_config(root / "minimax_h3_fast_full.yaml")
    backend.validate_config(legacy)
    assert "post_training" not in legacy.algorithm.options


def test_collect_uses_same_modal_selections_and_preserves_initial_noise_rng(tmp_path, monkeypatch):
    factory = config_factory.__wrapped__(tmp_path, monkeypatch)
    art = mixed_config(factory, "art")
    post = {**art.algorithm.options["post_training"], "video_pool": 15, "audio_pool": 10}
    art = replace(art, algorithm=replace(art.algorithm, options={**art.algorithm.options, "post_training": post}),
                  sampling=replace(art.sampling, steps=50, microbatch=2))
    fast = routed(art)
    model = Policy()
    generator = torch.Generator().manual_seed(9)
    noise = AVState(torch.randn(4, 3, generator=generator), torch.randn(4, 2, generator=generator))
    session = SimpleNamespace(distributed=True, device=torch.device("cpu"), identity=SimpleNamespace(rank=0))
    before = generator.get_state().clone()
    cache = runtime.collect(None, fast, model, noise, runtime.sigma_schedules(50), session, generator, rollout_index=3)
    expected = selected_positions(42, 4, video_pool=15, audio_pool=10)
    assert cache.video_selected == expected["video"] and cache.audio_selected == expected["audio"]
    assert torch.equal(before, generator.get_state())
    repeated = runtime.collect(None, fast, model, noise, runtime.sigma_schedules(50), session, generator, rollout_index=3)
    torch.testing.assert_close(cache.endpoints.video, repeated.endpoints.video, atol=0, rtol=0)
    evaluation = runtime.collect(None, fast, model, noise, runtime.sigma_schedules(50), session, generator,
                                 rollout_index=3, evaluation=True)
    baseline = runtime.collect(None, art, model, noise, runtime.sigma_schedules(50), session, generator,
                               rollout_index=3, evaluation=True)
    assert evaluation.selected == evaluation.stochastic == ()
    torch.testing.assert_close(evaluation.endpoints.video, baseline.endpoints.video, atol=0, rtol=0)
    torch.testing.assert_close(evaluation.endpoints.audio, baseline.endpoints.audio, atol=0, rtol=0)


def test_routed_fast_full_runtime_resume_restores_cps_and_reference(tmp_path, monkeypatch):
    factory = config_factory.__wrapped__(tmp_path, monkeypatch)
    config = routed(mixed_config(factory, "fast-continuous"))
    created = []
    result = runtime.run(config, backend_api=api(created))
    assert result["optimizer_steps"] == 8 and result["completed_rollouts"] == 4
    assert metric_frames(config.runtime.output_dir) == [124, 243, 345, 124]
    expected = checkpoint(config.runtime.output_dir)
    rows = [json.loads(line) for line in (Path(config.runtime.output_dir) / "metrics.jsonl").read_text().splitlines()]
    assert all(row["train/optimizer_steps_per_rollout"] == 2 for row in rows)
    assert all("av/video_ratio" in row and "av/audio_ratio" in row for row in rows)
    assert all("train/reference_mse_weighted" in row for row in rows)
    interrupted = routed(mixed_config(factory, "fast-resumed"))
    save = runtime.save_checkpoint
    receipts = []

    class StopAfterCheckpoint(Exception):
        pass

    def stop(*args, **kwargs):
        receipt = save(*args, **kwargs)
        receipts.append(receipt)
        raise StopAfterCheckpoint

    monkeypatch.setattr(runtime, "save_checkpoint", stop)
    with pytest.raises(StopAfterCheckpoint):
        runtime.run(interrupted, backend_api=api([]))
    monkeypatch.setattr(runtime, "save_checkpoint", save)
    runtime.run(interrupted, resume=receipts[0], backend_api=api([]))
    assert metric_frames(interrupted.runtime.output_dir) == [124, 243, 345, 124]
    actual = checkpoint(interrupted.runtime.output_dir)
    for name in ("model", "optimizer", "rng", "generators", "stateful"):
        equal_nested(actual[name], expected[name])
