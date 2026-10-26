"""Duration scheduling, isolated eval publication and missing-bucket recovery."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from visual_rl.backends import minimax_h3 as backend
from visual_rl.backends import minimax_h3_runtime as runtime
from visual_rl.backends.minimax_h3_posttraining import (
    POST_TRAINING,
    rollout_config,
    validate_post_training,
)
from visual_rl.rewards.audio_video import AVScores
from visual_rl.training.distributed import distributed_session
from visual_rl.wandb_export import MediaRecords, Records


@pytest.mark.parametrize("frames", [[107, 243], [124, 362], [124, 240], [124, 124], [243, 124], [], [True]])
def test_duration_validation_rejects_unaligned_out_of_range_or_ambiguous_buckets(frames):
    with pytest.raises(ValueError, match="duration"):
        validate_post_training({**POST_TRAINING, "duration_frames": frames, "full_duration_rewards": True}, 50)


def test_rotation_is_global_reproducible_and_does_not_consume_sampling_rng(tmp_path, monkeypatch):
    from test_minimax_h3_runtime import config_factory

    base = config_factory.__wrapped__(tmp_path, monkeypatch)("art", "full")
    c = replace(base, algorithm=replace(base.algorithm, options={**base.algorithm.options,
        "post_training": {**POST_TRAINING, "duration_frames": [124, 243, 345], "full_duration_rewards": True}}),
        sampling=replace(base.sampling, frames=124))
    before = torch.get_rng_state().clone()
    views = [rollout_config(c, i) for i in range(6)]
    assert [v.sampling.frames for v in views] == [124, 243, 345, 124, 243, 345]
    assert c.sampling.frames == 124 and torch.equal(before, torch.get_rng_state())
    assert rollout_config(c, 2).sampling.frames == views[2].sampling.frames
    legacy = replace(base, sampling=replace(base.sampling, frames=124))
    assert all(rollout_config(legacy, i).sampling.frames == 124 for i in range(4))


def test_eval_buckets_preserve_seeds_namespace_media_and_repair_only_missing_bucket(tmp_path, monkeypatch):
    from test_minimax_h3_runtime import TinyReward, config_factory, fixture_api

    base = config_factory.__wrapped__(tmp_path, monkeypatch)("art", "full", evaluation=True)
    dataset, _ = backend.load_datasets(base)
    c = replace(base, algorithm=replace(base.algorithm, options={**base.algorithm.options,
        "post_training": {**POST_TRAINING, "duration_frames": [124, 243, 345], "full_duration_rewards": True}}),
        sampling=replace(base.sampling, frames=124),
        reward=replace(base.reward, weights={**base.reward.weights, "imagebind": .5}))
    parts = fixture_api([]).load_components(c, torch.device("cpu"))

    class SixReward(TinyReward):
        def score(self, *args):
            result = super().score(*args)
            return AVScores({**result.components, "imagebind": result.components["video_align"] * .3}, result.diagnostics)

    parts.reward = SixReward()
    root = tmp_path / "results"
    root.mkdir()
    saved = []

    def save_fixture(media, target, **_):
        target = Path(target)
        target.parent.mkdir(parents=True)
        target.write_bytes(b"muxed AV fixture")
        target.with_suffix(".wav").write_bytes(b"audio master fixture")
        saved.append(media.video.shape[2])
        return target

    monkeypatch.setattr(runtime, "save_media", save_fixture)
    with distributed_session("none", device="cpu", timeout_seconds=30) as session:
        runtime.evaluate(session, c, parts, dataset, 0, root)
    assert saved == [124] * 3 + [243] * 3 + [345] * 3
    manifests = sorted((root / "evaluation").glob("*/*/manifest.json"))
    assert len(manifests) == 3
    for manifest in manifests:
        data = json.loads(manifest.read_text())
        # Independently verify each row rather than trusting a directory label.
        frames = {row["framecount"] for row in data["samples"]}
        assert len(frames) == 1
        frame = frames.pop()
        assert frame in (124, 243, 345)
        assert [row["seed"] for row in data["samples"]] == [42, 43, 44]
        assert [row["sample_id"] for row in data["samples"]] == [row.sample_id for row in dataset.rows]
        summary = json.loads(manifest.with_name("summary.json").read_text())
        assert f"eval/frames{frame}/reward/raw/imagebind" in summary
        assert "eval/reward" not in summary
    assert runtime._completed_evaluation(c, dataset, 0, root)
    assert len(list(Records(root).read())) == 3
    media = list(MediaRecords(root).read())
    assert {p[1]["metric_namespace"] for p in media} == {"eval/frames124", "eval/frames243", "eval/frames345"}
    # Only the missing long-duration receipt is retried; other buckets stay intact.
    long_manifest = next(p for p in manifests if "frames-345" in str(p))
    original = json.loads(long_manifest.read_text())
    bad = copy.deepcopy(original)
    bad["samples"][0]["framecount"] = 124
    long_manifest.write_text(json.dumps(bad))
    assert not runtime._completed_evaluation(c, dataset, 0, root)
    long_manifest.write_text(json.dumps(original))
    long_manifest.unlink()
    saved.clear()
    with distributed_session("none", device="cpu", timeout_seconds=30) as session:
        runtime.evaluate(session, c, parts, dataset, 0, root)
    assert saved == [345] * 3
    assert runtime._completed_evaluation(c, dataset, 0, root)
