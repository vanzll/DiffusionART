"""Scorer calibration/asset contracts; fixtures are not pretrained GPU evidence."""
import asyncio
import hashlib
from types import SimpleNamespace

import pytest
import torch

from visual_rl.rewards.audio_video import AudioVideoReward, validate_asset_inventory, validate_media
from visual_rl.training.audio_video_types import AVMedia


def test_media_native_rates_and_duration():
    m = AVMedia(torch.zeros(1, 3, 25, 2, 2), torch.zeros(1, 2, 32000), 24., 32000)
    validate_media(m)
    with pytest.raises(ValueError, match="durations"):
        validate_media(AVMedia(m.video, m.audio[..., :100], 24., 32000))
    with pytest.raises(ValueError, match="RGB"):
        validate_media(AVMedia(m.video + 2, m.audio, 24., 32000))


def test_inventory_binds_all_files_and_rejects_symlink(tmp_path):
    root = tmp_path / "scorer"
    root.mkdir()
    (root / "config.json").write_text("{}")
    sha = hashlib.sha256(b"{}").hexdigest()
    validate_asset_inventory({"a": str(root)}, {"a": {"config.json": sha}})
    (root / "extra").write_text("changed")
    with pytest.raises(ValueError, match="incomplete"):
        validate_asset_inventory({"a": str(root)}, {"a": {"config.json": sha}})
    (root / "extra").unlink()
    (root / "alias").symlink_to(root / "config.json")
    with pytest.raises(ValueError, match="symlink"):
        validate_asset_inventory({"a": str(root)}, {"a": {"config.json": sha}})


def test_audiobox_restores_axes_and_duration_weights():
    from visual_rl.rewards._native_av import audiobox
    out = {"predictions": {k: torch.tensor([1., 3.]) for k in audiobox._AXES},
           "target_transform": {k: (2., 4.) for k in audiobox._AXES}}
    score = audiobox._score_windows(out, 2, {"CE": 1, "CU": 1, "PC": -1, "PQ": 1}, .025)
    torch.testing.assert_close(score, torch.tensor([.3, .7]))
    windows, masks, weights = audiobox._make_windows([torch.ones(240000)])
    assert windows.shape == (2, 1, 160000)
    assert masks[1].sum() == 80000 and weights == [1., .5]


def test_clap_text_route_and_cosine_transform():
    from visual_rl.rewards._native_av import clap
    class Model:
        async def infer(self, waves, prompts):
            assert prompts == ["audio text"] and len(waves[0]) == 48000
            return {"audio_embeddings": torch.tensor([[3., 4.]]),
                    "text_embeddings": torch.tensor([[1., 0.]])}
    result = asyncio.run(clap.compute_score(None, None, "audio text",
        {"audio": torch.zeros(2, 48000), "audio_layout": "CT", "audio_sample_rate": 48000},
        reward_model=Model(), score_scale=.5, score_offset=.5, score_min=0., score_max=1.))
    assert result["score"] == pytest.approx(.8)
    assert result['cosine_raw'] == pytest.approx(.6)


def test_videoalign_calibration_and_video_text():
    from visual_rl.rewards._native_av import videoalign
    class Model:
        async def infer(self, videos, prompts):
            assert prompts == ["video text"] and videos[0].shape[1] == 3
            return torch.tensor([[3.6757 + 2.2476, 999., 2.8105 + 2 * 2.5121]])
    result = asyncio.run(videoalign.compute_score(solution_image=torch.zeros(24, 3, 16, 16, dtype=torch.uint8),
        ground_truth="video text", extra_info={"media_kind": "video", "fps": 24.},
        reward_model=Model(), fps=24.))
    assert result["score"] == pytest.approx(1.5, abs=1e-6)
    assert result["MQ_raw"] == 999.
    assert result["score"] == pytest.approx(.5*(result["VQ_raw"]-3.6757)/2.2476 + .5*(result["TA_raw"]-2.8105)/2.5121)


def test_hps_five_frames_top_fraction_cap():
    from visual_rl.rewards._native_av import hpsv3_reward
    class Model:
        async def infer(self, images, prompts):
            assert len(images) == 5 and prompts == ["video text"] * 5
            return torch.tensor([[1.], [2.], [3.], [10.], [100.]])
    result = asyncio.run(hpsv3_reward.compute_score_hpsv3(None, torch.zeros(25, 3, 4, 4, dtype=torch.uint8),
        "video text", {"media_kind": "video", "fps": 24.}, reward_model=Model(),
        num_frames=5, top_fraction=.3, score_cap=15., max_batch_size=5, reward_scale=1.))
    assert result == {"score": 12.5, "hpsv3_raw": 12.5}


@pytest.mark.parametrize("broken", [float("nan"), float("inf"), float("-inf")])
def test_hps_rejects_nonfinite_frame_before_cap_and_top_selection(broken):
    from visual_rl.rewards._native_av import hpsv3_reward

    class Model:
        async def infer(self, images, prompts):
            return torch.tensor([[broken], [2.], [3.], [10.], [100.]])

    with pytest.raises(ValueError, match="frame scores.*finite"):
        asyncio.run(hpsv3_reward.compute_score_hpsv3(
            None, torch.zeros(25, 3, 4, 4, dtype=torch.uint8), "video text",
            {"media_kind": "video", "fps": 24.}, reward_model=Model(),
            num_frames=5, top_fraction=.3, score_cap=15., max_batch_size=5,
            reward_scale=1.))


def test_hps_cached_batch_rejects_bad_request_without_dropping_healthy_peer(monkeypatch):
    from visual_rl.rewards._native_av import hpsv3_reward

    monkeypatch.setattr(hpsv3_reward, "_select_reward_frames", lambda *args: [object()])
    monkeypatch.setattr(hpsv3_reward, "_get_inferencer", lambda *args: SimpleNamespace(
        reward=lambda *args: torch.tensor([[float("inf")], [7.]])))
    requests = [SimpleNamespace(
        solution_image=None, extra_info={}, num_frames=1, batch_key=("local", "cpu", 2),
        prompt=prompt, top_fraction=.3, score_cap=15., reward_scale=1.)
        for prompt in ("broken", "healthy")]
    results = hpsv3_reward._score_batch(requests)
    assert isinstance(results[0], ValueError)
    assert results[1] == {"score": 7., "hpsv3_raw": 7.}


def test_desync_native_preprocess_and_class_offset():
    from visual_rl.rewards._native_av import desync
    class Model:
        async def infer(self, video, audio):
            assert video.shape[:3] == (1, 200, 3) and audio.shape == (1, 128000)
            logits = torch.zeros(2, 1, 21)
            logits[0, 0, 0] = 1  # -2 sec
            logits[1, 0, 15] = 1  # +1 sec
            return logits
    result = asyncio.run(desync.compute_score(solution_image=torch.zeros(24, 3, 16, 16, dtype=torch.uint8),
        extra_info={"audio": torch.zeros(2, 16000), "audio_layout": "CT", "audio_sample_rate": 16000,
                    "media_kind": "video", "fps": 24.}, reward_model=Model(), fps=24.))
    assert result["score"] == pytest.approx(1 / 2.5)
    assert result["absolute_offset_raw"] == pytest.approx(1.5)


@pytest.mark.parametrize("broken", [float("nan"), float("inf"), float("-inf")])
def test_desync_rejects_nonfinite_logits_before_argmax(broken):
    from visual_rl.rewards._native_av import desync

    class Model:
        async def infer(self, video, audio):
            logits = torch.zeros(2, 1, 21)
            logits[0, 0, 10] = broken
            return logits

    with pytest.raises(ValueError, match="DeSync logits.*finite"):
        asyncio.run(desync.compute_score(
            solution_image=torch.zeros(24, 3, 16, 16, dtype=torch.uint8),
            extra_info={"audio": torch.zeros(2, 16000), "audio_layout": "CT",
                        "audio_sample_rate": 16000, "media_kind": "video", "fps": 24.},
            reward_model=Model(), fps=24.))


def test_staging_restores_cpu_on_failure():
    class Model:
        def __init__(self):
            self.model = torch.nn.Linear(1, 1)
            self.device = "cpu"
        def close(self):
            pass
    reward = AudioVideoReward({k: Model() for k in ["video_align", "hpsv3", "audiobox", "clap", "desync"]})
    with pytest.raises(RuntimeError):
        with reward._active("clap"):
            raise RuntimeError("scoring failed")
    assert reward.models["clap"].device == torch.device("cpu")
