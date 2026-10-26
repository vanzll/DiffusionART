"""Full-duration reward coverage; mocks are not pretrained scorer evidence."""

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from visual_rl.rewards.audio_video import COMPONENTS, AudioVideoReward
from visual_rl.rewards.duration_windows import clap_window_seconds, media_windows, score_windows
from visual_rl.training.audio_video_types import AVMedia


def decoded(frames):
    rate, fps = 32000, 24.
    samples = round(frames / fps * rate)
    video = torch.linspace(0., 1., frames).view(1, 1, frames, 1, 1).expand(1, 3, frames, 2, 2)
    audio = torch.linspace(0., 1., samples).view(1, 1, samples).expand(1, 2, samples)
    return AVMedia(video, audio, fps, rate)


def native_row(media):
    return media.video[0].permute(1, 0, 2, 3), {
        "audio": media.audio[0], "fps": media.fps,
        "audio_sample_rate": media.sample_rate, "audio_layout": "CT"}


@pytest.mark.parametrize("frames", [124, 243, 345])
@pytest.mark.parametrize("seconds", [8., 10.])
def test_window_coverage_alignment_native_rates_and_no_rng_use(frames, seconds):
    video, extra = native_row(decoded(frames))
    state = torch.get_rng_state().clone()
    windows = media_windows(video, extra, seconds)
    torch.testing.assert_close(torch.get_rng_state(), state)
    assert len(windows) == (1 if frames == 124 else 2)
    assert windows[0][0][0, 0, 0, 0] == 0
    assert windows[0][1]["audio"][0, 0] == 0
    assert windows[-1][0][-1, 0, 0, 0] == 1
    assert windows[-1][1]["audio"][0, -1] == 1
    for frames_in_window, info in windows:
        assert info["fps"] == extra["fps"] and info["audio_sample_rate"] == extra["audio_sample_rate"]
        assert len(frames_in_window) <= seconds * extra["fps"]
        assert info["audio"].shape[-1] <= seconds * extra["audio_sample_rate"]
        # Temporal ramps identify the absolute AV timestamps independently.
        assert abs(float(frames_in_window[0, 0, 0, 0]) - float(info["audio"][0, 0])) < 1 / frames
        assert abs(len(frames_in_window) / info["fps"] - info["audio"].shape[-1] / info["audio_sample_rate"]) <= 1 / info["fps"]
    if frames == 124:
        assert windows[0][0] is video and windows[0][1] is extra


def test_clap_window_reads_real_processor_config_and_rejects_unknown_or_conflicting_limit():
    def model(**values):
        return SimpleNamespace(processor=SimpleNamespace(feature_extractor=SimpleNamespace(**values)))
    assert clap_window_seconds(model(nb_max_samples=336000, sampling_rate=48000, max_length_s=7)) == 7
    assert clap_window_seconds(model(max_length_s=6.)) == 6
    with pytest.raises(ValueError, match="explicit audio window"):
        clap_window_seconds(model())
    with pytest.raises(ValueError, match="inconsistent"):
        clap_window_seconds(model(nb_max_samples=480000, sampling_rate=48000, max_length_s=8))


def test_window_scores_average_transformed_values_and_matching_raw_diagnostics():
    video, extra = native_row(decoded(345))
    returned = []
    async def compute(frames, info):
        offset = 0. if not returned else 2.
        result = {"score": 1 / (1 + offset), "absolute_offset_raw": offset, "source_sample_rate": 32000}
        returned.append(result)
        return result
    result = asyncio.run(score_windows(video, extra, 8., compute))
    assert result["score"] == pytest.approx(2 / 3)
    assert result["absolute_offset_raw"] == 1.
    assert result["source_sample_rate"] == 32000
    # The transform belongs inside each window; never transform the raw mean
    # or sum window scores, either of which changes the declared clip reward.
    assert result["score"] != 1 / (1 + result["absolute_offset_raw"])


def test_short_clip_exactly_preserves_result_identity():
    video, extra = native_row(decoded(124))
    result = {"score": .7, "cosine_raw": .4, "source_sample_rate": 32000}
    async def compute(frames, info):
        assert frames is video and info is extra
        return result
    assert asyncio.run(score_windows(video, extra, 10., compute)) is result


def scorer_mocks(monkeypatch, calls):
    from visual_rl.rewards._native_av import audiobox, clap, desync, hpsv3_reward, videoalign
    async def generic(name, video=None, extra=None):
        calls[name].append((video, extra))
        value = float(extra["audio"][0, 0]) if extra else 0.
        return {"score": .5 + value / 4, "example_raw": value}
    async def va(**kwargs):
        assert kwargs["ground_truth"] == "video text"
        return await generic("video_align", kwargs["solution_image"], kwargs["extra_info"])
    async def hps(_source, video, prompt, extra, **kwargs):
        assert prompt == "video text" and kwargs["num_frames"] == 5
        return await generic("hpsv3", video, extra)
    async def ab(**kwargs):
        return await generic("audiobox", extra=kwargs["extra_info"])
    async def cl(_source, video, prompt, extra, **kwargs):
        assert prompt == "audio text"
        return await generic("clap", video, extra)
    async def ds(**kwargs):
        return await generic("desync", kwargs["solution_image"], kwargs["extra_info"])
    monkeypatch.setattr(videoalign, "compute_score", va)
    monkeypatch.setattr(hpsv3_reward, "compute_score_hpsv3", hps)
    monkeypatch.setattr(audiobox, "compute_score", ab)
    monkeypatch.setattr(clap, "compute_score", cl)
    monkeypatch.setattr(desync, "compute_score", ds)


@pytest.mark.parametrize("frames", [124, 243, 345])
@pytest.mark.parametrize("full_duration", [False, True])
def test_reward_integration_is_opt_in_and_full_span_only_for_fixed_window_scorers(monkeypatch, frames, full_duration):
    calls = {name: [] for name in COMPONENTS}
    scorer_mocks(monkeypatch, calls)
    processor = SimpleNamespace(feature_extractor=SimpleNamespace(nb_max_samples=480000, sampling_rate=48000))
    models = {name: SimpleNamespace(processor=processor) for name in COMPONENTS}
    reward = AudioVideoReward(models, full_duration=full_duration)
    @contextmanager
    def active(_name):
        yield object()
    monkeypatch.setattr(reward, "_active", active)
    media = decoded(frames)
    result = reward.score(media, ["video text"], ["audio text"])
    for name in COMPONENTS:
        expected = 2 if full_duration and frames > 124 and name in ("clap", "desync") else 1
        assert len(calls[name]) == expected
        window_scores = [.5 + float(info["audio"][0, 0]) / 4 for _, info in calls[name]]
        assert result.components[name][0] == pytest.approx(sum(window_scores) / expected)
        assert result.diagnostics[name + "/example_raw"][0] == pytest.approx(
            sum(float(info["audio"][0, 0]) for _, info in calls[name]) / expected)
        if expected == 2:
            assert calls[name][-1][1]["audio"][0, -1] == 1


def test_long_window_rejects_nonfinite_or_changed_diagnostic_schema():
    video, extra = native_row(decoded(345))
    async def bad(_frames, _extra):
        return {"score": float("nan")}
    with pytest.raises(ValueError, match="nonfinite"):
        asyncio.run(score_windows(video, extra, 8., bad))


def test_full_duration_flag_requires_boolean():
    with pytest.raises(ValueError, match="boolean"):
        AudioVideoReward({name: object() for name in COMPONENTS}, full_duration="yes")
