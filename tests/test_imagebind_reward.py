"""CPU contract checks; mocked embeddings are not native checkpoint acceptance."""

import asyncio
import hashlib
import sys
from types import ModuleType

import pytest
import torch

from visual_rl.rewards._native_av import imagebind as ib
from visual_rl.rewards.audio_video import COMPONENTS, SIX_COMPONENTS, AudioVideoReward


def test_six_is_explicit_and_five_stays_legacy():
    assert AudioVideoReward(dict.fromkeys(COMPONENTS)).components == COMPONENTS
    assert AudioVideoReward(dict.fromkeys(SIX_COMPONENTS)).components == SIX_COMPONENTS
    with pytest.raises(ValueError, match="exactly"):
        AudioVideoReward(dict.fromkeys((*COMPONENTS, "wrong")))


def test_video_uses_two_second_windows_and_all_three_crops():
    # 6 sec, 2 fps -> endpoint pairs (0,3),(2,5),(4,7),(6,9),(8,11).
    video = torch.arange(12, dtype=torch.uint8)[:, None, None, None].expand(-1, 3, 8, 12)
    out = ib.preprocess_video(video, 2.)
    assert out.shape == (1, 15, 3, 2, 224, 224)
    denormalized = out[0, :, 0, :, 0, 0] * ib._STD[0] + ib._MEAN[0]
    expected = torch.tensor([[i, i + 3] for i in (0, 2, 4, 6, 8)]).repeat_interleave(3, 0) / 255.
    torch.testing.assert_close(denormalized, expected, atol=1e-7, rtol=1e-5)


def test_video_short_clip_repeats_and_center_crop_rounds_up():
    video = torch.zeros(1, 3, 224, 227, dtype=torch.uint8)
    video[:, :, :, :] = torch.arange(227).to(torch.uint8)[None, None, :]
    out = ib.preprocess_video(video, 24.)
    values = out[0, :3, 0, :, 0, 0] * ib._STD[0] + ib._MEAN[0]
    # Upstream uniform_crop uses ceil for center offset, not floor.
    torch.testing.assert_close(values, torch.tensor([[0., 0.], [2., 2.], [3., 3.]]) / 255., atol=1e-7, rtol=1e-5)


def test_real_audio_preprocess_matches_kaldi_reference():
    torchaudio = pytest.importorskip("torchaudio")
    samples = torch.arange(96000) / 16000.
    wave = torch.sin(2 * torch.pi * 440 * samples)[None, :]
    original = wave.clone()
    out = ib.preprocess_audio(wave.expand(2, -1), 16000)
    assert out.shape == (1, 3, 1, 128, 204) and torch.isfinite(out).all()
    for index, start in enumerate((0, 32000, 64000)):
        clip = wave[:, start:start + 32000].clone()
        clip -= clip.mean()
        mel = torchaudio.compliance.kaldi.fbank(clip, htk_compat=True,
            sample_frequency=16000, use_energy=False, window_type="hanning",
            num_mel_bins=128, dither=0., frame_length=25, frame_shift=10).T
        expected = torch.nn.functional.pad(mel, (0, 204 - mel.shape[-1]))
        torch.testing.assert_close(out[0, index, 0], (expected + 4.268) / 9.138)
    torch.testing.assert_close(wave, original)
    assert ib.preprocess_audio(wave[:, :8000], 16000).shape == (1, 3, 1, 128, 204)
    assert ib.preprocess_audio(wave.repeat_interleave(3, -1), 48000).shape == out.shape


@pytest.mark.parametrize("audio_embedding,expected", [([3., 4.], .6), ([-1., 0.], -1.), ([1., 0.], 1.)])
def test_audio_video_cosine_is_unscaled_with_real_video_preprocessing(monkeypatch, audio_embedding, expected):
    monkeypatch.setattr(ib, "preprocess_audio", lambda audio, rate: torch.zeros(1, 3, 1, 128, 204))
    class Native:
        async def infer(self, video, audio):
            assert video.shape == (1, 15, 3, 2, 224, 224)
            assert audio.shape == (1, 3, 1, 128, 204)
            return {"vision": torch.tensor([[1., 0.]]), "audio": torch.tensor([audio_embedding])}
    result = asyncio.run(ib.compute_score(solution_image=torch.zeros(4, 3, 8, 8, dtype=torch.uint8),
        extra_info={"audio_layout": "CT", "audio": torch.zeros(2, 16000), "audio_sample_rate": 16000, "fps": 4.},
        reward_model=Native()))
    assert result["score"] == pytest.approx(expected)
    assert result["cosine_raw"] == result["score"]


def test_source_private_namespace_preserves_existing_global_package(tmp_path, monkeypatch):
    sources = {
        "imagebind/models/helpers.py": "VALUE = 17\n",
        "imagebind/models/imagebind_model.py": "from imagebind.models.helpers import VALUE\n",
    }
    hashes = {}
    for name, code in sources.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code)
        hashes[name] = hashlib.sha256(code.encode()).hexdigest()
    monkeypatch.setattr(ib, "SOURCE_SHA256", hashes)
    unrelated = ModuleType("imagebind")
    monkeypatch.setitem(sys.modules, "imagebind", unrelated)
    old_path = list(sys.path)
    module = ib.import_model_source(tmp_path)
    assert module.VALUE == 17 and sys.modules["imagebind"] is unrelated and sys.path == old_path
    assert ib.import_model_source(tmp_path) is module
    (tmp_path / "imagebind/models/helpers.py").write_text("VALUE = 99\n")
    with pytest.raises(ValueError, match="source differs"):
        ib.import_model_source(tmp_path)


def test_source_import_failure_does_not_leave_partial_namespace(tmp_path, monkeypatch):
    path = tmp_path / "imagebind/models/helpers.py"
    path.parent.mkdir(parents=True)
    path.write_text("import nonexistent_imagebind_test_dependency\n")
    monkeypatch.setattr(ib, "SOURCE_SHA256", {"imagebind/models/helpers.py": hashlib.sha256(path.read_bytes()).hexdigest()})
    before = set(sys.modules)
    with pytest.raises(ModuleNotFoundError):
        ib.import_model_source(tmp_path)
    assert not [name for name in set(sys.modules) - before if name.startswith("_visual_rl_imagebind_")]


def test_model_load_is_local_strict_and_never_pretrained(tmp_path, monkeypatch):
    checkpoint = tmp_path / "imagebind_huge.pth"
    torch.save(torch.nn.Linear(2, 2).state_dict(), checkpoint)
    def factory(*, pretrained):
        assert pretrained is False
        return torch.nn.Linear(2, 2)
    monkeypatch.setattr(ib, "import_model_source", lambda root: type("Source", (), {"imagebind_huge": staticmethod(factory)}))
    native = ib.ImageBindModel(checkpoint, source_root=tmp_path)
    assert not native.model.training and all(not p.requires_grad for p in native.model.parameters())
    torch.save({}, checkpoint)
    with pytest.raises(RuntimeError, match="Missing key"):
        ib.ImageBindModel(checkpoint, source_root=tmp_path)


def test_six_score_adds_cosine_and_preserves_all_five_transforms(monkeypatch):
    from contextlib import contextmanager

    from visual_rl.rewards._native_av import audiobox, clap, desync, hpsv3_reward, videoalign
    from visual_rl.training.audio_video_types import AVMedia
    expected = {"video_align": 1.5, "hpsv3": 12.5, "audiobox": .3, "clap": .8, "desync": .4}
    observed = []
    def scorer(name):
        async def compute(*args, **kwargs):
            observed.append((name, kwargs))
            return {"score": expected[name]}
        return compute
    for module, func, name in ((videoalign, "compute_score", "video_align"),
            (hpsv3_reward, "compute_score_hpsv3", "hpsv3"), (audiobox, "compute_score", "audiobox"),
            (clap, "compute_score", "clap"), (desync, "compute_score", "desync")):
        monkeypatch.setattr(module, func, scorer(name))
    monkeypatch.setattr(ib, "preprocess_audio", lambda audio, rate: torch.zeros(1, 3, 1, 128, 204))
    class Model:
        async def infer(self, video, audio):
            return {"vision": torch.tensor([[1., 0.]]), "audio": torch.tensor([[-1., 0.]])}
    @contextmanager
    def active(self, name):
        yield Model()
    monkeypatch.setattr(AudioVideoReward, "_active", active)
    media = AVMedia(torch.zeros(1, 3, 4, 8, 8), torch.zeros(1, 2, 16000), 4., 16000)
    five = AudioVideoReward(dict.fromkeys(COMPONENTS)).score(media, ["video"], ["audio"])
    six = AudioVideoReward(dict.fromkeys(SIX_COMPONENTS)).score(media, ["video"], ["audio"])
    assert set(five.components) == set(COMPONENTS) and set(six.components) == set(SIX_COMPONENTS)
    for name, value in expected.items():
        assert six.components[name].item() == pytest.approx(value)
        torch.testing.assert_close(five.components[name], six.components[name])
    assert six.components["imagebind"].item() == -1.
    assert six.diagnostics["imagebind/cosine_raw"].item() == -1.
    for key in ("ground_truth", "fps", "score_weights", "score_means", "score_stds"):
        assert observed[0][1][key] == observed[5][1][key]
    assert observed[0][1]["ground_truth"] == "video"
    assert observed[0][1]["score_weights"] == [.5, 0., .5]
    assert observed[3][1]["score_scale"] == .5 and observed[3][1]["score_offset"] == .5


def test_real_preprocessing_with_mock_native_embedding():
    pytest.importorskip("torchaudio")
    class Native:
        async def infer(self, video, audio):
            assert video.shape == (1, 15, 3, 2, 224, 224)
            assert audio.shape == (1, 3, 1, 128, 204)
            assert torch.isfinite(video).all() and torch.isfinite(audio).all()
            return {"vision": torch.tensor([[1., 0.]]), "audio": torch.tensor([[3., 4.]])}
    result = asyncio.run(ib.compute_score(solution_image=torch.zeros(24, 3, 8, 8, dtype=torch.uint8),
        extra_info={"audio_layout": "CT", "audio": torch.zeros(2, 48000), "audio_sample_rate": 48000, "fps": 24.},
        reward_model=Native()))
    assert result == pytest.approx({"score": .6, "cosine_raw": .6})
