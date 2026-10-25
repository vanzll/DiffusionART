import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from visual_rl import wandb_export
from visual_rl.evaluation import audio_video_gallery as gallery
from visual_rl.evaluation.manifest import SampleRecord
from visual_rl.rewards.scalar import weighted_score

KEYS = ("audiobox", "clap", "desync", "hpsv3", "video_align", "video_align_vq")


def sample(root, index=0, *, video=None, prompt='<script>alert("x")</script> & sound'):
    directory = root / "evaluation" / f"step-{index:09d}" / "id"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'row "quoted"' / "sample.mp4"
    path.parent.mkdir(exist_ok=True)
    if video is None:
        path.write_bytes(b"original muxed MP4 bytes")
        path.with_suffix(".wav").write_bytes(b"native wav master")
    else:
        shutil.copyfile(video, path)
        shutil.copyfile(video.with_suffix(".wav"), path.with_suffix(".wav"))
    raw = {key: index + i / 10 for i, key in enumerate(KEYS)}
    record = SampleRecord.from_file(media_root=root, path=path.relative_to(root).as_posix(),
        sample_id=f"sample-{index}", prompt=prompt, seed=index, optimizer_step=index,
        framecount=4, fps=4, scores=weighted_score(raw, dict.fromkeys(KEYS, 1), space="raw"))
    return directory, record


def test_gallery_escapes_text_links_all_samples_and_uses_no_blobs(tmp_path):
    directory, record = sample(tmp_path)
    records = [SampleRecord(sample_id=f"sample-{i}", prompt=record.prompt, seed=i,
                           optimizer_step=0, framecount=4, fps=4, path=record.path,
                           content_sha256=record.content_sha256, scores=record.scores)
               for i in range(32)]
    result = gallery.publish_av_gallery(tmp_path, directory, records, 0)
    page = (tmp_path / result["gallery"]).read_text()
    assert '<script>' not in page and '&lt;script&gt;' in page
    assert 'row%20%22quoted%22/sample.mp4' in page
    assert page.count('<video controls preload="none"') == 32
    assert 'muted' not in page and 'data:' not in page and 'base64' not in page
    assert page.count('WAV audio master') == 32
    manifest = gallery.read_media_manifest(tmp_path, tmp_path / result["wandb_media"])
    assert len(manifest["samples"]) == 32
    assert manifest["reward_columns"] == sorted(KEYS)
    assert manifest["samples"][0]["video"]["sha256"] == record.content_sha256


def test_root_alias_is_supported_but_wav_is_optional(tmp_path):
    directory, record = sample(tmp_path)
    (tmp_path / record.path).with_suffix(".wav").unlink()
    alias = tmp_path.parent / (tmp_path.name + "-alias")
    alias.symlink_to(tmp_path, target_is_directory=True)
    result = gallery.publish_av_gallery(alias, alias / directory.relative_to(tmp_path), [record], 0)
    manifest = gallery.read_media_manifest(alias, alias / result["wandb_media"])
    assert manifest["samples"][0]["audio_master"] is None
    assert "WAV audio master" not in (alias / result["gallery"]).read_text()


@pytest.mark.parametrize("problem", ["hash", "symlink", "traversal", "nonfinite", "count", "size"])
def test_media_manifest_rejects_bad_assets_and_unbounded_payload(tmp_path, problem):
    directory, record = sample(tmp_path)
    gallery.publish_av_gallery(tmp_path, directory, [record], 0)
    path = directory / gallery.MEDIA_FILENAME
    data = json.loads(path.read_text())
    asset = data["samples"][0]["video"]
    if problem == "hash":
        (tmp_path / asset["path"]).write_bytes(b"changed")
    elif problem == "symlink":
        original = tmp_path / asset["path"]
        outside = tmp_path.parent / (tmp_path.name + "-outside.mp4")
        outside.write_bytes(original.read_bytes())
        original.unlink()
        original.symlink_to(outside)
    elif problem == "traversal":
        asset["path"] = "../outside.mp4"
    elif problem == "nonfinite":
        data["samples"][0]["raw_rewards"][KEYS[0]] = float("nan")
    elif problem == "count":
        data["samples"] *= gallery.MAX_SAMPLES + 1
    else:
        asset["bytes"] = gallery.MAX_FILE_BYTES + 1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        gallery.read_media_manifest(tmp_path, path)


def sdk_stub(monkeypatch, *, video_failure=None):
    logged, videos, tables = [], [], []

    class Table:
        def __init__(self, *, columns):
            self.columns, self.rows = columns, []
            tables.append(self)

        def add_data(self, *row):
            self.rows.append(row)

    def video(path, **kwargs):
        if video_failure and video_failure(path):
            raise RuntimeError("injected upload failure")
        assert isinstance(path, str) and Path(path).is_file()
        assert kwargs == {"format": "mp4"}
        videos.append((path, Path(path).read_bytes()))
        return ("video", path)

    run = SimpleNamespace(id="stub", url=None, dir="stub", log=logged.append,
        define_metric=lambda *a, **k: None, finish=lambda: None)
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=lambda **k: run,
        Settings=lambda **k: k, Table=Table, Video=video))
    monkeypatch.setattr(wandb_export.os, "environ", dict(wandb_export.os.environ))
    return logged, videos, tables


def test_export_table_uses_original_paths_raw_scores_and_attempts_only_once(tmp_path, monkeypatch):
    directory, record = sample(tmp_path)
    gallery.publish_av_gallery(tmp_path, directory, [record], 0)
    (tmp_path / "resolved-config.json").write_text(json.dumps({"name": "media-test"}))
    summary = {"eval/optimizer_step": 0, "eval/reward": .2}
    (directory / "summary.json").write_text(json.dumps(summary))
    logged, videos, tables = sdk_stub(monkeypatch)
    # Simulate two live polls, followed by EOF. Neither scalars nor media repeat.
    polls = iter([False, False, False, False, True])
    monkeypatch.setattr(wandb_export.select, "select", lambda *a: ([1] if next(polls) else [], [], []))
    wandb_export.export(tmp_path, tmp_path / "export", follow_fd=1)
    assert logged[0] == summary and len(logged) == 2 and len(videos) == 1
    assert videos[0] == (str((tmp_path / record.path).resolve()), b"original muxed MP4 bytes")
    assert tables[0].columns == ["sample_id", "prompt", "seed", "optimizer_step", *sorted(KEYS), "video"]
    assert tables[0].rows[0][4:10] == tuple(record.scores.raw[k] for k in sorted(KEYS))
    status = json.loads((tmp_path / "export/status.json").read_text())
    assert status["state"] == "finished" and status["media_records"] == 1


@pytest.mark.parametrize("failure", ["manifest", "sdk"])
def test_bad_media_does_not_starve_scalars_or_next_gallery(tmp_path, monkeypatch, failure):
    for index in (0, 1):
        directory, record = sample(tmp_path, index)
        gallery.publish_av_gallery(tmp_path, directory, [record], index)
        (directory / "summary.json").write_text(json.dumps({"eval/optimizer_step": index}))
    if failure == "manifest":
        (tmp_path / "evaluation/step-000000000/id/wandb-media.json").write_text("invalid")
    (tmp_path / "resolved-config.json").write_text(json.dumps({"name": "media-test"}))
    (tmp_path / "metrics.jsonl").write_text('{"train/optimizer_step": 2}\n')
    logged, videos, tables = sdk_stub(monkeypatch,
        video_failure=(lambda path: "step-000000000" in path) if failure == "sdk" else None)
    with pytest.warns(RuntimeWarning, match="media export failed"):
        wandb_export.export(tmp_path, tmp_path / "export")
    assert logged[:3] == [{"train/optimizer_step": 2}, {"eval/optimizer_step": 0}, {"eval/optimizer_step": 1}]
    assert len(videos) == 1 and logged[-1]["eval/optimizer_step"] == 1
    status = json.loads((tmp_path / "export/status.json").read_text())
    assert status["state"] == "finished" and len(status["media_errors"]) == 1
    assert status["media_records"] == 1 and status["records"] == 3


def test_legacy_manifests_are_scalar_only_without_opt_in(tmp_path, monkeypatch):
    directory, record = sample(tmp_path)
    (directory / "manifest.json").write_text(json.dumps({"samples": [record.to_dict()]}))
    (directory / "summary.json").write_text('{"eval/optimizer_step":0}')
    (tmp_path / "resolved-config.json").write_text('{"name":"legacy"}')
    logged, videos, tables = sdk_stub(monkeypatch)
    wandb_export.export(tmp_path, tmp_path / "export")
    assert logged == [{"eval/optimizer_step": 0}] and not videos and not tables
    assert "media_records" not in json.loads((tmp_path / "export/status.json").read_text())


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg required")
def test_actual_ffmpeg_av_roundtrip_preserves_audio_in_sdk_input(tmp_path, monkeypatch):
    import torch

    from visual_rl.backends.minimax_h3_runtime import save_media
    from visual_rl.training.audio_video_types import AVMedia

    real_sdk_available = importlib.util.find_spec("wandb") is not None
    rate = 16000
    t = torch.arange(rate, dtype=torch.float32) / rate
    media = AVMedia(video=torch.full((1, 3, 4, 16, 16), .5),
                    audio=(.1 * torch.sin(2 * torch.pi * 440 * t))[None, None],
                    fps=4, sample_rate=rate)
    original = save_media(media, tmp_path / "original.mp4", timeout=30)
    directory, record = sample(tmp_path, video=original)
    gallery.publish_av_gallery(tmp_path, directory, [record], 0)
    (tmp_path / "resolved-config.json").write_text('{"name":"actual-av"}')
    logged, videos, tables = sdk_stub(monkeypatch)
    wandb_export.export(tmp_path, tmp_path / "export")
    assert hashlib.sha256(videos[0][1]).hexdigest() == record.content_sha256
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name,sample_rate",
                            "-of", "json", videos[0][0]], check=True, capture_output=True, timeout=10)
    streams = json.loads(probe.stdout)["streams"]
    assert {s["codec_type"] for s in streams} == {"video", "audio"}
    audio = next(s for s in streams if s["codec_type"] == "audio")
    assert audio["codec_name"] == "aac" and int(audio["sample_rate"]) == rate
    decoded = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", videos[0][0],
                              "-map", "0:a:0", "-f", "f32le", "-"],
                             check=True, capture_output=True, timeout=10)
    assert len(decoded.stdout) >= rate * 4
    if real_sdk_available:
        # Verify the real offline SDK stores byte-identical muxed AV as well;
        # keep it behind the production subprocess deadline, including atexit.
        destination = tmp_path / "real-sdk-export"
        result = subprocess.run([sys.executable, "-m", "visual_rl.wandb_export",
            "--root", str(tmp_path), "--destination", str(destination),
            "--timeout-seconds", "30"], env={**os.environ, "WANDB_MODE": "offline",
                "PYTHONPATH": str(Path(wandb_export.__file__).resolve().parents[1])},
            capture_output=True, timeout=40)
        assert result.returncode == 0, result.stderr.decode(errors="replace")
        # Tables store videos in artifact staging (some SDK versions omit the
        # filename extension there), rather than run/files/media/videos.
        staged = [path for path in (destination / "data/artifacts/staging").iterdir()
                  if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()
                  == record.content_sha256]
        assert staged
        status = json.loads((destination / "status.json").read_text())
        assert status["media_records"] == 1 and not status.get("media_errors")
