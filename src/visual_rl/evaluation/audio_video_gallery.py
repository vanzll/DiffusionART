"""Local AV gallery and explicit, bounded opt-in for the independent W&B owner.

Media files are linked and hashed, never decoded/re-encoded here. In particular,
the muxed MP4's audio track is retained by both HTML and wandb.Video(path).
"""

import html
import json
import math
import os
import re
import stat
from pathlib import Path
from urllib.parse import quote

from .manifest import SampleRecord, file_sha256, resolve_media

MEDIA_FILENAME = "wandb-media.json"
MEDIA_SCHEMA = "visual-rl-av-gallery-v1"
MAX_SAMPLES = 64
MAX_FILE_BYTES = 256 * 1024 * 1024
MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
MAX_MANIFEST_BYTES = 2 * 1024 * 1024


def _asset(root, path, expected_hash=None):
    candidate = resolve_media(root, path)
    digest = file_sha256(candidate, max_bytes=MAX_FILE_BYTES)
    if expected_hash is not None and digest != expected_hash:
        raise ValueError("gallery media hash mismatch")
    return {"path": path, "sha256": digest, "bytes": candidate.stat().st_size}


def _validate_payload(payload, root, *, verify_assets=True):
    if (not isinstance(payload, dict) or payload.get("schema") != MEDIA_SCHEMA
            or payload.get("wandb_media") is not True):
        raise ValueError("not an opted-in AV gallery manifest")
    if not re.fullmatch(r"eval(?:/frames[0-9]{1,4})?", payload.get("metric_namespace", "eval")):
        raise ValueError("invalid AV gallery metric namespace")
    step, columns, samples = (payload.get(key) for key in
                              ("optimizer_step", "reward_columns", "samples"))
    if type(step) is not int or not 0 <= step <= 2**53 - 1:
        raise ValueError("invalid gallery optimizer step")
    if (not isinstance(columns, list) or not 1 <= len(columns) <= 16
            or any(not isinstance(k, str) or not k or len(k) > 128 for k in columns)
            or len(set(columns)) != len(columns)):
        raise ValueError("invalid gallery reward columns")
    if not isinstance(samples, list) or not 1 <= len(samples) <= MAX_SAMPLES:
        raise ValueError("gallery sample count exceeds bound")
    seen, total = set(), 0
    for row in samples:
        if not isinstance(row, dict):
            raise ValueError("invalid gallery row")
        sample_id, prompt = row.get("sample_id"), row.get("prompt")
        if (not isinstance(sample_id, str) or not sample_id or len(sample_id) > 1024
                or sample_id in seen or not isinstance(prompt, str) or not prompt.strip()
                or len(prompt) > 32768):
            raise ValueError("invalid gallery sample identity/prompt")
        seen.add(sample_id)
        if (type(row.get("seed")) is not int or not 0 <= row["seed"] <= 2**53 - 1
                or type(row.get("optimizer_step")) is not int or row["optimizer_step"] != step):
            raise ValueError("invalid gallery seed/step")
        rewards = row.get("raw_rewards")
        if (not isinstance(rewards, dict) or set(rewards) != set(columns)
                or any(type(v) not in (int, float) or not math.isfinite(v)
                       for v in rewards.values())):
            raise ValueError("gallery rewards must be finite and match columns")
        for kind, suffix in (("video", ".mp4"), ("audio_master", ".wav")):
            asset = row.get(kind)
            if asset is None and kind == "audio_master":
                continue
            if not isinstance(asset, dict) or set(asset) != {"path", "sha256", "bytes"}:
                raise ValueError("invalid gallery asset")
            if not isinstance(asset["path"], str) or Path(asset["path"]).suffix.lower() != suffix:
                raise ValueError("gallery asset has wrong extension")
            # Bound the whole batch before hashing any asset beyond the budget.
            if type(asset["bytes"]) is not int or not 0 < asset["bytes"] <= MAX_FILE_BYTES:
                raise ValueError("gallery asset size exceeds bound")
            total += asset["bytes"]
            if total > MAX_TOTAL_BYTES:
                raise ValueError("gallery total media exceeds bound")
            if verify_assets and _asset(root, asset["path"], asset["sha256"]) != asset:
                raise ValueError("gallery asset metadata changed")
    return payload


def read_media_manifest(root, path):
    """Validate one opted-in manifest, including bounded hashes and local paths."""
    root, path = Path(root).absolute(), Path(path)
    relative = path.absolute().relative_to(root).as_posix()
    root = root.resolve(strict=True)
    path = resolve_media(root, relative)
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_MANIFEST_BYTES:
        raise ValueError("gallery manifest exceeds bound")
    with path.open("rb") as stream:
        data = stream.read(MAX_MANIFEST_BYTES + 1)
    if len(data) > MAX_MANIFEST_BYTES:
        raise ValueError("gallery manifest exceeds bound")
    return _validate_payload(json.loads(data), root)


def _publish(path, text):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def publish_av_gallery(root, directory, samples, step, *, metric_namespace="eval"):
    """Publish gallery.html then wandb-media.json; caller handles optional errors.

    ``samples`` are the same verified SampleRecord objects as manifest.json.
    Paths in the opt-in manifest are root-relative; HTML links are relative to
    its own directory, so moving the whole output root preserves playback.
    """
    root, directory = Path(root).absolute(), Path(directory).absolute()
    relative = directory.relative_to(root).as_posix()
    root = root.resolve(strict=True)
    directory = resolve_media(root, relative)
    samples = list(samples)
    if not 1 <= len(samples) <= MAX_SAMPLES:
        raise ValueError("gallery sample count exceeds bound")
    rows, total_bytes = [], 0

    def bounded_asset(path, expected_hash=None):
        nonlocal total_bytes
        total_bytes += resolve_media(root, path).stat().st_size
        if total_bytes > MAX_TOTAL_BYTES:
            raise ValueError("gallery total media exceeds bound")
        return _asset(root, path, expected_hash)

    for sample in samples:
        if not isinstance(sample, SampleRecord) or sample.scores is None:
            raise ValueError("gallery requires scored sample records")
        video = bounded_asset(sample.path, sample.content_sha256)
        wav_path = Path(sample.path).with_suffix(".wav").as_posix()
        wav = resolve_media(root, wav_path)
        rows.append({"sample_id": sample.sample_id, "prompt": sample.prompt,
                     "seed": sample.seed, "optimizer_step": sample.optimizer_step,
                     "raw_rewards": dict(sample.scores.raw), "video": video,
                     "audio_master": bounded_asset(wav_path) if wav.exists() else None})
    payload = {"schema": MEDIA_SCHEMA, "wandb_media": True, "optimizer_step": step,
               "reward_columns": sorted(samples[0].scores.raw), "samples": rows}
    if metric_namespace != "eval":
        payload["metric_namespace"] = metric_namespace
    encoded = json.dumps(payload, allow_nan=False, ensure_ascii=True, sort_keys=True) + "\n"
    if len(encoded.encode("utf-8")) > MAX_MANIFEST_BYTES:
        raise ValueError("gallery manifest exceeds bound")
    _validate_payload(payload, root, verify_assets=False)

    def link(path):
        return html.escape(quote(os.path.relpath(root / path, directory), safe="/"), quote=True)

    cards = []
    for row in rows:
        scores = " · ".join(f"{html.escape(k)}: {row['raw_rewards'][k]:.6g}"
                             for k in payload["reward_columns"])
        audio = (f'<a href="{link(row["audio_master"]["path"])}">WAV audio master</a>'
                 if row["audio_master"] else "")
        cards.append(f'<article><h2>{html.escape(row["sample_id"])}</h2>'
                     f'<p class="prompt">{html.escape(row["prompt"])}</p>'
                     f'<p>Seed {row["seed"]} · Step {step}</p><p>{scores}</p>'
                     f'<video controls preload="none" src="{link(row["video"]["path"])}">'
                     'Your browser does not support MP4 playback.</video>'
                     f'<p><a href="{link(row["video"]["path"])}">MP4 with audio</a> {audio}</p>'
                     '</article>')
    page = ('<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>AV evaluation · step {step}</title><style>'
            'body{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem}'
            'article{border-top:1px solid #ccc;padding:1rem 0}video{width:100%;max-width:720px}'
            '.prompt{white-space:pre-wrap}</style>'
            f'<h1>AV evaluation · step {step}</h1><p>{len(rows)} samples. '
            'Use the player controls to hear audio.</p>' + '\n'.join(cards) + '</html>\n')
    _publish(directory / "gallery.html", page)
    _publish(directory / MEDIA_FILENAME, encoded)
    return {"gallery": (directory / "gallery.html").relative_to(root).as_posix(),
            "wandb_media": (directory / MEDIA_FILENAME).relative_to(root).as_posix()}
