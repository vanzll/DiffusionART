"""Native SD3 integration with real tiny image modules and an injected JPEG scorer."""

import copy
import hashlib
import json
import socket
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from test_sd3_runner import config_file, tiny_parts

from visual_rl.backends import sd3
from visual_rl.config import ConfigError, load_config
from visual_rl.evaluation import EvaluationManifest, audit_manifest
from visual_rl.registry import get_backend
from visual_rl.rewards.geneval import TAGS, GenEvalScore, jpeg_bytes
from visual_rl.rewards.scalar import ScoreResult
from visual_rl.train import main


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def geneval_config(tmp_path):
    path = config_file(tmp_path)
    config = yaml.safe_load(path.read_text())
    dataset = Path(config["data"]["manifest"])
    rows = [json.loads(line) for line in dataset.read_text().splitlines()]
    for index, row in enumerate(rows):
        row["metadata"] = {
            "prompt": row["prompt"],
            "tag": "single_object" if index < 3 else "colors",
            "include": [{"class": "cat", "count": 1}],
        }
    config["data"]["sha256"] = write_rows(dataset, rows)
    validation = tmp_path / "validation.jsonl"
    config["evaluation"]["manifest"] = str(validation)
    config["evaluation"]["manifest_sha256"] = write_rows(validation, rows)
    # Asset loading is injected; this identity is not a native detector bundle.
    bundle = tmp_path / "reward-manifest.json"
    bundle.write_text(json.dumps({"fixture": "injected JPEG engine"}))
    config["reward"] = {
        "name": "geneval",
        "checkpoint": str(bundle),
        "revision": hashlib.sha256(bundle.read_bytes()).hexdigest(),
        "normalization": "raw",
        "weights": {"geneval": 1.0},
        "options": {
            "worker_python": sys.executable,
            "detector_microbatch": 1,
            "color_workers": 0,
            "microbatch": 2,
        },
    }
    path.write_text(yaml.safe_dump(config))
    return path, config, rows


def pixel_score(encoded):
    with Image.open(BytesIO(encoded)) as image:
        assert image.format == "JPEG" and image.mode == "RGB" and image.size == (4, 4)
        return -2.0 + float(np.asarray(image).mean()) / 255.0


class JPEGEngine:
    def __init__(self, rows):
        self.expected = {row["prompt"]: row["metadata"] for row in rows}
        self.calls = []
        self.closed = 0

    def __call__(self, images, metadata, *, only_strict):
        assert not torch.is_grad_enabled()
        assert 0 < len(images) == len(metadata) <= 2
        assert all(isinstance(image, bytes) for image in images)
        assert metadata == [self.expected[row["prompt"]] for row in metadata]
        dense = [pixel_score(image) for image in images]
        strict = [float(row["tag"] == "colors") for row in metadata]
        ordinary = [0.0 if only_strict else 1.0 - value for value in strict]
        grouped = {
            tag: [
                value if row["tag"] == tag else -10.0
                for value, row in zip(ordinary, metadata, strict=True)
            ]
            for tag in TAGS
        }
        grouped_strict = {
            tag: [
                value if row["tag"] == tag else -10.0
                for value, row in zip(strict, metadata, strict=True)
            ]
            for tag in TAGS
        }
        reply = dense, ordinary, strict, grouped, grouped_strict
        self.calls.append((list(images), copy.deepcopy(metadata), only_strict, reply))
        return reply

    def close(self):
        self.closed += 1


def assert_diagnostics(metrics, calls, *, evaluation):
    prefix = "eval" if evaluation else "train"
    rows = [row for _, batch, _, _ in calls for row in batch]
    for name, offset in (("strict_accuracy", 2), ("accuracy", 1)):
        key = f"{prefix}/diagnostics/geneval/{name}"
        if not evaluation and name == "accuracy":
            assert not any(k.startswith(key) for k in metrics)
            assert not any(k.endswith("/accuracy") for k in metrics)
            continue
        values = [value for _, _, _, reply in calls for value in reply[offset]]
        assert metrics[key] == pytest.approx(sum(values) / len(values))
        means = []
        for tag in TAGS:
            tag_key = f"{prefix}/diagnostics/geneval/tag/{tag}/{name}"
            members = [value for value, row in zip(values, rows, strict=True) if row["tag"] == tag]
            if not members:
                assert tag_key not in metrics
                continue
            means.append(sum(members) / len(members))
            assert metrics[tag_key] == pytest.approx(means[-1])
        assert metrics[f"{key}_macro"] == pytest.approx(sum(means) / len(means))
        assert metrics[f"{key}_tags_present"] == len(means)


@pytest.mark.parametrize("entrypoint,decode_microbatch", [("run", 1), ("cli", 2)])
def test_geneval_native_updates_jpeg_diagnostics_and_artifacts(
    tmp_path, monkeypatch, entrypoint, decode_microbatch
):
    path, config, rows = geneval_config(tmp_path)
    config["model"]["options"]["decode_microbatch"] = decode_microbatch
    path.write_text(yaml.safe_dump(config))
    engine = JPEGEngine(rows)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    def no_network(*args, **kwargs):
        pytest.fail("injected image assets must not download models")

    monkeypatch.setattr(socket.socket, "connect", no_network)

    def components(c):
        parts = tiny_parts(c)
        assert next(parts.adapter.parameters()).device.type == "cpu"
        return sd3.Components(parts.adapter, parts.text, GenEvalScore(engine, device="cpu"))

    monkeypatch.setattr(sd3, "load_components", components)
    original_update = sd3._art_update
    update_scores = []

    def art_update(*args, **kwargs):
        result = args[3]
        torch.testing.assert_close(kwargs["guidance_scores"], result.scores.raw["geneval"])
        update_scores.append(kwargs["guidance_scores"].detach().clone())
        return original_update(*args, **kwargs)

    monkeypatch.setattr(sd3, "_art_update", art_update)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        if entrypoint == "cli":
            assert main(["--config", str(path), "--validate"]) == 0
            assert not engine.calls and not (tmp_path / "run").exists()
            assert main(["--config", str(path)]) == 0
        else:
            sd3.run(load_config(path))
    finally:
        torch.set_num_threads(old_threads)

    root = tmp_path / "run"
    status = json.loads((root / "status.json").read_text())
    assert status["optimizer_steps"] == 4 and status["completed_rollouts"] == 2
    assert engine.closed == 1
    # Evaluation surrounds each rollout; decoding may further split score batches.
    calls_per_batch = 4 // decode_microbatch
    assert [call[2] for call in engine.calls] == [
        mode for mode in (False, True, False, True, False) for _ in range(calls_per_batch)
    ]
    assert all(len(call[0]) == decode_microbatch for call in engine.calls)
    training = [call for call in engine.calls if call[2]]
    evaluation = [call for call in engine.calls if not call[2]]
    training_rows = [row for _, batch, _, _ in training for row in batch]
    assert training_rows[::2] == training_rows[1::2]
    assert sorted(row["prompt"] for _, batch, _, _ in training for row in batch) == [
        row["prompt"] for row in rows for _ in range(2)
    ]
    assert len({score for _, _, _, reply in training for score in reply[0]}) > 1
    assert len(update_scores) == 2
    history = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
    assert [record["train/optimizer_step"] for record in history] == [1, 2, 3, 4]
    for rollout in range(2):
        calls = training[calls_per_batch * rollout : calls_per_batch * (rollout + 1)]
        dense = torch.tensor([value for _, _, _, reply in calls for value in reply[0]])
        torch.testing.assert_close(update_scores[rollout], dense, rtol=0, atol=0)
        assert dense.unique().numel() > 1 and (dense < 0).all()
        for record in history[2 * rollout : 2 * rollout + 2]:
            assert record["cost/elapsed_gpu_hours"] == 0
            assert record["train/reward"] == pytest.approx(float(dense.mean()))
            assert record["train/reward/raw/geneval"] == record["train/reward"]
            assert "train/reward/fixed_normalized/geneval" not in record
            assert_diagnostics(record, calls, evaluation=False)

    manifests = sorted(root.glob("evaluation/step-*/*/manifest.json"))
    assert len(manifests) == 3
    for index, manifest_path in enumerate(manifests):
        serialized = json.loads(manifest_path.read_text())
        manifest = EvaluationManifest.from_json(manifest_path.read_text())
        assert manifest.optimizer_step == 2 * index
        assert audit_manifest(manifest, media_root=root).complete
        assert len(manifest.samples) == 4
        calls = evaluation[calls_per_batch * index : calls_per_batch * (index + 1)]
        expected_jpegs = [image for images, _, _, _ in calls for image in images]
        for sample_index, sample in enumerate(manifest.samples):
            assert (sample.sample_id, sample.prompt, sample.seed) == (
                str(sample_index),
                rows[sample_index]["prompt"],
                config["seed"] + sample_index,
            )
            with Image.open(root / sample.path) as image:
                assert image.format == "PNG" and image.size == (4, 4)
                encoded = jpeg_bytes([image.convert("RGB")])[0]
            assert encoded == expected_jpegs[sample_index]
            scores = sample.scores
            assert scores.space == "raw" and scores.weights == {"geneval": 1.0}
            assert scores.fixed_normalized is None and scores.normalization is None
            assert scores.weighted == scores.raw["geneval"] == pytest.approx(pixel_score(encoded))
            stored = serialized["samples"][sample_index]["scores"]
            assert stored["diagnostics"] == scores.diagnostics
            assert ScoreResult.from_dict(stored).to_dict() == scores.to_dict() == stored
            row = rows[sample_index]["metadata"]
            for name, expected in (
                ("strict_accuracy", float(row["tag"] == "colors")),
                ("accuracy", float(row["tag"] != "colors")),
            ):
                assert scores.diagnostics[f"geneval/{name}"] == expected
                for tag in TAGS:
                    assert scores.diagnostics[f"geneval/tag/{tag}/{name}"] == (
                        expected if row["tag"] == tag else -10.0
                    )
        summary = json.loads(manifest_path.with_name("summary.json").read_text())
        assert summary["eval/optimizer_step"] == 2 * index
        assert summary["eval/reward"] == summary["eval/reward/raw/geneval"]
        assert summary["eval/reward"] == pytest.approx(
            sum(s.scores.weighted for s in manifest.samples) / 4
        )
        assert_diagnostics(summary, calls, evaluation=True)
        assert summary["eval/diagnostics/geneval/strict_accuracy"] == 0.25
        assert summary["eval/diagnostics/geneval/accuracy"] == 0.75
        assert summary["eval/diagnostics/geneval/strict_accuracy_macro"] == 0.5
        assert summary["eval/diagnostics/geneval/accuracy_macro"] == 0.5


@pytest.mark.parametrize("split", ["data", "evaluation"])
@pytest.mark.parametrize("defect", ["missing", "mismatched"])
def test_geneval_metadata_preflight_before_components(tmp_path, monkeypatch, split, defect):
    path, config, rows = geneval_config(tmp_path)
    if defect == "missing":
        rows[-1].pop("metadata")
    else:
        rows[-1]["metadata"]["prompt"] = "different prompt"
    checksum = "sha256" if split == "data" else "manifest_sha256"
    config[split][checksum] = write_rows(Path(config[split]["manifest"]), rows)
    path.write_text(yaml.safe_dump(config))

    def forbidden(*args, **kwargs):
        pytest.fail("invalid GenEval metadata must fail before loading components")

    monkeypatch.setattr(sd3, "load_components", forbidden)
    with pytest.raises(ConfigError, match="GenEval"):
        sd3.run(load_config(path))
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    "invalid",
    [
        {"normalization": "fixed"},
        {"weights": {"geneval": 0.5}},
        {"revision": "a" * 63},
        {"revision": "g" * 64},
        {"revision": "A" * 64},
        {"options": {"worker_python": "python"}},
        {"options": {"worker_python": True}},
        {"options": {"microbatch": 0}},
        {"options": {"microbatch": 257}},
        {"options": {"detector_microbatch": 0}},
        {"options": {"detector_microbatch": True}},
        {"options": {"color_workers": -1}},
        {"options": {"color_workers": 17}},
        {"options": {"color_workers": 1.5}},
        {"options": {"color_workers": True}},
        {"options": {"startup_timeout": 0}},
        {"options": {"request_timeout": 601}},
        {"options": {"request_timeout": True}},
        {"options": {"unknown": 1}},
    ],
)
def test_geneval_invalid_config_rejected(tmp_path, monkeypatch, invalid):
    path, _, _ = geneval_config(tmp_path)
    assert get_backend(load_config(path)).name == "sd3"

    def forbidden(*args, **kwargs):
        pytest.fail("invalid worker settings must fail before loading components")

    monkeypatch.setattr(sd3, "load_components", forbidden)
    with pytest.raises(ConfigError):
        sd3.run(load_config(path, {"reward": invalid}))
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    "option", ["worker_python", "detector_microbatch", "color_workers", "microbatch"]
)
def test_geneval_worker_options_must_be_explicit(tmp_path, monkeypatch, option):
    path, config, _ = geneval_config(tmp_path)
    del config["reward"]["options"][option]
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(
        sd3, "load_components", lambda *args: pytest.fail("must reject before assets")
    )
    with pytest.raises(ConfigError, match="GenEval options"):
        sd3.run(load_config(path))
    assert not (tmp_path / "run").exists()
