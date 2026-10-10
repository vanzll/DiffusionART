"""Portable evaluation assets, using small local fixtures only."""

import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from PIL import Image

from visual_rl.config import ConfigError, _read
from visual_rl.data import PromptDataset, file_sha256
from visual_rl.evaluation.package_i2v import package, publish
from visual_rl.evaluation.select_prompts import freeze_selection, select_ids
from visual_rl.prepare_assets import (
    _vidprom_manifest,
    plan_environment,
    prepare_assets,
    validate_assets,
)


def fixed_manifest(tmp_path):
    image = tmp_path / "original.jpg"
    Image.new("RGB", (16, 12), "green").save(image)
    body = {"schema_version": "wan-i2v-formal-validation/v1",
            "source": {"prompts_sha256": "a" * 64, "training_json": "/private/training.json"},
            "reference_model": {"repo_id": "black-forest-labs/FLUX.1-dev", "snapshot": "/private/model",
                                "revision": "b" * 40},
            "generation": {"num_inference_steps": 30, "guidance_scale": 3.5,
                           "output_dir": "/private/image-output", "token": "private-token"},
            "samples": [{"id": "original-prompt-id", "prompt": "A horse runs across a meadow.",
                         "image": str(image), "image_sha256": file_sha256(image), "image_seed": 312,
                         "metadata": {"source_path": "/private/input", "token": "private-token"}}]}
    checksum = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                         ensure_ascii=True, allow_nan=False).encode()).hexdigest()
    source = tmp_path / "manifest.json"
    source.write_text(json.dumps({**body, "checksum": checksum}))
    return source, image


def test_fixed_bundle_preserves_image_bytes_and_public_identity(tmp_path):
    source, image = fixed_manifest(tmp_path)
    output = tmp_path / "dataset"
    record = package(source, output, expected_count=1)
    data = PromptDataset.load(output / "prompts.jsonl", task="i2v",
                              expected_sha256=record["manifest_sha256"])
    assert data.rows[0].image.read_bytes() == image.read_bytes()
    assert data.rows[0].sample_id == "original-prompt-id"
    assert data.rows[0].metadata == {"image_seed": 312}
    assert "/private/" not in (output / "provenance.json").read_text()
    assert "private-token" not in (output / "provenance.json").read_text()
    assert "private-token" not in (output / "prompts.jsonl").read_text()
    assert record["generation"] == {"num_inference_steps": 30, "guidance_scale": 3.5}
    assert str(tmp_path) not in (output / "prompts.jsonl").read_text()
    assert "FLUX.1-dev" in (output / "README.md").read_text()
    with pytest.raises(ConfigError, match="overwrite"):
        package(source, output, expected_count=1)


def test_fixed_bundle_rejects_incomplete_or_corrupted_inputs(tmp_path):
    source, image = fixed_manifest(tmp_path)
    with pytest.raises(ConfigError, match="count differs"):
        package(source, tmp_path / "missing", expected_count=100)
    assert not (tmp_path / "missing").exists()
    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash differs"):
        package(source, tmp_path / "corrupt", expected_count=1)
    assert not (tmp_path / "corrupt").exists()


def test_fixed_bundle_relocation_uses_original_hashes(tmp_path):
    source, image = fixed_manifest(tmp_path)
    relocated = tmp_path / "copied-images"
    relocated.mkdir()
    copy = relocated / image.name
    copy.write_bytes(image.read_bytes())
    image.unlink()
    output = tmp_path / "dataset"
    record = package(source, output, expected_count=1, image_root=relocated)
    data = PromptDataset.load(output / "prompts.jsonl", task="i2v")
    assert data.rows[0].image.read_bytes() == copy.read_bytes()
    assert record["source_manifest_sha256"] == file_sha256(source)
    copy.write_bytes(b"not the original image")
    with pytest.raises(ValueError, match="hash differs"):
        package(source, tmp_path / "bad", expected_count=1, image_root=relocated)


def test_publish_checks_bundle_before_touching_hub(tmp_path, monkeypatch):
    source, image = fixed_manifest(tmp_path)
    output = tmp_path / "dataset"
    package(source, output, expected_count=1)
    calls = []

    class Api:
        def create_repo(self, repo, **kwargs):
            calls.append((repo, kwargs))

        def upload_folder(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(oid="c" * 40)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=Api))
    record = publish(output, "test/evaluation")
    assert record["revision"] == "c" * 40
    assert calls[0][1]["repo_type"] == "dataset"
    assert calls[1]["repo_type"] == "dataset"
    data = PromptDataset.load(output / "prompts.jsonl", task="i2v")
    data.rows[0].image.write_bytes(b"changed")
    calls.clear()
    with pytest.raises(ValueError, match="hash differs"):
        publish(output, "test/evaluation")
    assert not calls
    assert image.exists()


def test_publication_error_does_not_expose_credentials(tmp_path, monkeypatch):
    source, _ = fixed_manifest(tmp_path)
    output = tmp_path / "dataset"
    package(source, output, expected_count=1)

    class Api:
        def create_repo(self, *args, **kwargs):
            raise RuntimeError("signed-url?token=private-credential")

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=Api))
    with pytest.raises(ConfigError, match="publication failed") as error:
        publish(output, "test/evaluation")
    assert "private-credential" not in str(error.value)
    assert error.value.__suppress_context__


def test_dataset_snapshot_binds_manifest_and_preserves_custom_override(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "environ", {"VISUAL_RL_ASSET_ROOT": str(tmp_path / "cache")})
    source, image = fixed_manifest(tmp_path)
    bundle = tmp_path / "bundle"
    package(source, bundle, expected_count=1)
    spec = {"snapshots": [{"env": "RAW_FIXED", "repo_id": "test/eval", "revision": "a" * 40,
                           "repo_type": "dataset", "manifest_file": "prompts.jsonl", "for_env": "FIXED"}],
            "transformations": [{"action": "fixed_images", "env": "FIXED", "sha_env": "FIXED_SHA",
                                  "source_env": "RAW_FIXED", "format": "jsonl"}]}
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump({"assets": spec}))
    calls = []

    def download(**kwargs):
        import shutil
        calls.append(kwargs)
        shutil.copytree(bundle, kwargs["local_dir"])

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    plan = plan_environment(path)
    prepare_assets(plan)
    assert calls[0]["repo_type"] == "dataset"
    assert "/datasets/" in os.environ["RAW_FIXED"]
    data = PromptDataset.load(os.environ["FIXED"], task="i2v")
    assert data.rows[0].image.read_bytes() == image.read_bytes()
    prepare_assets(([], plan[1]))
    monkeypatch.setattr(os, "environ", {"VISUAL_RL_ASSET_ROOT": str(tmp_path / "other"),
                                       "FIXED": str(bundle / "prompts.jsonl")})
    custom = plan_environment(path)
    assert not custom[0] and len(custom[1]) == 1
    prepare_assets(custom)
    assert os.environ["FIXED_SHA"] == file_sha256(bundle / "prompts.jsonl")


@pytest.mark.parametrize("override", [{"repo_type": "space"}, {"repo_type": []}, {"manifest_file": "../bad.jsonl"},
                                      {"manifest_file": "good.jsonl", "repo_type": "model"}])
def test_dataset_snapshot_rejects_invalid_bindings(override):
    entry = {"env": "DATA", "repo_id": "test/eval", "revision": "a" * 40,
             "repo_type": "dataset", **override}
    with pytest.raises(ConfigError):
        validate_assets({"snapshots": [entry]})


def test_dataset_missing_manifest_does_not_publish(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "environ", {"VISUAL_RL_ASSET_ROOT": str(tmp_path / "cache")})
    spec = {"snapshots": [{"env": "DATA", "repo_id": "test/eval", "revision": "a" * 40,
                           "repo_type": "dataset", "manifest_file": "missing.jsonl"}]}
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump({"assets": spec}))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=lambda **k: None))
    with pytest.raises(ConfigError, match="lacks its manifest"):
        prepare_assets(plan_environment(path))


def test_frozen_eval1000_retains_original100_and_training_assets():
    base = _read(Path("configs/paper/hunyuan-assets.yaml"))["assets"]
    expanded = _read(Path("configs/paper/hunyuan-eval1000-assets.yaml"))["assets"]
    validate_assets(expanded)
    assert expanded["snapshots"] == base["snapshots"]
    assert expanded["transformations"] == base["transformations"]
    assert expanded["prompts"][0] == base["prompts"][0]
    original, larger = base["prompts"][1], expanded["prompts"][1]
    assert larger["ids"][:100] == original["ids"]
    assert len(larger["ids"]) == len(set(larger["ids"])) == 1000
    assert larger["url"] == original["url"] and larger["source_sha256"] == original["source_sha256"]
    assert larger["sha256"] == "3b3392989d399ec0b840fc1161d83858872d3843806f4800162dd728958f7e69"
    profile = _read(Path("configs/examples/hunyuan_art_eval1000.yaml"))
    assert profile["assets"] == expanded
    assert profile["provenance"]["run_id"] is None
    assert profile["runtime"]["output_dir"].endswith("/hunyuan-art-721-eval1000")


def selection_fixture(tmp_path):
    source = tmp_path / "source.csv"
    with source.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["uuid", "prompt"])
        writer.writeheader()
        writer.writerows({"uuid": f"{i:08d}-0000-0000-0000-000000000000",
                         "prompt": text} for i, text in enumerate([
            "A horse gallops across green hills", "The chef pours steaming tea slowly",
            "A train travels through snowy mountains", "Red flowers sway beside flowing water",
            "A silver robot walks across sand", "This excluded text is in training", "https://invalid.example",]))
    training = tmp_path / "train.txt"
    training.write_text("This excluded text is in training\n")
    ids = select_ids(source, training, count=2)
    manifest = tmp_path / "original.jsonl"
    _vidprom_manifest(source, manifest, ids)
    base = {"assets": {"prompts": [
        {"env": "HUNYUAN_TRAIN_RAW_MANIFEST", "sha_env": "TRAIN_SHA", "url": "https://example.com/train.txt",
         "source_sha256": file_sha256(training), "sha256": "a" * 64, "prefix": "train", "format": "text"},
        {"env": "HUNYUAN_EVAL_RAW_MANIFEST", "sha_env": "EVAL_SHA", "url": "https://example.com/eval.csv",
         "source_sha256": file_sha256(source), "sha256": file_sha256(manifest), "prefix": "eval",
         "format": "vidprom_ids", "ids": ids}]}}
    config = tmp_path / "base.yaml"
    config.write_text(yaml.safe_dump(base))
    return source, training, config


def test_freeze_selection_is_deterministic_and_preserves_prefix(tmp_path):
    source, training, config = selection_fixture(tmp_path)
    result = freeze_selection(config, source, training, tmp_path / "expanded.yaml", count=5)
    ids = _read(tmp_path / "expanded.yaml")["assets"]["prompts"][1]["ids"]
    assert result["count"] == len(ids) == 5
    assert ids[:2] == select_ids(source, training, count=2)
    assert ids == select_ids(source, training, count=5)
    with pytest.raises(ConfigError, match="overwrite"):
        freeze_selection(config, source, training, tmp_path / "expanded.yaml", count=5)
    with pytest.raises(ConfigError, match="not enough disjoint"):
        select_ids(source, training, count=6)


def test_selection_rejects_changed_public_inputs_or_prefix(tmp_path):
    source, training, config = selection_fixture(tmp_path)
    body = yaml.safe_load(config.read_text())
    body["assets"]["prompts"][1]["ids"].reverse()
    config.write_text(yaml.safe_dump(body))
    with pytest.raises(ConfigError, match="prefix"):
        freeze_selection(config, source, training, tmp_path / "wrong.yaml", count=5)
    training.write_text("changed training corpus\n")
    with pytest.raises(ConfigError, match="pinned public assets"):
        freeze_selection(config, source, training, tmp_path / "wrong.yaml", count=5)
