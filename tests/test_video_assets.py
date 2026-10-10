"""Asset automation: CPU fixtures only, no real models, CUDA or network."""

import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from PIL import Image

from visual_rl.config import ConfigError, _read, load_config
from visual_rl.data import PromptDataset
from visual_rl.prepare_assets import (
    _hash,
    _vidprom_manifest,
    environment_bindings,
    plan_environment,
    prepare_assets,
    validate_assets,
    validate_bindings,
)


@pytest.fixture(autouse=True)
def environment(monkeypatch, tmp_path):
    monkeypatch.setattr(os, "environ", {"VISUAL_RL_ASSET_ROOT": str(tmp_path / "cache")})


def config(tmp_path, transformations, **assets):
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump({"assets": {"transformations": transformations, **assets}}))
    return path


def fixed_recipe(format="jsonl"):
    return {"action": "fixed_images", "env": "FIXED_MANIFEST", "sha_env": "FIXED_SHA256",
            "source_env": "RAW_FIXED_MANIFEST", "format": format}


def fixed_source(tmp_path):
    image = tmp_path / "ref.png"
    Image.new("RGB", (16, 12), (100, 200, 40)).save(image)
    raw = tmp_path / "raw.jsonl"
    row = {"sample_id": "fixed-original-id", "prompt": "original prompt", "image": image.name,
           "image_sha256": _hash(image), "metadata": {"source": "custom fixture"}}
    raw.write_text(json.dumps(row) + "\n")
    os.environ["RAW_FIXED_MANIFEST"] = str(raw)
    return raw, image, row


def text_model(tmp_path):
    root = tmp_path / "model"
    for folder, files in {
        "text_encoder": ["config.json", "model.safetensors"],
        "text_encoder_2": ["config.json", "model.safetensors"],
        "tokenizer": ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"],
        "tokenizer_2": ["vocab.json", "merges.txt", "tokenizer_config.json", "special_tokens_map.json"],
    }.items():
        (root / folder).mkdir(parents=True)
        for name in files:
            (root / folder / name).write_text("{}")
    os.environ["TEXT_SOURCE"] = str(root)
    return root


def test_fixed_image_conversion_preserves_ids_metadata_and_bytes(tmp_path):
    raw, image, row = fixed_source(tmp_path)
    before = raw.read_bytes()
    plan = plan_environment(config(tmp_path, [fixed_recipe()]))
    assert "FIXED_SHA256" not in os.environ
    prepare_assets(plan)
    destination = plan[1][0][1]
    data = PromptDataset.load(destination, expected_sha256=os.environ["FIXED_SHA256"], task="i2v")
    assert data.rows[0].sample_id == row["sample_id"]
    assert data.rows[0].text == row["prompt"] and data.rows[0].metadata == row["metadata"]
    assert data.rows[0].image.read_bytes() == image.read_bytes()
    assert raw.read_bytes() == before
    prepare_assets(plan)
    receipt = json.loads((destination.parent / "preparation-receipt.json").read_text())
    assert receipt["identity"]["source_sha256"] == _hash(raw)
    assert receipt["output"] == str(destination)
    data.rows[0].image.write_bytes(b"corrupted")
    with pytest.raises(ConfigError, match="corrupted"):
        prepare_assets(plan)


def test_fixed_input_corruption_never_publishes(tmp_path):
    _, image, _ = fixed_source(tmp_path)
    plan = plan_environment(config(tmp_path, [fixed_recipe()]))
    image.write_bytes(b"changed")
    with pytest.raises(ConfigError, match="fixed_images failed"):
        prepare_assets(plan)
    assert not plan[1][0][1].exists()


def test_custom_fixed_images_do_not_require_manual_hashes(tmp_path):
    raw, image, row = fixed_source(tmp_path)
    row.pop("image_sha256")
    raw.write_text(json.dumps(row) + "\n")
    plan = plan_environment(config(tmp_path, [fixed_recipe()]))
    prepare_assets(plan)
    saved = json.loads(plan[1][0][1].read_text())
    assert saved["image_sha256"] == _hash(image)


def test_shared_cache_validates_custom_source_roles_and_inventory(tmp_path):
    from visual_rl.prepare_distributed import prepare_shared_assets

    raw, _, row = fixed_source(tmp_path)
    os.environ["RAW_INPUT"] = str(raw)
    recipe = {**fixed_recipe(), "source_env": "RAW_INPUT"}
    path = config(tmp_path, [recipe])
    plan = plan_environment(path)
    prepare_shared_assets(path, plan, prepare_assets)
    bindings = dict(os.environ)
    destination = plan[1][0][1]
    prepared = json.loads(destination.read_text())
    (destination.parent / prepared["image"]).unlink()
    os.environ.pop("FIXED_SHA256")
    with pytest.raises(ConfigError, match="fixed_images failed"):
        prepare_shared_assets(path, plan, prepare_assets)
    os.environ.update(bindings)
    os.environ.pop("FIXED_SHA256")
    row["prompt"] = "different scientific input"
    raw.write_text(json.dumps(row) + "\n")
    with pytest.raises(ConfigError, match="inputs differ"):
        prepare_shared_assets(path, plan, prepare_assets)


def test_wan_legacy_fixed_manifest_conversion_checks_provenance(tmp_path):
    raw, image, row = fixed_source(tmp_path)
    legacy = tmp_path / "legacy.json"
    body = {"schema_version": "wan-i2v-formal-validation/v1", "samples": [
        {"id": row["sample_id"], "prompt": row["prompt"], "image": str(image),
         "image_sha256": row["image_sha256"]}]}
    checksum = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                         ensure_ascii=True, allow_nan=False).encode()).hexdigest()
    legacy.write_text(json.dumps({**body, "checksum": checksum}))
    os.environ["RAW_FIXED_MANIFEST"] = str(legacy)
    plan = plan_environment(config(tmp_path, [fixed_recipe("wan_validation")]))
    prepare_assets(plan)
    assert json.loads(plan[1][0][1].read_text())["sample_id"] == row["sample_id"]
    legacy.write_text(json.dumps({**body, "checksum": "a" * 64}))
    with pytest.raises(ConfigError, match="inputs differ"):
        prepare_assets(plan)
    assert raw.exists()


def test_missing_fixed_source_fails_before_downloads(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        snapshot_download=lambda **k: pytest.fail("download despite missing evaluation images")))
    plan = plan_environment("configs/paper/video/wan_art_11.yaml")
    with pytest.raises(ConfigError, match="no public fixed Wan evaluation images"):
        prepare_assets(plan)


def test_missing_native_checkout_fails_before_downloads(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        snapshot_download=lambda **k: pytest.fail("download despite missing native source")))
    plan = plan_environment("configs/paper/video/hunyuan_art_721.yaml")
    with pytest.raises(ConfigError, match="requires DANCEGRPO_ROOT"):
        prepare_assets(plan)


def test_actual_native_text_view_reuses_and_validates_links(tmp_path):
    root = text_model(tmp_path)
    recipe = {"action": "hunyuan_text_view", "env": "TEXT_VIEW", "model_env": "TEXT_SOURCE"}
    plan = plan_environment(config(tmp_path, [recipe]))
    prepare_assets(plan)
    view = plan[1][0][1]
    assert (view / "text_encoder/model.safetensors").resolve() == root / "text_encoder/model.safetensors"
    prepare_assets(plan)
    (view / "text_encoder/model.safetensors").unlink()
    with pytest.raises(ConfigError, match="view link differs"):
        prepare_assets(plan)


def test_real_conditioning_serialization_with_mock_encoder(tmp_path, monkeypatch):
    from visual_rl.models import hunyuan_assets

    text_model(tmp_path)
    raw = tmp_path / "raw.jsonl"
    raw.write_text(json.dumps({"sample_id": "original-id", "prompt": "unaltered prompt"}) + "\n")
    os.environ["RAW_TEXT"] = str(raw)
    native = tmp_path / "native"
    native.mkdir()
    os.environ["NATIVE_ROOT"] = str(native)
    calls = []

    class Encoder(torch.nn.Module):
        def __init__(self, model, device):
            super().__init__()
            calls.append((model, device))

        def encode_prompt(self, prompts):
            assert prompts == ["unaltered prompt"]
            return torch.ones(1, 4, 8), torch.ones(1, 3, dtype=torch.bool)

    monkeypatch.setattr(hunyuan_assets, "native_checkout", lambda p: Path(p))
    monkeypatch.setitem(sys.modules, "fastvideo.utils.load", SimpleNamespace(HunyuanTextEncoderWrapper=Encoder))
    recipe = {"action": "hunyuan_conditioning", "env": "PREPARED_TEXT", "sha_env": "PREPARED_SHA",
              "source_env": "RAW_TEXT", "source_sha_env": "RAW_SHA", "model_env": "TEXT_SOURCE",
              "native_env": "NATIVE_ROOT", "device": "cpu"}
    plan = plan_environment(config(tmp_path, [recipe]))
    prepare_assets(plan)
    manifest = plan[1][0][1]
    dataset = PromptDataset.load(manifest, expected_sha256=os.environ["PREPARED_SHA"], task="t2v")
    embeddings, masks = hunyuan_assets.cached_conditioning(dataset.rows, manifest)
    assert embeddings.shape == (1, 4, 8) and masks.shape == (1, 3)
    assert dataset.rows[0].sample_id == "original-id" and os.environ["RAW_SHA"] == _hash(raw)
    prepare_assets(plan)
    assert len(calls) == 1 and calls[0][1] == "cpu"
    raw.write_text(raw.read_text().replace("unaltered", "changed"))
    with pytest.raises(ConfigError, match="raw conditioning input"):
        prepare_assets(plan)


def test_vidprom_extracts_frozen_order_and_hash_ids(tmp_path):
    ids = ["a" * 36, "b" * 36]
    source = tmp_path / "source.csv"
    with source.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["uuid", "prompt"])
        writer.writeheader()
        writer.writerows([{"uuid": ids[1], "prompt": " second  prompt "},
                         {"uuid": "unused", "prompt": "do not substitute"},
                         {"uuid": ids[0], "prompt": " first prompt "}])
    output = tmp_path / "prompts.jsonl"
    _vidprom_manifest(source, output, ids)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [r["prompt"] for r in rows] == ["first prompt", "second prompt"]
    assert rows[0]["sample_id"] == hashlib.sha256(b"first prompt").hexdigest()
    with pytest.raises(ConfigError, match="lacks frozen"):
        _vidprom_manifest(source, tmp_path / "missing.jsonl", ["absent"])


def test_selective_wan_download_and_pinned_receipt_reuse(tmp_path, monkeypatch):
    from visual_rl.models import convert_wan

    source_root, base = tmp_path / "source", tmp_path / "base"
    source_root.mkdir()
    (base / "transformer").mkdir(parents=True)
    source = source_root / "weights.safetensors"
    source.write_bytes(b"source fixture")
    original_config = base / "transformer/config.json"
    original_config.write_text("{}")
    os.environ["SOURCE_ROOT"] = str(source_root)
    os.environ["BASE_ROOT"] = str(base)
    recipe = {"action": "wan_transformer", "env": "TRANSFORMER", "source_env": "SOURCE_ROOT",
              "source_file": source.name, "config_env": "BASE_ROOT", "config_file": "transformer/config.json",
              "source_sha256": _hash(source), "revision": "a" * 40}
    calls = []

    def convert(source, config, output, **kwargs):
        calls.append(kwargs)
        output.mkdir()
        (output / "config.json").write_bytes(config.read_bytes())
        (output / "weights.safetensors").write_bytes(b"converted fixture")
        receipt = {"schema_version": "wan-checkpoint-conversion/v1", "source": {"sha256": _hash(source)},
                   "converter": {"diffusers": "0.33.1"},
                   "config": {"sha256": _hash(config)}, "value_transform": "none", "dtype_transform": "none",
                   "files": [{"path": "config.json", "sha256": _hash(output / "config.json"),
                              "size": (output / "config.json").stat().st_size},
                             {"path": "weights.safetensors", "sha256": _hash(output / "weights.safetensors"),
                              "size": (output / "weights.safetensors").stat().st_size}]}
        (output / "conversion-receipt.json").write_text(json.dumps(receipt))

    monkeypatch.setattr(convert_wan, "convert_checkpoint", convert)
    plan = plan_environment(config(tmp_path, [recipe]))
    prepare_assets(plan)

    def small_file_hash(path):
        assert path.suffix != ".safetensors", "cached startup must not rehash weights"
        return _hash(path)

    monkeypatch.setattr("visual_rl.prepare_assets._hash", small_file_hash)
    prepare_assets(plan)
    assert len(calls) == 1 and calls[0]["expected_source_sha256"] == _hash(source)
    original_config.write_text('{"changed": true}')
    with pytest.raises(ConfigError, match="receipt differs"):
        prepare_assets(plan)


def test_snapshot_download_filters(tmp_path, monkeypatch):
    calls = []
    contents = b"small pinned fixture"
    snapshot = {"env": "SNAPSHOT", "repo_id": "author/model", "revision": "a" * 40,
                "allow_patterns": ["weights.bin"]}

    def download(**kwargs):
        calls.append(kwargs)
        dest = Path(kwargs["local_dir"])
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "weights.bin").write_bytes(contents)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    plan = plan_environment(config(tmp_path, [], snapshots=[snapshot]))
    prepare_assets(plan)
    assert calls[0]["allow_patterns"] == ["weights.bin"]


def test_custom_completed_bindings_skip_prerequisite_assets(tmp_path):
    for name in ("HUNYUAN_MODEL", "VIDEO_REWARD_MODEL", "VIDEO_REWARD_BACKBONE",
                 "HUNYUAN_TRAIN_MANIFEST", "HUNYUAN_EVAL_MANIFEST"):
        os.environ[name] = str(tmp_path / name)
    for name in ("HUNYUAN_TRAIN_SHA256", "HUNYUAN_EVAL_SHA256"):
        os.environ[name] = "a" * 64
    plan = plan_environment("configs/paper/video/hunyuan_art_721.yaml")
    assert plan == ([], [])
    bindings = environment_bindings(_read(Path("configs/paper/hunyuan-assets.yaml"))["assets"])
    assert list(bindings) == sorted(bindings)
    assert "HUNYUAN_TRAIN_SHA256" in bindings and "WANDB_API_KEY" not in bindings


def test_wan_conversion_revision_conflict_rejected(tmp_path):
    os.environ["WAN_EVAL_MANIFEST"] = str(tmp_path / "fixed.jsonl")
    os.environ["WAN_EVAL_SHA256"] = "a" * 64
    os.environ["DANCEGRPO_ROOT"] = str(tmp_path / "native")
    plan = plan_environment("configs/paper/video/wan_art_11.yaml")
    c = load_config("configs/paper/video/wan_art_11.yaml")
    validate_bindings(c, plan)
    c = load_config("configs/paper/video/wan_art_11.yaml", {"model": {"revision": "b" * 40}})
    with pytest.raises(ConfigError, match="revision differs"):
        validate_bindings(c, plan)


@pytest.mark.parametrize("assets", [
    {"transformations": [{"action": "exec", "env": "OUTPUT", "command": "echo unsafe"}]},
    {"transformations": [{"action": [], "env": "OUTPUT"}]},
    {"transformations": [{**fixed_recipe(), "command": "ignored"}]},
    {"transformations": [{**fixed_recipe(), "source_env": "HF_TOKEN"}]},
    {"snapshots": [{"env": "MODEL", "repo_id": "a/b", "revision": "a" * 40, "allow_patterns": ["../x"]}]},
    {"transformations": [{**fixed_recipe(), "source_env": "FIXED_MANIFEST"}]},
])
def test_unsafe_or_malformed_transformations_are_rejected(assets):
    with pytest.raises(ConfigError):
        validate_assets(assets)


def test_all_video_profiles_inherit_asset_pins_without_network(tmp_path, monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: pytest.fail("network during planning"))
    for path in sorted(Path("configs/paper/video").glob("*.yaml")):
        os.environ.clear()
        os.environ["VISUAL_RL_ASSET_ROOT"] = str(tmp_path / "cache")
        os.environ["DANCEGRPO_ROOT"] = str(tmp_path / "native")
        os.environ["HUNYUAN_TRAIN_SHA256"] = "a" * 64
        os.environ["HUNYUAN_EVAL_SHA256"] = "a" * 64
        os.environ["WAN_EVAL_SHA256"] = "a" * 64
        plan = plan_environment(path)
        c = load_config(path)
        validate_bindings(c, plan)
        assert c.training.max_updates == 400 and len(plan[0]) == (5 if c.task == "i2v" else 3)
        assert all(not str(dest).startswith("/m2v") for group in plan for _, dest in group)
        assert c.model.options["native_root"] == str(tmp_path / "native")
