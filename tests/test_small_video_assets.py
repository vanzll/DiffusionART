"""CPU-only byte/provenance/data/receipt tests, with no model downloads."""

import copy
import hashlib
import json
import random
from types import SimpleNamespace

import pytest

from visual_rl import small_video_assets as assets


def digest(data):
    return hashlib.sha256(data).hexdigest()


def jsonl(rows):
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()


def data_fixture(train=None):
    train = train or [
        {"sample_id": "t0", "prompt": "training one"},
        {"sample_id": "t1", "prompt": "training two"},
    ]
    validation = [{"sample_id": f"v{i}", "prompt": f"held out {i}"} for i in range(4)]
    selected = random.Random(42).sample(validation, 2)
    data = {
        "asset": "prompts",
        "train": "t2v/train.jsonl",
        "validation": "t2v/validation.jsonl",
        "train_count": len(train),
        "validation_count": len(validation),
        "train_sha256": digest(jsonl(train)),
        "validation_sha256": digest(jsonl(validation)),
        "eval_count": 2,
        "eval_seed": 42,
        "eval_sha256": digest(jsonl(selected)),
        "eval_sample_ids": [row["sample_id"] for row in selected],
        "normalization": "NFKC/casefold/whitespace",
        "expected_train_validation_overlap": 0,
    }
    return data, {"t2v/train.jsonl": jsonl(train), "t2v/validation.jsonl": jsonl(validation)}


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    data, texts = data_fixture()
    config = b'{"scale":1.0}'
    model_files = {"weights.safetensors": b"immutable test bytes", "config.json": config}
    snapshots = []
    for name, files in (("wan", model_files), ("prompts", texts)):
        entries = []
        for path, content in files.items():
            row = {"path": path, "size": len(content)}
            if path == "config.json":
                row["git_blob_sha1"] = hashlib.sha1(
                    f"blob {len(content)}\0".encode() + content
                ).hexdigest()
            else:
                row["sha256"] = digest(content)
            entries.append(row)
        snapshots.append(
            {
                "id": name,
                "env": "TEST_" + name.upper(),
                "repo_id": "author/" + name,
                "revision": "a" * 40,
                "repo_type": "dataset" if name == "prompts" else "model",
                "profiles": ["wan"],
                "files": entries,
            }
        )
    spec = {
        "schema_version": assets.SCHEMA,
        "snapshots": snapshots,
        "sources": [
            {
                "id": "wan",
                "package": "wan",
                "revision": "b" * 40,
                "repo": "https://github.com/author/wan.git",
                "profiles": ["wan"],
            }
        ],
        "data": data,
    }
    root, native = tmp_path / "assets", tmp_path / "native"
    (native / "wan").mkdir(parents=True)
    for entry, directory in assets.snapshot_plan(spec, root, profile="wan"):
        for path, content in (model_files if entry["id"] == "wan" else texts).items():
            target = directory / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)

    def git(command, **kwargs):
        assert kwargs["timeout"] == 20
        return SimpleNamespace(stdout="b" * 40 if "rev-parse" in command else "")

    monkeypatch.setattr(assets.subprocess, "run", git)
    return spec, root, {"wan": native}, root / "receipt.json"


def test_checked_in_manifest_pins_only_exact_native_assets_and_data():
    spec = assets.load_manifest()
    by_id = {entry["id"]: entry for entry in spec["snapshots"]}
    assert by_id["wan"]["revision"] == "37ec512624d61f7aa208f7ea8140a131f93afc9a"
    ltx_weights = [row for row in by_id["ltx"]["files"] if row["path"].endswith("safetensors")]
    assert len(ltx_weights) == 1
    assert ltx_weights[0]["path"] == "ltxv-2b-0.9.6-distilled-04-25.safetensors"
    assert (
        ltx_weights[0]["sha256"]
        == "94891bd4bd08de30d484befbfc54fdcffe6d1596a131baad700b9baa5e1de86b"
    )
    assert by_id["reward"]["repo_id"] == "KlingTeam/VideoReward"
    assert by_id["prompts"]["revision"] == "9d947fb510d67a8e5334ec1d8c7d8927f7658c08"
    assert spec["data"]["eval_count"] == 8 and spec["data"]["eval_seed"] == 42
    assert (
        spec["data"]["eval_sha256"]
        == "06fb7106c1fa00cbe91c5f4b6e79cb3eb99c90d13f3c7995fb62d8486148c9ff"
    )


def test_download_reuses_existing_preparer_with_exact_revision_and_allowlist(prepared, monkeypatch):
    spec, root, _, _ = prepared
    calls = []
    monkeypatch.setattr(assets, "prepare_assets", lambda plan: calls.append(plan))
    result = assets.download_assets(spec, root, profile="wan")
    assert set(result) == {"wan", "prompts"}
    assert len(calls) == 1 and calls[0][1] == []
    for entry, _ in calls[0][0]:
        assert entry["revision"] == "a" * 40
        assert entry["allow_patterns"] and all(
            "*" not in value for value in entry["allow_patterns"]
        )


def test_master_full_verification_and_worker_no_rehash_identity(prepared, monkeypatch):
    spec, root, sources, receipt = prepared
    record = assets.preflight_assets(
        spec, root, source_roots=sources, profile="wan", receipt=receipt
    )
    assert record["data"]["normalized_train_validation_overlap"] == 0
    assert len(record["data"]["sample_ids"]) == 2
    monkeypatch.setattr(
        assets, "_hash_file", lambda *a, **k: pytest.fail("worker rehashed weights")
    )
    checked = assets.verify_receipt(spec, root, receipt, profile="wan", source_roots=sources)
    assert checked == record


def test_same_size_changed_asset_invalidates_fast_receipt(prepared):
    spec, root, sources, receipt = prepared
    assets.preflight_assets(spec, root, source_roots=sources, profile="wan", receipt=receipt)
    model = assets.snapshot_plan(spec, root, profile="wan")[0][1] / "weights.safetensors"
    model.write_bytes(b"x" * model.stat().st_size)
    with pytest.raises(ValueError, match="changed since master"):
        assets.verify_receipt(spec, root, receipt, profile="wan")
    with pytest.raises(ValueError, match="pinned asset bytes"):
        assets.preflight_assets(spec, root, source_roots=sources, profile="wan", receipt=receipt)


def test_missing_asset_fails_without_publishing_receipt(prepared):
    spec, root, sources, receipt = prepared
    (assets.snapshot_plan(spec, root, profile="wan")[0][1] / "weights.safetensors").unlink()
    with pytest.raises(ValueError, match="missing or inaccessible"):
        assets.preflight_assets(spec, root, source_roots=sources, profile="wan", receipt=receipt)
    assert not receipt.exists()


def test_wrong_source_identity_fails_before_weight_scanning(prepared, monkeypatch):
    spec, root, sources, receipt = prepared
    monkeypatch.setattr(assets.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="0" * 40))
    monkeypatch.setattr(
        assets, "_hash_file", lambda *a, **k: pytest.fail("scanned before source check")
    )
    with pytest.raises(ValueError, match="pinned Git revision"):
        assets.preflight_assets(spec, root, source_roots=sources, profile="wan", receipt=receipt)


def test_manifest_or_receipt_corruption_fails(prepared):
    spec, root, sources, receipt = prepared
    assets.preflight_assets(spec, root, source_roots=sources, profile="wan", receipt=receipt)
    different = copy.deepcopy(spec)
    different["snapshots"][0]["revision"] = "c" * 40
    with pytest.raises(ValueError, match="identity differs"):
        assets.verify_receipt(different, root, receipt, profile="wan")
    row = json.loads(receipt.read_text())
    row["data"]["sample_ids"].reverse()
    receipt.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="checksum"):
        assets.verify_receipt(spec, root, receipt, profile="wan")


def test_normalized_prompt_leakage_rejected_even_with_valid_source_hashes(tmp_path):
    # NFKC fullwidth characters, case and whitespace must not hide leakage.
    data, files = data_fixture([{"sample_id": "t", "prompt": "ＨＥＬＤ  OUT\t０"}])
    for name, content in files.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    with pytest.raises(ValueError, match="leakage"):
        assets.prepare_eval_subset(data, tmp_path, tmp_path / "eval.jsonl")


def test_changed_paired_subset_never_overwritten(prepared):
    spec, root, _, _ = prepared
    data_root = assets.snapshot_plan(spec, root, profile="wan")[1][1]
    output = root / "paired.jsonl"
    assets.prepare_eval_subset(spec["data"], data_root, output)
    output.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="existing paired"):
        assets.prepare_eval_subset(spec["data"], data_root, output)
    assert output.read_bytes() == b"corrupted"
    wrong = copy.deepcopy(spec["data"])
    wrong["eval_sample_ids"].reverse()
    with pytest.raises(ValueError, match="order/hash"):
        assets.prepare_eval_subset(wrong, data_root, root / "new.jsonl")


@pytest.mark.parametrize("value", ["../escape", "/absolute", "."])
def test_inventory_paths_cannot_escape_asset_root(prepared, value):
    spec = copy.deepcopy(prepared[0])
    spec["snapshots"][0]["files"][0]["path"] = value
    with pytest.raises(ValueError, match="inventory"):
        assets.validate_manifest(spec)


@pytest.mark.parametrize("payload", [b'{"scale":NaN}', b'{"scale":1e999}'])
def test_nonfinite_json_is_rejected_after_matching_content_pin(prepared, payload):
    spec, root, sources, receipt = prepared
    entry, directory = assets.snapshot_plan(spec, root, profile="wan")[0]
    (directory / "config.json").write_bytes(payload)
    row = next(row for row in entry["files"] if row["path"] == "config.json")
    row.update(
        size=len(payload),
        git_blob_sha1=hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest(),
    )
    with pytest.raises(ValueError, match="nonfinite"):
        assets.preflight_assets(spec, root, source_roots=sources, profile="wan", receipt=receipt)


def test_shard_index_requires_every_referenced_shard_pinned(prepared):
    spec, root, _, _ = prepared
    entry, directory = assets.snapshot_plan(spec, root, profile="wan")[0]
    name = "model.safetensors.index.json"
    content = b'{"weight_map":{"weight":"missing.safetensors"}}'
    (directory / name).write_bytes(content)
    entry["files"].append({"path": name, "size": len(content), "sha256": digest(content)})
    with pytest.raises(ValueError, match="absent/unpinned shard"):
        assets._json_file_checks(entry, directory)


def test_explicit_asset_paths_are_verified_not_treated_as_unpinned_overrides(prepared, tmp_path):
    spec, root, _, _ = prepared
    custom = tmp_path / "custom-model"
    planned = assets.snapshot_plan(spec, root, profile="wan", bindings={"wan": custom})
    assert planned[0][1] == custom
    assert planned[0][0]["revision"] == "a" * 40
    with pytest.raises(ValueError, match="unknown explicit"):
        assets.snapshot_plan(spec, root, profile="wan", bindings={"unknown": custom})


def test_runtime_receipt_preserves_master_custom_bindings_without_rehash(
    prepared, tmp_path, monkeypatch
):
    spec, root, sources, receipt = prepared
    original = assets.snapshot_plan(spec, root, profile="wan")[0][1]
    custom = tmp_path / "existing-immutable-model"
    original.rename(custom)
    assets.preflight_assets(
        spec, root, source_roots=sources, profile="wan", receipt=receipt, bindings={"wan": custom}
    )
    monkeypatch.setattr(
        assets, "_hash_file", lambda *args, **kwargs: pytest.fail("worker rehashed weights")
    )
    record = assets.verify_bound_receipt(spec, root, receipt, profile="wan")
    assert record["assets"]["wan"] == str(custom)
    (custom / "weights.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError):
        assets.verify_bound_receipt(spec, root, receipt, profile="wan")
