"""Preparation integrity tests use byte fixtures, never model weight downloads."""

import hashlib
import json

import pytest

from visual_rl.models.minimax_h3 import MODEL_ID, MODEL_REVISION
from visual_rl.models.prepare_minimax_h3 import (
    COMPONENT_CLASSES,
    MANIFEST_NAME,
    REWARD_ASSETS,
    WEIGHT_COMPONENTS,
    build_manifest,
    build_reward_manifest,
    validate_assets,
    write_manifest,
)


@pytest.fixture
def assets(tmp_path):
    index = {name: ["diffusers" if name in WEIGHT_COMPONENTS[:-1] else "transformers", cls,
                    {"pretrained_model_name_or_path": MODEL_ID, "subfolder": name}]
             for name, cls in COMPONENT_CLASSES.items()}
    (tmp_path / "model_index.json").write_text(json.dumps(index))
    for component in COMPONENT_CLASSES:
        (tmp_path / component).mkdir()
        (tmp_path / component / "config.json").write_text('{"test_fixture": true}')
    for component in WEIGHT_COMPONENTS:
        (tmp_path / component / "test.safetensors").write_bytes(b"explicit-byte-fixture")
    files = []
    for file in sorted(tmp_path.rglob("*")):
        if file.is_file():
            files.append({"path": file.relative_to(tmp_path).as_posix(),
                          "size": file.stat().st_size,
                          "git_blob_id": hashlib.sha1(
                              f"blob {file.stat().st_size}\0".encode() + file.read_bytes()).hexdigest(),
                          "hub_lfs_sha256": hashlib.sha256(file.read_bytes()).hexdigest()
                          if file.suffix == ".safetensors" else None})
    return tmp_path, {"model_id": MODEL_ID, "revision": MODEL_REVISION, "files": files}


def test_prepare_receipt_and_offline_validation_never_request_network(assets, monkeypatch):
    root, metadata = assets
    receipt = build_manifest(root, metadata)
    write_manifest(root / MANIFEST_NAME, receipt)
    import socket
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("network called"))
    actual = validate_assets(root)
    assert actual["ready"] and actual["weight_verification"] == "size_only"
    assert actual["gpu_validated"] is False
    assert actual["components"]["audio_vae"]["revision"] == MODEL_REVISION
    assert actual["components"]["audio_vae"]["config_sha256"]


def test_corrupted_same_size_config_is_rejected(assets):
    root, metadata = assets
    write_manifest(root / MANIFEST_NAME, build_manifest(root, metadata))
    config = root / "audio_vae/config.json"
    config.write_text(config.read_text().replace("true", "null"))
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_assets(root)


def test_corrupted_same_size_weight_is_rejected_with_strong_hash(assets):
    root, metadata = assets
    write_manifest(root / MANIFEST_NAME, build_manifest(root, metadata, hash_weights=True))
    file = root / "transformer/test.safetensors"
    file.write_bytes(b"X" * file.stat().st_size)
    with pytest.raises(ValueError, match="SHA256"):
        validate_assets(root)


def test_size_only_validation_does_not_claim_weight_byte_integrity(assets):
    root, metadata = assets
    receipt = build_manifest(root, metadata)
    write_manifest(root / MANIFEST_NAME, receipt)
    file = root / "transformer/test.safetensors"
    file.write_bytes(b"X" * file.stat().st_size)
    assert validate_assets(root)["weight_verification"] == "size_only"
    with pytest.raises(ValueError, match="SHA256"):
        validate_assets(root, hash_weights=True)


def test_missing_weight_never_becomes_ready(assets):
    root, metadata = assets
    (root / "vae/test.safetensors").unlink()
    with pytest.raises(FileNotFoundError):
        build_manifest(root, metadata)
    partial = build_manifest(root, metadata, require_weights=False)
    assert not partial["ready"]
    write_manifest(root / MANIFEST_NAME, partial)
    with pytest.raises(ValueError, match="incomplete"):
        validate_assets(root)


@pytest.mark.parametrize("path", ["../escape", "/escape", "transformer_ref/test.safetensors"])
def test_unsafe_or_wrong_partition_metadata_rejected(assets, path):
    root, metadata = assets
    metadata["files"].append({"path": path, "size": 1})
    with pytest.raises(ValueError):
        build_manifest(root, metadata)


def test_foreign_component_source_is_rejected(assets):
    root, metadata = assets
    path = root / "model_index.json"
    index = json.loads(path.read_text())
    index["audio_vae"][2]["revision"] = "a" * 40
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="foreign"):
        build_manifest(root, metadata)


def test_index_cannot_omit_shard_identity(assets):
    root, metadata = assets
    path = root / "transformer/model.safetensors.index.json"
    path.write_text(json.dumps({"weight_map": {"layer.weight": "absent.safetensors"}}))
    metadata["files"].append({"path": "transformer/model.safetensors.index.json",
                              "size": path.stat().st_size,
                              "git_blob_id": hashlib.sha1(
                                  f"blob {path.stat().st_size}\0".encode() + path.read_bytes()).hexdigest()})
    with pytest.raises(ValueError, match="shard"):
        build_manifest(root, metadata)


@pytest.fixture
def reward_assets(tmp_path):
    paths, provenance = {}, {}
    for name in REWARD_ASSETS:
        directory = tmp_path / name
        directory.mkdir()
        (directory / "config.json").write_text('{"fixture":true}')
        (directory / "weight.bin").write_bytes(b"test-only-fixture")
        paths[name] = name
        provenance[name] = {"repo": "fixture/repo", "revision": "a" * 40,
                            "license": "fixture-license-not-real-weights"}
    return tmp_path, paths, provenance


def test_reward_inventory_covers_every_file_and_preserves_identity(reward_assets):
    root, paths, provenance = reward_assets
    manifest = build_reward_manifest(paths, provenance, base_dir=root)
    assert set(manifest) == {"schema_version", "paths", "sha256", "provenance"}
    assert manifest["schema_version"] == 1
    assert set(manifest["sha256"]["clap"]) == {"config.json", "weight.bin"}
    assert manifest["sha256"]["clap"]["weight.bin"] == hashlib.sha256(b"test-only-fixture").hexdigest()
    paths["clap"] = str(root / "clap/config.json")
    assert set(build_reward_manifest(paths, provenance, base_dir=root)["sha256"]["clap"]) == {"."}


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_reward_symlink_inventory_is_rejected(reward_assets, kind):
    root, paths, provenance = reward_assets
    target = root / "clap" / ("weight.bin" if kind == "file" else ".")
    (root / "video_align" / "alias").symlink_to(target, target_is_directory=kind == "directory")
    with pytest.raises(ValueError, match="symlink"):
        build_reward_manifest(paths, provenance, base_dir=root)


def test_reward_manifest_rejects_empty_and_floating_assets(reward_assets):
    root, paths, provenance = reward_assets
    provenance["clap"]["revision"] = "main"
    with pytest.raises(ValueError, match="immutable"):
        build_reward_manifest(paths, provenance, base_dir=root)
    provenance["clap"]["revision"] = "b" * 40
    for file in (root / "clap").iterdir():
        file.unlink()
    with pytest.raises(ValueError, match="empty"):
        build_reward_manifest(paths, provenance, base_dir=root)


def test_prepared_extended_provenance_roundtrips_runtime(reward_assets, monkeypatch):
    from types import SimpleNamespace

    from visual_rl.backends.minimax_h3 import load_reward
    from visual_rl.rewards.audio_video import AudioVideoReward

    root, paths, provenance = reward_assets
    provenance['hpsv3']['base_model'] = dict(repo='fixture/base7b', revision='b' * 40, license='fixture')
    provenance['audiobox']['implementation'] = dict(repo='fixture/implementation', revision='c' * 40,
                                                   license='fixture', version='0.0.4')
    provenance['desync_source']['subtree'] = 'fixture/source'
    manifest = build_reward_manifest(paths, provenance, base_dir=root)
    path = root / 'rewards.json'
    write_manifest(path, manifest)
    observed = {}
    def make_reward(cls, actual_paths, *, device, expected_sha256):
        observed.update(paths=actual_paths, device=device, sha256=expected_sha256)
        return SimpleNamespace()
    monkeypatch.setattr(AudioVideoReward, 'from_paths', classmethod(make_reward))
    config = SimpleNamespace(reward=SimpleNamespace(options={'manifest': str(path)}))
    reward = load_reward(config, 'cpu')
    assert reward.asset_identity['provenance'] == provenance
    assert observed['sha256'] == manifest['sha256']
    assert observed['paths'] == manifest['paths']
    assert reward.asset_identity['manifest_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_nested_provenance_rejects_floating_revision(reward_assets):
    root, paths, provenance = reward_assets
    provenance['hpsv3']['base_model'] = dict(repo='fixture/base', revision='main', license='fixture')
    with pytest.raises(ValueError, match='immutable.*hpsv3/base_model'):
        build_reward_manifest(paths, provenance, base_dir=root)


def test_actual_loader_requires_ready_inventory_before_loading_components(assets, monkeypatch):
    from visual_rl.models.minimax_h3_assets import load_minimax_h3
    root, metadata = assets
    write_manifest(root / MANIFEST_NAME, {"revision": MODEL_REVISION})
    with pytest.raises(ValueError, match="incomplete"):
        load_minimax_h3(str(root))
    write_manifest(root / MANIFEST_NAME, build_manifest(root, metadata))
    file = root / "audio_vae/config.json"
    file.write_text(file.read_text().replace("true", "null"))
    with pytest.raises(ValueError, match="identity mismatch"):
        load_minimax_h3(str(root))


def test_initial_seal_rejects_same_size_configuration_tampering(assets):
    root, metadata = assets
    config = root / "audio_vae/config.json"
    config.write_text(config.read_text().replace("true", "null"))
    with pytest.raises(ValueError, match="Git blob identity"):
        build_manifest(root, metadata, hash_weights=True)


def test_strong_weight_verification_requires_authoritative_hash(assets):
    root, metadata = assets
    entry = next(v for v in metadata["files"] if v["path"] == "audio_vae/test.safetensors")
    entry["hub_lfs_sha256"] = None
    with pytest.raises(ValueError, match="authoritative"):
        build_manifest(root, metadata, hash_weights=True)


def test_hub_cache_file_symlink_is_verified_by_target_bytes(assets):
    root, metadata = assets
    config = root / "audio_vae/config.json"
    blob = root / "hub-blob"
    config.rename(blob)
    config.symlink_to(blob)
    assert build_manifest(root, metadata)["ready"]


def test_actual_loader_rejects_native_source_drift_before_weight_loading(assets, monkeypatch):
    pytest.importorskip("diffusers")
    from visual_rl.models import check_minimax_h3_environment as environment
    from visual_rl.models.minimax_h3_assets import load_minimax_h3
    root, metadata = assets
    write_manifest(root / MANIFEST_NAME, build_manifest(root, metadata))
    monkeypatch.setattr(environment, "check_native_sources", lambda root: ({}, ["fixture native drift"]))
    with pytest.raises(RuntimeError, match="fixture native drift"):
        load_minimax_h3(str(root))


def test_frozen_component_staging_rejects_policy_and_offloads_after_failure():
    import torch

    from visual_rl.models.minimax_h3_assets import stage_native_components
    policy = torch.nn.Linear(1, 1)
    with pytest.raises(ValueError, match="trainable"):
        with stage_native_components((policy,), "cpu"):
            pass
    frozen = [torch.nn.Linear(1, 1).requires_grad_(False) for _ in range(2)]
    with pytest.raises(RuntimeError, match="forward failure"):
        with stage_native_components(frozen, "cpu"):
            raise RuntimeError("forward failure")
    assert all(p.device.type == "cpu" for m in frozen for p in m.parameters())
    assert all(p.dtype == torch.float32 for m in frozen for p in m.parameters())
