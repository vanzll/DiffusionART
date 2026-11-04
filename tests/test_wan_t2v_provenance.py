"""Offline native source/assets checks, before importing or loading weights."""

import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
from torch import nn

from visual_rl.models import wan_t2v
from visual_rl.models.wan_t2v import WAN_T2V_NATIVE_REVISION, WanT2VAdapter
from visual_rl.models.wan_t2v_provenance import (
    import_native_components,
    verify_native_assets,
    verify_native_source,
)


def git(root, *args):
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True, timeout=10
    ).stdout.strip()


@pytest.fixture
def native_source(tmp_path, monkeypatch):
    # Every test owns its checkout and module cache changes. No network or global Git config.
    root = tmp_path / "source"
    modules = root / "wan/modules"
    modules.mkdir(parents=True)
    for name, text in {
        "wan/__init__.py": "",
        "wan/modules/__init__.py": "",
        "wan/modules/model.py": "class WanModel: pass\n",
        "wan/modules/vae.py": "class WanVAE: pass\n",
    }.items():
        (root / name).write_text(text)
    git(root, "init", "--quiet")
    git(root, "add", ".")
    git(
        root,
        "-c",
        "user.name=Offline Test",
        "-c",
        "user.email=offline@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "fixture",
    )
    monkeypatch.setattr(sys, "path", list(sys.path))
    for name in tuple(sys.modules):
        if name == "wan" or name.startswith("wan."):
            monkeypatch.delitem(sys.modules, name)
    yield root, git(root, "rev-parse", "HEAD")
    # Imported fixture modules were added by importlib, not monkeypatch.
    for name, module in tuple(sys.modules.items()):
        if name == "wan" or name.startswith("wan."):
            origin = getattr(module, "__file__", None)
            if origin and Path(origin).resolve().is_relative_to(root):
                sys.modules.pop(name, None)


@pytest.fixture
def native_assets(tmp_path):
    root = tmp_path / "assets"
    root.mkdir()
    metadata = {
        "_class_name": "WanModel",
        "model_type": "t2v",
        "dim": 1536,
        "ffn_dim": 8960,
        "freq_dim": 256,
        "num_heads": 12,
        "num_layers": 30,
        "text_len": 512,
        "in_dim": 16,
        "out_dim": 16,
        "eps": 1e-6,
    }
    (root / "config.json").write_text(json.dumps(metadata))
    # No real weights are needed: these tests verify that validation precedes loading.
    (root / "diffusion_pytorch_model.safetensors").write_bytes(b"fixture")
    (root / "Wan2.1_VAE.pth").write_bytes(b"fixture")
    return root, metadata


def test_native_source_requires_explicit_root_and_exact_git_commit(native_source):
    root, revision = native_source
    with pytest.raises(ValueError, match="native_root"):
        verify_native_source(None, expected_revision=revision)
    with pytest.raises(ValueError, match="pinned revision"):
        verify_native_source(root, expected_revision="0" * 40)
    assert verify_native_source(root, expected_revision=revision) == root


def test_native_source_rejects_tracked_changes_but_allows_untracked_logs(native_source):
    root, revision = native_source
    (root / "audit.log").write_text("untracked local observation\n")
    verify_native_source(root, expected_revision=revision)
    (root / "wan/modules/model.py").write_text("class ModifiedModel: pass\n")
    with pytest.raises(ValueError, match="pinned revision"):
        verify_native_source(root, expected_revision=revision)


def test_native_source_root_cannot_be_a_nested_directory(native_source):
    root, revision = native_source
    nested = root / "nested"
    (nested / "wan/modules").mkdir(parents=True)
    for name in ("wan/__init__.py", "wan/modules/model.py", "wan/modules/vae.py"):
        (nested / name).write_text("")
    with pytest.raises(ValueError, match="repository root"):
        verify_native_source(nested, expected_revision=revision)


@pytest.mark.parametrize("name", ["wan", "wan.modules.model", "wan.modules.vae"])
def test_native_source_rejects_conflicting_root_or_cached_submodule(
    native_source, monkeypatch, name
):
    root, revision = native_source
    module = ModuleType(name)
    module.__file__ = str(root.parent / "foreign" / "model.py")
    monkeypatch.setitem(sys.modules, name, module)
    with pytest.raises(ValueError, match="conflicting Wan"):
        verify_native_source(root, expected_revision=revision)


def test_native_import_is_from_verified_root(native_source):
    root, revision = native_source
    verify_native_source(root, expected_revision=revision)
    model, vae = import_native_components(root)
    assert model.__name__ == "WanModel" and vae.__name__ == "WanVAE"
    assert Path(sys.modules[model.__module__].__file__).resolve() == root / "wan/modules/model.py"


@pytest.mark.parametrize(
    "key,bad",
    [
        ("dim", 5120),
        ("ffn_dim", 13824),
        ("freq_dim", 128),
        ("num_heads", 40),
        ("num_layers", 40),
        ("text_len", 256),
        ("in_dim", 36),
        ("out_dim", 8),
        ("model_type", "i2v"),
    ],
)
def test_native_assets_reject_other_architectures_before_import(native_assets, key, bad):
    root, metadata = native_assets
    metadata[key] = bad
    (root / "config.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=key):
        verify_native_assets(
            root, revision=WAN_T2V_NATIVE_REVISION, expected_revision=WAN_T2V_NATIVE_REVISION
        )


def test_native_assets_revision_and_receipt_hook_fail_closed(native_assets):
    root, _ = native_assets
    with pytest.raises(ValueError, match="revision"):
        verify_native_assets(root, revision="main", expected_revision=WAN_T2V_NATIVE_REVISION)
    seen = []

    def receipt(path, revision):
        seen.append((path, revision))
        raise ValueError("master asset hash receipt mismatch")

    with pytest.raises(ValueError, match="receipt mismatch"):
        verify_native_assets(
            root,
            revision=WAN_T2V_NATIVE_REVISION,
            expected_revision=WAN_T2V_NATIVE_REVISION,
            asset_preflight=receipt,
        )
    assert seen == [(root, WAN_T2V_NATIVE_REVISION)]


def test_native_factory_checks_provenance_before_load_and_preserves_fp32(
    native_source, native_assets, monkeypatch
):
    root, revision = native_source
    assets, _ = native_assets
    monkeypatch.setattr(wan_t2v, "WAN_T2V_SOURCE_REVISION", revision)
    calls = []

    class Model(nn.Linear):
        model_type, in_dim, out_dim, text_len, patch_size = "t2v", 16, 16, 512, (1, 2, 2)

        def __init__(self):
            super().__init__(2, 2)
            self.time_embedding = nn.Linear(2, 2)
            self.time_projection = nn.Linear(2, 2)

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append((path, kwargs))
            return cls()

    class VAE:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.model = nn.Linear(2, 2)

    import visual_rl.models.wan_t2v_dependencies as dependencies
    import visual_rl.models.wan_t2v_provenance as provenance

    monkeypatch.setattr(provenance, "import_native_components", lambda source: (Model, VAE))
    monkeypatch.setattr(
        dependencies, "check_native_wan_dependencies", lambda: {"status": "passed", "mocked": True}
    )
    checked = []
    adapter = WanT2VAdapter.from_pretrained(
        assets,
        native_root=root,
        revision=WAN_T2V_NATIVE_REVISION,
        asset_preflight=lambda path, rev: checked.append((path, rev)),
    )
    assert checked == [(assets, WAN_T2V_NATIVE_REVISION)]
    assert calls[0] == (str(assets), {"local_files_only": True, "torch_dtype": torch.float32})
    assert all(p.dtype == torch.float32 for p in adapter.transformer.parameters())
    assert adapter.transformer.time_embedding.weight.dtype == torch.float32
    assert adapter.transformer.time_projection.weight.dtype == torch.float32
    assert calls[1]["dtype"] == torch.float32 and calls[1]["device"] == "cpu"
    assert not any(p.requires_grad for p in adapter.vae.parameters())
    assert adapter.native_dependency_report == {"status": "passed", "mocked": True}
    assert len(calls) == 2
    with pytest.raises(ValueError, match="FP32"):
        WanT2VAdapter.from_pretrained(assets, native_root=root, torch_dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="revision"):
        WanT2VAdapter.from_pretrained(assets, native_root=root, revision="main")
    assert len(calls) == 2
