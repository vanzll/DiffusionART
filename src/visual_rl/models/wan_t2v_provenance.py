"""Offline provenance checks before importing optional native Wan code."""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

from .source_pin import pinned_source


def _git(root: Path, *args):
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("cannot verify native Wan Git provenance") from exc
    if result.returncode:
        raise ValueError("cannot verify native Wan Git provenance")
    return result.stdout.strip()


def verify_native_source(native_root, *, expected_revision: str) -> Path:
    if native_root is None:
        raise ValueError("native_root must explicitly identify the pinned Wan source checkout")
    root = Path(native_root).expanduser().resolve()
    if not root.is_dir() or not (root / "wan/__init__.py").is_file():
        raise ValueError("native_root must contain the official Wan package")
    for name in ("model.py", "vae.py"):
        if not (root / "wan/modules" / name).is_file():
            raise ValueError("native_root is missing required Wan source modules")
    actual_root = Path(_git(root, "rev-parse", "--show-toplevel")).resolve()
    if actual_root != root:
        raise ValueError("native_root must be the Wan Git repository root")
    verify_wan_module_origins(root)
    pinned_source(root, package="wan", revision=expected_revision)
    return root


def verify_wan_module_origins(root: Path):
    """Never replace, reload, or silently reuse a different already imported Wan."""
    package_root = root / "wan"
    for name, module in tuple(sys.modules.items()):
        if name != "wan" and not name.startswith("wan."):
            continue
        origin = getattr(module, "__file__", None)
        if origin is None or not Path(origin).resolve().is_relative_to(package_root):
            raise ValueError("a conflicting Wan module is already imported outside native_root")
        for path in getattr(module, "__path__", ()):
            if not Path(path).resolve().is_relative_to(package_root):
                raise ValueError("a conflicting Wan package search path is outside native_root")


def import_native_components(root: Path):
    verify_wan_module_origins(root)
    importlib.invalidate_caches()
    model_module = importlib.import_module("wan.modules.model")
    vae_module = importlib.import_module("wan.modules.vae")
    verify_wan_module_origins(root)
    return model_module.WanModel, vae_module.WanVAE


def verify_native_assets(path, *, revision, expected_revision: str, asset_preflight=None) -> Path:
    """Check small metadata per rank; optionally validate a master hash receipt.

    The launch preflight must hash native weights/VAE/text/tokenizer once and
    verify the immutable snapshot receipt. This helper deliberately does not
    rehash17GB on every rank. asset_preflight(path,revision) is a caller hook to
    validate that receipt and must raise on failure; its return value is unused.
    """
    if revision != expected_revision:
        raise ValueError("native model revision does not match pinned Wan T2V1.3B weights")
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("native factory requires an existing local pinned snapshot directory")
    try:
        metadata = json.loads((root / "config.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("native snapshot requires readable root config.json") from exc
    if not isinstance(metadata, dict):
        raise ValueError("native snapshot root config.json must be an object")
    expected = {
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
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"native snapshot is not Wan T2V1.3B: {key}")
    patch = metadata.get("patch_size", (1, 2, 2))
    if not isinstance(patch, (tuple, list)) or tuple(patch) != (1, 2, 2):
        raise ValueError("native snapshot is not Wan T2V1.3B: patch_size")
    if metadata.get("eps") != 1e-6:
        raise ValueError("native snapshot is not Wan T2V1.3B: eps")
    for filename in ("diffusion_pytorch_model.safetensors", "Wan2.1_VAE.pth"):
        if not (root / filename).is_file():
            raise ValueError(f"native snapshot is missing required component: {filename}")
    if asset_preflight is not None:
        if not callable(asset_preflight):
            raise ValueError("asset_preflight must be a callable master-receipt validator")
        asset_preflight(root, revision)
    return root
