"""Public recipes use one HF dataset without changing existing training settings."""

import os
import re
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from visual_rl.backends import wan
from visual_rl.config import _read, load_config
from visual_rl.prepare_assets import plan_environment, validate_assets

ROOT = Path(__file__).resolve().parents[1]


def test_every_declared_paper_prompt_download_uses_immutable_owned_hf_dataset():
    entries = []
    for path in (ROOT / "configs/paper").rglob("*.yaml"):
        spec = _read(path).get("assets", {})
        validate_assets(spec)
        for entry in spec.get("prompts", []):
            url = urlsplit(entry["url"])
            assert url.hostname == "huggingface.co", (path, entry)
            assert re.fullmatch(r"/datasets/vanzl/DiffusionART-Prompts/resolve/[0-9a-f]{40}/sources/[^/]+",
                                url.path), (path, entry)
            entries.append(entry["env"])
    assert set(entries) >= {"PICKSCORE_TRAIN_MANIFEST", "GENEVAL_TRAIN_MANIFEST", "OCR_TRAIN_MANIFEST",
                           "HUNYUAN_TRAIN_RAW_MANIFEST", "WAN_TRAIN_MANIFEST", "WAN_EVAL_MANIFEST"}


@pytest.mark.parametrize("method,parent", [
    ("art", "wan_art_11"), ("nft", "wan_nft_ema_11"), ("fast", "wan_fast_11"),
    ("dance", "wan_dance_11"), ("refl", "wan_refl_11"),
])
def test_online_eval_profiles_keep_training_and_require_no_historical_images(tmp_path, monkeypatch, method, parent):
    monkeypatch.setattr(os, "environ", {"VISUAL_RL_ASSET_ROOT": str(tmp_path / "assets"),
                                      "DANCEGRPO_ROOT": "/native"})
    path = ROOT / f"configs/examples/wan_{method}_online_eval.yaml"
    original = _read(ROOT / f"configs/paper/video/{parent}.yaml")
    new = _read(path)
    for field in ("algorithm", "training", "sampling", "reward", "data", "evaluation"):
        assert new[field] == original[field], field
    assert new["runtime"]["checkpoint_every"] == original["runtime"]["checkpoint_every"]
    assert new["provenance"]["run_id"] is None
    snapshots, prepared = plan_environment(path)
    assert any(entry["env"] == "FLUX_REFERENCE_MODEL" for entry, _ in snapshots)
    assert not any(entry.get("action") == "fixed_images" for entry, _ in prepared)
    assert any(entry["env"] == "WAN_EVAL_MANIFEST" and entry["format"] == "vidprom_ids"
               for entry, _ in prepared)
    config = load_config(path)
    wan.validate_config(config)
    assert config.model.options["evaluation_reference_mode"] == "online_flux"
    assert config.training.max_updates == 400


def test_image_benchmarks_have_automatic_train_and_eval_prompts():
    for path in (ROOT / "configs/paper/image").glob("*.yaml"):
        doc = _read(path)
        if "data" not in doc:
            continue
        variables = {entry["env"] for entry in doc.get("assets", {}).get("prompts", [])}
        for key, field in (("data", "manifest"), ("evaluation", "manifest")):
            binding = doc[key][field]
            assert binding[2:-1] in variables, path
