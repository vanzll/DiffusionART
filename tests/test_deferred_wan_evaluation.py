"""Inactive fixed eval assets do not block real online-reference train preparation."""
from pathlib import Path

import pytest
import yaml

from visual_rl.config import ConfigError
from visual_rl.prepare_assets import plan_environment, prepare_assets


def clean_bindings(monkeypatch, tmp_path):
    import os
    for key in list(os.environ):
        if key.startswith(("WAN_", "VIDEO_REWARD_", "FLUX_REFERENCE_")):
            monkeypatch.delenv(key)
    monkeypatch.setenv("VISUAL_RL_ASSET_ROOT", str(tmp_path / "assets"))


def test_wan_short_defers_only_eval_not_online_train_assets(monkeypatch, tmp_path):
    clean_bindings(monkeypatch, tmp_path)
    original = Path(__file__).resolve().parents[1] / "configs/paper/video/wan_art_11.yaml"
    short = tmp_path / "short.yaml"
    short.write_text(yaml.safe_dump({"extends": str(original),
                                    "training": {"max_updates": 6},
                                    "evaluation": {"every_updates": 0, "at_updates": []},
                                    "runtime": {"checkpoint_every": 0, "checkpoint_at_updates": []}}))
    snapshots, transformations = plan_environment(short)
    assert {entry["env"] for entry, _ in snapshots} >= {
        "WAN_BASE_MODEL", "WAN_DISTILLED_SOURCE", "FLUX_REFERENCE_MODEL",
        "VIDEO_REWARD_MODEL", "VIDEO_REWARD_BACKBONE",
    }
    assert any(entry.get("action") == "wan_transformer" for entry, _ in transformations)
    assert any(entry["env"] == "WAN_TRAIN_MANIFEST" for entry, _ in transformations)
    assert not any(entry["env"] == "WAN_EVAL_MANIFEST" for entry, _ in transformations)
    # Deferral creates no evaluation images or substitute manifest.
    import os
    assert "WAN_EVAL_MANIFEST" in os.environ
    assert not Path(os.environ["WAN_EVAL_MANIFEST"]).exists()


def test_formal_wan_keeps_public_fixed_eval_asset_blocker(monkeypatch, tmp_path):
    clean_bindings(monkeypatch, tmp_path)
    original = Path(__file__).resolve().parents[1] / "configs/paper/video/wan_art_11.yaml"
    plan = plan_environment(original)
    assert any(entry.get("action") == "fixed_images" for entry, _ in plan[1])
    with pytest.raises(ConfigError, match="no public fixed Wan evaluation images"):
        prepare_assets(plan)
