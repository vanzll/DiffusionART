"""Paper short checks preserve their evaluation protocol when cadence is paused."""
from dataclasses import replace
from pathlib import Path

import pytest

from visual_rl.config import ConfigError, load_config
from visual_rl.prepare_assets import plan_environment
from visual_rl.registry import get_backend


@pytest.mark.parametrize("recipe", ["sd3_pickscore_art", "sd3_pickscore_nft_ema"])
def test_paper_short_check_preserves_paused_evaluation(recipe, monkeypatch, tmp_path):
    monkeypatch.setenv("VISUAL_RL_ASSET_ROOT", str(tmp_path / "assets"))
    path = Path(__file__).resolve().parents[1] / "configs/paper/image" / f"{recipe}.yaml"
    plan_environment(path)
    original = load_config(path)
    budget = 5 if "nft" in recipe else 6
    short = load_config(path, {
        "training": {"max_updates": budget},
        "evaluation": {"every_updates": 0, "at_updates": []},
        "runtime": {"checkpoint_every": 0, "checkpoint_at_updates": []},
    })
    get_backend(short)
    assert not short.evaluation.enabled
    assert not any(short.evaluation.due(step, budget) for step in range(budget + 1))
    assert short.evaluation.manifest == original.evaluation.manifest
    assert short.evaluation.manifest_sha256 == original.evaluation.manifest_sha256
    assert short.evaluation.sampling_steps == original.evaluation.sampling_steps
    assert short.algorithm == original.algorithm
    assert short.model == original.model
    assert short.reward == original.reward
    assert short.sampling == original.sampling
    assert replace(short.training, max_updates=original.training.max_updates) == original.training
    # Restoring the original cadence makes the original manifest active again.
    get_backend(replace(short, training=original.training, evaluation=original.evaluation))


def test_paused_manifest_does_not_allow_out_of_budget_eval(monkeypatch, tmp_path):
    monkeypatch.setenv("VISUAL_RL_ASSET_ROOT", str(tmp_path / "assets"))
    path = Path(__file__).resolve().parents[1] / "configs/paper/image/sd3_pickscore_art.yaml"
    plan_environment(path)
    with pytest.raises(ConfigError, match="explicit artifact steps"):
        load_config(path, {
            "training": {"max_updates": 6},
            "evaluation": {"every_updates": 0, "at_updates": [500]},
            "runtime": {"checkpoint_every": 0, "checkpoint_at_updates": []},
        })
