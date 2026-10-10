"""HPSv2.1 asset and scoring contracts stay explicit in composite recipes."""

from copy import deepcopy
from pathlib import Path

import pytest

from visual_rl.config import ConfigError, Reward, load_config
from visual_rl.prepare_assets import plan_environment, validate_bindings
from visual_rl.rewards.image_factory import load_image_reward, validate_image_reward


def config():
    return {
        "name": "hpsv2", "checkpoint": "/assets/HPS_v2.1_compressed.pt",
        "revision": "a" * 40, "normalization": "raw", "weights": {"hpsv2": 1},
        "options": {"microbatch": 2, "checkpoint_sha256": "b" * 64},
    }


@pytest.mark.parametrize("mutation", [
    lambda c: c.update(normalization="fixed"),
    lambda c: c.update(weights={"hpsv2": 0.25}),
    lambda c: c.update(revision="main"),
    lambda c: c.update(revision="z" * 40),
    lambda c: c["options"].update(checkpoint_sha256="bad"),
    lambda c: c["options"].update(checkpoint_sha256=1),
    lambda c: c["options"].update(microbatch=True),
    lambda c: c["options"].update(microbatch=0),
    lambda c: c["options"].update(version="v2.0"),
])
def test_invalid_hps_options_fail_before_loading(mutation):
    value = config()
    mutation(value)
    with pytest.raises(ConfigError):
        validate_image_reward(Reward(**value))


def test_factory_preserves_local_weight_pin_and_raw_cosine(monkeypatch):
    from visual_rl.rewards.hpsv2 import HPSv2Score

    value = config()
    before = deepcopy(value)
    monkeypatch.setattr(HPSv2Score, "from_pretrained", lambda **kwargs: kwargs)
    assert load_image_reward(Reward(**value), device="cpu") == {
        "model_path": value["checkpoint"], "model_revision": value["revision"],
        "checkpoint_sha256": value["options"]["checkpoint_sha256"], "device": "cpu",
    }
    assert value == before


def test_multireward_plans_only_needed_pinned_snapshots(tmp_path, monkeypatch):
    for name in ("SD3_MODEL", "PICKSCORE_MODEL", "PICKSCORE_PROCESSOR", "CLIP_MODEL", "HPS_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VISUAL_RL_ASSET_ROOT", str(tmp_path))
    monkeypatch.setenv("GENEVAL_MANIFEST", "/assets/geneval.json")
    monkeypatch.setenv("GENEVAL_MANIFEST_SHA256", "c" * 64)
    monkeypatch.setenv("GENEVAL_PYTHON", "/assets/geneval/bin/python")
    path = Path("configs/examples/sd3_multireward.yaml")
    plan = plan_environment(path)
    candidate = load_config(path)
    validate_image_reward(candidate.reward)
    validate_bindings(candidate, plan)
    assert len(plan[0]) == 5 and not plan[1]
    hps = next(entry for entry, _ in plan[0] if entry["env"] == "HPS_MODEL")
    assert hps["allow_patterns"] == ["HPS_v2.1_compressed.pt"]
    assert candidate.reward.weights == dict.fromkeys(("geneval", "pickscore", "clipscore", "hpsv2"), 1)
    assert candidate.reward.options["fixed_statistics"]["pickscore"]["std"] == 26
    assert candidate.reward.options["fixed_statistics"]["clipscore"]["std"] == 30
    candidate.reward.options["components"]["hpsv2"]["revision"] = "d" * 40
    with pytest.raises(ConfigError, match="automatic asset pin"):
        validate_bindings(candidate, plan)
