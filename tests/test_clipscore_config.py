"""CLIPScore assets and interpolation stay explicit at config boundaries."""

import importlib.util
from copy import deepcopy

import pytest
import torch

from visual_rl.config import ConfigError, Reward, load_config
from visual_rl.registry import get_backend
from visual_rl.rewards.image_factory import load_image_reward, validate_image_reward


def config():
    return {
        "name": "clipscore",
        "checkpoint": "model",
        "revision": "model-revision",
        "weights": {"clipscore": 1.0},
        "normalization": "fixed",
        "options": {
            "processor": "processor",
            "processor_revision": "processor-revision",
            "microbatch": 2,
            "antialias": True,
        },
    }


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])
def test_antialias_is_a_boolean_not_an_implicit_library_default(value):
    c = config()
    c["options"]["antialias"] = value
    with pytest.raises(ConfigError, match="antialias"):
        validate_image_reward(Reward(**c))


def test_missing_antialias_fails_preflight():
    c = config()
    del c["options"]["antialias"]
    with pytest.raises(ConfigError, match="reward.options"):
        validate_image_reward(Reward(**c))


def test_missing_dependency_explains_required_extra(monkeypatch):
    original = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: None if name == "torchvision" else original(name)
    )
    with pytest.raises(ConfigError, match="image-rewards"):
        validate_image_reward(Reward(**config()))


@pytest.mark.parametrize("space", ["raw", "fixed"])
@pytest.mark.parametrize("antialias", [True, False])
def test_factory_keeps_fp32_assets_and_requested_scale(monkeypatch, space, antialias):
    pytest.importorskip("torchvision")
    from visual_rl.rewards.clipscore import CLIPScore

    c = config()
    c["normalization"] = space
    c["options"]["antialias"] = antialias
    before = deepcopy(c)
    monkeypatch.setattr(CLIPScore, "from_pretrained", lambda **kw: kw)
    options = load_image_reward(Reward(**c), device="cpu")
    assert c == before
    assert options == {
        "model_path": "model",
        "model_revision": "model-revision",
        "processor_path": "processor",
        "processor_revision": "processor-revision",
        "device": "cpu",
        "dtype": torch.float32,
        "local_files_only": True,
        "antialias": antialias,
        "space": "raw" if space == "raw" else "fixed_normalized",
    }


def test_readme_example_preflight_keeps_only_clipscore():
    pytest.importorskip("torchvision")
    c = load_config("configs/examples/sd3_clipscore.yaml")
    assert c.reward.weights == {"clipscore": 1.0}
    assert c.reward.options["antialias"] is True
    assert c.reward.normalization == "fixed"
    assert get_backend(c).name == "sd3"
