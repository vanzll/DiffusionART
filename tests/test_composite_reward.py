"""Reward composition does not change component scales or algorithm advantages."""

from copy import deepcopy

import pytest
import torch

from visual_rl.config import ConfigError, Reward
from visual_rl.rewards.composite import CompositeImageReward
from visual_rl.rewards.image_factory import load_image_reward, validate_image_reward
from visual_rl.rewards.scalar import FixedNormalization, weighted_score


class Scorer(torch.nn.Module):
    def __init__(self, name, multiplier):
        super().__init__()
        self.name, self.multiplier = name, multiplier
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.calls = []

    def forward(self, images, prompts, *, microbatch):
        assert not self.training and not torch.is_grad_enabled()
        self.calls.append((tuple(prompts), microbatch))
        values = images[:, 0, 0, 0] * self.multiplier * self.scale
        # A child may expose a normalized total: composition must ignore it.
        return weighted_score(
            {self.name: values},
            {self.name: 1.0},
            space="fixed_normalized",
            normalization={self.name: FixedNormalization(50.0, 100.0, "child-not-used")},
        )


def config(normalization="fixed"):
    def component(aesthetic=False):
        options = {
            "processor": "processor",
            "processor_revision": "processor-rev",
            "microbatch": 2,
            "local_files_only": True,
        }
        if aesthetic:
            options.update(head_path="head.pth", head_sha256="a" * 64)
        return {"checkpoint": "model", "revision": "model-rev", "options": options}

    return {
        "name": "composite",
        "normalization": normalization,
        "weights": {"pickscore": 2.0, "aesthetic": 3.0},
        "options": {
            "microbatch": 4,
            "components": {"pickscore": component(), "aesthetic": component(True)},
            "fixed_statistics": {
                "pickscore": {"mean": 1.0, "std": 26.0, "source": "test"},
                "aesthetic": {"mean": -2.0, "std": 3.0, "source": "test"},
            },
        },
    }


@pytest.mark.parametrize("space", ["raw", "fixed_normalized"])
def test_literal_weights_once_only_scale_pairing_and_microbatches(space):
    scorers = {"pickscore": Scorer("pickscore", 26), "aesthetic": Scorer("aesthetic", 3)}
    norm = {
        "pickscore": FixedNormalization(1, 26, "test"),
        "aesthetic": FixedNormalization(-2, 3, "test"),
    }
    composite = CompositeImageReward(
        scorers,
        {"pickscore": 2, "aesthetic": 3},
        space=space,
        normalization=norm,
        microbatches={"pickscore": 2, "aesthetic": 5},
    )
    values = torch.tensor([4.0, 2.0, 1.0, 3.0])
    pixels = values[:, None, None, None].expand(-1, 3, 2, 2).requires_grad_()
    prompts = ["four", "two", "one", "three"]
    torch.nn.Sequential(composite).train()
    for scorer in scorers.values():
        scorer.train()
    result = composite(pixels, prompts, microbatch=3)
    expected = (
        2 * values * 26 + 3 * values * 3
        if space == "raw"
        else (2 * (values * 26 - 1) / 26 + 3 * (values * 3 + 2) / 3)
    )
    torch.testing.assert_close(result.weighted, expected)
    torch.testing.assert_close(result.raw["pickscore"], values * 26)
    torch.testing.assert_close(result.fixed_normalized["pickscore"], (values * 26 - 1) / 26)
    assert scorers["pickscore"].calls == [(tuple(prompts), 2)]
    assert scorers["aesthetic"].calls == [(tuple(prompts), 3)]
    assert not result.weighted.requires_grad and pixels.grad is None
    assert all(not p.requires_grad for p in composite.parameters())


def test_zero_weight_component_is_still_scored_and_invalid_values_fail():
    scorers = {"a": Scorer("a", 1), "b": Scorer("b", 2)}
    composite = CompositeImageReward(scorers, {"a": 1, "b": 0}, space="raw")
    images = torch.ones(2, 3, 2, 2)
    result = composite(images, ["one", "two"], microbatch=1)
    assert set(result.raw) == {"a", "b"} and len(scorers["b"].calls) == 1
    assert result.normalization is None
    scorers["b"].multiplier = float("nan")
    with pytest.raises(ValueError, match="finite"):
        composite(images, ["one", "two"], microbatch=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires actual CUDA score transport")
@pytest.mark.parametrize("first_device", ["cpu", "cuda:0"])
def test_mixed_cpu_cuda_scores_preserve_bits_and_once_only_normalization(first_device):
    class DeviceScore(torch.nn.Module):
        def __init__(self, name, values, device):
            super().__init__()
            self.name = name
            self.values = values.to(device=device)
        def forward(self, images, prompts, **kwargs):
            return weighted_score({self.name: self.values}, {self.name: 1}, space="raw")
    a = torch.tensor([13.0, 26.0], dtype=torch.float32)
    b = torch.tensor([0.25, 0.75], dtype=torch.float32)
    other = "cuda:0" if first_device == "cpu" else "cpu"
    scorer = CompositeImageReward(
        {"a": DeviceScore("a", a, first_device), "b": DeviceScore("b", b, other)},
        {"a": 1, "b": 1}, space="fixed_normalized",
        normalization={"a": FixedNormalization(0, 26, "test"), "b": FixedNormalization(0, 1, "test")})
    result = scorer(torch.ones(2, 3, 2, 2), ["one", "two"], microbatch=2)
    assert result.weighted.device == torch.device(first_device)
    assert result.weighted.dtype == torch.float32
    assert torch.equal(result.raw["a"].cpu(), a) and torch.equal(result.raw["b"].cpu(), b)
    expected = a.to(first_device) / 26 + b.to(first_device)
    assert torch.equal(result.weighted, expected)


def test_mixed_dtypes_remain_an_error_after_device_transport():
    class DoubleScore(Scorer):
        def forward(self, images, prompts, **kwargs):
            return weighted_score({self.name: torch.ones(len(prompts), dtype=torch.float64)}, {self.name: 1}, space="raw")
    scorer = CompositeImageReward({"a": Scorer("a", 1), "b": DoubleScore("b", 1)}, {"a": 1, "b": 1}, space="raw")
    with pytest.raises(ValueError, match="dtype"):
        scorer(torch.ones(2, 3, 2, 2), ["one", "two"], microbatch=1)


def test_misnamed_or_scalar_component_is_not_silently_broadcast():
    class Invalid(torch.nn.Module):
        def __init__(self, name, value):
            super().__init__()
            self.name, self.value = name, value

        def forward(self, images, prompts, **kwargs):
            return weighted_score({self.name: self.value}, {self.name: 1}, space="raw")

    for name, value in (("other", torch.ones(2)), ("a", torch.tensor(1.0)), ("a", 1.0)):
        composite = CompositeImageReward({"a": Invalid(name, value)}, {"a": 1}, space="raw")
        with pytest.raises(ValueError):
            composite(torch.ones(2, 3, 2, 2), ["one", "two"], microbatch=1)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: c.update(checkpoint="ignored"),
        lambda c: c.update(revision="ignored"),
        lambda c: c["options"].update(microbatch=True),
        lambda c: c["options"].update(microbatch=0),
        lambda c: c["options"].update(local_files_only=False),
        lambda c: c["options"].update(components=[]),
        lambda c: c["weights"].pop("aesthetic"),
        lambda c: c["options"].pop("fixed_statistics"),
        lambda c: c["options"]["fixed_statistics"].pop("aesthetic"),
        lambda c: c["options"]["fixed_statistics"]["aesthetic"].update(std=0),
        lambda c: c["options"]["fixed_statistics"]["aesthetic"].update(source=""),
        lambda c: c["options"]["fixed_statistics"]["aesthetic"].update(mean=float("nan")),
        lambda c: c["options"]["components"]["pickscore"].update(normalization="fixed"),
        lambda c: c["options"]["components"]["pickscore"].update(revision=""),
        lambda c: c["options"]["components"]["pickscore"]["options"].update(unknown=True),
        lambda c: c["options"]["components"]["pickscore"]["options"].update(microbatch=0),
        lambda c: c["options"]["components"]["aesthetic"]["options"].pop("head_sha256"),
    ],
)
def test_invalid_composition_rejected_before_loading(mutation):
    c = config()
    mutation(c)
    with pytest.raises(ConfigError):
        validate_image_reward(Reward(**c))


def test_raw_composition_can_omit_statistics_but_never_guess_them():
    c = config("raw")
    c["options"].pop("fixed_statistics")
    validate_image_reward(Reward(**c))
    c["normalization"] = "fixed"
    with pytest.raises(ConfigError, match="fixed_statistics"):
        validate_image_reward(Reward(**c))


def test_component_loaders_receive_raw_fp32_explicit_asset_contracts(monkeypatch):
    from visual_rl.rewards.aesthetic import AestheticScore
    from visual_rl.rewards.pickscore import PickScore

    calls = {}

    def load(name, **kwargs):
        calls[name] = kwargs
        return Scorer(name, 1)

    monkeypatch.setattr(PickScore, "from_pretrained", lambda **kw: load("pickscore", **kw))
    monkeypatch.setattr(AestheticScore, "from_pretrained", lambda **kw: load("aesthetic", **kw))
    c = config()
    before = deepcopy(c)
    scorer = load_image_reward(Reward(**c), device="cpu")
    assert isinstance(scorer, CompositeImageReward) and c == before
    for call in calls.values():
        assert call["device"] == "cpu" and call["dtype"] == torch.float32
        assert call["model_path"] == "model" and call["model_revision"] == "model-rev"
        assert call["processor_revision"] == "processor-rev" and call["local_files_only"] is True
    assert calls["pickscore"]["space"] == "raw"
    assert calls["aesthetic"]["head_sha256"] == "a" * 64
