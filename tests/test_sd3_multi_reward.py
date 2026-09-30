"""ART-specific joint guidance keeps scoring and update budgets independent."""

from types import SimpleNamespace

import pytest
import torch
import yaml
from test_sd3_runner import config_file

from visual_rl.algorithms import art_scalar_covariance
from visual_rl.backends import sd3
from visual_rl.config import ConfigError, load_config
from visual_rl.registry import get_backend
from visual_rl.rewards.scalar import FixedNormalization, weighted_score


def options(space="raw"):
    return {
        "weights": {"pickscore": 2.0, "aesthetic": 1.0},
        "score_space": space,
        "eps": 1e-4,
        "renormalize_combined": False,
    }


def result():
    return SimpleNamespace(
        scores=weighted_score(
            {
                "pickscore": torch.tensor([0.0, 26.0, 52.0, 78.0]),
                "aesthetic": torch.tensor([100.0, 10.0, 20.0, 500.0]),
            },
            {"pickscore": 5.0, "aesthetic": 4.0},
            space="raw",
            normalization={
                "pickscore": FixedNormalization(0, 26, "test"),
                "aesthetic": FixedNormalization(0, 1, "identity"),
            },
        )
    )


def test_default_path_returns_original_scalar_without_new_statistics():
    scored = result()
    c = SimpleNamespace(algorithm=SimpleNamespace(options={}))
    value, metrics = sd3._art_guidance_scores(c, scored, ["a", "a", "b", "b"])
    assert value is scored.scores.weighted and metrics == {}


@pytest.mark.parametrize("space", ["raw", "fixed_normalized"])
def test_joint_scores_not_scalar_total_reach_covariance_without_clipping(space, monkeypatch):
    c = SimpleNamespace(
        model=SimpleNamespace(name="sd3"),
        algorithm=SimpleNamespace(
            options={
                "multi_reward_advantage": options(space),
                "covariance": {},
                "restore_zero_signal": False,
                "loss": {},
            }
        )
    )
    scored = result()
    prompts = ["same", "different", "same", "different"]
    combined, metrics = sd3._art_guidance_scores(c, scored, prompts)
    expected = torch.zeros(4, dtype=torch.float64)
    values = getattr(scored.scores, space)
    for key, weight in options()["weights"].items():
        raw = values[key].double()
        for rows in ([0, 2], [1, 3]):
            r = raw[rows]
            expected[rows] += (weight / 3) * (r - r.mean()) / (r.std(unbiased=False) + 1e-4)
    torch.testing.assert_close(combined, expected, rtol=1e-12, atol=1e-12)
    assert metrics["train/multi_reward/combined_std"] >= 0
    assert not torch.allclose(combined.float(), scored.scores.weighted)
    captured = []

    def covariance(x, rewards, groups, count, **kwargs):
        captured.append(rewards.clone())
        return art_scalar_covariance(x, rewards, groups, count, **kwargs)

    monkeypatch.setattr(sd3, "art_scalar_covariance", covariance)
    cache = SimpleNamespace(endpoints=torch.arange(12.0).reshape(4, 3))
    objective, windows = sd3._art_update(
        c,
        cache,
        None,
        scored,
        torch.tensor([0, 0, 1, 1]),
        2,
        2,
        (0, 1),
        torch.Generator().manual_seed(4),
        torch.device("cpu"),
        guidance_scores=combined,
    )
    torch.testing.assert_close(captured[0], expected, rtol=1e-12, atol=1e-12)
    assert len(windows) == 2 and [len(window) for window in windows] == [4, 4]
    first_rows = {row for row, _ in windows[0]}
    second_rows = {row for row, _ in windows[1]}
    assert first_rows.isdisjoint(second_rows) and first_rows | second_rows == {0, 1, 2, 3}
    assert objective.cache is cache


@pytest.mark.parametrize(
    "bad",
    [
        None,
        {},
        {"unexpected": True},
        {"weights": {}},
        {"weights": {"pickscore": 0}},
        {"weights": {"pickscore": True}},
        {"weights": {"unknown": 1}},
        {"weights": {"pickscore": float("inf")}},
        {"eps": 0},
        {"eps": True},
        {"eps": float("nan")},
        {"score_space": "normalized"},
        {"renormalize_combined": 1},
    ],
)
def test_joint_bad_config_fails_before_assets(tmp_path, bad):
    path = config_file(tmp_path)
    data = yaml.safe_load(path.read_text())
    valid = options()
    valid["weights"] = {"pickscore": 1}
    data["algorithm"]["options"]["multi_reward_advantage"] = (
        None if bad is None else ({} if bad == {} else {**valid, **bad})
    )
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError):
        get_backend(load_config(path))
    assert not (tmp_path / "run").exists()


def test_fixed_advantages_require_available_component_statistics(tmp_path):
    from test_composite_reward import config as reward_config

    path = config_file(tmp_path)
    data = yaml.safe_load(path.read_text())
    data["reward"] = reward_config("raw")
    data["reward"]["options"].pop("fixed_statistics")
    data["algorithm"]["options"]["multi_reward_advantage"] = options("fixed_normalized")
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError, match="fixed statistics"):
        get_backend(load_config(path))
    data["algorithm"]["options"]["multi_reward_advantage"]["score_space"] = "raw"
    path.write_text(yaml.safe_dump(data))
    assert get_backend(load_config(path)).name == "sd3"


def test_reward_space_is_not_erased_by_group_normalization():
    scored = result()
    # This tiny signal distinguishes the two eps-in-score-units denominators.
    raw = dict(scored.scores.raw)
    raw["pickscore"] = torch.tensor([0.0, 1e-5, 2e-5, 3e-5])
    scored.scores = weighted_score(
        raw, scored.scores.weights, space="raw", normalization=scored.scores.normalization
    )
    outputs = []
    for space in ("raw", "fixed_normalized"):
        joint = options(space)
        joint["weights"] = {"pickscore": 1.0}
        c = SimpleNamespace(algorithm=SimpleNamespace(options={"multi_reward_advantage": joint}))
        values, _ = sd3._art_guidance_scores(c, scored, ["a", "a", "b", "b"])
        outputs.append(values)
    assert outputs[0].abs().mean() > outputs[1].abs().mean() * 20


def test_large_joint_advantage_is_not_clipped_before_covariance(monkeypatch):
    raw = torch.zeros(64)
    raw[-1] = 1
    scored = SimpleNamespace(
        scores=weighted_score({"pickscore": raw}, {"pickscore": 1}, space="raw")
    )
    joint = options()
    joint["weights"] = {"pickscore": 1}
    c = SimpleNamespace(
        model=SimpleNamespace(name="sd3"),
        algorithm=SimpleNamespace(
            options={
                "multi_reward_advantage": joint,
                "covariance": {},
                "loss": {},
                "restore_zero_signal": False,
            }
        )
    )
    combined, _ = sd3._art_guidance_scores(c, scored, ["one prompt"] * 64)
    expected = (raw.double() - raw.double().mean()) / (raw.double().std(unbiased=False) + 1e-4)
    assert expected.max() > 5
    torch.testing.assert_close(combined, expected, rtol=1e-12, atol=1e-12)
    original = sd3.art_scalar_covariance

    def check(x, reward, groups, count, **kw):
        torch.testing.assert_close(reward, expected, rtol=1e-12, atol=1e-12)
        assert reward.max() > 5
        return original(x, reward, groups, count, **kw)

    monkeypatch.setattr(sd3, "art_scalar_covariance", check)
    sd3._art_update(
        c,
        SimpleNamespace(endpoints=torch.arange(64.0)[:, None]),
        None,
        scored,
        torch.zeros(64, dtype=torch.long),
        1,
        64,
        (0,),
        torch.Generator().manual_seed(4),
        torch.device("cpu"),
        guidance_scores=combined,
    )
