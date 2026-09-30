from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_hunyuan_runner import config_file

from visual_rl.algorithms import art_pairwise_covariance
from visual_rl.backends import hunyuan
from visual_rl.config import ConfigError, load_config
from visual_rl.rewards.scalar import FixedNormalization, weighted_score
from visual_rl.training import video_pairwise
from visual_rl.training.rollout import collect_rollout


def scores(values):
    values = torch.as_tensor(values, dtype=torch.float32)
    raw = {"VQ": values, "MQ": values * .4, "TA": values * .2}
    return weighted_score(raw, {"VQ": 1., "MQ": 0., "TA": 0.}, space="fixed_normalized",
                          normalization={key: FixedNormalization(0., 1., "test") for key in raw})


@pytest.mark.parametrize("k", [2, 4, 8])
def test_full_group_binary_covariance_and_no_scalar_magnitudes(tmp_path, k):
    c = load_config(config_file(tmp_path, "pairwise"))
    c = replace(c, sampling=replace(c.sampling, samples_per_prompt=k))
    endpoints = torch.randn(2 * k, 3, generator=torch.Generator().manual_seed(9))
    cache = SimpleNamespace(endpoints=endpoints)
    before = torch.get_rng_state().clone()
    a = video_pairwise.prepare(c, cache, scores(torch.arange(2 * k)), attempt=7)
    b = video_pairwise.prepare(c, cache, scores(torch.arange(2 * k) * 100 + 20), attempt=7)
    assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(a["group_directions"], b["group_directions"])
    assert torch.equal(a["binary_rewards"], b["binary_rewards"])
    assert set(a["binary_rewards"].tolist()) == {0., 1.}
    for group in range(2):
        selection = slice(group * k, (group + 1) * k)
        labels = a["binary_rewards"][selection]
        expected = ((labels - .5)[:, None] * endpoints[selection]).sum(0) / (k - 1)
        torch.testing.assert_close(a["group_covariance"][group], expected)
        shard = video_pairwise.prepare(c, SimpleNamespace(endpoints=endpoints[selection]),
                                       scores(torch.arange(2 * k)[selection]), attempt=7, group_offset=group)
        assert torch.equal(shard["binary_rewards"], labels)
        assert torch.equal(shard["group_directions"][0], a["group_directions"][group])


def test_ties_have_restoration_only_and_global_activity_keeps_original_windows(tmp_path):
    c = load_config(config_file(tmp_path, "pairwise"))
    parameter = torch.tensor(.1, requires_grad=True)

    def predict(x, t, rows):
        return x * parameter

    cache = collect_rollout(predict, torch.arange(1., 9.).reshape(4, 2),
                            torch.tensor([1., .5, 0.]), selected=[0, 1])
    feedback = video_pairwise.prepare(c, cache, scores([2., 2., 0., 1.]), attempt=1)
    objective, windows, metrics = video_pairwise.finalize(c, cache, predict, feedback,
        video_pairwise.activity(feedback, "cpu"), torch.device("cpu"))
    assert objective.row_anchors.tolist() == [True, True, False, False]
    assert objective.row_magnitudes[:2].tolist() == [0., 0.]
    assert windows == (((0, 0), (0, 1), (1, 0), (1, 1)), ((2, 0), (2, 1), (3, 0), (3, 1)))
    assert metrics["train/pairwise/row_restore_anchor_fraction"] == .5
    assert torch.allclose(objective(windows[0]), torch.zeros(4), atol=1e-12)
    with torch.no_grad():
        parameter.add_(.1)
    assert (objective(windows[0]) > 0).all()

    local = video_pairwise.prepare(c, cache, scores([1., 1., 1., 1.]), attempt=1)
    assert video_pairwise.finalize(c, cache, predict, local, video_pairwise.activity(local, "cpu"), "cpu")[0] is None
    remote_active = torch.tensor([1, 1, 1])
    anchored, windows, _ = video_pairwise.finalize(c, cache, predict, local, remote_active, "cpu")
    assert anchored.row_anchors.all() and not anchored.row_magnitudes.any()
    assert len(windows[0]) == len(windows[1]) == 4


def test_nonzero_preferences_with_cancelled_covariance_do_not_spend_update(tmp_path):
    c = load_config(config_file(tmp_path, "pairwise"))
    cache = SimpleNamespace(endpoints=torch.ones(2, 3))
    feedback = video_pairwise.prepare(c, cache, scores([0., 1.]), attempt=2)
    flags = video_pairwise.activity(feedback, "cpu")
    assert flags.tolist() == [1, 0, 0]
    objective, windows, metrics = video_pairwise.finalize(c, cache, None, feedback, flags, "cpu")
    assert objective is None and windows == () and metrics["train/pairwise/inactive_rollout"] == 1


@pytest.mark.parametrize("change", ["weights", "normalization", "k", "anchor", "seed", "covariance", "model"])
def test_incompatible_pairwise_settings_rejected_before_assets(tmp_path, change):
    c = load_config(config_file(tmp_path, "pairwise"))
    if change == "weights":
        c = replace(c, reward=replace(c.reward, weights={"VQ": .5, "MQ": .5, "TA": 0.}))
    elif change == "normalization":
        c = replace(c, reward=replace(c.reward, normalization="raw"))
    elif change == "k":
        c = replace(c, sampling=replace(c.sampling, samples_per_prompt=6))
    elif change == "model":
        c = replace(c, task="i2v")
    else:
        options = dict(c.algorithm.options)
        if change == "anchor":
            options["restore_zero_signal"] = False
        elif change == "seed":
            options["pairwise"] = {"seed": True}
        else:
            options["covariance"] = {"guidance_gain": 2.}
        c = replace(c, algorithm=replace(c.algorithm, options=options))
    with pytest.raises(ConfigError):
        video_pairwise.validate_options(c)


def test_raw_reward_cannot_silently_replace_fixed_normalized_feedback(tmp_path):
    c = load_config(config_file(tmp_path, "pairwise"))
    raw = weighted_score({"VQ": torch.tensor([0., 1.])}, {"VQ": 1.}, space="raw")
    with pytest.raises(ValueError, match="fixed-normalized"):
        video_pairwise.prepare(c, SimpleNamespace(endpoints=torch.randn(2, 3)), raw, attempt=1)


def test_pairwise_examples_change_only_k_and_identity(monkeypatch):
    for name in ("HUNYUAN_MODEL", "HUNYUAN_REVISION", "DANCEGRPO_ROOT", "VIDEO_REWARD_MODEL",
                 "VIDEO_REWARD_REVISION", "VIDEO_REWARD_BACKBONE", "HUNYUAN_TRAIN_MANIFEST", "HUNYUAN_EVAL_MANIFEST"):
        monkeypatch.setenv(name, "configured-asset")
    for name in ("HUNYUAN_TRAIN_SHA256", "HUNYUAN_EVAL_SHA256"):
        monkeypatch.setenv(name, "a" * 64)
    baseline = None
    for k in (2, 4, 8):
        c = load_config(Path(__file__).resolve().parents[1] / f"configs/examples/hunyuan_pairwise_k{k}.yaml")
        hunyuan.validate_config(c)
        assert c.sampling.samples_per_prompt == k
        config = c.to_dict()
        config.pop("name")
        config["sampling"].pop("samples_per_prompt")
        config["runtime"].pop("output_dir")
        if baseline is None:
            baseline = config
        else:
            assert baseline == config


def test_imported_kernel_still_uses_exact_ties():
    result = art_pairwise_covariance(torch.arange(4.).reshape(2, 2), torch.tensor([1., 1.]), 2)
    assert result["binary_rewards"].tolist() == [.5, .5]
