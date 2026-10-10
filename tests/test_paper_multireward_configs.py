"""Offline evidence and executed-branch checks, without scorer assets or GPUs."""

import hashlib
import json
import math
import os
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from visual_rl.backends import sd3
from visual_rl.config import load_config
from visual_rl.registry import get_backend
from visual_rl.rewards.image_factory import validate_reward_prompts
from visual_rl.rewards.scalar import FixedNormalization, weighted_score
from visual_rl.training.grpo import prepare_grpo_update

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "configs/paper/image"
EVIDENCE = ROOT / "docs/paper-multireward-evidence.json"
RUNS = {
    "art": "3t35bwps",
    "grpo": "nv2hv2l1",
    "dance": "gpxub5g5",
    "fast": "ye14mp5t",
    "nft_ema": "ujnglsgl",
    "nft_onpolicy": "ldpbaea3",
    "dppo": "dvmnd177",
}
DIVISORS = {"geneval": 1, "pickscore": 26, "clipscore": 30, "hpsv2": 1}
JOINT = {"art", "grpo", "dance", "fast", "dppo"}


@pytest.fixture(autouse=True)
def asset_environment(monkeypatch):
    for directory in (PROFILES, ROOT / "configs/examples"):
        for path in directory.glob("*.yaml"):
            for name in re.findall(r"\$\{([A-Z_0-9]+)\}", path.read_text()):
                value = "a" * 64 if name.endswith(("SHA256", "SHA")) else f"/public-assets/{name.lower()}"
                monkeypatch.setenv(name, value)


@pytest.fixture(scope="module")
def evidence():
    return json.loads(EVIDENCE.read_text())


def profile(method):
    return load_config(PROFILES / f"sd3_multireward_{method}.yaml")


def test_seven_candidate_coverage_and_fragment_is_not_a_profile(evidence):
    expected = {f"sd3_multireward_{method}.yaml" for method in RUNS}
    assert {p.name for p in PROFILES.glob("sd3_multireward_*.yaml")} == (
        expected | {"sd3_multireward_rewards.yaml"}
    )
    fragment = yaml.safe_load((PROFILES / "sd3_multireward_rewards.yaml").read_text())
    assert set(fragment) == {"reward", "assets"}
    assert set(evidence["runs"]) == set(RUNS.values())
    assert set(evidence["coverage"]["emitted_methods"]) == set(RUNS)
    assert evidence["coverage"]["certified_historical_baselines"] == []


@pytest.mark.parametrize("method", RUNS)
def test_backend_assets_full_data_and_no_inherited_art_options(method, evidence):
    c = profile(method)
    assert get_backend(c).name == "sd3"
    assert c.runtime.distributed == "ddp" and c.runtime.log == "wandb"
    assert c.data.manifest == "/public-assets/geneval_train_manifest"
    assert c.data.sha256 == "a" * 64 and c.data.shuffle is True
    assert c.evaluation.manifest == "/public-assets/geneval_eval_manifest"
    assert c.evaluation.manifest_sha256 == "a" * 64
    assert c.evaluation.enabled and c.evaluation.every_updates == 0
    assert c.runtime.checkpoint_every == 200
    assert all(c.runtime.checkpoint_due(step, 1000) for step in (200, 400, 600, 800, 1000))
    assert c.evaluation.due(1000, 1000) and c.runtime.checkpoint_due(1000, 1000)
    assert c.reward.name == "composite" and c.reward.normalization == "fixed"
    assert c.reward.weights == {key: 1 for key in DIVISORS}
    stats = c.reward.options["fixed_statistics"]
    assert {key: item["std"] for key, item in stats.items()} == DIVISORS
    assert all(item["mean"] == 0 for item in stats.values())
    assert len(c.assets["snapshots"]) == 5
    hps = c.reward.options["components"]["hpsv2"]
    assert hps["checkpoint"] == "/public-assets/hps_model/HPS_v2.1_compressed.pt"
    assert hps["revision"] == "697403c78157020a1ae59d23f111aa58ced35b0a"
    assert hps["options"]["checkpoint_sha256"] == (
        "c57a38fb4a2f7e7c15bf00da2ea377cdf165448b4dd1052a484c215a998c9837"
    )
    assert c.provenance["source_run"] == RUNS[method]
    assert c.provenance["runtime_commit"] == evidence["runs"][RUNS[method]]["runtime_commit"]
    assert c.provenance["main_paper_plotted_steps"] == 500
    assert c.training.max_updates == 1000
    assert c.provenance["semantic_gaps"]
    if method != "art":
        assert "train_timesteps" not in c.sampling.options
        assert not {"covariance", "restore_zero_signal"} & set(c.algorithm.options)
        assert c.provenance["runtime_source_status"] == "commit_recorded_source_unrecovered"
    if method in JOINT:
        joint = c.algorithm.options["multi_reward_advantage"]
        assert joint == {
            "weights": {key: 0.25 for key in DIVISORS},
            "score_space": "fixed_normalized",
            "eps": 0.0001,
            "renormalize_combined": False,
        }
    else:
        assert "multi_reward_advantage" not in c.algorithm.options
        assert "metadata" not in c.provenance["candidate_advantage_branch"]
        assert "deliberately omits" in " ".join(c.provenance["semantic_gaps"])


@pytest.mark.parametrize("method", RUNS)
def test_scientific_settings_against_separate_api_projection(method, evidence):
    c = profile(method)
    source = evidence["runs"][RUNS[method]]["config"]
    sample, train = source["sample"], source["train"]
    assert c.seed == source["seed"]
    assert c.training.precision == source["mixed_precision"]
    assert c.training.allow_tf32 == source["allow_tf32"]
    assert c.model.options["gradient_checkpointing"] == source["activation_checkpointing"]
    assert c.training.learning_rate == train["learning_rate"]
    assert c.training.microbatch == train["batch_size"]
    assert c.training.betas == (train["adam_beta1"], train["adam_beta2"])
    assert c.training.weight_decay == train["adam_weight_decay"]
    assert c.training.epsilon == train["adam_epsilon"]
    assert c.training.grad_clip == train["max_grad_norm"]
    assert c.sampling.steps == sample["num_steps"]
    assert c.sampling.samples_per_prompt == sample["num_image_per_prompt"]
    assert c.sampling.prompts_per_rollout * c.sampling.samples_per_prompt == (
        sample["train_batch_size"] * sample["num_batches_per_epoch"] * 8
    )
    assert c.sampling.microbatch == sample["train_batch_size"]
    assert c.sampling.guidance_scale == sample["guidance_scale"]
    assert c.sampling.height == c.sampling.width == source["resolution"]
    assert c.sampling.options["shift"] == train["timestep_shift"]
    assert c.evaluation.sampling_steps == sample["eval_num_steps"]
    assert c.reward.weights == source["reward_fn"]
    assert train["multi_reward_advantage"]["enabled"] is True
    assert train["multi_reward_advantage"]["renormalize_combined"] is False
    updates = sample["num_batches_per_epoch"] // train["gradient_accumulation_steps"]
    assert c.algorithm.options["updates_per_rollout"] == updates
    assert c.provenance["source_gradient_accumulation_steps"] == train["gradient_accumulation_steps"]
    assert c.provenance["source_resume_global_step"] == source["resume_global_step"]
    if method == "art":
        assert c.evaluation.at_updates == tuple(source["paper_eval_wandb_steps"][1:])
        assert c.runtime.checkpoint_at_updates == tuple(source["paper_save_wandb_steps"])
        assert c.sampling.options["train_timesteps"] == list(range(10))
        assert train["covariance_guidance_reward_source"] == "combined_multi_reward_advantage"
    else:
        interval = source["eval_freq"] * updates
        assert c.evaluation.at_updates == tuple(range(interval, 1001, interval))
        assert c.runtime.checkpoint_at_updates == tuple(range(100, 1001, 100))
        count = int(sample["num_steps"] * train["timestep_fraction"])
        if method.startswith("nft"):
            assert c.sampling.options["solver"] == sample["solver"]
            assert c.sampling.options["train_timestep_count"] == count
            assert c.algorithm.options["loss"]["nft_beta"] == source["beta"]
            assert c.algorithm.options["old_policy"] == {
                "rate": 0.001 if source["decay_type"] else 0,
                "maximum": 0.5 if source["decay_type"] else 0,
                "delay": 0,
            }
        else:
            assert c.sampling.options["noise_level"] == sample["noise_level"]
            assert c.algorithm.options["loss"]["advantage_clip"] == train["adv_clip_max"]
            if method == "fast":
                assert c.sampling.options["pool"] == list(range(*sample["sde_window_range"]))
                assert c.sampling.options["count"] == sample["sde_window_size"]
                assert c.sampling.options["likelihood"] == "mse"
            else:
                assert c.sampling.options["count"] == count


def score_fixture(c):
    raw = {
        "geneval": torch.tensor([-1, 0, 0.25, 0.5, 0, 1, 0.75, 0.5], dtype=torch.float64),
        "pickscore": torch.tensor([0, 26, 1e-5, 52, 2e-5, 78, 3e-5, 104], dtype=torch.float64),
        "clipscore": torch.tensor([90, 1, 60, 2, 30, 4, 0, 8], dtype=torch.float64),
        "hpsv2": torch.tensor([0.2, 0.8, 0.6, 0.4, 0.4, 0.6, 0.8, 0.2], dtype=torch.float64),
    }
    return weighted_score(
        raw, c.reward.weights, space="fixed_normalized",
        normalization={key: FixedNormalization(**spec)
                       for key, spec in c.reward.options["fixed_statistics"].items()},
    )


def group_oracle(values, prompts, *, global_std=False):
    rows = values.tolist()
    overall_mean = sum(rows) / len(rows)
    overall_std = math.sqrt(sum((v - overall_mean) ** 2 for v in rows) / len(rows))
    result = []
    for value, prompt in zip(rows, prompts, strict=True):
        group = [v for v, p in zip(rows, prompts, strict=True) if p == prompt]
        mean = sum(group) / len(group)
        std = overall_std if global_std else math.sqrt(
            sum((v - mean) ** 2 for v in group) / len(group)
        )
        result.append((value - mean) / (std + 0.0001))
    return torch.tensor(result, dtype=torch.float64)


@pytest.mark.parametrize("method", RUNS)
def test_reporting_is_sum_not_quarter_mean(method):
    scores = score_fixture(profile(method))
    expected = sum(scores.raw[key] / divisor for key, divisor in DIVISORS.items())
    torch.testing.assert_close(scores.weighted, expected, rtol=1e-14, atol=1e-14)
    assert not torch.allclose(scores.weighted, expected * 0.25)


@pytest.mark.parametrize("method", sorted(JOINT - {"art"}))
def test_joint_policy_normalizes_once_in_fixed_units_and_ignores_scalar_global_std(method, monkeypatch):
    c = profile(method)
    scores = score_fixture(c)
    prompts = ["a", "b"] * 4
    expected = sum(group_oracle(scores.raw[key] / scale, prompts) * 0.25
                   for key, scale in DIVISORS.items())

    def unexpected(*args, **kwargs):
        pytest.fail("Joint branch must not renormalize the combined scalar")

    monkeypatch.setattr(sd3, "prompt_advantages", unexpected)
    monkeypatch.setattr(sd3, "dance_advantages", unexpected)
    actual = sd3._policy_advantages(c, scores, prompts)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    assert not torch.allclose(actual, group_oracle(scores.weighted, prompts, global_std=True))
    assert not torch.allclose(actual, group_oracle(expected, prompts, global_std=True))
    raw_joint = sum(group_oracle(scores.raw[key], prompts) * 0.25 for key in DIVISORS)
    assert not torch.allclose(actual, raw_joint)


@pytest.mark.parametrize("method", ["nft_ema", "nft_onpolicy"])
def test_nft_uses_canonical_scalar_total_not_metadata_joint_flag(method, monkeypatch):
    c = profile(method)
    scores = score_fixture(c)
    prompts = ["a", "b"] * 4

    def unexpected(*args, **kwargs):
        pytest.fail("Canonical NFT ignores the archived multi-reward metadata flag")

    monkeypatch.setattr(sd3, "_art_guidance_scores", unexpected)
    actual = sd3._policy_advantages(c, scores, prompts)
    expected = group_oracle(scores.weighted, prompts, global_std=True)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual.double(), expected, rtol=1e-7, atol=1e-7)
    joint = sum(group_oracle(scores.raw[key] / scale, prompts) * 0.25
                for key, scale in DIVISORS.items())
    assert not torch.allclose(actual.double(), joint)


def test_art_covariance_receives_joint_advantage_and_two_complete_disjoint_windows(monkeypatch):
    c = profile("art")
    scores = score_fixture(c)
    prompts = ["a", "b"] * 4
    expected = sum(group_oracle(scores.raw[key] / scale, prompts) * 0.25
                   for key, scale in DIVISORS.items())
    guidance, _ = sd3._art_guidance_scores(c, SimpleNamespace(scores=scores), prompts)
    torch.testing.assert_close(guidance, expected, rtol=1e-12, atol=1e-12)
    assert not torch.allclose(guidance, scores.weighted)
    original = sd3.art_scalar_covariance

    def capture(endpoints, rewards, *args, **kwargs):
        torch.testing.assert_close(rewards, expected, rtol=1e-12, atol=1e-12)
        return original(endpoints, rewards, *args, **kwargs)

    monkeypatch.setattr(sd3, "art_scalar_covariance", capture)
    groups = torch.tensor([0, 1] * 4)
    _, windows = sd3._art_update(
        c, SimpleNamespace(endpoints=torch.arange(24.0).reshape(8, 3)), None,
        SimpleNamespace(scores=scores), groups, 2, 4, tuple(range(10)),
        torch.Generator().manual_seed(42), torch.device("cpu"), guidance_scores=guidance,
    )
    assert len(windows) == 2 and [len(w) for w in windows] == [40, 40]
    assert {row for row, _ in windows[0]}.isdisjoint({row for row, _ in windows[1]})
    assert set(windows[0]) | set(windows[1]) == {(row, slot) for row in range(8) for slot in range(10)}
    assert c.algorithm.options["restore_zero_signal"] is False
    assert c.provenance["approved_public_scalar_semantics"]["denominator"] == "original_full_update_window"


@pytest.mark.parametrize("method", sorted(set(RUNS) - {"art"}))
def test_baseline_clamp_is_after_advantage_construction(method):
    from visual_rl.algorithms import nft_loss

    c = profile(method)
    signal = torch.zeros(64, dtype=torch.float64)
    signal[-1] = 1
    scores = weighted_score(
        {key: signal * scale for key, scale in DIVISORS.items()}, c.reward.weights,
        space="fixed_normalized",
        normalization={key: FixedNormalization(**spec)
                       for key, spec in c.reward.options["fixed_statistics"].items()},
    )
    prompts = ["one prompt"] * 64
    advantages = sd3._policy_advantages(c, scores, prompts)
    assert advantages[-1] > 5
    if method.startswith("nft"):
        terms = nft_loss(
            torch.zeros(64, 1), torch.zeros(64, 1), torch.zeros(64, 1),
            torch.ones(64, 1), torch.full((64,), 0.5), advantages,
            **c.algorithm.options["loss"],
        )
        expected = (advantages.clamp(-5, 5) / 5) / 2 + 0.5
        torch.testing.assert_close(terms["positive_probability"], expected.float())
    else:
        objective, _, _ = prepare_grpo_update(
            c, SimpleNamespace(selected=(0,)), None, scores.weighted, prompts,
            torch.Generator().manual_seed(42), torch.device("cpu"), advantages=advantages,
        )
        torch.testing.assert_close(objective.advantages, advantages.clamp(-5, 5))
        assert objective.advantages[-1] == 5


def test_full_prompt_metadata_required_without_truncation():
    c = profile("art")
    prompt = "a red cat " + "beside a blue bicycle " * 100
    metadata = {"prompt": prompt, "tag": "two_object", "include": [
        {"class": "cat", "count": 1, "color": "red"},
        {"class": "bicycle", "count": 1, "color": "blue"},
    ], "exclude": []}
    validate_reward_prompts(c.reward, [prompt], metadata=[metadata])
    with pytest.raises(ValueError, match="GenEval"):
        validate_reward_prompts(c.reward, [prompt], metadata=[{**metadata, "prompt": prompt[:128]}])


def test_public_evidence_distinguishes_read_source_from_runtime_certification(evidence):
    assert not re.search(r"/m2v_|/Users/|WANDB_API_KEY|api_key|password", EVIDENCE.read_text())
    nft_sources = [s for s in evidence["sources"] if s["path"] == "scripts/train_nft_sd3.py"]
    assert len(nft_sources) == 2
    assert all(s["multi_reward_advantage_symbol_occurrences"] == 0 for s in nft_sources)
    assert all(re.fullmatch(r"[0-9a-f]{64}", s["sha256"]) for s in evidence["sources"])
    for method, run in RUNS.items():
        source = evidence["runs"][run]
        assert source["config_status"] == "bounded_wandb_api_config_verified"
        if method != "art":
            assert source["archived_advantage_execution"] == "unknown_actual_runtime_source"
            assert source["archived_multi_flag_is_execution_proof"] is False
    assert evidence["runs"]["gpxub5g5"]["config"]["resume_global_step"] == 100
    assert evidence["runs"]["dvmnd177"]["config"]["resume_global_step"] == 600


def test_optional_reference_git_hashes_and_excerpts(evidence):
    repository = os.environ.get("PAPER_MULTIREWARD_REFERENCE_REPO")
    if not repository:
        pytest.skip("Optional source-repository cross-check; bundled semantic checks always run")
    for source in evidence["sources"]:
        raw = subprocess.check_output(
            ["git", "-C", repository, "show", f"{source['commit']}:{source['path']}"], timeout=10,
        )
        assert hashlib.sha256(raw).hexdigest() == source["sha256"]
        lines = raw.decode().splitlines()
        for excerpt in source["excerpts"]:
            start = excerpt["start_line"] - 1
            expected = excerpt["text"].splitlines()
            assert lines[start:start + len(expected)] == expected
