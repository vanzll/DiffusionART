"""Paper science contracts, frozen independently of the example/YAML defaults.

Constants were checked with git show against the runtime sources named below.
Tests do not require the private research checkout, pretrained weights or CUDA.
"""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from visual_rl.backends.hunyuan import updates_per_rollout
from visual_rl.config import ConfigError, load_config
from visual_rl.registry import get_backend
from visual_rl.rewards.scalar import FixedNormalization, weighted_score
from visual_rl.rewards.video_reward import resized_shape, uniform_indices
from visual_rl.sampling import shifted_schedule
from visual_rl.training import video_pairwise
from visual_rl.training.video_nft import VideoNFTState

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs/paper/video"
# Expected coverage is fixed rather than discovered by glob (deletions must fail).
RUNS = {
    "hunyuan_art_721": "gyvtocue",
    "hunyuan_fast_11": "ontwcush",
    "hunyuan_dance_11": "1a9gbm4p",
    "hunyuan_nft_ema_721": "bjb9ed8b",
    "hunyuan_nft_onpolicy_721": "y5vn56yr",
    "hunyuan_scalar_vq_k8": "1iwf5sj9",
    "hunyuan_pairwise_vq_k2": "x9mzbm4c",
    "hunyuan_pairwise_vq_k4": "sdi90cn1",
    "hunyuan_pairwise_vq_k8": "p6jdpon4",
    "cover_hunyuan_art_11": "4dnkgeto",
    "cover_hunyuan_nft_ema_11": "njpjfulm",
    "cover_hunyuan_fast_11_long": "76zs2v12",
    "wan_art_11": "08ca4146a532752c473f1f37",
    "wan_fast_11": "31223c0be629580537f55ae2",
    "wan_dance_11": "ef06ee7200e554ef1d5f4e5f",
    "wan_nft_ema_11": "a6ce1ae3fe255baf889d5c03",
    "hunyuan_refl_11": "a0ba6dc83810b89403515067",
    "wan_refl_11": "16cc3f7450393e017fd6e1de",
}
# ccab6ad8:hunyuan_public_flowart.py loss_terms, also 8b2f8a8d cover;
# 3ca2a36f:wan_i2v_distilled_algorithms.py reuses the same ART objective.
ART_LOSS = {
    "loss_space": "velocity_linear_sigma",
    "direction_normalization": "rms",
    "mask_space": "velocity",
    "velocity_mask_metric": "scheduler_reference_metric",
    "corrective_mask_enabled": True,
    "q2_reference_sde_noise_level": 0.8,
    "q2_reference_kl_threshold": 1e-5,
}
# a2a3dad8:hunyuan_public_fast_sde.py MAIN_PLAN, NOISE_LEVEL, RATIO_CLIP;
# a2a3dad8:hunyuan_public_dancegrpo.py; c56245ec/51269742 Wan contracts.
GRPO = {"flow_grpo_fast": (0.8, 1e-5, 1e-4), "dancegrpo": (0.25, 1e-4, 1e-8)}


@pytest.fixture
def configured_env(monkeypatch, tmp_path):
    # Synthetic identities only. Validation must not open assets or touch CUDA.
    names = (
        "HUNYUAN_MODEL", "DANCEGRPO_ROOT", "VIDEO_REWARD_MODEL", "VIDEO_REWARD_BACKBONE",
        "HUNYUAN_TRAIN_MANIFEST", "HUNYUAN_EVAL_MANIFEST", "WAN_DISTILLED_TRANSFORMER",
        "WAN_BASE_MODEL", "FLUX_REFERENCE_MODEL", "WAN_TRAIN_MANIFEST", "WAN_EVAL_MANIFEST",
        "PAPER_VIDEO_OUTPUT_ROOT",
    )
    for name in names:
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
    monkeypatch.setenv("WAN_DISTILLED_REVISION", "converted-source-fixture")
    for prefix in ("HUNYUAN_TRAIN", "HUNYUAN_EVAL", "WAN_TRAIN", "WAN_EVAL"):
        monkeypatch.setenv(prefix + "_SHA256", "a" * 64)
    return tmp_path


def profile(name):
    return load_config(CONFIGS / f"{name}.yaml")


@pytest.mark.parametrize("name", RUNS)
def test_real_backend_validation_budget_and_inventory(configured_env, name):
    c = profile(name)
    backend = get_backend(c)  # Invokes the actual native backend config validator.
    assert backend.name == ("wan_i2v_distilled" if name.startswith("wan_") else "hunyuan_public")
    assert c.training.max_updates == 400
    assert c.sampling.prompts_per_rollout == 32
    assert c.sampling.samples_per_prompt == (
        1 if "refl" in name else int(name[-1]) if "pairwise" in name else 8
    )
    assert c.runtime.distributed == "fsdp" and c.runtime.device == "cuda"
    assert c.seed == 42 and c.training.precision == "bf16"
    assert c.training.optimizer == "adamw"
    assert c.training.learning_rate == 1e-5
    assert c.training.betas == (0.9, 0.999)
    assert c.training.epsilon == 1e-8 and c.training.weight_decay == 1e-4
    assert c.training.grad_clip == 1 and c.model.options["allow_tf32"] is True
    assert c.reward.normalization == "fixed"
    assert c.reward.revision == "4f26600130683e6f1de9f5d463887f28e8ef995c"
    assert c.sampling.guidance_scale == 1 and c.sampling.frames == 53
    assert c.sampling.fps == c.evaluation.fps == 8 and c.evaluation.reward_fps == 4
    assert c.evaluation.every_updates == 10 and c.evaluation.retain_all is True
    assert c.evaluation.seed == 312
    assert c.runtime.checkpoint_every == 100 * updates_per_rollout(c)
    assert c.runtime.output_dir.startswith(str(configured_env))
    assert c.provenance["run_id"] == RUNS[name]
    inventory = json.loads((ROOT / "docs/reproduction-inventory.json").read_text())
    run = next(r for g in inventory["groups"] for r in g["runs"] if r["run_id"] == RUNS[name])
    assert c.provenance["runtime_sha"] == run["runtime_commit"]
    assert len(c.provenance["reference_sha"]) == 40
    assert c.provenance["reference_location"]
    assert c.provenance["historical_evaluation"] != c.provenance["evaluation_protocol"]
    assert c.provenance["kind"].startswith("corrected_eval")


@pytest.mark.parametrize("name", RUNS)
def test_frozen_schedule_loss_and_no_stale_inheritance(configured_env, name):
    c = profile(name)
    wan = c.task == "i2v"
    algorithm = c.algorithm.name
    assert c.sampling.steps == (4 if wan else 16)
    assert (c.sampling.height, c.sampling.width) == ((400, 640) if wan else (480, 480))
    assert c.evaluation.sampling_steps == (4 if wan else 30)
    assert (c.evaluation.height, c.evaluation.width) == ((400, 640) if wan else (640, 640))
    assert c.sampling.options["shift"] == 5
    assert c.evaluation.shift == (None if wan else 7.0)
    if algorithm == "diffusionnft":
        count, pool = (4, 4) if wan else (9, 16)
    elif algorithm == "refl":
        count, pool = 1, None
    elif algorithm == "flow_grpo_fast":
        count, pool = (1, 3) if wan else (2, 6)
    else:
        count, pool = (3, 3) if wan else (9, 15)
    assert c.sampling.options["train_timestep_count"] == count
    assert c.sampling.options["train_timestep_pool"] == (
        [c.sampling.steps - 1] if pool is None else list(range(pool))
    )
    keys = {"shift", "train_timestep_count", "train_timestep_pool"}
    if algorithm in GRPO:
        keys.add("noise_level")
        noise, clip, eps = GRPO[algorithm]
        assert c.sampling.options["noise_level"] == noise
        assert c.algorithm.options == {
            "loss": {"ratio_clip": clip, "advantage_clip": 5.0}, "advantage_epsilon": eps,
        }
    elif algorithm == "diffusionnft":
        # 75778e76 NFT runtime and nft_objective.py normalized-x0 floor.
        assert set(c.algorithm.options) == {"old_policy", "loss", "advantage_epsilon"}
        assert c.algorithm.options["advantage_epsilon"] == 1e-4
        assert c.algorithm.options["loss"] == {
            "nft_beta": 1.0, "adv_clip_max": 5.0, "adv_mode": "all", "eps": 1e-5,
        }
        assert c.algorithm.options["old_policy"] == {
            "mode": "no-ema" if "onpolicy" in name else "slow-ema", "rate": 0.00125,
            "maximum": 0.6,
        }
    elif algorithm == "refl":
        # 14379aef final-gpu contract; NOT random-late or first-raw ReFL.
        assert c.algorithm.options == {"reward_loss_scale": 1.0}
    else:
        assert c.algorithm.options["loss"] == ART_LOSS
        assert c.algorithm.options["restore_zero_signal"] is True
        assert c.algorithm.options["covariance"] == {
            "reduction": "unbiased", "direction_mode": "historical",
            "scale_mode": "unit_normalized_covariance", "guidance_gain": 1.0,
            "zero_signal_mode": "historical",
        }
        assert set(c.algorithm.options) == {
            "loss", "covariance", "restore_zero_signal",
        } | ({"pairwise"} if "pairwise" in name else set())
    assert set(c.sampling.options) == keys
    weights = ({"VQ": 1., "MQ": 0., "TA": 0.} if "vq_k" in name else
               {"VQ": .2, "MQ": .1, "TA": .7} if name.endswith("721") else
               {"VQ": .5, "MQ": .5, "TA": 0.})
    assert c.reward.weights == weights
    if wan:
        torch.testing.assert_close(shifted_schedule(c.sampling.steps, 5),
                                   torch.tensor([1., .9375, 5 / 6, .625, 0.]), rtol=0, atol=0)


@pytest.mark.parametrize("k", [2, 4, 8])
def test_pairwise_preserves_p_and_uses_only_binary_feedback(configured_env, k):
    c = profile(f"hunyuan_pairwise_vq_k{k}")
    scalar = profile("hunyuan_scalar_vq_k8")
    assert c.sampling.prompts_per_rollout == scalar.sampling.prompts_per_rollout == 32
    assert c.sampling.prompts_per_rollout * k == {2: 64, 4: 128, 8: 256}[k]
    assert c.training == scalar.training
    assert c.algorithm.options["pairwise"] == {"seed": 42}
    endpoints = torch.randn(k, 3, generator=torch.Generator().manual_seed(19))
    cache = SimpleNamespace(endpoints=endpoints)

    def scores(values):
        raw = {"VQ": values, "MQ": values * 0, "TA": values * 0}
        return weighted_score(raw, c.reward.weights, space="fixed_normalized",
                              normalization={key: FixedNormalization(0., 1., "fixture") for key in raw})

    rewards = torch.arange(k, dtype=torch.float32)
    a = video_pairwise.prepare(c, cache, scores(rewards), attempt=7)
    b = video_pairwise.prepare(c, cache, scores(rewards * 100 + 3), attempt=7)
    assert set(a["binary_rewards"].tolist()) == {0., 1.}
    assert torch.equal(a["binary_rewards"], b["binary_rewards"])
    assert torch.equal(a["group_covariance"], b["group_covariance"])
    ties = video_pairwise.prepare(c, cache, scores(torch.ones(k)), attempt=7)
    assert ties["binary_rewards"].tolist() == [.5] * k


def test_nft_old_policy_zero_based_decay_not_evaluation_ema(configured_env):
    for name in ("hunyuan_nft_ema_721", "hunyuan_nft_onpolicy_721", "wan_nft_ema_11"):
        state = VideoNFTState(torch.nn.Linear(1, 1), profile(name).algorithm.options)
        for step in (1, 2, 401, 600):
            metrics = state.after_update(step)
            expected = 0 if "onpolicy" in name else min((step - 1) * .00125, .6)
            assert metrics["nft/old_policy_decay"] == expected
            assert metrics["nft/evaluation_ema_decay"] == 0


def test_reward_preprocess_and_required_output_binding(configured_env, monkeypatch):
    # Frozen RM4 uniform/even-frame rule: 53 video8 frames -> 26 RM frames.
    assert uniform_indices(53, 8, 4).tolist() == torch.linspace(0, 52, 26).round().long().tolist()
    assert resized_shape(480, 480) == (448, 448)
    assert resized_shape(400, 640) == (336, 560)
    monkeypatch.delenv("PAPER_VIDEO_OUTPUT_ROOT")
    with pytest.raises(ConfigError, match="PAPER_VIDEO_OUTPUT_ROOT"):
        profile("wan_art_11")


def test_unsupported_fields_rejected_not_ignored(configured_env):
    c = profile("hunyuan_fast_11")
    stale = replace(c, algorithm=replace(c.algorithm, options={**c.algorithm.options, "ema": True}))
    with pytest.raises(ConfigError, match="no EMA"):
        get_backend(stale)
    stale = replace(c, sampling=replace(c.sampling, options={**c.sampling.options, "cps_window": 3}))
    with pytest.raises(ConfigError, match="sampling requires"):
        get_backend(stale)


def test_unbounded_run_horizon_not_fabricated(configured_env):
    c = profile("cover_hunyuan_fast_11_long")
    assert c.provenance["kind"] == "corrected_eval_bounded_longrun_reference"
    assert c.provenance["paper_plotted_horizon"] is None
    assert "not_unbounded_run_endpoint" in c.provenance["budget_basis"]
    assert c.provenance["unresolved"]
