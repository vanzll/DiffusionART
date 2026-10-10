"""Paper candidates: backend validation plus independent archived-config checks."""

import base64
import hashlib
import json
import os
import re
import zlib
from pathlib import Path

import pytest

from visual_rl.config import load_config
from visual_rl.registry import get_backend

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "configs/paper/image"
EXPECTED_RUNS = {
    "sd3_geneval_art.yaml": "11pq0iy3",
    "sd3_geneval_grpo.yaml": "wxissl19",
    "sd3_geneval_dance.yaml": "kj09p7ee",
    "sd3_geneval_fast.yaml": "ienj38uu",
    "sd3_geneval_nft_ema.yaml": "7zirilwj",
    "sd3_geneval_nft_onpolicy.yaml": "u64dtm5f",
    "sd3_geneval_dppo.yaml": "2fryhvvf",
    "sd3_ocr_art.yaml": "9c0zrzlw",
    "sd3_ocr_art_one_update.yaml": "v4nt8roi",
    "sd3_ocr_grpo.yaml": "x4cywydr",
    "sd3_ocr_dance.yaml": "g3949kwe",
    "sd3_ocr_fast.yaml": "x76d50t3",
    "sd3_ocr_nft_ema.yaml": "uxe86tf6",
    "sd3_ocr_nft_onpolicy.yaml": "s408j3eh",
    "sd3_ocr_dppo.yaml": "92ahoxm2",
    "sd3_pickscore_art.yaml": "mofumvbw",
    "sd3_pickscore_grpo.yaml": "dosfn01k",
    "sd3_pickscore_dance.yaml": "bt84xzy0",
    "sd3_pickscore_fast.yaml": "0qvzsp36",
    "sd3_pickscore_nft_ema.yaml": "pm9ouohi",
    "sd3_pickscore_nft_onpolicy.yaml": "6wvdti47",
    "sd3_pickscore_dppo.yaml": "gzqhxeg0",
    "flux_geneval_art.yaml": "z9gkdemj",
    "flux_geneval_fast.yaml": "dmrybr9d",
    "flux_geneval_dppo.yaml": "yghomcbn",
    "flux_geneval_nft_ema.yaml": "asge8y6v",
}
METHODS = ("art", "grpo", "fast", "dppo", "nft_ema", "nft_onpolicy", "dance")


@pytest.fixture(autouse=True)
def asset_environment(monkeypatch):
    names = set()
    for directory in (PROFILES, ROOT / "configs/examples"):
        for path in directory.glob("*.yaml"):
            names.update(re.findall(r"\$\{([A-Z_0-9]+)\}", path.read_text()))
    for name in names:
        value = "a" * 64 if name.endswith(("SHA256", "SHA")) else f"/public-assets/{name.lower()}"
        monkeypatch.setenv(name, value)


@pytest.fixture(scope="module")
def archive():
    override = os.environ.get("PAPER_IMAGE_ARCHIVE")
    if override:
        return json.loads(zlib.decompress(base64.b64decode(Path(override).read_bytes())))
    return json.loads((ROOT / "docs/paper-image-evidence.json").read_text())["runs"]


def test_candidate_coverage():
    assert set(EXPECTED_RUNS) <= {p.name for p in PROFILES.glob("*.yaml")}
    for benchmark in ("geneval", "ocr", "pickscore"):
        for method in METHODS:
            assert (PROFILES / f"sd3_{benchmark}_{method}.yaml").is_file()
    assert (PROFILES / "flux_geneval_nft_ema.yaml").is_file()


@pytest.mark.parametrize("filename", sorted(EXPECTED_RUNS))
def test_backend_and_public_candidate_contract(filename):
    c = load_config(PROFILES / filename)
    assert get_backend(c).name == c.model.name
    assert c.runtime.distributed == "ddp"
    assert c.runtime.log == "wandb"
    assert c.training.max_updates == 1000
    assert c.provenance["source_run"] == EXPECTED_RUNS[filename]
    assert c.provenance["kind"] == "training_candidate_not_fullhistorical_reproduction"
    assert c.provenance["semantic_gaps"]
    assert c.training.allow_tf32 is True
    assert c.evaluation.every_updates == 0
    assert c.runtime.checkpoint_every == 200
    assert all(c.runtime.checkpoint_due(step, 1000) for step in (200, 400, 600, 800, 1000))
    assert c.evaluation.enabled
    assert c.evaluation.due(1000, 1000)
    assert c.runtime.checkpoint_due(1000, 1000)
    assert re.fullmatch(r"[0-9a-f]{40}", c.model.revision)
    assert len(c.data.sha256) == len(c.evaluation.manifest_sha256) == 64
    if c.algorithm.name == "diffusionart":
        assert c.evaluation.at_updates == (500, 800, 1000)
        assert c.runtime.checkpoint_at_updates == (1000,)
    if c.algorithm.name == "dancegrpo":
        assert c.algorithm.options["advantage_mode"] == "prompt_global_std"
        assert c.sampling.options["endpoint_kind"] == "last_x0"
        assert c.provenance["runtime_commit"]
        assert "source unavailable" in " ".join(c.provenance["semantic_gaps"])


@pytest.mark.parametrize("filename", sorted(set(EXPECTED_RUNS) - {"flux_geneval_nft_ema.yaml"}))
def test_scientific_parameters_against_independent_archive(filename, archive):
    c = load_config(PROFILES / filename)
    raw = archive[EXPECTED_RUNS[filename]]["config"]
    source = {key: item["value"] for key, item in raw.items()}
    sample, train = source["sample"], source["train"]
    assert c.seed == source["seed"]
    assert c.training.precision == source["mixed_precision"]
    assert c.training.allow_tf32 == source["allow_tf32"]
    assert c.training.learning_rate == train["learning_rate"]
    assert c.training.microbatch == train["batch_size"]
    assert c.training.weight_decay == train["adam_weight_decay"]
    assert c.training.betas == (train["adam_beta1"], train["adam_beta2"])
    assert c.training.epsilon == train["adam_epsilon"]
    assert c.training.grad_clip == train["max_grad_norm"]
    assert c.sampling.steps == sample["num_steps"]
    assert c.sampling.samples_per_prompt == sample["num_image_per_prompt"]
    assert c.sampling.prompts_per_rollout * sample["num_image_per_prompt"] == (
        sample["train_batch_size"] * sample["num_batches_per_epoch"] * 8
    )
    assert c.sampling.microbatch == sample["train_batch_size"]
    assert c.sampling.guidance_scale == sample["guidance_scale"]
    assert c.sampling.height == c.sampling.width == source["resolution"]
    assert c.evaluation.sampling_steps == sample["eval_num_steps"]
    assert c.reward.weights == source["reward_fn"]
    assert c.provenance["source_gradient_accumulation_steps"] == train["gradient_accumulation_steps"]
    assert c.provenance["source_resume_global_step"] == source.get("resume_global_step", 0)
    if source.get("paper_eval_wandb_steps"):
        assert c.evaluation.at_updates == tuple(i for i in source["paper_eval_wandb_steps"] if i)
        assert c.runtime.checkpoint_at_updates == tuple(source["paper_save_wandb_steps"])
    else:
        updates = c.provenance["candidate_updates_per_rollout"]
        interval = source["eval_freq"] * updates
        assert c.evaluation.at_updates == tuple(range(interval, 1001, interval))
    a, s = c.algorithm.options, c.sampling.options
    count = int(sample["num_steps"] * train["timestep_fraction"])
    if c.model.name == "sd3":
        assert s["shift"] == train["timestep_shift"]
    if c.algorithm.name == "diffusionart":
        assert s["train_timesteps"] == list(range(count))
        assert a.get("updates_per_rollout", 2) == (
            sample["num_batches_per_epoch"] // train["gradient_accumulation_steps"]
        )
        assert a["covariance"]["scale_mode"] == train["covariance_guidance_scale_mode"]
        assert a["covariance"]["guidance_gain"] == train["covariance_guidance_gain"]
        for target, original in (
            ("loss_space", "covariance_guidance_loss_space"),
            ("mask_space", "x0_covariance_mask_space"),
            ("velocity_mask_metric", "x0_covariance_velocity_mask_metric"),
            ("corrective_mask_enabled", "x0_covariance_corrective_mask_enabled"),
            ("q2_reference_sde_noise_level", "x0_covariance_q2_reference_sde_noise_level"),
            ("q2_reference_kl_threshold", "x0_covariance_q2_reference_kl_threshold"),
        ):
            assert a["loss"][target] == train[original]
    elif c.algorithm.name == "diffusionnft":
        assert a["loss"]["nft_beta"] == source["beta"]
        assert a["loss"]["adv_clip_max"] == train["adv_clip_max"]
        assert a["loss"]["adv_mode"] == train["adv_mode"]
        assert s["solver"] == sample["solver"]
        assert s["train_timestep_count"] == count
        expected_decay = {
            0: {"rate": 0, "maximum": 0, "delay": 0},
            1: {"rate": 0.001, "maximum": 0.5, "delay": 0},
            2: {"rate": 0.0075, "maximum": 0.999, "delay": 75},
        }
        assert a["old_policy"] == expected_decay[source["decay_type"]]
        assert a["evaluation_ema"]["enabled"] == train["ema"]
    else:
        assert s["noise_level"] == sample["noise_level"]
        assert s["logical_batch_size"] == sample["train_batch_size"]
        assert a["training_batches"] == sample["num_batches_per_epoch"]
        assert a["evaluation_ema"]["enabled"] == train["ema"]
        assert a["loss"]["advantage_clip"] == train["adv_clip_max"]
        if c.algorithm.name == "flow_dppo":
            assert a["loss"]["kl_mask_threshold"] == train["flow_dppo_kl_threshold"]
        else:
            assert a["loss"]["ratio_clip"] == train["clip_range"]
        if sample["sde_window_size"] > 0:
            start, stop = sample["sde_window_range"]
            assert s["pool"] == list(range(start, stop))
            assert s["count"] == sample["sde_window_size"]
        else:
            assert s["count"] == count
        if c.algorithm.name == "dancegrpo":
            assert s["same_latent"] == sample["same_latent"]
            assert c.provenance["source_global_std"] == sample["global_std"]


def test_ocr_art_one_update_preserves_restoration_and_full_window():
    from visual_rl.training.objectives import disjoint_windows

    one = load_config(PROFILES / "sd3_ocr_art_one_update.yaml")
    two = load_config(PROFILES / "sd3_ocr_art.yaml")
    assert get_backend(one).name == "sd3"
    options = dict(one.algorithm.options)
    assert options.pop("updates_per_rollout") == 1
    assert options == two.algorithm.options
    for field in ("model", "reward", "data", "sampling", "training", "evaluation"):
        assert getattr(one, field) == getattr(two, field)
    assert one.runtime.output_dir != two.runtime.output_dir
    assert one.provenance["source_gradient_accumulation_steps"] == 16
    assert one.provenance["candidate_updates_per_rollout"] == 1
    assert one.provenance["semantic_gaps"] == two.provenance["semantic_gaps"]
    rows = one.sampling.prompts_per_rollout * one.sampling.samples_per_prompt
    slots = len(one.sampling.options["train_timesteps"])
    windows = disjoint_windows(rows, slots, updates=one.algorithm.options["updates_per_rollout"])
    assert len(windows) == 1
    assert len(windows[0]) == rows * slots
    assert {row for row, _ in windows[0]} == set(range(rows))


def test_known_unresolved_semantics_remain_declared():
    c = load_config(PROFILES / "sd3_ocr_nft_ema.yaml")
    assert c.provenance["source_gradient_accumulation_steps"] == 8
    assert c.provenance["source_updates_per_rollout"] == 2
    assert c.provenance["candidate_updates_per_rollout"] == 2
    assert c.algorithm.options["updates_per_rollout"] == 2
    for filename in ("sd3_geneval_dppo.yaml", "sd3_ocr_dppo.yaml", "sd3_pickscore_dppo.yaml"):
        c = load_config(PROFILES / filename)
        assert c.provenance["source_resume_global_step"] > 0
        assert "checkpoint unavailable" in " ".join(c.provenance["semantic_gaps"])


def test_flux_nft_against_separately_archived_runtime():
    override = os.environ.get("PAPER_FLUX_NFT_ARCHIVE")
    if override:
        document = json.loads(Path(override).read_text())
        run = next(row for row in document["records"] if row["id"] == "asge8y6v")["data"]["project"]["run"]
        raw = json.loads(run["config"])
    else:
        raw = json.loads((ROOT / "docs/paper-image-evidence.json").read_text())["runs"]["asge8y6v"]["config"]
    archived = {k: v["value"] for k, v in raw.items()}
    c = load_config(PROFILES / "flux_geneval_nft_ema.yaml")
    native = archived["nft_flux"]
    assert c.provenance["runtime_commit"] == native["runtime_commit"]
    assert c.sampling.prompts_per_rollout == native["prompts_per_rollout"] == 48
    assert c.sampling.samples_per_prompt == archived["sample"]["num_image_per_prompt"] == 24
    assert c.sampling.microbatch == native["rollout_microbatch"]
    assert c.training.microbatch == native["update_microbatch"]
    assert c.training.max_updates == native["max_successful_updates"]
    assert c.evaluation.at_updates == tuple(x for x in native["eval_updates"] if x)
    assert c.runtime.checkpoint_at_updates == tuple(native["save_updates"])
    assert c.sampling.steps == archived["sample"]["num_steps"] == 6
    assert c.sampling.options["train_timestep_count"] == int(6 * archived["train"]["timestep_fraction"])
    assert c.training.precision == archived["mixed_precision"] == "bf16"
    assert c.training.learning_rate == archived["train"]["learning_rate"]
    assert c.algorithm.options["loss"]["nft_beta"] == archived["beta"]
    assert c.algorithm.options["updates_per_rollout"] == 1
    assert c.model.options["gradient_checkpointing"] == archived["activation_checkpointing"]


def test_public_scientific_evidence_has_no_private_fields():
    text = (ROOT / "docs/paper-image-evidence.json").read_text()
    evidence = json.loads(text)
    assert len(evidence["runs"]) == 26
    assert not re.search(r"/m2v_|/Users/|WANDB_API_KEY|10\.83\.", text)
    assert {load_config(PROFILES / name).provenance["source_run"]
            for name in EXPECTED_RUNS} == set(evidence["runs"])
    assert all(re.fullmatch(r"[0-9a-f]{64}", source["sha256"]) for source in evidence["sources"])


def test_public_evidence_matches_original_snapshot_when_available():
    default = ROOT.parent / "paper_data/t2i-draft-20260910/t2i_draft_metadata_compact.b64"
    path = Path(os.environ.get("PAPER_IMAGE_ARCHIVE", default))
    if not path.is_file():
        pytest.skip("Original private archive is optional; bundled scientific checks still run")
    evidence = json.loads((ROOT / "docs/paper-image-evidence.json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == evidence["sources"][0]["sha256"]
    original = json.loads(zlib.decompress(base64.b64decode(path.read_bytes())))
    for run, source in evidence["runs"].items():
        if run == "asge8y6v":
            continue
        config = original[run]["config"]
        for key, wrapped in source["config"].items():
            expected = config[key]["value"]
            if isinstance(wrapped["value"], dict):
                assert wrapped["value"] == {name: expected[name] for name in wrapped["value"]}
            else:
                assert wrapped["value"] == expected
