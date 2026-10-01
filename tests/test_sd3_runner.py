"""Native runner integration: same command path with small real model modules."""

import hashlib
import json
import socket
import time
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml
from test_sd3_text import real_pipeline  # noqa: F401 -- shared offline asset fixture

from visual_rl.backends import sd3
from visual_rl.config import ConfigError, load_config
from visual_rl.evaluation import EvaluationManifest, audit_manifest
from visual_rl.models import SD3Adapter, SD3Conditioning, SD3Config
from visual_rl.registry import get_backend
from visual_rl.rewards.pickscore import PickScore
from visual_rl.train import main


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def config_file(tmp_path):
    dataset = tmp_path / "prompts.jsonl"
    dataset.write_text(
        "".join(json.dumps({"sample_id": str(i), "prompt": str(i + 2)}) + "\n" for i in range(4))
    )
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    config = {
        "name": "sd3-native-test",
        "task": "t2i",
        "seed": 17,
        "algorithm": {
            "name": "diffusionart",
            "options": {"covariance": {}, "loss": {}, "restore_zero_signal": False},
        },
        "model": {
            "name": "sd3",
            "checkpoint": "injected",
            "revision": "test-only",
            "options": {
                "scaling_factor": 2.0,
                "shift_factor": 0.25,
                "latent_channels": 2,
                "spatial_factor": 1,
                "max_sequence_length": 3,
                "finetune": {"mode": "full"},
                "decode_microbatch": 1,
                "gradient_checkpointing": False,
            },
        },
        "reward": {
            "name": "pickscore",
            "checkpoint": "injected",
            "revision": "test-only",
            "normalization": "fixed",
            "weights": {"pickscore": 1.0},
            "options": {
                "processor": "injected",
                "processor_revision": "test-only",
                "microbatch": 2,
            },
        },
        "sampling": {
            "steps": 3,
            "height": 4,
            "width": 4,
            "frames": 1,
            "guidance_scale": 1,
            "prompts_per_rollout": 2,
            "samples_per_prompt": 2,
            "microbatch": 2,
            "options": {"shift": 1.0, "train_timesteps": [0, 1]},
        },
        "training": {"max_updates": 4, "learning_rate": 1e-3, "precision": "fp32", "microbatch": 3},
        "evaluation": {
            "every_updates": 2,
            "manifest": str(dataset),
            "manifest_sha256": digest,
            "sampling_steps": 3,
        },
        "data": {"manifest": str(dataset), "sha256": digest, "shuffle": True},
        "runtime": {"device": "cpu", "output_dir": str(tmp_path / "run"), "checkpoint_every": 2},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def tiny_parts(c):
    diffusers = pytest.importorskip("diffusers")
    transformers = pytest.importorskip("transformers")
    from test_pickscore import Processor

    model = diffusers.SD3Transformer2DModel(
        sample_size=4,
        patch_size=2,
        in_channels=2,
        out_channels=2,
        num_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=12,
        caption_projection_dim=16,
        pooled_projection_dim=8,
        pos_embed_max_size=4,
    ).eval()
    vae = diffusers.AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(8,),
        layers_per_block=1,
        latent_channels=2,
        norm_num_groups=4,
        sample_size=4,
        scaling_factor=2.0,
        shift_factor=0.25,
    )
    adapter = SD3Adapter(model, vae, SD3Config(2.0, 0.25, latent_channels=2)).eval()

    class Text:
        def encode(self, prompts, *, num_images_per_prompt, **kwargs):
            x = torch.tensor([float(p) for p in prompts])
            return SD3Conditioning(
                x[:, None, None].expand(-1, 3, 12).repeat_interleave(num_images_per_prompt, 0),
                x[:, None].expand(-1, 8).repeat_interleave(num_images_per_prompt, 0),
            )

    clip = transformers.CLIPModel(
        transformers.CLIPConfig(
            projection_dim=4,
            text_config=dict(
                vocab_size=16,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                max_position_embeddings=8,
            ),
            vision_config=dict(
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=4,
                patch_size=2,
            ),
        )
    )
    return sd3.Components(adapter, Text(), PickScore(clip, Processor(), space="fixed_normalized"))


def payload(root):
    latest = json.loads((root / "checkpoints/latest.json").read_text())
    return torch.load(root / "checkpoints" / latest["file"], weights_only=True)


def test_native_cli_train_eval_and_resume(tmp_path, monkeypatch):
    path = config_file(tmp_path)
    monkeypatch.setattr(sd3, "load_components", tiny_parts)
    assert main(["--config", str(path), "--validate"]) == 0
    assert main(["--config", str(path)]) == 0
    root = tmp_path / "run"
    status = json.loads((root / "status.json").read_text())
    assert status["optimizer_steps"] == 4 and status["completed_rollouts"] == 2
    logs = [json.loads(x) for x in (root / "metrics.jsonl").read_text().splitlines()]
    assert [x["train/optimizer_step"] for x in logs] == [1, 2, 3, 4]
    assert [x["system/completed_rollouts"] for x in logs] == [0, 1, 1, 2]
    for record in logs:
        assert record["cost/elapsed_gpu_hours"] == 0
        assert record["train/reward"] == pytest.approx(
            record["train/reward/raw/pickscore"] / 26, abs=1e-7
        )
    manifests = sorted(root.glob("evaluation/step-*/*/manifest.json"))
    assert len(manifests) == 3
    for path_ in manifests:
        manifest = EvaluationManifest.from_json(path_.read_text())
        assert audit_manifest(manifest, media_root=root).complete
        assert manifest.optimizer_step in (0, 2, 4)
        assert len(manifest.samples) == 4
        assert [(s.sample_id, s.seed) for s in manifest.samples] == [
            (str(i), 17 + i) for i in range(4)
        ]
    before = payload(root)
    assert "prompts" in before["stateful"] and "sampling" in before["generators"]
    assert main(["--config", str(path), "--resume", str(root / "checkpoints/latest.json")]) == 0
    assert len((root / "metrics.jsonl").read_text().splitlines()) == 4
    assert len(list(root.glob("evaluation/step-*/*/manifest.json"))) == 3
    records = {name: (root / name).read_bytes() for name in ("metrics.jsonl", "status.json", "cost.json")}
    assert main(["--config", str(path), "--resume", str(root / "checkpoints/latest.json"), "--eval-only"]) == 0
    assert records == {name: (root / name).read_bytes() for name in records}
    assert len(list(root.glob("evaluation/step-*/*/manifest.json"))) == 4


@pytest.mark.parametrize("precision", ["fp32", "bf16", "fp16"])
def test_interrupted_boundary_resume_matches_uninterrupted(tmp_path, monkeypatch, precision):
    path = config_file(tmp_path)
    c = load_config(path, {"training": {"precision": precision}})
    monkeypatch.setattr(sd3, "load_components", tiny_parts)
    sd3.run(c)
    expected = payload(Path(c.runtime.output_dir))
    assert ("grad_scaler" in expected["stateful"]) == (precision == "fp16")
    interrupted = replace(c, runtime=replace(c.runtime, output_dir=str(tmp_path / "interrupted")))
    save = sd3.save_checkpoint

    def stop_after_boundary(*args, **kwargs):
        receipt = save(*args, **kwargs)
        if kwargs["progress"].optimizer_steps == 2:
            raise RuntimeError("injected shutdown after full checkpoint")
        return receipt

    monkeypatch.setattr(sd3, "save_checkpoint", stop_after_boundary)
    with pytest.raises(RuntimeError, match="injected shutdown"):
        sd3.run(interrupted)
    monkeypatch.setattr(sd3, "save_checkpoint", save)
    root = Path(interrupted.runtime.output_dir)
    sd3.run(interrupted, resume=root / "checkpoints/latest.json")
    actual = payload(root)
    for key in expected["model"]:
        torch.testing.assert_close(actual["model"][key], expected["model"][key], rtol=0, atol=0)
    assert actual["stateful"] == expected["stateful"]
    assert torch.equal(actual["generators"]["sampling"], expected["generators"]["sampling"])


def test_resume_refuses_foreign_output_before_loading_assets(tmp_path, monkeypatch):
    path = config_file(tmp_path)
    config = load_config(path)
    monkeypatch.setattr(sd3, "load_components", tiny_parts)
    sd3.run(config)
    root = Path(config.runtime.output_dir)
    foreign = replace(config, seed=config.seed + 1)
    (root / "resolved-config.json").write_text(json.dumps(foreign.to_dict()))

    def forbidden(*args):
        raise AssertionError("identity must be checked before asset loading")

    monkeypatch.setattr(sd3, "load_components", forbidden)
    with pytest.raises(ConfigError, match="different run/config"):
        sd3.run(config, resume=root / "checkpoints/latest.json")


def test_rewind_after_updates_beyond_checkpoint_is_exact(tmp_path, monkeypatch):
    from test_distributed_checkpoint import assert_tree_equal

    c = load_config(config_file(tmp_path))
    monkeypatch.setattr(sd3, "load_components", tiny_parts)
    sd3.run(c)
    expected = payload(Path(c.runtime.output_dir))
    interrupted = replace(c, runtime=replace(c.runtime, output_dir=str(tmp_path / "interrupted")))
    save = sd3.save_checkpoint

    def fail_before_next_checkpoint(*args, **kwargs):
        if kwargs["progress"].optimizer_steps == 4:
            raise RuntimeError("crash between checkpoints")
        return save(*args, **kwargs)

    monkeypatch.setattr(sd3, "save_checkpoint", fail_before_next_checkpoint)
    with pytest.raises(RuntimeError, match="between checkpoints"):
        sd3.run(interrupted)
    monkeypatch.setattr(sd3, "save_checkpoint", save)
    root = Path(interrupted.runtime.output_dir)
    receipt = root / "checkpoints/latest.json"
    with pytest.raises(ConfigError, match="--resume-rewind"):
        sd3.run(interrupted, resume=receipt)
    sd3.run(interrupted, resume=receipt, resume_rewind=True)
    actual = payload(root)
    for key in ("model", "optimizer", "stateful", "generators"):
        assert_tree_equal(expected[key], actual[key])
    assert [json.loads(line)["train/optimizer_step"] for line in
            (root / "metrics.jsonl").read_text().splitlines()] == [1, 2, 3, 4]
    archive, = (root / "resume-history").iterdir()
    assert len((archive / "metrics.jsonl").read_text().splitlines()) == 4
    assert list((archive / "evaluation/step-000000004").glob("*/manifest.json"))
    assert actual["progress"]["training_seconds"] > expected["progress"]["training_seconds"]


def test_checkpoint_cost_persisted_and_completed_resume_does_not_regress(tmp_path, monkeypatch):
    c = load_config(config_file(tmp_path))
    monkeypatch.setattr(sd3, "load_components", tiny_parts)
    save = sd3.save_checkpoint

    def slow_save(*args, **kwargs):
        result = save(*args, **kwargs)
        time.sleep(0.2)
        return result

    monkeypatch.setattr(sd3, "save_checkpoint", slow_save)
    sd3.run(c)
    root = Path(c.runtime.output_dir)
    status = (root / "status.json").read_text()
    cost = json.loads((root / "cost.json").read_text())["progress"]
    checkpoint = payload(root)["progress"]
    assert cost["elapsed_seconds"] - checkpoint["elapsed_seconds"] >= 0.2
    assert cost["training_seconds"] == checkpoint["training_seconds"]
    sd3.run(c, resume=root / "checkpoints/latest.json")
    assert (root / "status.json").read_text() == status


@pytest.mark.parametrize(
    "override",
    [
        {"training": {"max_updates": 3}},
        {"algorithm": "diffusionnft"},
        {"runtime": {"distributed": "fsdp"}},
        {"reward": {"weights": {"pickscore": 0.5}}},
        {"sampling": {"options": {"train_timesteps": [0, 0]}}},
        {"model": {"options": {"misspelt_option": True}}},
        {"algorithm": {"options": {"loss": {"reference_mse_beta": 0.01}}}},
    ],
)
def test_unsupported_settings_fail_preflight(tmp_path, override):
    path = config_file(tmp_path)
    with pytest.raises(ConfigError):
        get_backend(load_config(path, override))
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    "bad",
    [
        {"normalization": "fixed"},
        {"options": {"head_sha256": "not-a-digest"}},
        {"options": {"head_path": ""}},
        {"options": {"head_path": False}},
        {"weights": {"pickscore": 1.0}},
    ],
)
def test_aesthetic_config_does_not_inherit_pickscore_scale(tmp_path, bad):
    path = config_file(tmp_path)
    config = yaml.safe_load(path.read_text())
    config["reward"].update(name="aesthetic", normalization="raw", weights={"aesthetic": 1.0})
    config["reward"]["options"].update(head_path="head.pth", head_sha256="a" * 64)
    path.write_text(yaml.safe_dump(config))
    assert get_backend(load_config(path)).name == "sd3"
    with pytest.raises(ConfigError):
        get_backend(load_config(path, {"reward": bad}))
    assert not (tmp_path / "run").exists()


def test_fp16_real_asset_cli(tmp_path, monkeypatch, request):
    test_actual_local_asset_loaders_run_without_network(
        tmp_path, monkeypatch, request, "diffusionart", "pickscore", False, precision="fp16"
    )


@pytest.mark.parametrize(
    "algorithm,joint",
    [
        ("diffusionart", False),
        ("diffusionart", True),
        ("diffusionnft", False),
        ("flow_grpo", False),
        ("flow_grpo_fast", False),
        ("flow_dppo", False),
    ],
)
@pytest.mark.parametrize(
    "reward", ["pickscore", "aesthetic", "composite", "clipscore", "composite_clip"]
)
def test_actual_local_asset_loaders_run_without_network(
    tmp_path, monkeypatch, request, algorithm, reward, joint, precision="fp32"
):
    if reward in ("clipscore", "composite_clip"):
        pytest.importorskip("torchvision")
    pipeline = request.getfixturevalue("real_pipeline")
    diffusers = pytest.importorskip("diffusers")
    transformers = pytest.importorskip("transformers")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    def no_network(*args, **kwargs):
        raise AssertionError("local-asset run must not use network")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    pipeline.register_modules(
        transformer=diffusers.SD3Transformer2DModel(
            sample_size=4,
            patch_size=2,
            in_channels=2,
            out_channels=2,
            num_layers=1,
            attention_head_dim=8,
            num_attention_heads=2,
            joint_attention_dim=24,
            caption_projection_dim=16,
            pooled_projection_dim=10,
            pos_embed_max_size=4,
        ),
        vae=diffusers.AutoencoderKL(
            in_channels=3,
            out_channels=3,
            down_block_types=("DownEncoderBlock2D",),
            up_block_types=("UpDecoderBlock2D",),
            block_out_channels=(8,),
            layers_per_block=1,
            latent_channels=2,
            norm_num_groups=4,
            sample_size=4,
            scaling_factor=2.0,
            shift_factor=0.25,
        ),
    )
    base = tmp_path / "sd3-asset"
    pipeline.save_pretrained(base, safe_serialization=True)
    rm = tmp_path / "reward-asset"
    model = transformers.CLIPModel(
        transformers.CLIPConfig(
            projection_dim=4 if reward in ("pickscore", "clipscore") else 768,
            text_config=dict(
                vocab_size=8,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                max_position_embeddings=77,
                bos_token_id=0,
                eos_token_id=1,
            ),
            vision_config=dict(
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=4,
                patch_size=2,
            ),
        )
    )
    model.save_pretrained(rm)
    vocab = {
        "<|startoftext|>": 0,
        "<|endoftext|>": 1,
        "2</w>": 2,
        "3</w>": 3,
        "4</w>": 4,
        "5</w>": 5,
    }
    import inspect

    if "vocab" in inspect.signature(transformers.CLIPTokenizer).parameters:
        tokenizer = transformers.CLIPTokenizer(vocab=vocab, merges=[], model_max_length=77)
    else:
        (rm / "vocab.json").write_text(json.dumps(vocab))
        (rm / "merges.txt").write_text("#version: 0.2\n")
        tokenizer = transformers.CLIPTokenizer(
            vocab_file=str(rm / "vocab.json"),
            merges_file=str(rm / "merges.txt"),
            model_max_length=77,
        )
    image_cls = getattr(transformers, "CLIPImageProcessorPil", None)
    if image_cls is None:
        image_cls = transformers.CLIPImageProcessor
    processor = transformers.CLIPProcessor(
        tokenizer=tokenizer,
        image_processor=image_cls(size={"shortest_edge": 4}, crop_size={"height": 4, "width": 4}),
    )
    processor.save_pretrained(rm)
    path = config_file(tmp_path)
    config = yaml.safe_load(path.read_text())
    config["training"]["precision"] = precision
    config["model"]["checkpoint"] = str(base)
    config["reward"]["checkpoint"] = str(rm)
    config["reward"]["options"]["processor"] = str(rm)
    if reward == "clipscore":
        config["reward"]["name"] = "clipscore"
        config["reward"]["weights"] = {"clipscore": 1.0}
        config["reward"]["options"]["antialias"] = True
    if reward in ("aesthetic", "composite", "composite_clip"):
        from visual_rl.rewards.aesthetic import AestheticMLP

        head = tmp_path / "aesthetic-head.pth"
        torch.save(AestheticMLP().state_dict(), head)
        config["reward"]["name"] = "aesthetic"
        config["reward"]["normalization"] = "raw"
        config["reward"]["weights"] = {"aesthetic": 1.0}
        config["reward"]["options"].update(
            head_path=str(head), head_sha256=hashlib.sha256(head.read_bytes()).hexdigest()
        )
        if reward.startswith("composite"):
            aesthetic = {
                key: config["reward"][key] for key in ("checkpoint", "revision", "options")
            }
            pickscore = dict(aesthetic)
            pickscore["options"] = {
                key: value
                for key, value in aesthetic["options"].items()
                if key not in ("head_path", "head_sha256")
            }
            config["reward"] = {
                "name": "composite",
                "normalization": "fixed",
                "weights": {"aesthetic": 3.0, "pickscore": 2.0},
                "options": {
                    "microbatch": 2,
                    "components": {"aesthetic": aesthetic, "pickscore": pickscore},
                    "fixed_statistics": {
                        "aesthetic": {"mean": 0.2, "std": 2.0, "source": "fixture-only"},
                        "pickscore": {"mean": 1.0, "std": 26.0, "source": "fixture-only"},
                    },
                },
            }
            if reward == "composite_clip":
                clip = {**pickscore, "options": {**pickscore["options"], "antialias": True}}
                config["reward"]["weights"]["clipscore"] = 0.5
                config["reward"]["options"]["components"]["clipscore"] = clip
                config["reward"]["options"]["fixed_statistics"]["clipscore"] = {
                    "mean": 0.0,
                    "std": 30.0,
                    "source": "fixture-clip-logit-divisor-30",
                }
    if joint:
        config["algorithm"]["options"]["multi_reward_advantage"] = {
            "weights": dict(config["reward"]["weights"]),
            "score_space": "fixed_normalized"
            if config["reward"]["normalization"] == "fixed"
            else "raw",
            "eps": 1e-4,
            "renormalize_combined": False,
        }
    if algorithm == "diffusionnft":
        config["algorithm"] = {
            "name": "diffusionnft",
            "options": {
                "loss": {"nft_beta": 1.0},
                "old_policy": {"rate": 0.001, "maximum": 0.5, "delay": 0},
                "evaluation_ema": {"enabled": True, "decay": 0.9},
            },
        }
        config["sampling"]["options"] = {
            "shift": 1.0,
            "solver": "dpm2",
            "train_timestep_count": 2,
            "timestep_selection": "random_per_row",
        }
    elif algorithm in ("flow_grpo", "flow_grpo_fast", "flow_dppo"):
        config["algorithm"] = {
            "name": algorithm,
            "options": {
                "loss": (
                    {"kl_mask_threshold": 1e-5, "advantage_clip": 5.0}
                    if algorithm == "flow_dppo"
                    else {"ratio_clip": 1e-5, "advantage_clip": 5.0}
                ),
                "updates_per_rollout": 2,
                "training_batches": 2,
                "shuffle_rows": algorithm != "flow_grpo_fast",
                "zero_advantage_mode": "retain",
                "max_empty_rollouts": 2,
                "evaluation_ema": {"enabled": True, "decay": 0.9, "refresh": "logical_batch"},
            },
        }
        config["sampling"]["options"] = {
            "shift": 1.0,
            "transition_kind": "sde" if algorithm == "flow_grpo" else "cps",
            "likelihood": "gaussian" if algorithm == "flow_grpo" else "mse",
            "noise_level": 0.8,
            "selection": "fixed",
            "pool": [0, 1],
            "count": 2,
            "logical_batch_size": 2,
            "unselected_noise": "all" if algorithm == "flow_grpo" else "ode",
        }
    path.write_text(yaml.safe_dump(config))
    # Do not replace load_components: exercise the real loaders and tokenizer.
    assert main(["--config", str(path)]) == 0
    assert json.loads((tmp_path / "run/status.json").read_text())["optimizer_steps"] == 4
    assert len(list((tmp_path / "run").glob("evaluation/step-*/*/manifest.json"))) == 3
    history = [
        json.loads(line) for line in (tmp_path / "run/metrics.jsonl").read_text().splitlines()
    ]
    assert len(history) == 4
    assert ("grad_scaler" in payload(tmp_path / "run")["stateful"]) == (precision == "fp16")
    for row in history:
        assert not any(key.startswith("train/multi_reward/") for key in row)
        if reward.startswith("composite"):
            expected = (
                3 * (row["train/reward/raw/aesthetic"] - 0.2) / 2
                + 2 * (row["train/reward/raw/pickscore"] - 1) / 26
            )
            if reward == "composite_clip":
                expected += 0.5 * row["train/reward/raw/clipscore"] / 30
                assert "train/reward/fixed_normalized/clipscore" in row
            assert row["train/reward"] == pytest.approx(expected, abs=1e-6)
            assert "train/reward/fixed_normalized/aesthetic" in row
            assert "train/reward/fixed_normalized/pickscore" in row
        else:
            assert f"train/reward/raw/{reward}" in row
        if reward == "aesthetic":
            assert row["train/reward"] == row["train/reward/raw/aesthetic"]
            assert not any("pickscore" in key for key in row)
        if reward == "clipscore":
            assert row["train/reward"] == pytest.approx(
                row["train/reward/raw/clipscore"] / 30, abs=1e-6
            )
    for path in (tmp_path / "run").glob("evaluation/step-*/*/summary.json"):
        summary = json.loads(path.read_text())
        if reward.startswith("composite"):
            expected = (
                3 * (summary["eval/reward/raw/aesthetic"] - 0.2) / 2
                + 2 * (summary["eval/reward/raw/pickscore"] - 1) / 26
            )
            if reward == "composite_clip":
                expected += 0.5 * summary["eval/reward/raw/clipscore"] / 30
            assert summary["eval/reward"] == pytest.approx(expected, abs=1e-6)
            manifest = EvaluationManifest.from_json(path.with_name("manifest.json").read_text())
            for sample in manifest.samples:
                assert set(sample.scores.raw) == set(config["reward"]["weights"])
                assert sample.scores.weights == config["reward"]["weights"]
                assert sample.scores.space == "fixed_normalized"
        else:
            assert f"eval/reward/raw/{reward}" in summary
        if reward == "aesthetic":
            assert summary["eval/reward"] == summary["eval/reward/raw/aesthetic"]
            assert not any("pickscore" in key for key in summary)
        if reward == "clipscore":
            assert summary["eval/reward"] == pytest.approx(
                summary["eval/reward/raw/clipscore"] / 30, abs=1e-6
            )
