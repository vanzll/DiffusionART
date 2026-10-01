import hashlib
import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import yaml
from torch import nn

from visual_rl.backends import hunyuan
from visual_rl.config import load_config
from visual_rl.evaluation.manifest import EvaluationManifest, audit_manifest, file_sha256
from visual_rl.evaluation.video import quantize_rgb
from visual_rl.models.hunyuan import HunyuanVideoAdapter
from visual_rl.models.hunyuan_assets import NATIVE_REVISION, cached_conditioning
from visual_rl.rewards.scalar import FixedNormalization, weighted_score
from visual_rl.sampling import shifted_schedule
from visual_rl.train import main


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(.1))
        self.config = SimpleNamespace(in_channels=16, out_channels=16)

    def forward(self, hidden_states, encoder_hidden_states, timestep, guidance, **kwargs):
        assert guidance.item() == torch.tensor(6018., dtype=torch.bfloat16).item()
        assert timestep.dtype == torch.int64
        return (hidden_states * self.scale + encoder_hidden_states.mean() * .001,)


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(1.))
        self.config = SimpleNamespace(scaling_factor=.476986, latent_channels=16)

    def decode(self, latents, **kwargs):
        return (F.interpolate(latents[:, :3], size=((latents.shape[2] - 1) * 4 + 1,
                                                  latents.shape[3] * 8, latents.shape[4] * 8)).tanh(),)


class Scorer(nn.Module):
    def score_paths(self, paths, prompts):
        values = []
        for path in paths:
            raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo",
                                  "-pix_fmt", "rgb24", "-"], capture_output=True, check=True, timeout=10).stdout
            values.append(sum(raw) / len(raw) / 255)
        x = torch.tensor(values)
        return weighted_score({"VQ": x, "MQ": x * .5, "TA": x * .25},
                              {"VQ": .2, "MQ": .1, "TA": .7}, space="raw")


class ReFLScorer(Scorer):
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(1, 1, bias=False).requires_grad_(False)
        self.model.weight.data.fill_(1)
        self.weights = {"VQ": .2, "MQ": .1, "TA": .7}
        self.normalization = {key: FixedNormalization(0., 1., "test") for key in self.weights}

    def forward(self, inputs):
        inputs.validate()
        x = self.model(((inputs.frames.float() + 1) * .5).flatten(1).mean(1, keepdim=True)).flatten()
        return weighted_score({"VQ": x, "MQ": x * .5, "TA": x * .25}, self.weights,
                              space="fixed_normalized", normalization=self.normalization)

    def score_paths(self, paths, prompts):
        raw = super().score_paths(paths, prompts).raw
        return weighted_score(raw, self.weights, space="fixed_normalized", normalization=self.normalization)


class PairwiseScorer(Scorer):
    def score_paths(self, paths, prompts):
        raw = super().score_paths(paths, prompts).raw
        return weighted_score(raw, {"VQ": 1., "MQ": 0., "TA": 0.}, space="fixed_normalized",
                              normalization={key: FixedNormalization(0., 1., "test") for key in raw})


def components(c, device):
    scorer = (ReFLScorer() if c.algorithm.name == "refl" else
              PairwiseScorer() if "pairwise" in c.algorithm.options else Scorer())
    return hunyuan.Components(HunyuanVideoAdapter(Transformer(), VAE()).to(device).eval(), scorer)


def config_file(tmp_path, algorithm="diffusionart"):
    from safetensors.torch import save_file

    records = []
    for i in range(2):
        text = f"scene {i}"
        tensor = tmp_path / f"embedding-{i}.safetensors"
        save_file({"prompt_embeds": torch.ones(5, 4) * i, "attention_mask": torch.ones(4)}, str(tensor),
                  metadata={"prompt_sha256": hashlib.sha256(text.encode()).hexdigest(), "native_revision": NATIVE_REVISION})
        records.append({"sample_id": str(i), "prompt": text, "metadata": {
            "hunyuan_conditioning": {"path": tensor.name, "sha256": file_sha256(tensor)}}})
    manifest = tmp_path / "prompts.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in records))
    c = {"name": "tiny-hunyuan-art", "task": "t2v", "seed": 42,
         "algorithm": {"name": "diffusionart", "options": {"covariance": {}, "loss": {}, "restore_zero_signal": True}},
         "model": {"name": "hunyuan_public", "checkpoint": "injected", "revision": "test", "options": {
             "native_root": "injected", "finetune": {"mode": "full"}, "gradient_checkpointing": False,
             "ffmpeg": "ffmpeg", "ffprobe": "ffprobe", "allow_tf32": False}},
         "reward": {"name": "video_reward", "checkpoint": "injected", "revision": "test", "normalization": "raw",
                    "weights": {"VQ": .2, "MQ": .1, "TA": .7}, "options": {"backbone": "injected"}},
         "data": {"manifest": str(manifest), "sha256": file_sha256(manifest), "shuffle": True},
         "sampling": {"steps": 4, "prompts_per_rollout": 1, "samples_per_prompt": 2, "microbatch": 2,
                      "height": 16, "width": 16, "frames": 5, "fps": 8, "guidance_scale": 1,
                      "options": {"shift": 5., "train_timestep_count": 2, "train_timestep_pool": [0, 1, 2]}},
         "training": {"max_updates": 4, "microbatch": 1, "learning_rate": .001, "precision": "bf16"},
         "evaluation": {"every_updates": 2, "sampling_steps": 5, "fps": 8, "reward_fps": 4,
                        "manifest": str(manifest), "manifest_sha256": file_sha256(manifest), "seed": 312,
                        "height": 32, "width": 16},
         "runtime": {"output_dir": str(tmp_path / "run"), "device": "cpu", "distributed": "none", "checkpoint_every": 2}}
    if algorithm == "pairwise":
        c["algorithm"]["options"]["pairwise"] = {"seed": 42}
        c["reward"].update(normalization="fixed", weights={"VQ": 1., "MQ": 0., "TA": 0.})
    elif algorithm == "refl":
        c["algorithm"] = {"name": "refl", "options": {"reward_loss_scale": 1.0}}
        c["sampling"]["samples_per_prompt"] = 1
        c["sampling"]["options"].update(train_timestep_pool=[3], train_timestep_count=1)
        c["reward"]["normalization"] = "fixed"
    elif algorithm == "diffusionnft":
        c["algorithm"] = {"name": algorithm, "options": {
            "old_policy": {"mode": "slow-ema", "rate": .1, "maximum": .6},
            "advantage_epsilon": 1e-4, "loss": {"nft_beta": 1.0, "adv_clip_max": 5.0, "adv_mode": "all"}}}
        c["sampling"]["options"]["train_timestep_pool"] = [0, 1, 2, 3]
    elif algorithm != "diffusionart":
        c["algorithm"] = {"name": algorithm, "options": {
            "advantage_epsilon": 1e-4 if algorithm == "flow_grpo_fast" else 1e-8,
            "loss": {"ratio_clip": 1e-5 if algorithm == "flow_grpo_fast" else 1e-4, "advantage_clip": 5.0}}}
        c["sampling"]["options"]["noise_level"] = .8 if algorithm == "flow_grpo_fast" else .25
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(c))
    return path


def test_video_eval_shift_and_explicit_cadence_are_independent_of_train(tmp_path, monkeypatch):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("video integration requires FFmpeg and ffprobe")
    torch.set_num_threads(1)
    c = load_config(config_file(tmp_path))
    c = replace(c, evaluation=replace(c.evaluation, shift=7., every_updates=0, at_updates=(2,)),
                runtime=replace(c.runtime, checkpoint_every=0, checkpoint_at_updates=(2,)))
    endpoint = hunyuan.evaluation_endpoint
    observed = []

    def evaluate(predict, noise, sigmas):
        torch.testing.assert_close(sigmas, shifted_schedule(5, 7), rtol=0, atol=0)
        observed.append(sigmas.clone())
        return endpoint(predict, noise, sigmas)

    monkeypatch.setattr(hunyuan, "evaluation_endpoint", evaluate)
    monkeypatch.setattr(hunyuan, "load_components", components)
    hunyuan.validate_config(c)
    hunyuan.run(c)
    assert c.sampling.options["shift"] == 5
    assert len(observed) == 6  # two prompts at steps 0/2/4
    assert len(list((tmp_path / "run").glob("evaluation/*/*/manifest.json"))) == 3


def test_published_video_configs_select_their_own_algorithm_contract(monkeypatch):
    for name in ("HUNYUAN_MODEL", "HUNYUAN_REVISION", "DANCEGRPO_ROOT", "VIDEO_REWARD_MODEL",
                 "VIDEO_REWARD_REVISION", "VIDEO_REWARD_BACKBONE", "HUNYUAN_TRAIN_MANIFEST",
                 "HUNYUAN_EVAL_MANIFEST"):
        monkeypatch.setenv(name, "configured-asset")
    for name in ("HUNYUAN_TRAIN_SHA256", "HUNYUAN_EVAL_SHA256"):
        monkeypatch.setenv(name, "a" * 64)
    configs = Path(__file__).resolve().parents[1] / "configs/examples"
    for suffix, algorithm, positions in (("art", "diffusionart", 9),
                                          ("fast", "flow_grpo_fast", 2),
                                          ("dance", "dancegrpo", 9),
                                          ("nft", "diffusionnft", 9),
                                          ("nft_onpolicy", "diffusionnft", 9),
                                          ("refl", "refl", 1)):
        config = load_config(configs / f"hunyuan_{suffix}.yaml")
        hunyuan.validate_config(config)
        assert config.algorithm.name == algorithm
        assert config.sampling.options["train_timestep_count"] == positions
        assert config.sampling.samples_per_prompt == (1 if suffix == "refl" else 8)
        assert config.runtime.distributed == "fsdp"
        assert ("covariance" in config.algorithm.options) == (suffix == "art")
        assert ("noise_level" in config.sampling.options) == (suffix in ("fast", "dance"))
        if algorithm == "diffusionnft":
            assert config.algorithm.options["old_policy"]["mode"] == ("no-ema" if suffix == "nft_onpolicy" else "slow-ema")
            assert config.sampling.options["train_timestep_pool"] == list(range(16))
        if algorithm == "refl":
            assert config.sampling.options["train_timestep_pool"] == [15]
            assert config.reward.weights == {"VQ": .5, "MQ": .5, "TA": 0.}


@pytest.mark.parametrize("algorithm", ["diffusionart", "flow_grpo_fast", "dancegrpo", "diffusionnft", "refl", "pairwise"])
def test_native_video_cli_encoding_eval_and_exact_resume(tmp_path, monkeypatch, algorithm):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("native video integration requires FFmpeg and ffprobe")
    torch.set_num_threads(1)
    path = config_file(tmp_path, algorithm)
    monkeypatch.setattr(hunyuan, "load_components", components)
    assert main(["--config", str(path)]) == 0
    root = tmp_path / "run"
    checkpoint = json.loads((root / "checkpoints/latest.json").read_text())
    expected = torch.load(root / "checkpoints" / checkpoint["file"], weights_only=True)
    manifests = list(root.glob("evaluation/*/*/manifest.json"))
    assert len(manifests) == 3
    for file in manifests:
        manifest = EvaluationManifest.from_json(file.read_text())
        assert audit_manifest(manifest, media_root=root).complete
        assert [s.seed for s in manifest.samples] == [312, 313]
        assert all(s.fps == 8 and s.framecount == 5 for s in manifest.samples)
        assert (file.parent / "index.html").exists()
    status = json.loads((root / "status.json").read_text())
    assert status["optimizer_steps"] == 4
    assert status["completed_rollouts"] == (4 if algorithm in ("diffusionnft", "refl") else 2)
    metrics = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
    assert [row["train/optimizer_step"] for row in metrics] == [1, 2, 3, 4]
    assert all(row["cost/elapsed_gpu_hours"] == 0 for row in metrics)
    if algorithm == "diffusionnft":
        assert [row["nft/old_policy_decay"] for row in metrics] == pytest.approx([0., .1, .2, .3])
        assert "nft_old_policy" in expected["stateful"]
    if algorithm == "refl":
        assert not any(key.startswith("train/refl/update_") for row in metrics for key in row)
        assert all(row["train/loss"] == pytest.approx(row["train/refl/loss"]) for row in metrics)
        assert all(row["train/grad_norm/before_clip"] > 0 for row in metrics)
    save = hunyuan.save_checkpoint

    def stop(*args, **kwargs):
        value = save(*args, **kwargs)
        if kwargs["progress"].optimizer_steps == 2:
            raise RuntimeError("test checkpoint interruption")
        return value

    monkeypatch.setattr(hunyuan, "save_checkpoint", stop)
    interrupted = tmp_path / "interrupted"
    with pytest.raises(RuntimeError, match="test checkpoint"):
        main(["--config", str(path), "--output-dir", str(interrupted)])
    monkeypatch.setattr(hunyuan, "save_checkpoint", save)
    receipt = json.loads((interrupted / "checkpoints/latest.json").read_text())["receipt"]
    assert main(["--config", str(path), "--output-dir", str(interrupted), "--resume", str(interrupted / "checkpoints" / receipt)]) == 0
    actual_path = json.loads((interrupted / "checkpoints/latest.json").read_text())["file"]
    actual = torch.load(interrupted / "checkpoints" / actual_path, weights_only=True)
    from test_distributed_checkpoint import assert_tree_equal

    assert_tree_equal(expected["model"], actual["model"])
    assert_tree_equal(expected["optimizer"], actual["optimizer"])
    assert_tree_equal(expected["stateful"], actual["stateful"])


def test_cached_conditioning_identity_and_pooled_token(tmp_path):
    from visual_rl.data import PromptDataset

    path = config_file(tmp_path)
    c = load_config(path)
    dataset = PromptDataset.load(c.data.manifest, expected_sha256=c.data.sha256, task="t2v")
    embeddings, masks = cached_conditioning(dataset.rows, c.data.manifest)
    assert embeddings.shape == (2, 5, 4) and masks.shape == (2, 4)
    from dataclasses import replace

    with pytest.raises(ValueError, match="identity"):
        cached_conditioning([replace(dataset.rows[0], text="different")], c.data.manifest)
    (tmp_path / "embedding-0.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        cached_conditioning(dataset.rows, c.data.manifest)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_native_rgb_quantization_matches_video_processor(dtype):
    import numpy as np
    from diffusers.video_processor import VideoProcessor

    frames = torch.linspace(-1, 1, 96).reshape(2, 3, 4, 4).to(dtype)
    reference = VideoProcessor(vae_scale_factor=8).postprocess_video(
        frames.permute(1, 0, 2, 3)[None], output_type="np")[0]
    expected = (reference * 255).astype(np.uint8)
    np.testing.assert_array_equal(quantize_rgb(frames).numpy(), expected)
    if dtype == torch.bfloat16:
        assert quantize_rgb(torch.full((1, 3, 2, 2), -.2, dtype=dtype)).flatten()[0] == 102


def test_native_hunyuan_rejects_fp32_before_loading(tmp_path):
    path = config_file(tmp_path)
    values = yaml.safe_load(path.read_text())
    values["training"]["precision"] = "fp32"
    path.write_text(yaml.safe_dump(values))
    assert main(["--config", str(path), "--validate"]) == 2
