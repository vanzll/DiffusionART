import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_hunyuan_runner as video_test
import torch
import torch.multiprocessing as mp
import yaml
from PIL import Image
from torch import nn

from visual_rl.backends import wan
from visual_rl.config import load_config
from visual_rl.evaluation import EvaluationManifest, audit_manifest
from visual_rl.evaluation.manifest import file_sha256
from visual_rl.models.wan import (
    WanI2VAdapter,
    WanI2VConditioning,
    WanI2VConfig,
    first_frame_condition,
)
from visual_rl.registry import get_backend
from visual_rl.rewards.scalar import weighted_score

_hunyuan_config = video_test.config_file


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(.1))
        self.config = SimpleNamespace(in_channels=36, out_channels=16, image_dim=4)

    def forward(self, hidden_states, timestep, encoder_hidden_states, encoder_hidden_states_image, **kwargs):
        assert timestep.is_floating_point()
        assert hidden_states.shape[1] == 36
        assert len(encoder_hidden_states) == len(hidden_states)
        offset = encoder_hidden_states.flatten(1).mean(1).reshape(-1, 1, 1, 1, 1)
        return (hidden_states[:, :16] * self.scale + offset * .001,)


class Scorer(video_test.Scorer):
    def score_paths(self, paths, prompts):
        return weighted_score(super().score_paths(paths, prompts).raw,
                              {"VQ": .5, "MQ": .5, "TA": 0.}, space="raw")


class ReFLScorer(video_test.ReFLScorer):
    def __init__(self):
        super().__init__()
        self.weights = {"VQ": .5, "MQ": .5, "TA": 0.}


def components(c, device):
    vae = video_test.VAE()
    vae.config = SimpleNamespace(z_dim=16, latents_mean=(.1,) * 16, latents_std=(.5,) * 16)
    adapter = WanI2VAdapter(Transformer(), vae, WanI2VConfig((.1,) * 16, (.5,) * 16, cfg_scale=None)).to(device).eval()
    scorer = ReFLScorer() if c.algorithm.name == "refl" else Scorer()
    decoder = (lambda raw: adapter.vae.decode(raw, return_dict=False)[0]) if c.algorithm.name == "refl" else None
    return wan.Components(adapter, scorer, None, None, decoder)


def reference(flux, prompt, *, height, width, device, seed):
    return Image.new("RGB", (width, height), (seed % 256, 40, 80))


def condition(pipe, prompt, image, *, height, width, frames, device, seed, cfg):
    assert not cfg
    value = sum(image.getpixel((0, 0))) / 255
    image_latents = torch.full((1, 16, (frames - 1) // 4 + 1, height // 8, width // 8), value)
    return WanI2VConditioning(torch.full((1, 4, 4), value), torch.ones(1, 3, 4), first_frame_condition(image_latents))


def install_components():
    from visual_rl.models import wan_assets

    wan.load_components = components
    wan_assets.generate_reference = reference
    wan_assets.condition_image = condition


def config_file(tmp_path, algorithm="diffusionart"):
    path = _hunyuan_config(tmp_path, algorithm)
    c = yaml.safe_load(path.read_text())
    c.update(name="tiny-wan", task="i2v")
    c["model"]["name"] = "wan_i2v_distilled"
    c["model"]["options"].update({key: "injected" for key in wan.ASSET_KEYS})
    c["reward"]["weights"] = {"VQ": .5, "MQ": .5, "TA": 0.}
    count = 4 if algorithm == "diffusionnft" else 1 if algorithm in ("flow_grpo_fast", "refl") else 3
    c["sampling"]["options"].update(train_timestep_count=count,
        train_timestep_pool=[3] if algorithm == "refl" else list(range(4 if algorithm == "diffusionnft" else 3)))
    evaluation = tmp_path / "wan-eval.jsonl"
    rows = []
    for i in range(2):
        image = tmp_path / f"reference-{i}.png"
        Image.new("RGB", (16, 16), (i * 50, 80, 30)).save(image)
        rows.append({"sample_id": str(i), "prompt": f"scene {i}", "image": image.name, "image_sha256": file_sha256(image)})
    evaluation.write_text("".join(json.dumps(row) + "\n" for row in rows))
    c["evaluation"].update(sampling_steps=4, height=16, width=16,
                           manifest=str(evaluation), manifest_sha256=file_sha256(evaluation))
    path.write_text(yaml.safe_dump(c))
    return path


@pytest.mark.parametrize("algorithm", ["diffusionart", "flow_grpo_fast", "dancegrpo", "diffusionnft", "refl"])
def test_wan_cli_media_eval_and_exact_resume(tmp_path, monkeypatch, algorithm):
    from visual_rl.models import wan_assets

    monkeypatch.setattr(wan, "load_components", components)
    monkeypatch.setattr(wan_assets, "generate_reference", reference)
    monkeypatch.setattr(wan_assets, "condition_image", condition)
    monkeypatch.setattr(video_test, "config_file", config_file)
    video_test.test_native_video_cli_encoding_eval_and_exact_resume(tmp_path, monkeypatch, algorithm)


@pytest.mark.parametrize("algorithm", ["diffusionart", "flow_grpo_fast", "dancegrpo", "diffusionnft", "refl"])
@pytest.mark.parametrize("shift", [None, 5, 7])
def test_wan_eval_shift_contract(tmp_path, algorithm, shift):
    from visual_rl.config import ConfigError

    path = config_file(tmp_path, algorithm)
    values = yaml.safe_load(path.read_text())
    values["evaluation"]["shift"] = shift
    path.write_text(yaml.safe_dump(values))
    c = load_config(path)
    if shift == 7:
        with pytest.raises(ConfigError, match="shift five"):
            wan.validate_config(c)
    else:
        wan.validate_config(c)


def test_example_configs(monkeypatch):
    for name in ("WAN_DISTILLED_TRANSFORMER", "WAN_DISTILLED_REVISION", "WAN_BASE_MODEL", "WAN_BASE_REVISION",
                 "FLUX_REFERENCE_MODEL", "FLUX_REFERENCE_REVISION", "DANCEGRPO_ROOT", "VIDEO_REWARD_MODEL",
                 "VIDEO_REWARD_REVISION", "VIDEO_REWARD_BACKBONE", "WAN_TRAIN_MANIFEST", "WAN_EVAL_MANIFEST"):
        monkeypatch.setenv(name, "configured-asset")
    for name in ("WAN_TRAIN_SHA256", "WAN_EVAL_SHA256"):
        monkeypatch.setenv(name, "a" * 64)
    for suffix, algorithm in (("art", "diffusionart"), ("fast", "flow_grpo_fast"), ("dance", "dancegrpo"), ("nft", "diffusionnft"), ("refl", "refl")):
        c = load_config(Path(__file__).resolve().parents[1] / f"configs/examples/wan_{suffix}.yaml")
        get_backend(c)
        assert c.algorithm.name == algorithm
        assert c.sampling.steps == 4 and c.sampling.samples_per_prompt == (1 if suffix == "refl" else 8)
        assert c.reward.weights == {"VQ": .5, "MQ": .5, "TA": 0.}


@pytest.mark.parametrize("algorithm", ["diffusionart", "flow_grpo_fast", "dancegrpo", "diffusionnft", "refl"])
def test_wan_fsdp_eval_and_exact_resume(tmp_path, algorithm):
    from test_hunyuan_fsdp import worker

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg/ffprobe required")
    path = config_file(tmp_path, algorithm)
    c = yaml.safe_load(path.read_text())
    c["runtime"]["distributed"] = "fsdp"
    c["sampling"]["prompts_per_rollout"] = 2
    validation = Path(c["evaluation"]["manifest"])
    rows = [json.loads(line) for line in validation.read_text().splitlines()]
    rows.append({**rows[0], "sample_id": "third"})
    validation.write_text("".join(json.dumps(row) + "\n" for row in rows))
    c["evaluation"]["manifest_sha256"] = file_sha256(validation)
    path.write_text(yaml.safe_dump(c))
    context = mp.spawn(worker, args=(str(path), str(tmp_path), "wan"), nprocs=2, join=False)
    deadline = time.monotonic() + 120
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("Wan FSDP integration exceeded 120s")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    for rank in range(2):
        assert json.loads((tmp_path / f"rank-{rank}-passed.json").read_text())["passed"]
    for name in ("run", "interrupted"):
        root = tmp_path / name
        for file in root.glob("evaluation/*/*/manifest.json"):
            manifest = EvaluationManifest.from_json(file.read_text())
            assert len(manifest.samples) == 3
            assert audit_manifest(manifest, media_root=root).complete
        metrics = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
        assert [row["train/optimizer_step"] for row in metrics] == [1, 2, 3, 4]
