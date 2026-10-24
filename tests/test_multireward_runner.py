"""Four component scores reach actual small-model ART updates and evaluation."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from test_geneval_runner import JPEGEngine, geneval_config
from test_sd3_runner import tiny_parts

from visual_rl.backends import sd3
from visual_rl.config import load_config
from visual_rl.rewards.composite import CompositeImageReward
from visual_rl.rewards.geneval import GenEvalScore
from visual_rl.rewards.hpsv2 import HPSv2Score
from visual_rl.rewards.scalar import FixedNormalization, weighted_score


class PixelLogits(torch.nn.Module):
    def forward(self, images, prompts, *, microbatch):
        if not isinstance(images, torch.Tensor):
            images = torch.stack([torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255
                                  for im in images])
        return weighted_score({"clipscore": images.flatten(1).mean(1) * 30},
                              {"clipscore": 1}, space="raw")


class FeatureModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(3))

    def forward(self, images, tokens):
        return {
            "image_features": torch.nn.functional.normalize(images.mean((2, 3)), dim=-1),
            "text_features": torch.nn.functional.normalize(tokens.float() * self.weight, dim=-1),
        }


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_four_reward_art_native_updates_and_eval(tmp_path, monkeypatch):
    pytest.importorskip("diffusers")
    path, config, rows = geneval_config(tmp_path)
    geneval_spec = {k: config["reward"][k] for k in ("checkpoint", "revision", "options")}
    pick_spec = {
        "checkpoint": "fixture", "revision": "fixture",
        "options": {"processor": "fixture", "processor_revision": "fixture", "microbatch": 1},
    }
    names = ("geneval", "pickscore", "clipscore", "hpsv2")
    divisors = dict(zip(names, (1, 26, 30, 1), strict=True))
    config["reward"] = {
        "name": "composite", "normalization": "fixed", "weights": dict.fromkeys(names, 1),
        "options": {
            "microbatch": 2,
            "fixed_statistics": {n: {"mean": 0, "std": d, "source": "fixture"}
                                 for n, d in divisors.items()},
            "components": {
                "geneval": geneval_spec, "pickscore": pick_spec,
                "clipscore": {**pick_spec, "options": {**pick_spec["options"], "antialias": True}},
                "hpsv2": {"checkpoint": "fixture.pt", "revision": "a" * 40,
                          "options": {"microbatch": 1, "checkpoint_sha256": "b" * 64}},
            },
        },
    }
    config["algorithm"]["options"]["multi_reward_advantage"] = {
        "weights": dict.fromkeys(names, 0.25), "score_space": "fixed_normalized",
        "eps": 0.0001, "renormalize_combined": False,
    }
    path.write_text(yaml.safe_dump(config))

    def load(candidate):
        parts = tiny_parts(candidate)
        hps = HPSv2Score(
            FeatureModel(),
            lambda image: torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255,
            lambda prompts: torch.tensor([[int(p), 1, 2] for p in prompts]),
        )
        parts.scorer = CompositeImageReward(
            {"geneval": GenEvalScore(JPEGEngine(rows), device="cpu"),
             "pickscore": parts.scorer, "clipscore": PixelLogits(), "hpsv2": hps},
            dict.fromkeys(names, 1), space="fixed_normalized",
            normalization={n: FixedNormalization(0, d, "fixture") for n, d in divisors.items()},
        )
        return parts

    monkeypatch.setattr(sd3, "load_components", load)
    candidate = load_config(path)
    sd3.run(candidate)
    root = Path(candidate.runtime.output_dir)
    assert json.loads((root / "status.json").read_text())["optimizer_steps"] == 4
    logs = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
    updates = [row for row in logs if "train/optimizer_step" in row]
    assert [row["train/optimizer_step"] for row in updates] == [1, 2, 3, 4]
    for row in updates:
        components = [row[f"train/reward/raw/{name}"] / divisor
                      for name, divisor in divisors.items()]
        expected = sum(components)
        # Runtime averages FP32 per-sample sums; these logged component means
        # are separately reduced. Bound absolute rounding by component scale,
        # rather than a tiny cancelled total (or a changed runtime reduction).
        rounding = 8 * torch.finfo(torch.float32).eps * sum(abs(v) for v in components)
        assert row["train/reward"] == pytest.approx(expected, rel=0, abs=rounding)
    assert len(list(root.glob("evaluation/*/*/manifest.json"))) == 3
