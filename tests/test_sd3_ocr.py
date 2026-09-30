"""Native SD3 loop with the OCR factory; Paddle inference itself is injected."""

import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from test_paddle_loader import bundle as bundle
from test_sd3_runner import config_file, tiny_parts

from visual_rl.backends import sd3
from visual_rl.config import ConfigError, load_config
from visual_rl.rewards.image_factory import load_image_reward
from visual_rl.rewards.paddle_loader import VERSIONS, _load_native_engine
from visual_rl.train import main


def setup_config(tmp_path, bundle, *, valid=True):
    path = config_file(tmp_path)
    c = yaml.safe_load(path.read_text())
    manifest = Path(c["data"]["manifest"])
    manifest.write_text(
        "".join(
            json.dumps(
                {"sample_id": str(i), "prompt": f'A sign "{i + 2}"' if valid else "no quotes"}
            )
            + "\n"
            for i in range(4)
        )
    )
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    c["data"]["sha256"] = digest
    c["evaluation"]["manifest_sha256"] = digest
    c["reward"] = {
        "name": "ocr",
        "checkpoint": str(bundle[0]),
        "revision": bundle[1],
        "normalization": "raw",
        "weights": {"ocr": 1.0},
        "options": {"microbatch": 2, "error_policy": "zero"},
    }
    return path, c


@pytest.mark.parametrize("algorithm", ["diffusionart", "diffusionnft"])
def test_ocr_rewards_reach_updates_eval_and_full_checkpoints(
    tmp_path, monkeypatch, bundle, algorithm
):
    pytest.importorskip("rapidfuzz")
    pytest.importorskip("diffusers")
    path, c = setup_config(tmp_path, bundle)
    if algorithm == "diffusionnft":
        c["algorithm"] = {
            "name": algorithm,
            "options": {
                "loss": {"nft_beta": 1.0},
                "old_policy": {"rate": 0.001, "maximum": 0.5, "delay": 0},
                "evaluation_ema": {"enabled": True, "decay": 0.9},
            },
        }
        c["sampling"]["options"] = {
            "shift": 1.0,
            "solver": "dpm2",
            "train_timestep_count": 2,
            "timestep_selection": "random_per_row",
        }
    path.write_text(yaml.safe_dump(c))
    seen = []
    closed = []

    def recognize(image, *, cls):
        assert image.dtype.name == "uint8" and image.shape == (4, 4, 3) and cls is False
        seen.append(image.copy())
        # Deterministic image-dependent responses; this is not a real OCR model.
        text = str(2 + int(image[0, 0, 0]) % 4)
        return [[[[0, 0], (text, 0.99)]]]

    original_version = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: VERSIONS[name] if name in VERSIONS else original_version(name),
    )
    monkeypatch.setitem(
        sys.modules,
        "paddleocr",
        SimpleNamespace(
            PaddleOCR=lambda **kw: SimpleNamespace(ocr=recognize, close=lambda: closed.append(True))
        ),
    )

    monkeypatch.setattr("visual_rl.rewards.ocr_process.PaddleWorker", _load_native_engine)

    def components(config):
        parts = tiny_parts(config)

        class Text:
            def encode(self, prompts, **kwargs):
                values = [prompt.split('"')[1] for prompt in prompts]
                return parts.text.encode(values, **kwargs)

        return sd3.Components(parts.adapter, Text(), load_image_reward(config.reward, device="cpu"))

    monkeypatch.setattr(sd3, "load_components", components)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        assert main(["--config", str(path)]) == 0
    finally:
        torch.set_num_threads(old_threads)
    output = tmp_path / "run"
    history = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert len(history) == 4 and seen
    assert closed == [True]
    for row in history:
        assert 0 <= row["train/reward"] <= 1
        assert row["train/reward"] == row["train/reward/raw/ocr"]
    evaluations = list(output.glob("evaluation/step-*/*/summary.json"))
    assert len(evaluations) == 3
    for summary in evaluations:
        values = json.loads(summary.read_text())
        assert values["eval/reward"] == values["eval/reward/raw/ocr"]
    assert (output / "checkpoints/latest.json").is_file()
    assert json.loads((output / "status.json").read_text())["optimizer_steps"] == 4


def test_bad_ocr_prompt_fails_before_output_creation_and_model_load(tmp_path, monkeypatch, bundle):
    pytest.importorskip("rapidfuzz")
    path, c = setup_config(tmp_path, bundle, valid=False)
    path.write_text(yaml.safe_dump(c))

    def should_not_load(*args):
        pytest.fail("invalid OCR prompts must be rejected before loading models")

    monkeypatch.setattr(sd3, "load_components", should_not_load)
    with pytest.raises(ConfigError, match="quoted target"):
        sd3.run(load_config(path))
    assert not (tmp_path / "run").exists()


def test_bad_eval_target_is_rejected_even_with_valid_training_data(tmp_path, monkeypatch, bundle):
    pytest.importorskip("rapidfuzz")
    path, c = setup_config(tmp_path, bundle)
    validation = tmp_path / "bad-eval.jsonl"
    validation.write_text(json.dumps({"sample_id": "eval", "prompt": 'A sign "  "'}) + "\n")
    c["evaluation"]["manifest"] = str(validation)
    c["evaluation"]["manifest_sha256"] = hashlib.sha256(validation.read_bytes()).hexdigest()
    path.write_text(yaml.safe_dump(c))
    monkeypatch.setattr(sd3, "load_components", lambda *a: pytest.fail("must reject before assets"))
    with pytest.raises(ConfigError, match="quoted target"):
        sd3.run(load_config(path))
    assert not (tmp_path / "run").exists()


def test_ocr_combines_with_neural_raw_scores_without_normalizing_twice():
    pytest.importorskip("rapidfuzz")
    from test_composite_reward import Scorer

    from visual_rl.rewards.composite import CompositeImageReward
    from visual_rl.rewards.ocr import OCRScore
    from visual_rl.rewards.scalar import FixedNormalization

    engine = SimpleNamespace(ocr=lambda *a, **k: [[[[0, 0], ("abc", 1.0)]]])
    scorer = CompositeImageReward(
        {"pickscore": Scorer("pickscore", 26), "ocr": OCRScore(engine, error_policy="raise")},
        {"pickscore": 2.0, "ocr": 0.5},
        space="fixed_normalized",
        normalization={
            "pickscore": FixedNormalization(0, 26, "fixture"),
            "ocr": FixedNormalization(0, 1, "identity"),
        },
    )
    images = torch.tensor([0.1, 0.8])[:, None, None, None].expand(-1, 3, 4, 4)
    result = scorer(images, ['"abc"', '"abd"'], microbatch=1)
    torch.testing.assert_close(result.raw["ocr"], torch.tensor([1.0, 2.0 / 3]))
    torch.testing.assert_close(result.weighted, torch.tensor([0.2 + 0.5, 1.6 + 1.0 / 3]))
