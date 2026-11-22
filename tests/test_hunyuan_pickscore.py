import json
import shutil

import pytest
import torch
import yaml
from test_hunyuan_runner import VAE, Transformer, config_file
from test_pickscore import Model, Processor
from torch import nn

from visual_rl.backends import hunyuan
from visual_rl.config import ConfigError, load_config
from visual_rl.evaluation.manifest import EvaluationManifest, audit_manifest
from visual_rl.models.hunyuan import HunyuanVideoAdapter
from visual_rl.registry import get_backend
from visual_rl.rewards.pickscore import PickScore
from visual_rl.train import main


def pickscore_config(tmp_path, algorithm="diffusionart"):
    path = config_file(tmp_path, algorithm)
    config = yaml.safe_load(path.read_text())
    config["reward"] = {
        "name": "pickscore", "checkpoint": "pickscore-path", "revision": "model-pin",
        "normalization": "fixed", "weights": {"pickscore": 1.0},
        "options": {"processor": "processor-path", "processor_revision": "processor-pin", "microbatch": 8},
    }
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.mark.parametrize("algorithm", ["refl", "dancegrpo", "pairwise"])
def test_unsupported_algorithms_reject_first_frame_pickscore(tmp_path, algorithm):
    with pytest.raises(ConfigError, match="PickScore"):
        get_backend(load_config(pickscore_config(tmp_path, algorithm)))


@pytest.mark.parametrize("field,change", [
    ("normalization", "raw"),
    ("weights", {"pickscore": 0.5}),
    ("options", {"processor": "processor-path", "processor_revision": "processor-pin", "microbatch": 8,
                 "local_files_only": False}),
    ("options", {"processor": "processor-path", "microbatch": 8}),
])
def test_invalid_pickscore_settings_fail_in_cli(tmp_path, field, change):
    path = pickscore_config(tmp_path)
    config = yaml.safe_load(path.read_text())
    config["reward"][field] = change
    path.write_text(yaml.safe_dump(config))
    assert main(["--config", str(path), "--validate"]) == 2


@pytest.mark.parametrize("evaluation", [False, True])
def test_first_decoded_frame_before_lossy_encoding(tmp_path, monkeypatch, evaluation):
    c = load_config(pickscore_config(tmp_path))
    height = c.evaluation.height if evaluation else c.sampling.height
    width = c.evaluation.width if evaluation else c.sampling.width

    class FixedVAE(nn.Module):
        def decode(self, raw, *, return_dict):
            assert raw.dtype == torch.float32 and not torch.is_autocast_enabled("cpu")
            video = torch.ones(1, 3, c.sampling.frames, height, width)
            # The other frames are white. This first-frame color distinguishes
            # truncation from rounding, selection of the middle frame and RGB order.
            video[0, :, 0] = torch.tensor([0.0, -0.5, 0.5])[:, None, None]
            return (video,)

    def encode(frames, target, **options):
        assert evaluation and len(frames) == c.sampling.frames
        target.write_bytes(b"media stub with no decodable frames")
        return target

    monkeypatch.setattr(hunyuan, "encode_video", encode)
    processor = Processor()
    scorer = PickScore(Model(), processor, space="fixed_normalized")
    parts = hunyuan.Components(HunyuanVideoAdapter(Transformer(), FixedVAE()), scorer)
    endpoints = torch.zeros(hunyuan._shape(c, 1, evaluation=evaluation), dtype=torch.bfloat16)
    paths, scores = hunyuan.score_videos(c, parts, endpoints, ["2"], tmp_path / "samples",
                                         torch.device("cpu"), evaluation=evaluation)
    image = processor.calls[0]["images"][0]
    assert image.mode == "RGB" and image.size == (width, height)
    assert image.getpixel((0, 0)) == (127, 63, 191)
    assert paths == ([tmp_path / "samples" / "000000.mp4"] if evaluation else [])
    cosine = torch.nn.functional.cosine_similarity(torch.tensor([[127., 63., 191.]]), torch.tensor([[2., 1., 1.]]))
    torch.testing.assert_close(scores.raw["pickscore"], torch.tensor(2.).exp() * cosine)
    torch.testing.assert_close(scores.weighted, torch.tensor(2.).exp() / 26 * cosine)


@pytest.mark.parametrize("algorithm", ["diffusionart", "diffusionnft", "flow_grpo_fast"])
def test_native_pickscore_training_entrypoint(tmp_path, monkeypatch, algorithm):
    from visual_rl.models import hunyuan_assets

    class SceneProcessor(Processor):
        def __call__(self, **kwargs):
            if "text" in kwargs:
                kwargs["text"] = [prompt.split()[-1] for prompt in kwargs["text"]]
            return super().__call__(**kwargs)

    def load(**kwargs):
        assert kwargs == dict(model_path="pickscore-path", model_revision="model-pin",
                              processor_path="processor-path", processor_revision="processor-pin",
                              dtype=torch.float32, local_files_only=True, device="cpu", space="fixed_normalized")
        return PickScore(Model(), SceneProcessor(), space=kwargs["space"])

    monkeypatch.setattr(PickScore, "from_pretrained", load)
    monkeypatch.setattr(hunyuan, "load_adapter", lambda c, device: HunyuanVideoAdapter(Transformer(), VAE()))
    monkeypatch.setattr(hunyuan_assets, "native_checkout", lambda root: pytest.fail("VideoReward loaded"))
    path = pickscore_config(tmp_path, algorithm)
    config = yaml.safe_load(path.read_text())
    config["training"]["max_updates"] = 2
    evaluate = algorithm == "diffusionart" and shutil.which("ffmpeg") and shutil.which("ffprobe")
    if not evaluate:
        config["evaluation"] = {"every_updates": 0}
    path.write_text(yaml.safe_dump(config))
    assert main(["--config", str(path)]) == 0
    rows = [json.loads(line) for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    assert [row["train/optimizer_step"] for row in rows] == [1, 2]
    for row in rows:
        assert row["train/reward"] == pytest.approx(row["train/reward/raw/pickscore"] / 26)
        assert row["train/reward"] == row["train/reward/fixed_normalized/pickscore"]
    if evaluate:
        root = tmp_path / "run"
        manifests = list(root.glob("evaluation/*/*/manifest.json"))
        assert len(manifests) == 2
        for file in manifests:
            manifest = EvaluationManifest.from_json(file.read_text())
            assert audit_manifest(manifest, media_root=root).complete
            for sample in manifest.samples:
                assert sample.scores.weighted == pytest.approx(sample.scores.raw["pickscore"] / 26)
                assert sample.framecount == 5 and sample.fps == 8
