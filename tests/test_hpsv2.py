"""CPU HPSv2 contract checks with tiny injected models and no weight downloads."""

import builtins
import hashlib
import pickle
import sys
from contextlib import contextmanager
from types import ModuleType

import numpy as np
import pytest
import torch
from PIL import Image

from visual_rl.rewards.composite import CompositeImageReward
from visual_rl.rewards.hpsv2 import HPSv2Score
from visual_rl.rewards.scalar import FixedNormalization, ScoreResult

MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
REVISION = "697403c78157020a1ae59d23f111aa58ced35b0a"


class Preprocess:
    def __init__(self):
        self.images = []

    def __call__(self, image):
        self.images.append(image.copy())
        pixels = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255
        return pixels.mean(dim=(1, 2), keepdim=True)


class Tokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, prompts):
        self.calls.append(prompts)
        return torch.tensor([[int(prompt), 1] for prompt in prompts])


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(3))
        self.dropout = torch.nn.Dropout(0.9)
        self.calls = []
        self.strict = []

    def forward(self, pixels, tokens):
        self.calls.append((pixels.clone(), tokens.clone(), torch.is_autocast_enabled("cpu")))
        image = self.dropout(pixels.flatten(1) * self.weight)
        text = torch.stack((tokens[:, 0], tokens[:, 1], -tokens[:, 0]), dim=1).float()
        return {"image_features": torch.nn.functional.normalize(image, dim=-1),
                "text_features": torch.nn.functional.normalize(text, dim=-1),
                "logit_scale": torch.tensor(1000.0)}

    def load_state_dict(self, state_dict, strict=True):
        self.strict.append(strict)
        return super().load_state_dict(state_dict, strict=strict)


def make_scorer():
    return HPSv2Score(TinyModel(), Preprocess(), Tokenizer())


@pytest.mark.parametrize("microbatch", [1, 2, 10])
def test_historical_score_pairing_quantization_and_microbatches(microbatch):
    scorer = make_scorer()
    images = torch.rand(5, 3, 7, 11, generator=torch.Generator().manual_seed(11))
    images.requires_grad_()
    prompts = ["3", "1", "8", "2", "4"]
    torch.nn.Sequential(scorer).train()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = scorer(images, prompts, [{"id": i} for i in range(5)], microbatch=microbatch)
    pixels = (images.detach() * 255).round().clamp(0, 255).to(torch.uint8)
    image_features = torch.nn.functional.normalize(pixels.float().mean((2, 3)) / 255, dim=-1)
    text_features = torch.nn.functional.normalize(
        torch.tensor([[int(p), 1, -int(p)] for p in prompts]).float(), dim=-1
    )
    expected = (image_features * text_features).sum(-1)
    assert isinstance(result, ScoreResult)
    assert set(result.raw) == {"hpsv2"}
    torch.testing.assert_close(result.raw["hpsv2"], expected)
    torch.testing.assert_close(result.weighted, expected)
    assert result.space == "raw" and result.fixed_normalized is None
    assert result.normalization is None and result.weights == {"hpsv2": 1.0}
    assert result.weighted.device.type == "cpu" and result.weighted.dtype == torch.float32
    assert not result.weighted.requires_grad and images.grad is None
    assert not scorer.training and not scorer.model.training
    assert all(not p.requires_grad for p in scorer.parameters())
    assert all(not call[2] for call in scorer.model.calls)
    assert [p for batch in scorer.tokenizer.calls for p in batch] == prompts
    assert [len(batch) for batch in scorer.tokenizer.calls] == [
        len(prompts[i:i + microbatch]) for i in range(0, 5, microbatch)
    ]
    actual_pixels = torch.stack([
        torch.from_numpy(np.array(im)).permute(2, 0, 1) for im in scorer.preprocess.images
    ])
    torch.testing.assert_close(actual_pixels, pixels, rtol=0, atol=0)


def test_rounding_ties_and_rectangular_pil_inputs():
    scorer = make_scorer()
    images = torch.tensor([0.5, 1.5, 2.5, 3.5, 254.5, 255.0]).reshape(1, 3, 1, 2) / 255
    scorer(images, ["1"], microbatch=1)
    assert np.array(scorer.preprocess.images[0]).reshape(-1).tolist() == [0, 2, 254, 2, 4, 255]
    pil = [Image.new("RGB", (9, 3), (17, 127, 255)),
           Image.new("RGB", (3, 12), (233, 16, 83))]
    first = scorer(pil, ["2", "3"], microbatch=2)
    second = scorer(pil, ["2", "3"], microbatch=1)
    torch.testing.assert_close(first.weighted, second.weighted, rtol=0, atol=0)
    assert [im.size for im in scorer.preprocess.images[1:3]] == [(9, 3), (3, 12)]


def test_no_renormalization_logit_scale_or_early_fp32_cast(monkeypatch):
    scorer = make_scorer()
    image = torch.tensor([[2.001, 3.002, 4.004]], dtype=torch.float16)
    text = torch.tensor([[5.003, 6.002, 7.003]], dtype=torch.float16)
    monkeypatch.setattr(scorer.model, "forward", lambda *args: {
        "image_features": image, "text_features": text, "logit_scale": float("nan")
    })
    result = scorer(torch.ones(1, 3, 2, 2), ["1"], microbatch=1)
    torch.testing.assert_close(result.weighted, (image * text).sum(-1).float(), rtol=0, atol=0)
    assert result.weighted.item() > 1
    assert not torch.equal(result.weighted, (image.float() * text.float()).sum(-1))


def test_cuda_autocast_policy_without_gpu(monkeypatch):
    scorer = make_scorer()
    monkeypatch.setattr(torch.Tensor, "device", property(lambda self: torch.device("cuda:0")))
    original_to = torch.Tensor.to

    def host_only_to(self, *args, **kwargs):
        kwargs.pop("device", None)
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", host_only_to)
    contexts = []

    @contextmanager
    def autocast(**kwargs):
        contexts.append(kwargs)
        yield

    monkeypatch.setattr(torch, "autocast", autocast)
    monkeypatch.setattr(scorer.model, "forward", lambda *args: {
        "image_features": torch.tensor([[0.6, 0.8]]),
        "text_features": torch.tensor([[0.8, 0.6]])
    })
    scorer(torch.ones(1, 3, 2, 2), ["1"], microbatch=1)
    assert contexts == [{"device_type": "cuda", "enabled": True}]


def test_composite_owns_weights_normalization_and_microbatch():
    scorer = make_scorer()
    images = torch.rand(5, 3, 4, 7)
    prompts = ["1", "2", "3", "4", "5"]
    expected = scorer(images, prompts, microbatch=5).weighted
    scorer.tokenizer.calls.clear()
    composite = CompositeImageReward(
        {"hpsv2": scorer}, {"hpsv2": 2.0}, space="fixed_normalized",
        normalization={"hpsv2": FixedNormalization(0.1, 0.5, "test")},
        microbatches={"hpsv2": 2},
    )
    result = composite(images, prompts, microbatch=4, metadata=[None] * 5)
    torch.testing.assert_close(result.raw["hpsv2"], expected)
    torch.testing.assert_close(result.weighted, 2 * (expected - 0.1) / 0.5)
    assert [len(batch) for batch in scorer.tokenizer.calls] == [2, 2, 1]


@pytest.mark.parametrize("images", [
    None, [], ["bad"], [Image.new("L", (2, 2))], [Image.new("RGB", (0, 2))],
    torch.zeros(1, 2, 2, 2), torch.zeros(3, 2, 2), torch.zeros(0, 3, 2, 2),
    torch.zeros(1, 3, 0, 2), torch.ones(1, 3, 2, 2, dtype=torch.uint8),
    torch.full((1, 3, 2, 2), -0.1), torch.full((1, 3, 2, 2), 1.1),
    torch.full((1, 3, 2, 2), float("nan")), torch.full((1, 3, 2, 2), float("inf")),
])
def test_invalid_images(images):
    with pytest.raises(ValueError):
        make_scorer()(images, ["1"], microbatch=1)


@pytest.mark.parametrize("prompts", [None, "1", b"1", [], [""], [" "], [1], ["1", "2"]])
def test_invalid_prompts_and_lengths(prompts):
    with pytest.raises(ValueError):
        make_scorer()(torch.ones(1, 3, 2, 2), prompts, microbatch=1)


@pytest.mark.parametrize("microbatch", [0, -1, True, 1.5, None])
def test_invalid_microbatch(microbatch):
    with pytest.raises(ValueError, match="microbatch"):
        make_scorer()(torch.ones(1, 3, 2, 2), ["1"], microbatch=microbatch)


@pytest.mark.parametrize("metadata", [[], {}, "bad", [1], [None, None]])
def test_invalid_metadata(metadata):
    with pytest.raises(ValueError, match="metadata"):
        make_scorer()(torch.ones(1, 3, 2, 2), ["1"], metadata, microbatch=1)


@pytest.mark.parametrize("value", [None, torch.zeros(3), torch.zeros(1, 0),
                                      torch.zeros(2, 3), torch.ones(1, 3, dtype=torch.int64),
                                      torch.full((1, 3), float("nan")),
                                      torch.full((1, 3), float("inf"))])
@pytest.mark.parametrize("key", ["image_features", "text_features"])
def test_invalid_features(monkeypatch, value, key):
    scorer = make_scorer()
    outputs = {"image_features": torch.ones(1, 3), "text_features": torch.ones(1, 3)}
    outputs[key] = value
    monkeypatch.setattr(scorer.model, "forward", lambda *args: outputs)
    with pytest.raises(ValueError, match="features"):
        scorer(torch.ones(1, 3, 2, 2), ["1"], microbatch=1)


@pytest.mark.parametrize("outputs", [None, (), {},
    {"image_features": torch.ones(1, 3), "text_features": torch.ones(1, 2)},
    {"image_features": torch.full((1, 3), 1e30), "text_features": torch.full((1, 3), 1e30)},
])
def test_invalid_model_outputs(monkeypatch, outputs):
    scorer = make_scorer()
    monkeypatch.setattr(scorer.model, "forward", lambda *args: outputs)
    with pytest.raises(ValueError):
        scorer(torch.ones(1, 3, 2, 2), ["1"], microbatch=1)


@pytest.mark.parametrize("value", [None, torch.zeros(2, 2), torch.zeros(4, 2, 2),
    torch.zeros(3, 0, 2), torch.ones(3, 2, 2, dtype=torch.uint8),
    torch.full((3, 2, 2), float("nan"))])
def test_invalid_preprocess(monkeypatch, value):
    scorer = make_scorer()
    scorer.preprocess = lambda image: value
    with pytest.raises(ValueError, match="preprocess"):
        scorer(torch.ones(1, 3, 2, 2), ["1"], microbatch=1)


@pytest.mark.parametrize("value", [None, torch.ones(2), torch.ones(2, 77, dtype=torch.int64),
                                  torch.zeros(1, 0, dtype=torch.int64), torch.ones(1, 77)])
def test_invalid_tokenizer(value):
    scorer = make_scorer()
    scorer.tokenizer = lambda prompts: value
    with pytest.raises(ValueError, match="tokenizer"):
        scorer(torch.ones(1, 3, 2, 2), ["1"], microbatch=1)


@pytest.fixture
def vendor(monkeypatch):
    api = ModuleType("hpsv2.src.open_clip")
    constants = ModuleType("hpsv2.src.open_clip.constants")
    constants.OPENAI_DATASET_MEAN, constants.OPENAI_DATASET_STD = MEAN, STD
    api.calls = []
    api.model = TinyModel()
    api.preprocess, api.tokenizer = Preprocess(), Tokenizer()

    def create(name, **kwargs):
        api.calls.append((name, kwargs))
        assert kwargs["pretrained"] is None
        return api.model, None, api.preprocess

    def tokenizer(name):
        api.calls.append(("tokenizer", name))
        return api.tokenizer

    api.create_model_and_transforms, api.get_tokenizer = create, tokenizer
    monkeypatch.setitem(sys.modules, "hpsv2.src.open_clip", api)
    monkeypatch.setitem(sys.modules, "hpsv2.src.open_clip.constants", constants)
    return api


def save_checkpoint(tmp_path, checkpoint=None):
    path = tmp_path / "HPS_v2.1_compressed.pt"
    if checkpoint is None:
        checkpoint = {"state_dict": {"weight": torch.tensor([1.0, 2.0, 3.0])}}
    torch.save(checkpoint, path)
    return dict(model_path=str(path), model_revision=REVISION, device="cpu",
                checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def test_loader_local_strict_weights_only_and_vendor_settings(tmp_path, vendor, monkeypatch):
    options = save_checkpoint(tmp_path)
    original_load = torch.load
    loads = []

    def checked_load(stream, **kwargs):
        assert stream.tell() == 0
        assert hashlib.sha256(stream.read()).hexdigest() == options["checkpoint_sha256"].lower()
        stream.seek(0)
        loads.append(kwargs)
        return original_load(stream, **kwargs)

    monkeypatch.setattr(torch, "load", checked_load)
    options["checkpoint_sha256"] = options["checkpoint_sha256"].upper()
    scorer = HPSv2Score.from_pretrained(**options)
    assert loads == [{"map_location": "cpu", "weights_only": True}]
    assert vendor.model.strict == [True]
    torch.testing.assert_close(scorer.model.weight, torch.tensor([1.0, 2.0, 3.0]))
    assert vendor.calls == [
        ("ViT-H-14", dict(pretrained=None, precision="amp", device="cpu", jit=False,
                         force_quick_gelu=False, force_custom_text=False,
                         force_patch_dropout=False, force_image_size=None, pretrained_image=False,
                         image_mean=MEAN, image_std=STD, light_augmentation=True, aug_cfg={},
                         output_dict=True, with_score_predictor=False, with_region_predictor=False)),
        ("tokenizer", "ViT-H-14"),
    ]
    assert scorer.model_revision == REVISION
    assert scorer.model_path == options["model_path"]
    assert scorer.checkpoint_sha256 == options["checkpoint_sha256"].lower()
    assert scorer.model.weight.dtype == torch.float32 and not scorer.model.weight.requires_grad


@pytest.mark.parametrize("key,value", [
    ("model_path", ""), ("model_path", "xswu/HPSv2"), ("model_path", None),
    ("model_revision", ""), ("model_revision", None), ("checkpoint_sha256", None),
    ("checkpoint_sha256", "0" * 63), ("checkpoint_sha256", "g" * 64),
    ("checkpoint_sha256", "0" * 64),
])
def test_loader_rejects_invalid_options_before_deserialization(tmp_path, vendor, monkeypatch,
                                                              key, value):
    options = save_checkpoint(tmp_path)
    options[key] = value
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("unexpected load"))
    with pytest.raises(ValueError):
        HPSv2Score.from_pretrained(**options)
    assert vendor.calls == []


@pytest.mark.parametrize("checkpoint", [[], {"weight": torch.ones(3)}, {"state_dict": {}},
    {"state_dict": None}, {"state_dict": {"weight": "bad"}},
    {"state_dict": {1: torch.ones(3)}},
    {"state_dict": {"weight": torch.full((3,), float("nan"))}},
    {"state_dict": {"weight": torch.full((3,), float("inf"))}},
    {"state_dict": {"weight": torch.ones(3, dtype=torch.complex64)}},
    {"state_dict": {"weight": torch.ones(3, device="meta")}},
])
def test_malformed_checkpoint_before_model_construction(tmp_path, vendor, checkpoint):
    with pytest.raises(ValueError, match="state_dict"):
        HPSv2Score.from_pretrained(**save_checkpoint(tmp_path, checkpoint))
    assert vendor.calls == []


def test_unsafe_serialized_object_is_not_loaded(tmp_path, vendor):
    options = save_checkpoint(tmp_path, {"state_dict": {"weight": object()}})
    with pytest.raises(pickle.UnpicklingError):
        HPSv2Score.from_pretrained(**options)
    assert vendor.calls == []


def test_corrupt_serialization_is_not_loaded(tmp_path, vendor):
    options = save_checkpoint(tmp_path)
    from pathlib import Path

    path = Path(options["model_path"])
    path.write_bytes(b"not a torch checkpoint")
    options["checkpoint_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(pickle.UnpicklingError):
        HPSv2Score.from_pretrained(**options)
    assert vendor.calls == []


@pytest.mark.parametrize("state", [{"unexpected": torch.ones(3)},
    {"weight": torch.ones(4)}, {"weight": torch.ones(3), "extra": torch.ones(1)}])
def test_partial_or_wrong_architecture_never_falls_back_to_laion(tmp_path, vendor, state):
    with pytest.raises(RuntimeError):
        HPSv2Score.from_pretrained(**save_checkpoint(tmp_path, {"state_dict": state}))
    assert vendor.model.strict == [True]
    assert len(vendor.calls) == 1 and vendor.calls[0][1]["pretrained"] is None


def test_missing_optional_vendor_only_fails_at_load(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "hpsv2.src.open_clip", None)
    make_scorer()(torch.ones(1, 3, 2, 2), ["1"], microbatch=1)
    with pytest.raises(ImportError, match="hpsv2==1.2.0"):
        HPSv2Score.from_pretrained(**save_checkpoint(tmp_path))


def test_missing_vendor_bpe_fails_before_checkpoint_read(tmp_path, monkeypatch):
    options = save_checkpoint(tmp_path)
    original_import = builtins.__import__

    def missing_bpe(name, *args, **kwargs):
        if name == "hpsv2.src.open_clip":
            raise FileNotFoundError("bpe_simple_vocab_16e6.txt.gz")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_bpe)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("unexpected load"))
    with pytest.raises(ImportError, match="asset-complete.*bpe_simple_vocab_16e6"):
        HPSv2Score.from_pretrained(**options)


def test_vendor_tiny_architecture_without_pretrained_weights(tmp_path, monkeypatch):
    pytest.importorskip("hpsv2")
    from hpsv2.src.open_clip import create_model_and_transforms, factory, get_tokenizer
    from hpsv2.src.open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

    assert OPENAI_DATASET_MEAN == MEAN and OPENAI_DATASET_STD == STD
    laion = factory.get_pretrained_cfg("ViT-H-14", "laion2B-s32B-b79K")
    assert laion["mean"] is None and laion["std"] is None
    config = factory.get_model_config("ViT-H-14")
    assert config["embed_dim"] == 1024
    assert config["vision_cfg"] == {
        "image_size": 224, "layers": 32, "width": 1280, "head_width": 80, "patch_size": 14,
    }
    assert config["text_cfg"] == {
        "context_length": 77, "vocab_size": 49408, "width": 1024, "heads": 16, "layers": 24,
    }

    tiny_config = {
        "embed_dim": 8,
        "vision_cfg": {"image_size": 28, "layers": 1, "width": 32,
                       "head_width": 16, "patch_size": 14},
        "text_cfg": {"context_length": 77, "vocab_size": 49408,
                     "width": 32, "heads": 2, "layers": 1},
    }
    monkeypatch.setitem(factory._MODEL_CONFIGS, "ViT-H-14", tiny_config)
    monkeypatch.setattr(factory, "download_pretrained",
                        lambda *args, **kwargs: pytest.fail("unexpected pretrained download"))
    model, _, preprocess = create_model_and_transforms(
        "ViT-H-14", pretrained=None, precision="amp", device="cpu", output_dict=True,
        light_augmentation=True, image_mean=None, image_std=None,
    )
    tokenizer = get_tokenizer("ViT-H-14")
    images = [Image.new("RGB", (41, 19), (100, 170, 50)),
              Image.new("RGB", (17, 39), (70, 23, 210))]
    prompts = ["a green landscape", "a blue flower"]
    model.eval()
    pixels = torch.stack([preprocess(im) for im in images])
    tokens = tokenizer(prompts)
    assert pixels.shape == (2, 3, 28, 28) and tokens.shape == (2, 77)
    with torch.no_grad():
        outputs = model(pixels, tokens)
    for key in ("image_features", "text_features"):
        torch.testing.assert_close(outputs[key].norm(dim=-1), torch.ones(2))
    reference = (outputs["image_features"] * outputs["text_features"]).sum(-1)
    options = save_checkpoint(tmp_path, {"state_dict": model.state_dict()})
    scorer = HPSv2Score.from_pretrained(**options)
    torch.testing.assert_close(torch.stack([scorer.preprocess(im) for im in images]),
                               pixels, rtol=0, atol=0)
    actual = scorer(images, prompts, microbatch=2)
    torch.testing.assert_close(actual.weighted, reference, rtol=0, atol=0)
    torch.testing.assert_close(scorer(images, prompts, microbatch=1).weighted, reference)
