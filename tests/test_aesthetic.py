"""CPU evidence with real tiny CLIP, production head and slow PIL processing.

All weights are randomly initialized or generated locally, never downloaded.
"""

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from visual_rl.rewards.aesthetic import AestheticMLP, AestheticScore

transformers = pytest.importorskip("transformers")
Image = pytest.importorskip("PIL.Image")


@pytest.fixture(autouse=True)
def cpu_offline(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        yield
    torch.set_num_threads(threads)


def image_processor_class():
    cls = getattr(transformers, "CLIPImageProcessorPil", None)
    return cls if cls is not None else transformers.CLIPImageProcessor


def tiny_clip():
    return transformers.CLIPModel(transformers.CLIPConfig(
        projection_dim=768,
        text_config=dict(vocab_size=16, hidden_size=8, intermediate_size=16,
                         num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=8),
        vision_config=dict(hidden_size=8, intermediate_size=16, num_hidden_layers=1,
                           num_attention_heads=2, image_size=8, patch_size=2),
    ))


@pytest.fixture
def scorer():
    processor = image_processor_class()(
        size={"shortest_edge": 8}, crop_size={"height": 8, "width": 8},
    )
    return AestheticScore(tiny_clip(), processor, AestheticMLP())


def independent_pixels(images):
    pixels = (images.detach() * 255).round().clamp(0, 255).to(torch.uint8)
    pil = [Image.fromarray(p.permute(1, 2, 0).numpy()) for p in pixels]
    return pixels, pil


def test_production_head_architecture_and_checkpoint_keys():
    head = AestheticMLP()
    assert [type(layer) for layer in head.layers] == [
        torch.nn.Linear, torch.nn.Dropout, torch.nn.Linear, torch.nn.Dropout,
        torch.nn.Linear, torch.nn.Dropout, torch.nn.Linear, torch.nn.Linear,
    ]
    assert [(layer.in_features, layer.out_features) for layer in head.layers
            if isinstance(layer, torch.nn.Linear)] == [
        (768, 1024), (1024, 128), (128, 64), (64, 16), (16, 1),
    ]
    assert [layer.p for layer in head.layers if isinstance(layer, torch.nn.Dropout)] == [
        0.2, 0.2, 0.1,
    ]
    assert list(head.state_dict()) == [
        f"layers.{i}.{name}" for i in (0, 2, 4, 6, 7) for name in ("weight", "bias")
    ]


@pytest.mark.parametrize("microbatch", [1, 2, 5, 8])
def test_real_clip_raw_reference_microbatch_and_frozen_state(scorer, microbatch):
    images = torch.rand(5, 3, 11, 17, requires_grad=True)
    prompts = ["fifth", "first", "fourth", "second", "third"]
    pixels, _ = independent_pixels(images)
    # Independent historical path: uint8 BCHW -> actual slow processor -> CLIP.
    with torch.no_grad():
        inputs = scorer.processor(images=pixels, return_tensors="pt")
        vision = scorer.model.vision_model(pixel_values=inputs["pixel_values"])
        features = scorer.model.visual_projection(vision.pooler_output)
        features = features / torch.linalg.vector_norm(features, dim=-1, keepdim=True)
        expected = scorer.head(features).squeeze(1)
    observed_batches = []

    def observe(module, inputs):
        assert not module.training and not torch.is_grad_enabled()
        assert inputs[0].dtype == torch.float32
        torch.testing.assert_close(inputs[0].norm(dim=-1), torch.ones(len(inputs[0])))
        observed_batches.append(len(inputs[0]))

    handle = scorer.head.register_forward_pre_hook(observe)
    torch.nn.Sequential(scorer).train()
    # Also defend against an external caller directly toggling the children.
    scorer.model.train()
    scorer.head.train()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = scorer(images, prompts, microbatch=microbatch)
    handle.remove()
    torch.testing.assert_close(result.raw["aesthetic"], expected, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(result.weighted, expected, rtol=1e-5, atol=1e-7)
    assert result.weighted.shape == (5,) and result.weighted.dtype == torch.float32
    assert result.weights == {"aesthetic": 1.0}
    assert result.space == "raw"
    assert result.normalization is None and result.fixed_normalized is None
    assert observed_batches == [min(microbatch, 5 - i) for i in range(0, 5, microbatch)]
    assert all(not module.training for module in scorer.modules())
    assert all(not p.requires_grad and p.grad is None for p in scorer.parameters())
    assert not result.weighted.requires_grad and images.grad is None


def test_actual_pil_uint8_preprocessing_and_prompt_independence(scorer):
    images = torch.rand(3, 3, 13, 9)
    images.flatten()[:6] = torch.tensor([0, 0.5 / 255, 1.5 / 255, 0.5, 1, 254.5 / 255])
    pixels, pil = independent_pixels(images)
    np.testing.assert_array_equal(np.asarray(pil[0])[0, :6, 0], [0, 0, 2, 128, 255, 254])
    from_pil = scorer.processor(images=pil, return_tensors="pt")["pixel_values"]
    from_uint8 = scorer.processor(images=pixels, return_tensors="pt")["pixel_values"]
    torch.testing.assert_close(from_pil, from_uint8, rtol=0, atol=0)
    tensor_result = scorer(images, ["a", "b", "c"], microbatch=2)
    pil_result = scorer(pil, ["unrelated", "prompts", "ignored"], microbatch=2)
    torch.testing.assert_close(tensor_result.weighted, pil_result.weighted, rtol=0, atol=0)
    permuted = scorer([pil[2], pil[0], pil[1]], ["x"] * 3, microbatch=2)
    torch.testing.assert_close(permuted.weighted, pil_result.weighted[[2, 0, 1]])
    repeated = scorer(images, ["a", "b", "c"], microbatch=2)
    torch.testing.assert_close(tensor_result.weighted, repeated.weighted, rtol=0, atol=0)


@pytest.mark.parametrize("images", [
    torch.zeros(1, 2, 4, 4), torch.zeros(1, 3, 4), torch.zeros(0, 3, 4, 4),
    torch.zeros(1, 3, 0, 4), torch.ones(1, 3, 4, 4, dtype=torch.uint8),
    torch.full((1, 3, 4, 4), -0.01), torch.full((1, 3, 4, 4), 1.01),
    torch.full((1, 3, 4, 4), float("nan")), torch.full((1, 3, 4, 4), float("inf")),
    [], [Image.new("L", (4, 4))], [Image.new("RGBA", (4, 4))],
    [Image.new("RGB", (0, 4))], [object()], np.zeros((1, 3, 4, 4)), None,
])
def test_invalid_images(scorer, images):
    with pytest.raises(ValueError):
        scorer(images, ["prompt"], microbatch=1)


@pytest.mark.parametrize("prompts", ["prompt", b"prompt", [], [""], ["  "], [None],
                                    [1], None, ["one", "two"]])
def test_invalid_pairing(scorer, prompts):
    with pytest.raises(ValueError):
        scorer(torch.ones(1, 3, 4, 4), prompts, microbatch=1)


@pytest.mark.parametrize("microbatch", [0, -1, True, 1.0, None])
def test_invalid_microbatch(scorer, microbatch):
    with pytest.raises(ValueError, match="microbatch"):
        scorer(torch.ones(1, 3, 4, 4), ["prompt"], microbatch=microbatch)


@pytest.mark.parametrize("structured", [False, True])
def test_tensor_and_structured_projected_features(scorer, monkeypatch, structured):
    features = torch.randn(2, 768)
    result = SimpleNamespace(pooler_output=features) if structured else features
    monkeypatch.setattr(scorer.model, "get_image_features", lambda **kwargs: result)
    with torch.no_grad():
        expected = scorer.head(features / features.norm(dim=-1, keepdim=True)).squeeze(1)
    actual = scorer(torch.ones(2, 3, 4, 4), ["a", "b"], microbatch=2)
    torch.testing.assert_close(actual.weighted, expected)


@pytest.mark.parametrize("features", [
    torch.zeros(1, 768), torch.full((1, 768), float("nan")),
    torch.full((1, 768), float("inf")), torch.full((1, 768), 1e38),
    torch.ones(2, 768), torch.ones(1, 4), torch.ones(1, 768, 1),
    torch.ones(1, 768, dtype=torch.int64), None,
])
def test_invalid_features(scorer, monkeypatch, features):
    monkeypatch.setattr(scorer.model, "get_image_features", lambda **kwargs: features)
    with pytest.raises(ValueError):
        scorer(torch.ones(1, 3, 4, 4), ["a"], microbatch=1)


@pytest.mark.parametrize("value", [torch.zeros(1), torch.zeros(1, 2), torch.zeros(2, 1),
                                  torch.tensor([[float("nan")]]),
                                  torch.tensor([[float("inf")]]), torch.ones(1, 1).long(), None])
def test_invalid_head_output(scorer, monkeypatch, value):
    monkeypatch.setattr(scorer.head, "forward", lambda features: value)
    with pytest.raises(ValueError, match="head"):
        scorer(torch.ones(1, 3, 4, 4), ["a"], microbatch=1)


def test_mismatched_head_precision(scorer):
    scorer.head.double()
    with pytest.raises(ValueError, match="device and dtype"):
        scorer(torch.ones(1, 3, 4, 4), ["a"], microbatch=1)


@pytest.fixture
def checkpoint(tmp_path):
    state = AestheticMLP().state_dict()
    path = tmp_path / "head.pth"
    torch.save(state, path)
    return state, path


def loader_options(path):
    return dict(model_path="clip-local", model_revision="model-sha",
                processor_path="processor-local", processor_revision="processor-sha",
                head_path=str(path), head_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_loader_real_head_strict_weights_only_pil_revisions(checkpoint, monkeypatch, dtype):
    state, path = checkpoint
    calls = []
    processor = SimpleNamespace(image_processor=None)
    image_processor = image_processor_class()()

    def loader(name, value):
        def load(path, **kwargs):
            calls.append((name, path, kwargs))
            return value
        return load

    monkeypatch.setattr(transformers.CLIPProcessor, "from_pretrained", loader("processor", processor))
    monkeypatch.setattr(image_processor_class(), "from_pretrained", loader("pil", image_processor))
    monkeypatch.setattr(transformers.CLIPModel, "from_pretrained", loader("model", tiny_clip()))
    original_load = torch.load

    def checked_load(stream, **kwargs):
        assert kwargs == {"map_location": "cpu", "weights_only": True}
        assert hashlib.sha256(stream.getvalue()).hexdigest() == loader_options(path)["head_sha256"]
        calls.append(("head",))
        return original_load(stream, **kwargs)

    monkeypatch.setattr(torch, "load", checked_load)
    options = loader_options(path)
    options["head_sha256"] = options["head_sha256"].upper()
    scorer = AestheticScore.from_pretrained(**options, dtype=dtype)
    assert calls == [
        ("head",),
        ("processor", "processor-local", {"revision": "processor-sha", "local_files_only": True}),
        ("pil", "processor-local", {"revision": "processor-sha", "local_files_only": True}),
        ("model", "clip-local", {"revision": "model-sha", "local_files_only": True}),
    ]
    assert scorer.processor.image_processor is image_processor
    assert all(p.dtype == dtype and p.device.type == "cpu" and not p.requires_grad
               for p in scorer.parameters())
    assert all(not module.training for module in scorer.modules())
    for key, value in scorer.head.state_dict().items():
        torch.testing.assert_close(value, state[key].to(dtype))


@pytest.mark.parametrize("key,value", [
    ("model_path", ""), ("model_revision", None), ("processor_path", "  "),
    ("processor_revision", ""), ("head_path", None), ("head_path", "https://example.org/head"),
    ("head_sha256", "0" * 64), ("head_sha256", "g" * 64), ("head_sha256", "0" * 63),
    ("head_sha256", None), ("dtype", torch.int64), ("local_files_only", 1),
])
def test_loader_rejects_invalid_config_before_loading(checkpoint, monkeypatch, key, value):
    _, path = checkpoint

    def unexpected(*args, **kwargs):
        pytest.fail("invalid config/hash must fail before deserialization or HF loading")

    monkeypatch.setattr(torch, "load", unexpected)
    monkeypatch.setattr(transformers.CLIPModel, "from_pretrained", unexpected)
    monkeypatch.setattr(transformers.CLIPProcessor, "from_pretrained", unexpected)
    options = loader_options(path)
    options[key] = value
    with pytest.raises(ValueError):
        AestheticScore.from_pretrained(**options)


@pytest.mark.parametrize("corruption", ["missing", "extra", "shape", "nested", "nan"])
def test_loader_strict_head_failure_before_clip_loading(checkpoint, monkeypatch, corruption):
    state, path = checkpoint
    if corruption == "missing":
        del state["layers.0.bias"]
    elif corruption == "extra":
        state["extra"] = torch.ones(1)
    elif corruption == "shape":
        state["layers.0.weight"] = torch.ones(1024, 4)
    elif corruption == "nested":
        state = {"state_dict": state}
    else:
        state["layers.0.bias"][0] = float("nan")
    torch.save(state, path)

    def unexpected(*args, **kwargs):
        pytest.fail("invalid head must fail before HF loading")

    monkeypatch.setattr(transformers.CLIPModel, "from_pretrained", unexpected)
    monkeypatch.setattr(transformers.CLIPProcessor, "from_pretrained", unexpected)
    with pytest.raises(ValueError if corruption == "nan" else RuntimeError):
        AestheticScore.from_pretrained(**loader_options(path))


def test_loader_caps_head_read_before_deserialization(checkpoint, monkeypatch):
    _, path = checkpoint
    options = loader_options(path)
    with path.open("wb") as stream:
        stream.truncate(16 * 1024 * 1024 + 1)

    def unexpected(*args, **kwargs):
        pytest.fail("oversized head must fail before deserialization")

    monkeypatch.setattr(torch, "load", unexpected)
    with pytest.raises(ValueError, match="16 MiB"):
        AestheticScore.from_pretrained(**options)
