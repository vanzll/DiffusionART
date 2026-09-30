"""Portable CPU checks for canonical CLIPScore semantics, without downloads."""

from types import SimpleNamespace

import pytest
import torch

from visual_rl.rewards.clipscore import CLIPScore
from visual_rl.rewards.scalar import ScoreResult

T = pytest.importorskip("torchvision.transforms")
Image = pytest.importorskip("PIL.Image")


class ImageConfig:
    image_mean = [0.48145466, 0.4578275, 0.40821073]
    image_std = [0.26862954, 0.26130258, 0.27577711]

    def __init__(self, **overrides):
        self.config = dict(
            size={"shortest_edge": 6},
            crop_size={"height": 4, "width": 4},
            do_resize=True,
            do_center_crop=True,
            do_normalize=True,
            resample=3,
            do_rescale=True,
            rescale_factor=1 / 255,
        )
        self.config.update(overrides)

    def to_dict(self):
        return dict(self.config)

    def __call__(self, *args, **kwargs):
        raise AssertionError("image processor must not process pixels")


class Processor:
    def __init__(self, image_processor=None):
        self.image_processor = image_processor or ImageConfig()
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        assert set(kwargs) == {"text", "padding", "truncation", "return_tensors"}
        assert kwargs["padding"] == "max_length"
        assert kwargs["truncation"] is True and kwargs["return_tensors"] == "pt"
        return tokens(kwargs["text"])


def tokens(prompts):
    # Eight is this asset's context length, not a hard-coded scorer limit of 77.
    return {
        "input_ids": torch.tensor([[1, int(p), 2, 0, 0, 0, 0, 0] for p in prompts]),
        "attention_mask": torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0] for _ in prompts]),
    }


def canonical_transform(config, *, antialias=True, interpolation=T.InterpolationMode.BILINEAR):
    """Independent transcription of the reference's size and transform rules."""

    def size(value):
        if isinstance(value, int):
            return (value, value)
        if "height" in value and "width" in value:
            return (value["height"], value["width"])
        return value["shortest_edge"]

    options = config.to_dict()
    return T.Compose(
        [
            T.Resize(size(options["size"]), interpolation=interpolation, antialias=antialias)
            if options["do_resize"]
            else torch.nn.Identity(),
            T.CenterCrop(size(options["crop_size"]))
            if options["do_center_crop"]
            else torch.nn.Identity(),
            T.Normalize(config.image_mean, config.image_std)
            if options["do_normalize"]
            else torch.nn.Identity(),
        ]
    )


class CaptureModel(torch.nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0, dtype=dtype))
        self.dropout = torch.nn.Dropout(0.9)
        self.pixels = []
        self.text = []

    def forward(self, *, pixel_values, input_ids, attention_mask):
        assert not self.training and not self.dropout.training
        assert not torch.is_grad_enabled() and not torch.is_autocast_enabled("cpu")
        assert pixel_values.dtype == self.weight.dtype
        assert input_ids.dtype == attention_mask.dtype == torch.int64
        self.pixels.append(pixel_values.detach().clone())
        self.text.append(input_ids.detach().clone())
        values = self.dropout(pixel_values).mean((1, 2, 3)) * self.weight
        logits = values[:, None] * input_ids[:, 1].to(values.dtype)[None, :]
        return SimpleNamespace(logits_per_image=logits)


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def tiny_clip():
    transformers = pytest.importorskip("transformers")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        config = transformers.CLIPConfig(
            projection_dim=8,
            text_config=dict(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=24,
                num_hidden_layers=1,
                num_attention_heads=2,
                max_position_embeddings=8,
                bos_token_id=1,
                eos_token_id=2,
                pad_token_id=0,
                attention_dropout=0.5,
            ),
            vision_config=dict(
                hidden_size=16,
                intermediate_size=24,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=4,
                patch_size=2,
                attention_dropout=0.5,
            ),
        )
        return transformers.CLIPModel(config).eval()


@pytest.mark.parametrize("space", ["raw", "fixed_normalized"])
@pytest.mark.parametrize("microbatch", [1, 2, 8])
def test_tiny_clip_logits_pairing_and_scaling(tiny_clip, space, microbatch):
    generator = torch.Generator().manual_seed(29)
    images = torch.rand(5, 3, 13, 9, generator=generator).requires_grad_()
    prompts = ["5", "3", "5", "7", "4"]
    processor = Processor()
    scorer = CLIPScore(tiny_clip, processor, space=space)
    torch.nn.Sequential(scorer).train()
    assert not scorer.training and all(not module.training for module in tiny_clip.modules())
    with torch.no_grad():
        pixels = canonical_transform(processor.image_processor)(images)
        raw = tiny_clip(pixel_values=pixels, **tokens(prompts)).logits_per_image.diagonal()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = scorer(images, prompts, microbatch=microbatch)
    assert isinstance(result, ScoreResult)
    assert result.space == space and dict(result.weights) == {"clipscore": 1.0}
    torch.testing.assert_close(result.raw["clipscore"], raw, atol=3e-6, rtol=2e-5)
    torch.testing.assert_close(result.fixed_normalized["clipscore"], raw / 30, atol=2e-7, rtol=2e-5)
    torch.testing.assert_close(
        result.weighted,
        result.raw["clipscore"] if space == "raw" else result.fixed_normalized["clipscore"],
    )
    assert result.weighted.dtype == torch.float32 and not result.weighted.requires_grad
    assert images.grad is None and all(not p.requires_grad for p in tiny_clip.parameters())
    assert [p for call in processor.calls for p in call["text"]] == prompts
    assert [len(call["text"]) for call in processor.calls] == [
        len(prompts[start : start + microbatch]) for start in range(0, len(prompts), microbatch)
    ]


def test_sub_uint8_pixels_are_not_quantized(tiny_clip):
    generator = torch.Generator().manual_seed(6)
    images = torch.randint(20, 230, (2, 3, 9, 13), generator=generator).float() / 255
    images += torch.rand(images.shape, generator=generator) * (0.49 / 255)
    quantized = (images * 255).round() / 255
    processor = Processor()
    scorer = CLIPScore(tiny_clip, processor, space="fixed_normalized")
    actual = scorer(images, ["3", "4"], microbatch=2)
    transform = canonical_transform(processor.image_processor)
    with torch.no_grad():
        raw = tiny_clip(pixel_values=transform(images), **tokens(["3", "4"]))
        rounded = tiny_clip(pixel_values=transform(quantized), **tokens(["3", "4"]))
    torch.testing.assert_close(actual.raw["clipscore"], raw.logits_per_image.diagonal())
    torch.testing.assert_close(actual.weighted, raw.logits_per_image.diagonal() / 30)
    assert (actual.raw["clipscore"] - rounded.logits_per_image.diagonal()).abs().max() > 1e-5


@pytest.mark.parametrize("antialias", [True, False])
def test_bilinear_not_processor_bicubic_or_rescale(antialias):
    images = torch.rand(2, 3, 17, 11, generator=torch.Generator().manual_seed(1))
    processor, model = Processor(), CaptureModel()
    scorer = CLIPScore(model, processor, space="raw", antialias=antialias)
    scorer(images, ["3", "4"], microbatch=2)
    actual = model.pixels[0]
    expected = canonical_transform(processor.image_processor, antialias=antialias)(images)
    bicubic = canonical_transform(
        processor.image_processor, antialias=antialias, interpolation=T.InterpolationMode.BICUBIC
    )(images)
    other_aa = canonical_transform(processor.image_processor, antialias=not antialias)(images)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.allclose(actual, bicubic)
    assert not torch.allclose(actual, other_aa)
    assert not torch.allclose(actual, expected / 255)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_original_precision_until_model_and_frozen_dropout(dtype):
    images = torch.rand(
        2, 3, 11, 7, dtype=torch.float64, generator=torch.Generator().manual_seed(8)
    ).requires_grad_()
    model, processor = CaptureModel(dtype), Processor()
    scorer = CLIPScore(model, processor, space="raw")
    torch.nn.Sequential(scorer).train()
    transform = scorer.transform
    seen_dtypes = []

    def observe(value):
        seen_dtypes.append(value.dtype)
        return transform(value)

    scorer.transform = observe
    with torch.autocast("cpu", dtype=torch.bfloat16):
        first = scorer(images, ["4", "3"], microbatch=2)
        second = scorer(images, ["4", "3"], microbatch=2)
    expected = canonical_transform(processor.image_processor)(images).to(dtype)
    torch.testing.assert_close(model.pixels[0], expected, rtol=0, atol=0)
    torch.testing.assert_close(first.weighted, second.weighted, rtol=0, atol=0)
    assert seen_dtypes == [torch.float64, torch.float64]
    assert not model.weight.requires_grad and not first.weighted.requires_grad
    assert images.grad is None and model.weight.grad is None


@pytest.mark.parametrize("size", [6, {"height": 5, "width": 7}, {"shortest_edge": 6}])
@pytest.mark.parametrize(
    "flags", [(True, True, True), (False, False, False), (True, False, True), (False, True, False)]
)
def test_config_sizes_and_transform_flags(size, flags):
    config = ImageConfig(
        size=size, crop_size=4, do_resize=flags[0], do_center_crop=flags[1], do_normalize=flags[2]
    )
    model = CaptureModel()
    images = torch.linspace(0, 1, 3 * 7 * 11).reshape(1, 3, 7, 11)
    CLIPScore(model, Processor(config), space="raw")(images, ["3"], microbatch=1)
    torch.testing.assert_close(model.pixels[0], canonical_transform(config)(images), rtol=0, atol=0)


def test_rgb_pil_rectangular_sequences():
    images = [Image.new("RGB", (11, 7), (17, 127, 233)), Image.new("RGB", (7, 13), (100, 19, 201))]
    model, processor = CaptureModel(), Processor()
    scorer = CLIPScore(model, processor, space="raw")
    scorer(images, ["3", "4"], microbatch=2)
    expected = torch.stack(
        [
            canonical_transform(processor.image_processor)(T.functional.to_tensor(im))
            for im in images
        ]
    )
    torch.testing.assert_close(model.pixels[0], expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "images",
    [
        torch.zeros(1, 2, 4, 4),
        torch.zeros(3, 4, 4),
        torch.zeros(0, 3, 4, 4),
        torch.zeros(1, 3, 0, 4),
        torch.ones(1, 3, 4, 4, dtype=torch.uint8),
        torch.full((1, 3, 4, 4), -0.01),
        torch.full((1, 3, 4, 4), 1.01),
        torch.full((1, 3, 4, 4), float("nan")),
        torch.full((1, 3, 4, 4), float("inf")),
        [],
        None,
        [Image.new("L", (4, 4))],
        [Image.new("RGBA", (4, 4))],
        [Image.new("RGB", (0, 4))],
        [torch.zeros(3, 4, 4)],
    ],
)
def test_invalid_images(images):
    with pytest.raises(ValueError):
        CLIPScore(CaptureModel(), Processor(), space="raw")(images, ["3"], microbatch=1)


@pytest.mark.parametrize("prompts", [[], "3", b"3", None, [""], ["  "], [3], ["3", "4"]])
def test_invalid_or_unpaired_prompts(prompts):
    with pytest.raises(ValueError, match="prompt|paired"):
        CLIPScore(CaptureModel(), Processor(), space="raw")(
            torch.zeros(1, 3, 4, 4),
            prompts,
            microbatch=1,
        )


@pytest.mark.parametrize("microbatch", [0, -1, True, 1.5, None])
def test_invalid_microbatch(microbatch):
    with pytest.raises(ValueError, match="microbatch"):
        CLIPScore(CaptureModel(), Processor(), space="raw")(
            torch.zeros(1, 3, 4, 4),
            ["3"],
            microbatch=microbatch,
        )


@pytest.mark.parametrize(
    "logits",
    [
        torch.tensor([[float("nan"), 0.0], [0.0, 1.0]]),
        torch.tensor([[1.0, float("inf")], [0.0, 1.0]]),
        torch.ones(2),
        torch.ones(2, 1),
        torch.ones(2, 2, dtype=torch.int64),
        None,
    ],
)
def test_invalid_model_outputs(logits):
    class BadModel(CaptureModel):
        def forward(self, **kwargs):
            return SimpleNamespace(logits_per_image=logits)

    with pytest.raises(ValueError, match="logits"):
        CLIPScore(BadModel(), Processor(), space="raw")(
            torch.zeros(2, 3, 4, 4),
            ["3", "4"],
            microbatch=2,
        )


@pytest.mark.parametrize("options", [{"space": "softmax"}, {"antialias": None}, {"antialias": 1}])
def test_invalid_constructor_options(options):
    with pytest.raises(ValueError):
        CLIPScore(CaptureModel(), Processor(), **({"space": "raw"} | options))


@pytest.mark.parametrize(
    "size", [None, 0, -1, True, {}, {"shortest_edge": 0}, {"height": 4, "width": -1}]
)
def test_invalid_image_config_size(size):
    with pytest.raises(ValueError, match="size"):
        CLIPScore(CaptureModel(), Processor(ImageConfig(size=size)), space="raw")


@pytest.mark.parametrize("local_files_only", [True, False])
def test_loader_offline_revision_routing_and_pil_config(monkeypatch, local_files_only):
    transformers = pytest.importorskip("transformers")
    processor, image_config, calls = Processor(), ImageConfig(), []

    def loader(name, result):
        def load(path, **kwargs):
            calls.append((name, path, kwargs))
            return result

        return load

    image_class = getattr(transformers, "CLIPImageProcessorPil", None)
    if image_class is None:
        image_class = transformers.CLIPImageProcessor
    monkeypatch.setattr(
        transformers.CLIPProcessor, "from_pretrained", loader("processor", processor)
    )
    monkeypatch.setattr(image_class, "from_pretrained", loader("image", image_config))
    monkeypatch.setattr(transformers.CLIPModel, "from_pretrained", loader("model", CaptureModel()))
    options = {} if local_files_only else {"local_files_only": False}
    scorer = CLIPScore.from_pretrained(
        model_path="weights",
        processor_path="processor",
        model_revision="model-sha",
        processor_revision="processor-sha",
        space="fixed_normalized",
        antialias=False,
        **options,
    )
    assert calls == [
        (
            "processor",
            "processor",
            {"revision": "processor-sha", "local_files_only": local_files_only},
        ),
        ("image", "processor", {"revision": "processor-sha", "local_files_only": local_files_only}),
        ("model", "weights", {"revision": "model-sha", "local_files_only": local_files_only}),
    ]
    assert scorer.processor.image_processor is image_config
    assert next(scorer.model.parameters()).dtype == torch.float32
    assert next(scorer.model.parameters()).device.type == "cpu"
    assert not scorer.antialias and not scorer.model.training
    assert not scorer.model.weight.requires_grad


@pytest.mark.parametrize(
    "override",
    [
        {"model_path": ""},
        {"processor_path": None},
        {"model_revision": " "},
        {"processor_revision": 3},
        {"dtype": torch.int64},
        {"space": "probability"},
        {"local_files_only": 1},
        {"antialias": None},
    ],
)
def test_loader_invalid_arguments_before_loading(override):
    options = dict(
        model_path="weights",
        processor_path="processor",
        model_revision="model-sha",
        processor_revision="processor-sha",
        space="raw",
    )
    with pytest.raises(ValueError):
        CLIPScore.from_pretrained(**(options | override))
