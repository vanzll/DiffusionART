"""Native scorer semantics without checkpoint or tokenizer downloads."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from visual_rl.rewards.pickscore import PickScore, quantized_rgb

pytest.importorskip("PIL")


class Processor:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if "images" in kwargs:
            return {"pixel_values": torch.stack([
                torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255
                for im in kwargs["images"]
            ])}
        return {"input_ids": torch.tensor([[int(p), 1] for p in kwargs["text"]])}


class Model(torch.nn.Module):
    def __init__(self, structured=False):
        super().__init__()
        self.logit_scale = torch.nn.Parameter(torch.tensor(2.0))
        self.structured = structured

    def wrap(self, value):
        return SimpleNamespace(pooler_output=value) if self.structured else value

    def get_image_features(self, pixel_values):
        assert not self.training and not torch.is_grad_enabled()
        return self.wrap(pixel_values.mean((2, 3)))

    def get_text_features(self, input_ids):
        assert input_ids.dtype == torch.int64
        return self.wrap(torch.cat((input_ids.float(), torch.ones(len(input_ids), 1)), 1))


@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize("space", ["raw", "fixed_normalized"])
def test_pairing_and_historical_scaling(structured, space):
    images = torch.linspace(0.05, 0.95, 5 * 3 * 2 * 2).reshape(5, 3, 2, 2).requires_grad_()
    prompts = ["5", "1", "3", "2", "4"]
    processor, model = Processor(), Model(structured)
    scorer = PickScore(model, processor, space=space).train()
    output = scorer(images, prompts, microbatch=2)
    pixel = (images.detach() * 255).round() / 255
    image_features = pixel.mean((2, 3))
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text = torch.tensor([[int(p), 1.0, 1.0] for p in prompts])
    text = text / text.norm(dim=-1, keepdim=True)
    raw = (torch.tensor(2.0).exp() * (text @ image_features.T)).diag()
    torch.testing.assert_close(output.raw["pickscore"], raw)
    torch.testing.assert_close(output.fixed_normalized["pickscore"], raw / 26)
    torch.testing.assert_close(output.weighted, raw if space == "raw" else raw / 26)
    assert not output.weighted.requires_grad and images.grad is None
    assert not model.logit_scale.requires_grad and not model.training
    for call in processor.calls:
        assert {k: call[k] for k in ("padding", "truncation", "max_length", "return_tensors")} == {
            "padding": True, "truncation": True, "max_length": 77, "return_tensors": "pt"
        }


def test_quantization_rounds_not_truncates():
    x = torch.tensor([0.0, 0.5 / 255, 1.5 / 255, 0.5, 1.0, 254.5 / 255]).repeat(3)
    image = quantized_rgb(x.reshape(1, 3, 2, 3))[0]
    np.testing.assert_array_equal(np.asarray(image)[:, :, 0], [[0, 0, 2], [128, 255, 254]])


@pytest.mark.parametrize("value", [torch.zeros(1, 2, 4, 4), torch.ones(1, 3, 4, 4) * 1.1,
                                    torch.full((1, 3, 4, 4), float("nan")),
                                    torch.zeros(0, 3, 4, 4), torch.ones(1, 3, 4, 4).long()])
def test_bad_pixels_fail(value):
    with pytest.raises(ValueError):
        quantized_rgb(value)


def test_pair_mismatch_and_zero_features_fail():
    scorer = PickScore(Model(), Processor(), space="raw")
    with pytest.raises(ValueError, match="paired"):
        scorer(torch.ones(2, 3, 2, 2), ["1"], microbatch=2)
    with pytest.raises(ValueError, match="positive"):
        scorer(torch.zeros(1, 3, 2, 2), ["1"], microbatch=2)
    with pytest.raises(ValueError, match="prompt"):
        scorer(torch.ones(1, 3, 2, 2), "1", microbatch=1)


def test_real_clip_projection_output_parity():
    transformers = pytest.importorskip("transformers")
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        config = transformers.CLIPConfig(
            projection_dim=4,
            text_config=dict(vocab_size=16, hidden_size=8, intermediate_size=16,
                             num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=8),
            vision_config=dict(hidden_size=8, intermediate_size=16, num_hidden_layers=1,
                               num_attention_heads=2, image_size=4, patch_size=2),
        )
        model = transformers.CLIPModel(config).eval()
        images, prompts = torch.rand(2, 3, 4, 4), ["2", "4"]
        scorer = PickScore(model, Processor(), space="raw")
        result = scorer(images, prompts, microbatch=2)
        pixels = (images * 255).round() / 255
        with torch.no_grad():
            direct = model(pixel_values=pixels, input_ids=torch.tensor([[2, 1], [4, 1]]))
        torch.testing.assert_close(result.raw["pickscore"], direct.logits_per_text.diag())
    finally:
        torch.set_num_threads(old_threads)


def test_loader_pins_pil_preprocessor_and_offline_revisions(monkeypatch):
    transformers = pytest.importorskip("transformers")
    calls = []
    processor, image_processor = Processor(), object()

    def load_processor(path, **kwargs):
        calls.append(("processor", path, kwargs))
        return processor

    def load_image(path, **kwargs):
        calls.append(("image", path, kwargs))
        return image_processor

    def load_model(path, **kwargs):
        calls.append(("model", path, kwargs))
        return Model()

    image_class = getattr(transformers, "CLIPImageProcessorPil", None)
    if image_class is None:
        image_class = transformers.CLIPImageProcessor
    monkeypatch.setattr(transformers.CLIPProcessor, "from_pretrained", load_processor)
    monkeypatch.setattr(image_class, "from_pretrained", load_image)
    monkeypatch.setattr(transformers.CLIPModel, "from_pretrained", load_model)
    scorer = PickScore.from_pretrained(
        model_path="weights", processor_path="processor", model_revision="model-sha",
        processor_revision="processor-sha", space="fixed_normalized",
    )
    assert scorer.processor.image_processor is image_processor
    assert calls == [
        ("processor", "processor", {"revision": "processor-sha", "local_files_only": True}),
        ("image", "processor", {"revision": "processor-sha", "local_files_only": True}),
        ("model", "weights", {"revision": "model-sha", "local_files_only": True}),
    ]
    assert next(scorer.model.parameters()).dtype == torch.float32
    assert not scorer.model.training
