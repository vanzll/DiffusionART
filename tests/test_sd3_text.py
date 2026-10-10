"""Offline SD3 text contracts, including tiny real Diffusers/Transformers APIs."""

import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from visual_rl.models.sd3 import SD3Adapter, SD3Conditioning, SD3Config
from visual_rl.models.sd3_text import SD3TextEncoder


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("network access is forbidden in SD3 text tests")

    monkeypatch.setattr(socket.socket, "connect", fail)
    monkeypatch.setattr(socket, "create_connection", fail)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


class Pipeline:
    def __init__(self):
        for suffix in ("", "_2", "_3"):
            setattr(self, "text_encoder" + suffix, nn.Sequential(nn.Linear(1, 2), nn.Dropout(.8)))
            setattr(self, "tokenizer" + suffix, lambda *args, **kwargs: None)
        self.calls = []
        self.output = None

    def encode_prompt(self, **kwargs):
        self.calls.append(kwargs)
        assert not torch.is_grad_enabled()
        assert all(not module.training for encoder in encoders(self) for module in encoder.modules())

        def encode(text):
            ids = torch.tensor([[sum(map(ord, item)) + 1.] for item in text],
                               device=kwargs["device"], dtype=self.text_encoder[0].weight.dtype)
            repeats = kwargs["num_images_per_prompt"]
            tokens = self.text_encoder(ids)[:, None].expand(-1, kwargs["max_sequence_length"], -1)
            # Exercise all encoders and keep pooled/token channels distinguishable.
            tokens = tokens + self.text_encoder_3(ids)[:, None]
            pooled = self.text_encoder_2(ids)
            return tokens.repeat_interleave(repeats, 0), pooled.repeat_interleave(repeats, 0)

        pos, pooled_pos = encode(kwargs["prompt"])
        neg, pooled_neg = None, None
        if kwargs["do_classifier_free_guidance"]:
            neg, pooled_neg = encode(kwargs["negative_prompt"])
        self.output = (pos, neg, pooled_pos, pooled_neg)
        return self.output


def encoders(pipeline):
    return (pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3)


def tensors(conditioning):
    return (conditioning.prompt_embeds, conditioning.negative_prompt_embeds,
            conditioning.pooled_prompt_embeds, conditioning.negative_pooled_prompt_embeds)


def helper(pipeline=None, dtype=torch.float32):
    return SD3TextEncoder(pipeline or Pipeline(), device="cpu", torch_dtype=dtype)


@pytest.mark.parametrize("scale", [None, 0., 1., 7.])
def test_named_fields_cfg_repeats_and_adapter_order(scale):
    pipeline = Pipeline()
    text = helper(pipeline)
    options = dict(max_sequence_length=5, cfg_scale=scale)
    if scale is not None:
        options["negative_prompt"] = ["", "negative"]
    single = text.encode(["red cube", "blue sphere"], **options)
    result = text.encode(["red cube", "blue sphere"], num_images_per_prompt=3, **options)
    assert isinstance(result, SD3Conditioning)
    for one, repeated in zip(tensors(single), tensors(result)):
        if one is None:
            assert repeated is None
        else:
            torch.testing.assert_close(repeated, one.repeat_interleave(3, 0))
    call = pipeline.calls[-1]
    assert call["do_classifier_free_guidance"] is (scale is not None)
    assert call["prompt_2"] is None and call["prompt_3"] is None
    assert call["max_sequence_length"] == 5
    assert call["num_images_per_prompt"] == 3

    class Field(nn.Module):
        def forward(self, **kwargs):
            self.last = kwargs
            return (kwargs["hidden_states"],)

    field = Field()
    adapter = SD3Adapter(field, nn.Linear(1, 1), SD3Config(1., 0., latent_channels=2, cfg_scale=scale))
    adapter.predict_velocity(torch.ones(6, 2, 2, 2), torch.linspace(.1, .6, 6), result)
    expected = result.prompt_embeds
    pooled = result.pooled_prompt_embeds
    if scale is not None:
        expected = torch.cat([result.negative_prompt_embeds, expected])
        pooled = torch.cat([result.negative_pooled_prompt_embeds, pooled])
    torch.testing.assert_close(field.last["encoder_hidden_states"], expected)
    torch.testing.assert_close(field.last["pooled_projections"], pooled)


def test_frozen_dropout_fresh_storage_dtype_and_downstream_autograd():
    pipeline = Pipeline()
    for encoder in encoders(pipeline):
        for parameter in encoder.parameters():
            parameter.grad = torch.ones_like(parameter)
    text = helper(pipeline, torch.float64)
    for encoder in encoders(pipeline):
        assert not encoder.training
        assert all(not p.requires_grad and p.grad is None and p.dtype == torch.float64 for p in encoder.parameters())
    first = text.encode("red", max_sequence_length=3, cfg_scale=0)
    # An external train() must not turn dropout back on during the next encode.
    for encoder in encoders(pipeline):
        encoder.train().requires_grad_(True)
        for parameter in encoder.parameters():
            parameter.grad = torch.ones_like(parameter)
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        second = text.encode("red", max_sequence_length=3, cfg_scale=0)
    for a, b, raw in zip(tensors(first), tensors(second), pipeline.output):
        torch.testing.assert_close(a, b)
        assert b.dtype == torch.float64 and b.device.type == "cpu"
        assert not b.requires_grad and b.grad_fn is None and not b.is_inference()
        assert a.data_ptr() != b.data_ptr() != raw.data_ptr()
    policy = nn.Linear(2, 1, dtype=torch.float64)
    sum(policy(t).sum() for t in tensors(second)).backward()
    assert policy.weight.grad is not None
    assert all(p.grad is None and not p.requires_grad for e in encoders(pipeline) for p in e.parameters())
    second.prompt_embeds.zero_()
    fresh = text.encode("red", max_sequence_length=3, cfg_scale=0)
    torch.testing.assert_close(fresh.prompt_embeds, first.prompt_embeds)
    changed = text.encode("blue", max_sequence_length=3, cfg_scale=0)
    assert not torch.equal(changed.prompt_embeds, fresh.prompt_embeds)
    assert len(pipeline.calls) == 4


def test_empty_negative_is_encoded_not_zero_and_scalar_broadcasts():
    text = helper()
    options = dict(max_sequence_length=3, cfg_scale=1, num_images_per_prompt=2)
    implicit = text.encode(("red", "blue"), **options)
    explicit = text.encode(["red", "blue"], negative_prompt="", **options)
    assert text.pipeline.calls[-1]["negative_prompt"] == ["", ""]
    for a, b in zip(tensors(implicit), tensors(explicit)):
        torch.testing.assert_close(a, b)
    assert implicit.negative_prompt_embeds.count_nonzero() > 0
    scalar = text.encode(["red", "blue"], negative_prompt="negative", **options)
    listed = text.encode(["red", "blue"], negative_prompt=["negative", "negative"], **options)
    torch.testing.assert_close(scalar.negative_prompt_embeds, listed.negative_prompt_embeds)


@pytest.mark.parametrize("prompt, options", [
    (None, {}), ("", {}), (" \t", {}), ([], {}), (["red", ""], {}),
    (["red", 2], {}), ({"red"}, {}), (iter(["red"]), {}),
    ("red", {"max_sequence_length": 0}), ("red", {"max_sequence_length": 513}),
    ("red", {"max_sequence_length": True}), ("red", {"max_sequence_length": 3.5}),
    ("red", {"num_images_per_prompt": 0}), ("red", {"num_images_per_prompt": -1}),
    ("red", {"num_images_per_prompt": True}), ("red", {"num_images_per_prompt": 1.5}),
    ("red", {"cfg_scale": -1}), ("red", {"cfg_scale": float("inf")}),
    ("red", {"cfg_scale": float("nan")}), ("red", {"cfg_scale": True}),
    ("red", {"cfg_scale": "7"}), ("red", {"negative_prompt": ""}),
    ("red", {"cfg_scale": 1, "negative_prompt": []}),
    ("red", {"cfg_scale": 1, "negative_prompt": [None]}),
    (["red", "blue"], {"cfg_scale": 1, "negative_prompt": [""]}),
])
def test_invalid_requests_fail_before_encoding(prompt, options):
    pipeline = Pipeline()
    text = helper(pipeline)
    with pytest.raises(ValueError):
        text.encode(prompt, **(dict(max_sequence_length=4) | options))
    assert pipeline.calls == []


@pytest.mark.parametrize("options", [
    {"model_id_or_path": " "}, {"model_id_or_path": None},
    {"revision": None}, {"revision": ""}, {"revision": 42},
    {"device": None}, {"device": "invalid-device"}, {"device": "meta"},
    {"torch_dtype": None}, {"torch_dtype": "float32"}, {"torch_dtype": torch.int64},
    {"local_files_only": "false"},
])
def test_invalid_loader_options_do_not_import_or_load(monkeypatch, options):
    def fail(*args, **kwargs):
        raise AssertionError("loader must not run for invalid options")

    monkeypatch.setitem(sys.modules, "diffusers", SimpleNamespace(
        StableDiffusion3Pipeline=SimpleNamespace(from_pretrained=fail)))
    arguments = dict(model_id_or_path="org/model", revision="pinned", device="cpu",
                     torch_dtype=torch.float32, local_files_only=False)
    with pytest.raises(ValueError):
        SD3TextEncoder.from_pretrained(**(arguments | options))


@pytest.mark.parametrize("online", [False, True])
def test_loader_is_text_only_revision_pinned_and_offline_by_default(monkeypatch, online):
    calls = []
    pipeline = Pipeline()

    def load(path, **kwargs):
        calls.append((path, kwargs))
        return pipeline

    monkeypatch.setitem(sys.modules, "diffusers", SimpleNamespace(
        StableDiffusion3Pipeline=SimpleNamespace(from_pretrained=load)))
    options = {"local_files_only": False} if online else {}
    result = SD3TextEncoder.from_pretrained(
        Path("local-model"), revision="immutable-revision", device="cpu", torch_dtype=torch.float64, **options)
    assert result.pipeline is pipeline
    assert calls == [("local-model", dict(revision="immutable-revision", local_files_only=not online,
                                         torch_dtype=torch.float64, transformer=None, vae=None))]


def test_incomplete_pipeline_rejected_without_partial_freeze():
    pipeline = Pipeline()
    pipeline.text_encoder_3 = None
    with pytest.raises(ValueError, match="all three"):
        helper(pipeline)
    assert pipeline.text_encoder.training


def test_bad_pipeline_repeat_contract_is_rejected():
    pipeline = Pipeline()
    original = pipeline.encode_prompt

    def bad(**kwargs):
        pos, neg, pooled, negative_pooled = original(**kwargs)
        return pos, neg, pooled[:1], negative_pooled

    pipeline.encode_prompt = bad
    with pytest.raises(ValueError, match="pooled_prompt_embeds"):
        helper(pipeline).encode(["red", "blue"], max_sequence_length=4, num_images_per_prompt=2)


def test_optional_imports_are_lazy_in_clean_process():
    source = Path(__file__).resolve().parents[1] / "src"
    code = f"""
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('diffusers', 'transformers'):
            raise AssertionError('optional dependency imported: ' + fullname)
sys.meta_path.insert(0, Block())
import visual_rl
visual_rl.__path__.insert(0, {str(source / 'visual_rl')!r})
import visual_rl.models.sd3_text as module
assert module.__file__ == {str(source / 'visual_rl/models/sd3_text.py')!r}
"""
    result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True,
                            timeout=30, env=dict(os.environ, PYTHONPATH=str(source)))
    assert result.returncode == 0, result.stderr


@pytest.fixture
def real_pipeline():
    diffusers = pytest.importorskip("diffusers")
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(
        {"<pad>": 0, "<unk>": 1, "red": 2, "blue": 3, "cube": 4,
         "sphere": 5, "negative": 6, "<s>": 7, "</s>": 8}, unk_token="<unk>"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    backend.post_processor = tokenizers.processors.TemplateProcessing(
        single="<s> $A </s>", special_tokens=[("<s>", 7), ("</s>", 8)])
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, model_max_length=6,
        pad_token="<pad>", unk_token="<unk>", bos_token="<s>", eos_token="</s>")

    def clip(projection):
        return transformers.CLIPTextModelWithProjection(transformers.CLIPTextConfig(
            vocab_size=9, hidden_size=8, intermediate_size=16, projection_dim=projection,
            num_hidden_layers=2, num_attention_heads=2, max_position_embeddings=6,
            attention_dropout=.7, pad_token_id=0, bos_token_id=7, eos_token_id=8))

    torch.manual_seed(123)
    return diffusers.StableDiffusion3Pipeline(
        transformer=None, vae=None, scheduler=diffusers.FlowMatchEulerDiscreteScheduler(),
        text_encoder=clip(4), text_encoder_2=clip(6),
        text_encoder_3=transformers.T5EncoderModel(transformers.T5Config(
            vocab_size=9, d_model=24, d_ff=32, d_kv=12, num_heads=2, num_layers=1,
            dropout_rate=.7, pad_token_id=0, eos_token_id=8)),
        tokenizer=tokenizer, tokenizer_2=tokenizer, tokenizer_3=tokenizer,
    )


def test_real_clip_t5_structure_repetition_empty_negatives_and_dropout(real_pipeline):
    pipeline = real_pipeline
    text = helper(pipeline)
    prompts = ["red cube", "blue sphere"]
    options = dict(max_sequence_length=5, cfg_scale=0)
    result = text.encode(prompts, num_images_per_prompt=3, **options)
    assert result.prompt_embeds.shape == (6, 11, 24)
    assert result.pooled_prompt_embeds.shape == (6, 10)
    for actual, direct in zip(tensors(result), pipeline.encode_prompt(
            prompts, None, None, device=torch.device("cpu"), max_sequence_length=5,
            num_images_per_prompt=3, do_classifier_free_guidance=True)):
        torch.testing.assert_close(actual, direct.detach())

    with torch.no_grad():
        ids = pipeline.tokenizer(prompts, padding="max_length", max_length=6,
                                 truncation=True, return_tensors="pt").input_ids
        first = pipeline.text_encoder(ids, output_hidden_states=True)
        second = pipeline.text_encoder_2(ids, output_hidden_states=True)
        clip_tokens = torch.nn.functional.pad(torch.cat(
            [first.hidden_states[-2], second.hidden_states[-2]], -1), (0, 8))
        t5_ids = pipeline.tokenizer_3(prompts, padding="max_length", max_length=5,
                                     truncation=True, return_tensors="pt").input_ids
        expected = torch.cat([clip_tokens, pipeline.text_encoder_3(t5_ids)[0]], 1)
        pooled = torch.cat([first.text_embeds, second.text_embeds], -1)
    torch.testing.assert_close(result.prompt_embeds, expected.repeat_interleave(3, 0))
    torch.testing.assert_close(result.pooled_prompt_embeds, pooled.repeat_interleave(3, 0))
    assert not torch.equal(expected[0], expected[1])
    assert result.negative_prompt_embeds.count_nonzero() > 0

    for encoder in encoders(pipeline):
        encoder.train()
    again = text.encode(prompts, negative_prompt="", num_images_per_prompt=3, **options)
    for a, b in zip(tensors(result), tensors(again)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert not b.requires_grad and not b.is_inference()
    policy = nn.Linear(24, 1)
    policy(result.prompt_embeds).sum().backward()
    assert policy.weight.grad is not None
    assert all(p.grad is None for encoder in encoders(pipeline) for p in encoder.parameters())
    no_cfg = text.encode(prompts, max_sequence_length=2)
    assert no_cfg.prompt_embeds.shape == (2, 8, 24)
    assert no_cfg.negative_prompt_embeds is None


def test_real_local_pretrained_loader_never_needs_transformer_or_vae(real_pipeline, tmp_path):
    real_pipeline.save_pretrained(tmp_path, safe_serialization=True)
    # Advertise heavy components but supply no weights: explicit None overrides
    # must prevent loading them, just as with a complete SD3 checkpoint.
    index = tmp_path / "model_index.json"
    config = json.loads(index.read_text())
    config["transformer"] = ["diffusers", "SD3Transformer2DModel"]
    config["vae"] = ["diffusers", "AutoencoderKL"]
    index.write_text(json.dumps(config))
    loaded = SD3TextEncoder.from_pretrained(
        tmp_path, revision="offline-fixture", device="cpu", torch_dtype=torch.float32)
    assert loaded.pipeline.transformer is None and loaded.pipeline.vae is None
    expected = helper(real_pipeline).encode("red cube", max_sequence_length=5, cfg_scale=1)
    actual = loaded.encode("red cube", max_sequence_length=5, cfg_scale=1)
    for a, b in zip(tensors(expected), tensors(actual)):
        torch.testing.assert_close(a, b)
