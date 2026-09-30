"""Offline FLUX text contracts; real_flux_pipeline is reusable by runner tests."""

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

from visual_rl.models.flux import FluxAdapter, FluxConditioning, FluxConfig, pack_latents
from visual_rl.models.flux_text import FluxTextEncoder
from visual_rl.training.image import ConditionedImagePolicy


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("network access is forbidden in FLUX text tests")

    monkeypatch.setattr(socket.socket, "connect", fail)
    monkeypatch.setattr(socket, "create_connection", fail)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


def encoders(pipeline):
    return (pipeline.text_encoder, pipeline.text_encoder_2)


def tensors(conditioning):
    return (conditioning.prompt_embeds, conditioning.pooled_prompt_embeds, conditioning.text_ids)


class Pipeline:
    def __init__(self):
        for suffix in ("", "_2"):
            setattr(self, "text_encoder" + suffix, nn.Sequential(nn.Linear(1, 2), nn.Dropout(.8)))
            setattr(self, "tokenizer" + suffix, lambda *args, **kwargs: None)
        self.calls = []
        self.output = None
        self.transformer = nn.Linear(1, 1)
        self.vae = nn.Linear(1, 1)

    def encode_prompt(self, **kwargs):
        self.calls.append(kwargs)
        assert not torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
        assert not torch.is_autocast_enabled("cpu")
        assert all(not m.training for encoder in encoders(self) for m in encoder.modules())
        ids = torch.tensor([[sum(map(ord, item)) + 1.] for item in kwargs["prompt"]],
                           device=kwargs["device"], dtype=self.text_encoder[0].weight.dtype)
        repeats, length = kwargs["num_images_per_prompt"], kwargs["max_sequence_length"]
        tokens = self.text_encoder_2(ids)[:, None].expand(-1, length, -1)
        pooled = self.text_encoder(ids)
        self.output = (tokens.repeat_interleave(repeats, 0), pooled.repeat_interleave(repeats, 0),
                       torch.zeros(length, 3, device=ids.device, dtype=ids.dtype))
        return self.output


def helper(pipeline=None, dtype=torch.float32):
    return FluxTextEncoder(pipeline or Pipeline(), device="cpu", torch_dtype=dtype)


def test_named_fields_prompt_major_repeats_and_embedded_guidance():
    pipeline = Pipeline()
    text = helper(pipeline)
    single = text.encode(("red cube", "blue sphere"), max_sequence_length=5)
    result = text.encode(["red cube", "blue sphere"], max_sequence_length=5,
                         num_images_per_prompt=3)
    assert isinstance(result, FluxConditioning)
    for one, repeated in zip(tensors(single)[:2], tensors(result)[:2]):
        torch.testing.assert_close(repeated, one.repeat_interleave(3, 0))
    torch.testing.assert_close(result.text_ids, torch.zeros(5, 3))
    assert pipeline.calls[-1] == dict(prompt=["red cube", "blue sphere"], prompt_2=None,
                                     device=torch.device("cpu"), num_images_per_prompt=3,
                                     max_sequence_length=5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16, torch.float16])
def test_freeze_fresh_storage_outer_contexts_and_no_cache(dtype):
    pipeline = Pipeline()
    for encoder in encoders(pipeline):
        for parameter in encoder.parameters():
            parameter.grad = torch.ones_like(parameter)
    text = helper(pipeline, dtype)
    for encoder in encoders(pipeline):
        assert not encoder.training
        assert all(not p.requires_grad and p.grad is None and p.dtype == dtype
                   and p.device.type == "cpu" for p in encoder.parameters())
    assert pipeline.transformer.training and pipeline.vae.training
    assert all(p.requires_grad and p.dtype == torch.float32
               for m in (pipeline.transformer, pipeline.vae) for p in m.parameters())
    first = text.encode("red", max_sequence_length=3)
    for encoder in encoders(pipeline):
        encoder.train().requires_grad_(True)
        for parameter in encoder.parameters():
            parameter.grad = torch.ones_like(parameter)
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        second = text.encode("red", max_sequence_length=3)
    for a, b, raw in zip(tensors(first), tensors(second), pipeline.output):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert b.dtype == dtype and b.device.type == "cpu"
        assert not b.requires_grad and b.grad_fn is None and not b.is_inference()
        assert len({a.data_ptr(), b.data_ptr(), raw.data_ptr()}) == 3
    policies = [nn.Linear(value.shape[-1], 1, dtype=dtype) for value in tensors(second)]
    sum(policy(value).sum() for policy, value in zip(policies, tensors(second))).backward()
    assert all(p.weight.grad is not None for p in policies)
    assert all(p.grad is None and not p.requires_grad for e in encoders(pipeline) for p in e.parameters())
    for value in tensors(second):
        value.fill_(123)
    fresh = text.encode("red", max_sequence_length=3)
    for a, b in zip(tensors(first), tensors(fresh)):
        torch.testing.assert_close(a, b)
    changed = text.encode("blue", max_sequence_length=3)
    assert not torch.equal(changed.prompt_embeds, fresh.prompt_embeds)
    assert len(pipeline.calls) == 4


@pytest.mark.parametrize("length", [1, 128, 512])
def test_sequence_bounds(length):
    result = helper().encode("red", max_sequence_length=length)
    assert result.prompt_embeds.shape == (1, length, 2)
    assert result.text_ids.shape == (length, 3)


@pytest.mark.parametrize("prompt,options", [
    (None, {}), ("", {}), (" \t", {}), ([], {}), (["red", ""], {}),
    (["red", 2], {}), ({"red"}, {}), (iter(["red"]), {}),
    ("red", {"max_sequence_length": 0}), ("red", {"max_sequence_length": 513}),
    ("red", {"max_sequence_length": True}), ("red", {"max_sequence_length": 3.5}),
    ("red", {"num_images_per_prompt": 0}), ("red", {"num_images_per_prompt": -1}),
    ("red", {"num_images_per_prompt": True}), ("red", {"num_images_per_prompt": 1.5}),
])
def test_invalid_requests_fail_before_encoding(prompt, options):
    pipeline = Pipeline()
    text = helper(pipeline)
    with pytest.raises(ValueError):
        text.encode(prompt, **(dict(max_sequence_length=4) | options))
    assert pipeline.calls == []


@pytest.mark.parametrize("options", [{"cfg_scale": 1}, {"negative_prompt": ""},
                                     {"guidance_scale": 3.5}, {"prompt_2": "blue"}])
def test_no_cfg_negative_or_override_api(options):
    pipeline = Pipeline()
    with pytest.raises(TypeError):
        helper(pipeline).encode("red", max_sequence_length=4, **options)
    assert pipeline.calls == []


@pytest.mark.parametrize("attribute", ["encode_prompt", "text_encoder", "text_encoder_2",
                                       "tokenizer", "tokenizer_2"])
def test_incomplete_pipeline_rejected_without_partial_freeze(attribute):
    pipeline = Pipeline()
    original = encoders(pipeline)
    setattr(pipeline, attribute, None)
    with pytest.raises(ValueError):
        helper(pipeline)
    assert all(e.training and all(p.requires_grad for p in e.parameters()) for e in original)


@pytest.mark.parametrize("slot,bad", [
    (0, None), (0, torch.zeros(4, 4)), (0, torch.zeros(1, 4, 2)),
    (0, torch.zeros(4, 3, 2)), (0, torch.zeros(4, 4, 0)),
    (0, torch.zeros(4, 4, 2, dtype=torch.int64)),
    (0, torch.zeros(4, 4, 2, device="meta")),
    (0, torch.zeros(4, 4, 2).to_sparse()),
    (1, None), (1, torch.zeros(1, 2)), (1, torch.zeros(4, 2, 1)),
    (1, torch.zeros(4, 0)), (1, torch.zeros(4, 2, dtype=torch.int64)),
    (2, None), (2, torch.zeros(4, 4, 3)), (2, torch.zeros(3, 3)),
    (2, torch.zeros(4, 2)), (2, torch.ones(4, 3)),
    (2, torch.full((4, 3), float("nan"))),
    (2, torch.zeros(4, 3, dtype=torch.int64)),
])
def test_bad_pipeline_outputs_rejected(slot, bad):
    pipeline = Pipeline()
    original = pipeline.encode_prompt

    def invalid(**kwargs):
        output = list(original(**kwargs))
        output[slot] = bad
        return tuple(output)

    pipeline.encode_prompt = invalid
    name = ("prompt_embeds", "pooled_prompt_embeds", "text_ids")[slot]
    with pytest.raises(ValueError, match=name):
        helper(pipeline).encode(["red", "blue"], max_sequence_length=4, num_images_per_prompt=2)


@pytest.mark.parametrize("options", [
    {"model_id_or_path": " "}, {"model_id_or_path": None},
    {"revision": None}, {"revision": ""}, {"revision": " "}, {"revision": 42},
    {"device": None}, {"device": "invalid-device"}, {"device": "meta"},
    {"torch_dtype": None}, {"torch_dtype": "float32"}, {"torch_dtype": torch.int64},
    {"local_files_only": "false"},
])
def test_invalid_loader_options_do_not_import_or_load(monkeypatch, options):
    def fail(*args, **kwargs):
        raise AssertionError("loader must not run for invalid options")

    monkeypatch.setitem(sys.modules, "diffusers", SimpleNamespace(
        FluxPipeline=SimpleNamespace(from_pretrained=fail)))
    arguments = dict(model_id_or_path="org/model", revision="pinned", device="cpu",
                     torch_dtype=torch.float32, local_files_only=False)
    with pytest.raises(ValueError):
        FluxTextEncoder.from_pretrained(**(arguments | options))


@pytest.mark.parametrize("online", [False, True])
def test_loader_text_only_strict_revision_route_and_offline_default(monkeypatch, online):
    calls = []
    pipeline = Pipeline()

    def load(path, **kwargs):
        calls.append((path, kwargs))
        return pipeline

    monkeypatch.setitem(sys.modules, "diffusers", SimpleNamespace(
        FluxPipeline=SimpleNamespace(from_pretrained=load)))
    options = {"local_files_only": False} if online else {}
    result = FluxTextEncoder.from_pretrained(
        Path("local-model"), revision="immutable-revision", device="cpu", torch_dtype=torch.float64,
        **options)
    assert result.pipeline is pipeline
    assert calls == [("local-model", dict(revision="immutable-revision", local_files_only=not online,
                                         torch_dtype=torch.float64, transformer=None, vae=None))]
    with pytest.raises(TypeError, match="revision"):
        FluxTextEncoder.from_pretrained("local-model", device="cpu", torch_dtype=torch.float32)
    assert len(calls) == 1


def test_revision_failure_is_not_retried_without_revision(monkeypatch):
    calls = []

    def load(path, **kwargs):
        calls.append(kwargs)
        raise OSError("revision not cached")

    monkeypatch.setitem(sys.modules, "diffusers", SimpleNamespace(
        FluxPipeline=SimpleNamespace(from_pretrained=load)))
    with pytest.raises(OSError, match="revision not cached"):
        FluxTextEncoder.from_pretrained("org/model", revision="missing-revision",
                                       device="cpu", torch_dtype=torch.float32)
    assert len(calls) == 1 and calls[0]["revision"] == "missing-revision"


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
import visual_rl.models.flux_text as module
assert module.__file__ == {str(source / 'visual_rl/models/flux_text.py')!r}
"""
    result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True,
                            timeout=30, env=dict(os.environ, PYTHONPATH=str(source)))
    assert result.returncode == 0, result.stderr


@pytest.fixture
def real_flux_pipeline():
    """Tiny local CLIP(8)/T5(24), 77-token CLIP; import this fixture in runner tests."""
    diffusers = pytest.importorskip("diffusers")
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")

    def tokenizer(length):
        backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(
            {"<pad>": 0, "<unk>": 1, "red": 2, "blue": 3, "cube": 4,
             "sphere": 5, "negative": 6, "<s>": 7, "</s>": 8}, unk_token="<unk>"))
        backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
        backend.post_processor = tokenizers.processors.TemplateProcessing(
            single="<s> $A </s>" if length == 77 else "$A </s>",
            special_tokens=[("<s>", 7), ("</s>", 8)])
        return transformers.PreTrainedTokenizerFast(
            tokenizer_object=backend, model_max_length=length, pad_token="<pad>",
            unk_token="<unk>", bos_token="<s>", eos_token="</s>")

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        return diffusers.FluxPipeline(
            transformer=None, vae=None, scheduler=diffusers.FlowMatchEulerDiscreteScheduler(),
            text_encoder=transformers.CLIPTextModel(transformers.CLIPTextConfig(
                vocab_size=9, hidden_size=8, intermediate_size=16, projection_dim=5,
                num_hidden_layers=2, num_attention_heads=2, max_position_embeddings=77,
                attention_dropout=.7, pad_token_id=0, bos_token_id=7, eos_token_id=8)),
            text_encoder_2=transformers.T5EncoderModel(transformers.T5Config(
                vocab_size=9, d_model=24, d_ff=32, d_kv=12, num_heads=2, num_layers=1,
                dropout_rate=.7, pad_token_id=0, eos_token_id=8)),
            tokenizer=tokenizer(77), tokenizer_2=tokenizer(512),
        )


def test_real_clip_pool_t5_tokens_repetition_and_dropout(real_flux_pipeline):
    pipeline = real_flux_pipeline
    text = helper(pipeline)
    prompts = ["red cube", "blue sphere"]
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        result = text.encode(prompts, max_sequence_length=5, num_images_per_prompt=3)
    assert result.prompt_embeds.shape == (6, 5, 24)
    assert result.pooled_prompt_embeds.shape == (6, 8)
    assert not hasattr(pipeline.text_encoder, "text_projection")
    with torch.no_grad():
        clip_ids = pipeline.tokenizer(prompts, padding="max_length", max_length=77,
                                      truncation=True, return_tensors="pt").input_ids
        pooled = pipeline.text_encoder(clip_ids, output_hidden_states=False).pooler_output
        t5_ids = pipeline.tokenizer_2(prompts, padding="max_length", max_length=5,
                                    truncation=True, return_tensors="pt").input_ids
        tokens = pipeline.text_encoder_2(t5_ids)[0]
        direct = pipeline.encode_prompt(prompts, device=torch.device("cpu"),
                                        max_sequence_length=5, num_images_per_prompt=3)
    torch.testing.assert_close(result.prompt_embeds, tokens.repeat_interleave(3, 0))
    torch.testing.assert_close(result.pooled_prompt_embeds, pooled.repeat_interleave(3, 0))
    torch.testing.assert_close(result.text_ids, torch.zeros(5, 3))
    assert not torch.equal(tokens[0], tokens[1]) and not torch.equal(pooled[0], pooled[1])
    for a, b in zip(tensors(result), direct):
        torch.testing.assert_close(a, b)
        assert a.dtype == torch.float32 and not a.is_inference() and not a.requires_grad
    for encoder in encoders(pipeline):
        encoder.train()
    again = text.encode(prompts, max_sequence_length=5, num_images_per_prompt=3)
    for a, b in zip(tensors(result), tensors(again)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert text.encode(prompts, max_sequence_length=1).prompt_embeds.shape == (2, 1, 24)


def test_real_clip_truncates_at_77_and_t5_length_is_caller_owned(real_flux_pipeline):
    pipeline = real_flux_pipeline
    observed = []
    handle = pipeline.text_encoder.register_forward_pre_hook(
        lambda module, args: observed.append(tuple(args[0].shape)))
    text = helper(pipeline)
    try:
        first = text.encode("red " * 100 + "cube", max_sequence_length=128)
        second = text.encode("red " * 100 + "sphere", max_sequence_length=128)
    finally:
        handle.remove()
    assert observed == [(1, 77), (1, 77)]
    assert first.prompt_embeds.shape == (1, 128, 24)
    torch.testing.assert_close(first.pooled_prompt_embeds, second.pooled_prompt_embeds)
    assert not torch.equal(first.prompt_embeds, second.prompt_embeds)


def test_real_local_pretrained_text_assets_only(real_flux_pipeline, tmp_path):
    real_flux_pipeline.save_pretrained(tmp_path, safe_serialization=True)
    index = tmp_path / "model_index.json"
    config = json.loads(index.read_text())
    config["transformer"] = ["diffusers", "FluxTransformer2DModel"]
    config["vae"] = ["diffusers", "AutoencoderKL"]
    index.write_text(json.dumps(config))
    assert not (tmp_path / "transformer").exists() and not (tmp_path / "vae").exists()
    loaded = FluxTextEncoder.from_pretrained(
        tmp_path, revision="offline-fixture", device="cpu", torch_dtype=torch.float32)
    assert loaded.pipeline.transformer is None and loaded.pipeline.vae is None
    assert type(loaded.pipeline.text_encoder).__name__ == "CLIPTextModel"
    assert type(loaded.pipeline.text_encoder_2).__name__ == "T5EncoderModel"
    expected = helper(real_flux_pipeline).encode(["red cube", "blue sphere"], max_sequence_length=5,
                                               num_images_per_prompt=2)
    actual = loaded.encode(["red cube", "blue sphere"], max_sequence_length=5,
                           num_images_per_prompt=2)
    for a, b in zip(tensors(expected), tensors(actual)):
        torch.testing.assert_close(a, b)
    assert all(not e.training and all(not p.requires_grad for p in e.parameters())
               for e in encoders(loaded.pipeline))


def test_real_hub_cache_uses_exact_revision_offline(real_flux_pipeline, tmp_path, monkeypatch):
    from huggingface_hub import constants
    from huggingface_hub.errors import LocalEntryNotFoundError

    revision = "a" * 40
    cache = tmp_path / "hub"
    repo = cache / "models--offline-fixtures--tiny-flux"
    snapshot = repo / "snapshots" / revision
    real_flux_pipeline.save_pretrained(snapshot, safe_serialization=True)
    # A cached main must not be used as a fallback for a missing explicit revision.
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(revision)
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(cache))
    loaded = FluxTextEncoder.from_pretrained(
        "offline-fixtures/tiny-flux", revision=revision, device="cpu", torch_dtype=torch.float32)
    expected = helper(real_flux_pipeline).encode("red cube", max_sequence_length=5)
    actual = loaded.encode("red cube", max_sequence_length=5)
    for a, b in zip(tensors(expected), tensors(actual)):
        torch.testing.assert_close(a, b)
    with pytest.raises(LocalEntryNotFoundError):
        FluxTextEncoder.from_pretrained(
            "offline-fixtures/tiny-flux", revision="b" * 40, device="cpu", torch_dtype=torch.float32)


@pytest.mark.parametrize("guidance", [None, 3.5])
def test_real_flux_adapter_gradients_and_replay_pack_conditioning(real_flux_pipeline, guidance):
    diffusers = pytest.importorskip("diffusers")
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        conditioning = helper(real_flux_pipeline).encode(
            ["red cube", "blue sphere"], max_sequence_length=5, num_images_per_prompt=2)
    model = diffusers.FluxTransformer2DModel(
        patch_size=1, in_channels=8, out_channels=8, num_layers=1, num_single_layers=1,
        attention_head_dim=16, num_attention_heads=2, joint_attention_dim=24,
        pooled_projection_dim=8, guidance_embeds=guidance is not None, axes_dims_rope=(4, 6, 6),
    ).eval()
    adapter = FluxAdapter(model, nn.Linear(1, 1), FluxConfig(
        1., 0., latent_channels=2, embedded_guidance_scale=guidance))
    policy = ConditionedImagePolicy(adapter, conditioning)
    rows = torch.tensor([3, 0, 3, 1, 2, 0])
    latents = torch.randn(6, 2, 4, 6, requires_grad=True)
    sigma = torch.linspace(.1, .6, 6)
    calls = []
    handle = model.register_forward_pre_hook(lambda module, args, kwargs: calls.append(kwargs),
                                              with_kwargs=True)
    try:
        output = policy(latents, sigma, rows)
    finally:
        handle.remove()
    call = calls[0]
    torch.testing.assert_close(call["hidden_states"], pack_latents(latents))
    torch.testing.assert_close(call["encoder_hidden_states"], conditioning.prompt_embeds[rows])
    torch.testing.assert_close(call["pooled_projections"], conditioning.pooled_prompt_embeds[rows])
    torch.testing.assert_close(call["txt_ids"], conditioning.text_ids)
    assert call["txt_ids"].shape == (5, 3)
    if guidance is None:
        assert call["guidance"] is None
    else:
        torch.testing.assert_close(call["guidance"], torch.full((6,), guidance))
    assert output.shape == latents.shape and torch.isfinite(output).all()
    output.square().mean().backward()
    for parameter in (model.context_embedder.weight,
                      model.time_text_embed.text_embedder.linear_1.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.count_nonzero() > 0
    assert latents.grad is not None and torch.isfinite(latents.grad).all()
    assert all(p.grad is None and not p.requires_grad
               for e in encoders(real_flux_pipeline) for p in e.parameters())
