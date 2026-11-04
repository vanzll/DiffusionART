import json
import struct
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from visual_rl.models.ltx_video import (
    LTX_VIDEO_TEXT_ENCODER_ID,
    LTX_VIDEO_TEXT_ENCODER_REVISION,
    LTXVideoAdapter,
    LTXVideoConditioning,
    LTXVideoConfig,
    encode_ltx_video_prompts,
    load_ltx_video_components,
    ltx_video_coordinates,
    ltx_video_latent_shape,
    read_ltx_video_checkpoint_config,
)
from visual_rl.models.ltx_video_sampling import (
    LTX_VIDEO_ALLOWED_INFERENCE_STEPS,
    LTX_VIDEO_CHECKPOINT,
    LTX_VIDEO_CODE_REVISION,
)


class NativeTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5))
        self.calls = []

    def forward(self, hidden_states, **kwargs):
        self.calls.append((hidden_states, kwargs))
        text = kwargs["encoder_hidden_states"]
        mask = kwargs["encoder_attention_mask"].unsqueeze(-1)
        context = (text * mask).sum((1, 2)) / mask.sum((1, 2))
        return (hidden_states * self.weight + context[:, None, None],)


class NativeVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))
        self.register_buffer("mean_of_means", torch.tensor([1.0, -2.0]))
        self.register_buffer("std_of_means", torch.tensor([2.0, 4.0]))
        self.calls = []
        self.decoder = nn.Identity()

    def decode(self, latents, **kwargs):
        self.calls.append((latents, kwargs))
        return (
            self.decoder(latents.mean(1, keepdim=True).expand(-1, 3, -1, -1, -1) * self.weight),
        )


def adapter():
    return LTXVideoAdapter(
        NativeTransformer(), NativeVAE(), LTXVideoConfig(latent_channels=2, caption_channels=3)
    )


def test_real_batch_variable_length_text_masks_positions_and_sigma():
    model = adapter()
    latents = torch.randn(2, 2, 3, 2, 2)
    text = torch.tensor(
        [[[1.0, 2.0, 3.0], [999.0, 999.0, 999.0]], [[2.0, 4.0, 6.0], [3.0, 6.0, 9.0]]]
    )
    mask = torch.tensor([[1, 0], [1, 1]])
    c = LTXVideoConditioning(text, mask, frame_rate=30.0)
    velocity = model.predict_velocity(latents, torch.tensor([1.0, 0.4219]), c)
    assert len(model.transformer.calls) == 1
    packed, kwargs = model.transformer.calls[0]
    assert packed.shape == (2, 12, 2)
    torch.testing.assert_close(kwargs["timestep"], torch.tensor([[1.0], [0.4219]]))
    assert torch.equal(kwargs["encoder_attention_mask"], mask)
    torch.testing.assert_close(velocity[0], latents[0] * 0.5 + 6)
    torch.testing.assert_close(velocity[1], latents[1] * 0.5 + 15)
    coords = kwargs["indices_grid"]
    torch.testing.assert_close(coords[0, 0, ::4], torch.tensor([0.0, 1 / 30, 9 / 30]))
    torch.testing.assert_close(coords[0, 1, :4], torch.tensor([0.0, 0.0, 32.0, 32.0]))
    assert torch.equal(coords[0], coords[1])
    assert c.select([1]).attention_mask.shape == (1, 2)


def test_bf16_native_forward_overrides_outer_amp_and_explicitly_casts_cached_states():
    model = adapter()
    model.transformer.bfloat16()
    observed = []
    hook = model.transformer.register_forward_pre_hook(
        lambda module, args, kwargs: observed.append(
            (
                torch.is_autocast_enabled("cpu"),
                args[0].dtype,
                args[0].stride(),
                kwargs["encoder_hidden_states"].dtype,
                kwargs["timestep"].dtype,
            )
        ),
        with_kwargs=True,
    )
    # Collector states are FP32; native packed inputs remain explicitly BF16.
    packed = torch.arange(16.0).reshape(2, 4, 2)
    latents = packed.reshape(2, 1, 2, 2, 2).permute(0, 4, 1, 2, 3)
    c = LTXVideoConditioning(torch.ones(2, 2, 3).bfloat16(), torch.ones(2, 2))
    expected = model.predict_velocity(latents, 0.725, c)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert torch.is_autocast_enabled("cpu")
        actual = model.predict_velocity(latents, 0.725, c)
        assert torch.is_autocast_enabled("cpu")
    hook.remove()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert len(observed) == 2 and observed[0] == observed[1]
    enabled, dtype, stride, text_dtype, time_dtype = observed[1]
    assert not enabled and dtype == text_dtype == torch.bfloat16
    assert time_dtype == torch.float32 and stride == (8, 2, 1)


def test_valid_five_second_native_grid_and_causal_compression():
    assert ltx_video_latent_shape(2) == (2, 128, 20, 22, 38)
    assert 153 / 30 == 5.1
    for height, width, frames in [(703, 1216, 153), (704, 1215, 153), (704, 1216, 150)]:
        with pytest.raises(ValueError):
            ltx_video_latent_shape(2, height, width, frames)
    with pytest.raises(ValueError):
        ltx_video_coordinates((2, 128, 2, 2, 2), frame_rate=0)


def test_conditioned_decode_noise_external_statistics_and_latent_gradient():
    model = adapter().train()
    assert not model.vae.training and not model.vae.weight.requires_grad
    z = torch.ones(2, 2, 2, 1, 1, requires_grad=True)
    noise = torch.full_like(z, 3.0)
    pixels = model.decode(z, innovation=noise)
    raw, kwargs = model.vae.calls[0]
    # 0.975*1+0.025*3=1.05, then external per-channel inverse.
    torch.testing.assert_close(raw[:, 0], torch.full_like(raw[:, 0], 3.1))
    torch.testing.assert_close(raw[:, 1], torch.full_like(raw[:, 1], 2.2))
    torch.testing.assert_close(kwargs["timestep"], torch.tensor([0.05, 0.05]))
    assert kwargs["target_shape"] == (2, 3, 16, 32, 32)
    pixels.sum().backward()
    assert z.grad is not None and (z.grad > 0).all()
    assert model.vae.weight.grad is None


def test_decode_reproducibility_and_nonfinite_fail_closed():
    model = adapter()
    z = torch.ones(2, 2, 2, 1, 1)
    a = model.decode(z, generator=torch.Generator().manual_seed(42))
    b = model.decode(z, generator=torch.Generator().manual_seed(42))
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    with pytest.raises(ValueError, match="finite"):
        model.decode(z, innovation=torch.full_like(z, float("nan")))
    model.vae.weight.data.fill_(float("inf"))
    with pytest.raises(ValueError, match="nonfinite"):
        model.decode(z, innovation=torch.zeros_like(z))


def test_decode_default_and_generator_preserve_native_unpacked_strides():
    model = adapter()
    # Official prepare_latents/unpatchify creates C-stride1 unpacked views.
    z = torch.arange(16.0).reshape(2, 2, 1, 2, 2).permute(0, 4, 1, 2, 3)
    assert not z.is_contiguous()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(719)
        official_noise = torch.randn_like(z)
        expected = model.decode(z, innovation=official_noise)
        torch.manual_seed(719)
        actual = model.decode(z)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    explicit = model.decode(z, generator=torch.Generator().manual_seed(719))
    torch.testing.assert_close(explicit, expected, rtol=0, atol=0)


def test_mask_and_statistics_validation():
    model = adapter()
    z = torch.ones(2, 2, 1, 1, 1)
    for mask in [torch.zeros(2, 2), torch.tensor([[1.0, 0.5], [1.0, 1.0]])]:
        with pytest.raises(ValueError, match="mask"):
            model.predict_velocity(z, 1.0, LTXVideoConditioning(torch.ones(2, 2, 3), mask))
    with pytest.raises(ValueError, match="no implicit repeat"):
        model.predict_velocity(z, 1.0, LTXVideoConditioning(torch.ones(1, 2, 3), torch.ones(1, 2)))
    vae = NativeVAE()
    vae.std_of_means[0] = 0
    with pytest.raises(ValueError, match="std_of_means"):
        LTXVideoAdapter(
            NativeTransformer(), vae, LTXVideoConfig(latent_channels=2, caption_channels=3)
        )


def test_upstream_tiling_cannot_silently_drop_decode_conditioning():
    model = adapter()
    model.vae.use_hw_tiling = True
    with pytest.raises(ValueError, match="timestep conditioning"):
        model.decode(torch.ones(2, 2, 2, 1, 1))


def test_batch_text_encoder_preserves_native_mask_and_strips_without_enhancement():
    class Tokenizer:
        def __call__(self, texts, **kwargs):
            assert texts == ["short", "two words"]
            assert kwargs["max_length"] == 256 and kwargs["padding"] == "max_length"
            return SimpleNamespace(
                input_ids=torch.tensor([[1, 0, 0], [2, 3, 0]]),
                attention_mask=torch.tensor([[1, 0, 0], [1, 1, 0]]),
            )

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))
            self.calls = 0

        def forward(self, ids, attention_mask):
            self.calls += 1
            return (ids[..., None].expand(-1, -1, 3) * self.weight,)

    encoder = Encoder()
    c = encode_ltx_video_prompts([" short ", "two words"], Tokenizer(), encoder)
    assert encoder.calls == 1
    assert torch.equal(c.attention_mask, torch.tensor([[1, 0, 0], [1, 1, 0]]))
    assert not c.prompt_embeds.requires_grad


def test_bounded_header_identity_exact_checkpoint_and_native_schedule(tmp_path):
    checkpoint = tmp_path / LTX_VIDEO_CHECKPOINT
    config = {
        "transformer": {"in_channels": 128, "out_channels": 128},
        "vae": {"latent_channels": 128},
        "scheduler": {"sampler": "LinearQuadratic", "shifting": None},
        "allowed_inference_steps": list(LTX_VIDEO_ALLOWED_INFERENCE_STEPS),
    }

    def write(cfg):
        header = json.dumps({"__metadata__": {"config": json.dumps(cfg)}}).encode()
        checkpoint.write_bytes(struct.pack("<Q", len(header)) + header)

    write(config)
    assert read_ltx_video_checkpoint_config(checkpoint) == config
    config["allowed_inference_steps"] = [1.0, 0.5]
    write(config)
    with pytest.raises(ValueError, match="metadata"):
        read_ltx_video_checkpoint_config(checkpoint)
    with pytest.raises(ValueError, match="exact"):
        read_ltx_video_checkpoint_config(tmp_path / "ltx2.safetensors")
    checkpoint.write_bytes(struct.pack("<Q", 1 << 30))
    with pytest.raises(ValueError, match="excessive"):
        read_ltx_video_checkpoint_config(checkpoint)


@pytest.fixture
def native_loader_inputs(tmp_path):
    checkpoint = tmp_path / LTX_VIDEO_CHECKPOINT
    config = {
        "transformer": {"in_channels": 128, "out_channels": 128},
        "vae": {"latent_channels": 128},
        "scheduler": {"sampler": "LinearQuadratic", "shifting": None},
        "allowed_inference_steps": list(LTX_VIDEO_ALLOWED_INFERENCE_STEPS),
    }
    header = json.dumps({"__metadata__": {"config": json.dumps(config)}}).encode()
    checkpoint.write_bytes(struct.pack("<Q", len(header)) + header)
    native_root = tmp_path / "native"
    (native_root / "ltx_video").mkdir(parents=True)
    (native_root / "ltx_video" / "__init__.py").write_text("")
    return {
        "checkpoint": checkpoint,
        "text_encoder_path": LTX_VIDEO_TEXT_ENCODER_ID,
        "text_encoder_revision": LTX_VIDEO_TEXT_ENCODER_REVISION,
        "native_root": native_root,
    }


def test_native_loader_requires_explicit_source_root(native_loader_inputs):
    values = dict(native_loader_inputs)
    values.pop("native_root")
    with pytest.raises(TypeError, match="native_root"):
        load_ltx_video_components(**values)
    values["native_root"] = None
    with pytest.raises(ValueError, match="explicit source root"):
        load_ltx_video_components(**values)


@pytest.mark.parametrize(
    "revision,dirty", [("0" * 40, ""), (LTX_VIDEO_CODE_REVISION, " M native.py")]
)
def test_native_loader_rejects_wrong_sha_or_dirty_source(
    native_loader_inputs, monkeypatch, revision, dirty
):
    from visual_rl.models import source_pin

    calls = []

    def git_result(command, **kwargs):
        calls.append(command)
        assert kwargs["timeout"] == 15
        if "--show-toplevel" in command:
            return SimpleNamespace(stdout=str(native_loader_inputs["native_root"]))
        return SimpleNamespace(stdout=revision if "rev-parse" in command else dirty)

    monkeypatch.setattr(source_pin.subprocess, "run", git_result)
    with pytest.raises(ValueError, match="clean at its pinned revision"):
        load_ltx_video_components(**native_loader_inputs)
    assert calls
    if dirty:
        assert any("status" in command for command in calls)


def test_native_loader_rejects_an_already_imported_wrong_package_origin(
    native_loader_inputs, monkeypatch, tmp_path
):
    from visual_rl.models import source_pin

    def git_result(command, **kwargs):
        if "--show-toplevel" in command:
            return SimpleNamespace(stdout=str(native_loader_inputs["native_root"]))
        return SimpleNamespace(stdout=LTX_VIDEO_CODE_REVISION if "rev-parse" in command else "")

    monkeypatch.setattr(source_pin.subprocess, "run", git_result)
    monkeypatch.setitem(
        sys.modules,
        "ltx_video",
        SimpleNamespace(__file__=str(tmp_path / "untrusted" / "__init__.py")),
    )
    with pytest.raises(ValueError, match="another native package.*origin"):
        load_ltx_video_components(**native_loader_inputs)


def test_native_loader_rejects_cached_component_from_another_source(
    native_loader_inputs, monkeypatch, tmp_path
):
    from visual_rl.models import ltx_video

    def pinned_package(root, *, package, revision):
        assert root == native_loader_inputs["native_root"]
        assert package == "ltx_video" and revision == LTX_VIDEO_CODE_REVISION
        return SimpleNamespace(__file__=str(root / "ltx_video" / "__init__.py"))

    monkeypatch.setattr(ltx_video, "import_pinned_source", pinned_package)
    wrong_component = SimpleNamespace(__file__=str(tmp_path / "untrusted" / "vae.py"))
    monkeypatch.setattr(ltx_video.importlib, "import_module", lambda name: wrong_component)
    with pytest.raises(ValueError, match="component origin"):
        load_ltx_video_components(**native_loader_inputs)


@pytest.mark.parametrize(
    "repo,revision",
    [
        (LTX_VIDEO_TEXT_ENCODER_ID, None),
        (LTX_VIDEO_TEXT_ENCODER_ID, "main"),
        (LTX_VIDEO_TEXT_ENCODER_ID, "0" * 40),
        ("other/T5", LTX_VIDEO_TEXT_ENCODER_REVISION),
    ],
)
def test_native_loader_rejects_unpinned_remote_text_assets(native_loader_inputs, repo, revision):
    values = dict(native_loader_inputs, text_encoder_path=repo, text_encoder_revision=revision)
    with pytest.raises(ValueError, match="exact pinned PixArt"):
        load_ltx_video_components(**values)
