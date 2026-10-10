"""CPU contract tests; no model assets, network, or research imports."""

import os
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from visual_rl.models import (
    FluxAdapter,
    FluxConditioning,
    FluxConfig,
    HunyuanVideoAdapter,
    HunyuanVideoConditioning,
    HunyuanVideoConfig,
    SD3Adapter,
    SD3Conditioning,
    SD3Config,
    WanI2VAdapter,
    WanI2VConditioning,
    WanI2VConfig,
    first_frame_condition,
    pack_latents,
    unpack_latents,
)


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))
        self.last = None

    def decode(self, z, return_dict=False):
        self.last = z
        return (z * self.weight,)


class Field(nn.Module):
    def __init__(self, channels=2):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.7))
        self.channels = channels
        self.calls = []

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        z = kwargs["hidden_states"]
        if z.ndim != 3:
            z = z[:, : self.channels]
        prompt = kwargs["encoder_hidden_states"].mean(
            tuple(range(1, kwargs["encoder_hidden_states"].ndim))
        )
        shape = (-1,) + (1,) * (z.ndim - 1)
        return (z * self.weight + prompt.reshape(shape) + kwargs["timestep"].reshape(shape) / 1000,)


def sd3_condition():
    return SD3Conditioning(
        torch.tensor([2.0, 3.0]).view(2, 1, 1).requires_grad_(),
        torch.ones(2, 4),
        torch.tensor([-1.0, -2.0]).view(2, 1, 1).requires_grad_(),
        torch.zeros(2, 4),
    )


@pytest.mark.parametrize("scale", [None, 0.0, 1.0, 4.0])
def test_sd3_cfg_order_timestep_and_branch_gradients(scale):
    field, vae = Field(), VAE()
    adapter = SD3Adapter(field, vae, SD3Config(2.0, 0.25, latent_channels=2, cfg_scale=scale))
    z = torch.randn(2, 2, 4, 6, requires_grad=True)
    sigma = torch.tensor([0.1234, 0.7891])
    c = sd3_condition()
    result = adapter.predict_velocity(z, sigma, c)
    call = field.calls[-1]
    expected_prompt = c.prompt_embeds.flatten()
    if scale is not None:
        torch.testing.assert_close(
            call["encoder_hidden_states"], torch.cat([c.negative_prompt_embeds, c.prompt_embeds])
        )
        torch.testing.assert_close(
            call["pooled_projections"],
            torch.cat([c.negative_pooled_prompt_embeds, c.pooled_prompt_embeds]),
        )
        torch.testing.assert_close(call["timestep"], sigma.repeat(2) * 1000)
        expected_prompt = c.negative_prompt_embeds.flatten() + scale * (
            expected_prompt - c.negative_prompt_embeds.flatten()
        )
    else:
        torch.testing.assert_close(call["timestep"], sigma * 1000)
    expected = z * field.weight + (expected_prompt + sigma).view(2, 1, 1, 1)
    torch.testing.assert_close(result, expected)
    result.sum().backward()
    torch.testing.assert_close(z.grad, torch.full_like(z, 0.7))
    per_row = z[0].numel()
    torch.testing.assert_close(
        c.prompt_embeds.grad,
        torch.full_like(c.prompt_embeds, per_row * (1 if scale is None else scale)),
    )
    if scale is not None:
        torch.testing.assert_close(
            c.negative_prompt_embeds.grad,
            torch.full_like(c.negative_prompt_embeds, per_row * (1 - scale)),
        )
    assert field.weight.grad is not None
    assert vae.weight.grad is None


@pytest.mark.parametrize("family", ["sd3", "flux", "hy"])
def test_affine_decode_once_and_frozen_vae_refl_chain(family):
    field, vae = Field(), VAE()
    if family == "sd3":
        adapter = SD3Adapter(field, vae, SD3Config(2.0, 0.25, latent_channels=2))
        z = torch.randn(2, 2, 4, 6, requires_grad=True)
        conditioning = sd3_condition()
        shift = 0.25
    elif family == "flux":
        adapter = FluxAdapter(field, vae, FluxConfig(2.0, 0.25, latent_channels=2))
        z = torch.randn(2, 2, 4, 6, requires_grad=True)
        conditioning = FluxConditioning(torch.ones(2, 3, 4), torch.ones(2, 4), torch.zeros(3, 3))
        shift = 0.25
    else:
        adapter = HunyuanVideoAdapter(
            field, vae, HunyuanVideoConfig(scaling_factor=2.0, latent_channels=2)
        )
        z = torch.randn(2, 2, 3, 4, 6, requires_grad=True)
        conditioning = HunyuanVideoConditioning(torch.ones(2, 4, 4), torch.ones(2, 3))
        shift = 0.0
    vae.weight.grad = None
    adapter.train()
    assert adapter.trainable_module is field
    assert field.training and not vae.training and not vae.weight.requires_grad
    pixels = adapter.decode(z)
    torch.testing.assert_close(vae.last, z / 2 + shift)
    pixels.sum().backward()
    torch.testing.assert_close(z.grad, torch.ones_like(z))
    assert vae.weight.grad is None
    velocity = adapter(z, 0.5, conditioning)
    x0 = z - 0.5 * velocity
    loss = adapter.decode(x0).square().mean()
    loss.backward()
    assert torch.isfinite(field.weight.grad) and field.weight.grad.abs() > 0
    assert vae.weight.grad is None


def test_flux_pack_exact_order_rectangular_noncontiguous_and_gradient():
    z = torch.arange(2 * 2 * 4 * 6.0).reshape(2, 2, 6, 4).transpose(-1, -2).requires_grad_()
    packed = pack_latents(z)
    oracle = torch.stack(
        [
            torch.stack(
                [
                    z[b, :, r : r + 2, c : c + 2].reshape(-1)
                    for r in range(0, 4, 2)
                    for c in range(0, 6, 2)
                ]
            )
            for b in range(2)
        ]
    )
    torch.testing.assert_close(packed, oracle)
    torch.testing.assert_close(unpack_latents(packed, 4, 6), z)
    packed.square().sum().backward()
    torch.testing.assert_close(z.grad, 2 * z)


@pytest.mark.parametrize("guidance", [None, 3.5])
def test_flux_time_guidance_and_position_ids(guidance):
    field = Field()
    adapter = FluxAdapter(
        field, VAE(), FluxConfig(2.0, 0.25, latent_channels=2, embedded_guidance_scale=guidance)
    )
    z = torch.randn(2, 2, 4, 6, dtype=torch.bfloat16)
    c = FluxConditioning(torch.ones(2, 3, 4), torch.ones(2, 4), torch.zeros(3, 3))
    sigma = torch.tensor([0.1237, 0.8753])
    output = adapter(z, sigma, c)
    call = field.calls[-1]
    torch.testing.assert_close(call["timestep"], (sigma * 1000).to(torch.bfloat16) / 1000)
    torch.testing.assert_close(
        call["img_ids"].float(),
        torch.tensor([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 1, 1], [0, 1, 2]]).float(),
    )
    assert call["txt_ids"] is c.text_ids
    assert output.shape == z.shape
    if guidance is None:
        assert call["guidance"] is None
    else:
        torch.testing.assert_close(call["guidance"], torch.full((2,), guidance))


def test_hunyuan_native_abi_and_historical_bfloat_guidance_rounding():
    field = Field()
    adapter = HunyuanVideoAdapter(field, VAE(), HunyuanVideoConfig(latent_channels=2))
    z = torch.randn(2, 2, 3, 4, 6, requires_grad=True)
    c = HunyuanVideoConditioning(torch.randn(2, 4, 6), torch.ones(2, 3))
    result = adapter(z, torch.tensor([0.1239, 0.7899]), c)
    call = field.calls[-1]
    assert set(call) == {
        "hidden_states",
        "encoder_hidden_states",
        "timestep",
        "guidance",
        "encoder_attention_mask",
        "return_dict",
    }
    assert call["encoder_hidden_states"] is c.prompt_embeds
    assert call["encoder_attention_mask"] is c.attention_mask
    assert call["timestep"].tolist() == [123, 789]
    assert call["timestep"].dtype == torch.long
    assert call["guidance"].shape == (1,)
    assert call["guidance"].dtype == torch.bfloat16
    assert call["guidance"].item() == 6016  # Public 6018 rounds to this in BF16.
    result.sum().backward()
    torch.testing.assert_close(z.grad, torch.full_like(z, 0.7))


@pytest.mark.parametrize("cfg", [None, 0.0, 1.0, 5.0])
def test_wan_conditional_first_cfg_and_image_condition_kept_in_both_branches(cfg):
    field = Field()
    adapter = WanI2VAdapter(field, VAE(), WanI2VConfig((0.1, -0.2), (2.0, 3.0), cfg_scale=cfg))
    z = torch.randn(2, 2, 3, 4, 6, requires_grad=True)
    c = WanI2VConditioning(
        torch.randn(2, 3, 4, requires_grad=True),
        torch.randn(2, 2, 5),
        first_frame_condition(torch.randn_like(z)),
        torch.randn(2, 3, 4, requires_grad=True),
    )
    sigma = torch.tensor([0.1234, 0.7891])
    result = adapter(z, sigma, c)
    call = field.calls[-1]
    hidden = torch.cat([z, c.image_condition], 1)
    prompt = c.prompt_embeds.mean((1, 2))
    if cfg is not None:
        torch.testing.assert_close(call["hidden_states"], torch.cat([hidden, hidden]))
        torch.testing.assert_close(
            call["encoder_hidden_states"], torch.cat([c.prompt_embeds, c.negative_prompt_embeds])
        )
        torch.testing.assert_close(
            call["encoder_hidden_states_image"], c.image_embeds.repeat(2, 1, 1)
        )
        torch.testing.assert_close(call["timestep"], sigma.repeat(2) * 1000)
        negative = c.negative_prompt_embeds.mean((1, 2))
        prompt = negative + cfg * (prompt - negative)
    else:
        torch.testing.assert_close(call["hidden_states"], hidden)
    torch.testing.assert_close(
        result, z * 0.7 + (prompt + sigma).view(2, 1, 1, 1, 1), atol=2e-6, rtol=2e-6
    )
    result.sum().backward()
    torch.testing.assert_close(z.grad, torch.full_like(z, 0.7))
    row_factor = z[0].numel() / c.prompt_embeds[0].numel()
    torch.testing.assert_close(
        c.prompt_embeds.grad,
        torch.full_like(c.prompt_embeds, row_factor * (1 if cfg is None else cfg)),
    )
    if cfg is not None:
        torch.testing.assert_close(
            c.negative_prompt_embeds.grad,
            torch.full_like(c.negative_prompt_embeds, row_factor * (1 - cfg)),
        )


def test_wan_normalization_and_first_frame_mask_not_state_replacement():
    adapter = WanI2VAdapter(Field(), VAE(), WanI2VConfig((0.1, -0.2), (2.0, 3.0)))
    z = torch.randn(2, 2, 3, 4, 6, requires_grad=True)
    raw = adapter.to_vae_latents(z)
    expected = z * torch.tensor([2.0, 3.0]).view(1, 2, 1, 1, 1) + torch.tensor([0.1, -0.2]).view(
        1, 2, 1, 1, 1
    )
    torch.testing.assert_close(raw, expected)
    raw.sum().backward()
    torch.testing.assert_close(z.grad, torch.tensor([2.0, 3.0]).view(1, 2, 1, 1, 1).expand_as(z))
    condition = first_frame_condition(z)
    assert condition.shape == (2, 6, 3, 4, 6)
    assert torch.all(condition[:, :4, 0] == 1) and torch.all(condition[:, :4, 1:] == 0)
    torch.testing.assert_close(condition[:, 4:], z)


@pytest.mark.parametrize(
    "sigma", [-0.01, 1.01, float("nan"), float("inf"), torch.zeros(2, 1), torch.ones(3)]
)
def test_reject_invalid_sigma(sigma):
    adapter = SD3Adapter(Field(), VAE(), SD3Config(2.0, 0.0, latent_channels=2))
    with pytest.raises(ValueError, match="sigma"):
        adapter(torch.ones(2, 2, 4, 6), sigma, sd3_condition())


@pytest.mark.parametrize("sigma", [0.0, 1.0, torch.tensor([0.2]), torch.tensor([0.2, 0.3])])
def test_sigma_broadcast_is_explicit(sigma):
    adapter = SD3Adapter(Field(), VAE(), SD3Config(2.0, 0.0, latent_channels=2))
    result = adapter(torch.ones(2, 2, 4, 6), sigma, sd3_condition())
    assert result.shape == (2, 2, 4, 6)


def test_invalid_conditioning_shapes_and_missing_cfg_branches():
    adapter = SD3Adapter(Field(), VAE(), SD3Config(2.0, 0.0, latent_channels=2, cfg_scale=4.0))
    z = torch.ones(2, 2, 4, 6)
    with pytest.raises(ValueError, match="negative_prompt_embeds"):
        adapter(z, 0.5, replace(sd3_condition(), negative_prompt_embeds=None))
    with pytest.raises(ValueError, match="batch"):
        adapter(z, 0.5, replace(sd3_condition(), prompt_embeds=torch.zeros(1, 1, 1)))
    with pytest.raises(ValueError, match="shapes"):
        adapter(z, 0.5, replace(sd3_condition(), negative_prompt_embeds=torch.zeros(2, 2, 1)))
    with pytest.raises(ValueError, match="channels"):
        adapter(torch.ones(2, 3, 4, 6), 0.5, sd3_condition())


def test_invalid_geometry_and_config():
    with pytest.raises(ValueError, match="even"):
        pack_latents(torch.zeros(2, 2, 3, 4))
    with pytest.raises(ValueError, match="geometry"):
        unpack_latents(torch.zeros(2, 3, 8), 4, 4)
    with pytest.raises(ValueError, match="scaling_factor"):
        SD3Config(0.0, 0.0)
    with pytest.raises(ValueError, match="latents_std"):
        WanI2VConfig((0.0,), (-1.0,))
    with pytest.raises(ValueError, match="guidance"):
        FluxConfig(1.0, 0.0, embedded_guidance_scale=float("nan"))
    field = Field()
    field.config = SimpleNamespace(guidance_embeds=False)
    with pytest.raises(ValueError, match="guidance_embeds"):
        FluxAdapter(field, VAE(), FluxConfig(1.0, 0.0))


def test_wan_rejects_last_frame_conditioning_and_condition_channel_velocity():
    z = torch.zeros(2, 2, 3, 4, 6)
    c = WanI2VConditioning(
        torch.ones(2, 3, 4), torch.ones(2, 2, 5), first_frame_condition(z), torch.zeros(2, 3, 4)
    )
    adapter = WanI2VAdapter(Field(), VAE(), WanI2VConfig((0.0, 0.0), (1.0, 1.0)))
    c.image_condition[:, :4, -1] = 1
    with pytest.raises(ValueError, match="first-frame"):
        adapter(z, 0.5, c)
    c.image_condition[:, :4, -1] = 0
    adapter.transformer = Field(channels=8)
    with pytest.raises(ValueError, match="exclude conditioning"):
        adapter(z, 0.5, c)


def test_decode_rejects_inference_only_vae():
    vae = VAE()
    vae.decode = torch.no_grad()(vae.decode)
    adapter = SD3Adapter(Field(), vae, SD3Config(2.0, 0.25, latent_channels=2))
    with pytest.raises(RuntimeError, match="detached"):
        adapter.decode(torch.ones(2, 2, 4, 6, requires_grad=True))


@pytest.mark.parametrize(
    "adapter_cls, config, vae_config",
    [
        (SD3Adapter, SD3Config(2.0, 0.25), SimpleNamespace(scaling_factor=3.0)),
        (FluxAdapter, FluxConfig(2.0, 0.25), SimpleNamespace(shift_factor=0.0)),
        (HunyuanVideoAdapter, HunyuanVideoConfig(), SimpleNamespace(shift_factor=0.1)),
        (WanI2VAdapter, WanI2VConfig((0.0,), (1.0,)), SimpleNamespace(latents_std=[2.0])),
    ],
)
def test_declared_vae_metadata_must_match_explicit_config(adapter_cls, config, vae_config):
    vae = VAE()
    vae.config = vae_config
    with pytest.raises(ValueError, match="component config"):
        adapter_cls(Field(), vae, config)
    assert vae.weight.requires_grad  # Validation precedes the freeze side effect.


def test_decode_disables_outer_autocast():
    class LinearVAE(VAE):
        def decode(self, z, return_dict=False):
            assert not torch.is_autocast_enabled("cpu")
            return (z * self.weight,)

    adapter = SD3Adapter(Field(), LinearVAE(), SD3Config(2.0, 0.25, latent_channels=2))
    z = torch.ones(2, 2, 4, 6, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = adapter.decode(z)
    assert output.dtype == torch.float32
    output.sum().backward()
    torch.testing.assert_close(z.grad, torch.ones_like(z))


def test_import_does_not_load_optional_backends_or_contact_network():
    code = """
import sys
class Reject:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'diffusers', 'transformers', 'huggingface_hub', 'fastvideo'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Reject())
import visual_rl.models
"""
    subprocess.run(
        [sys.executable, "-B", "-c", code], check=True, timeout=30, env=os.environ.copy()
    )
