"""Tiny real Diffusers modules. Optional dependency, no pretrained downloads."""

from types import SimpleNamespace

import pytest
import torch

from visual_rl.models import (
    FluxAdapter,
    FluxConditioning,
    FluxConfig,
    HunyuanVideoAdapter,
    HunyuanVideoConfig,
    SD3Adapter,
    SD3Conditioning,
    SD3Config,
    WanI2VAdapter,
    WanI2VConditioning,
    WanI2VConfig,
    first_frame_condition,
    pack_latents,
)
from visual_rl.models.flux import image_ids
from visual_rl.models.wan import decode_wan_differentiable

diffusers = pytest.importorskip("diffusers")


@pytest.fixture(autouse=True)
def tiny_cpu_runtime():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(17)
        yield
    torch.set_num_threads(old_threads)


def image_vae():
    return diffusers.AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(8,),
        layers_per_block=1,
        latent_channels=2,
        norm_num_groups=4,
        sample_size=8,
        scaling_factor=2.0,
        shift_factor=0.25,
    )


def assert_policy_gradient(adapter, pixels):
    pixels.square().mean().backward()
    grads = [p.grad for p in adapter.trainable_module.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum() for g in grads) > 0
    assert all(p.grad is None and not p.requires_grad for p in adapter.vae.parameters())


def test_real_sd3_cfg_matches_two_forwards_and_decode_vjp():
    model = diffusers.SD3Transformer2DModel(
        sample_size=4,
        patch_size=2,
        in_channels=2,
        out_channels=2,
        num_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=12,
        caption_projection_dim=16,
        pooled_projection_dim=8,
        pos_embed_max_size=4,
    ).eval()
    adapter = SD3Adapter(model, image_vae(), SD3Config(2.0, 0.25, latent_channels=2, cfg_scale=3.0))
    z = torch.randn(2, 2, 4, 6, requires_grad=True)
    sigma = torch.tensor([0.27, 0.83])
    c = SD3Conditioning(
        torch.randn(2, 3, 12), torch.randn(2, 8), torch.randn(2, 3, 12), torch.randn(2, 8)
    )
    output = adapter(z, sigma, c)

    def direct(prompt, pooled):
        return model(
            hidden_states=z,
            timestep=sigma * 1000,
            encoder_hidden_states=prompt,
            pooled_projections=pooled,
            return_dict=False,
        )[0]

    negative = direct(c.negative_prompt_embeds, c.negative_pooled_prompt_embeds)
    positive = direct(c.prompt_embeds, c.pooled_prompt_embeds)
    expected = negative + 3 * (positive - negative)
    torch.testing.assert_close(output, expected, atol=2e-6, rtol=2e-5)
    grad = torch.autograd.grad(output.square().sum(), z, retain_graph=True)[0]
    oracle_grad = torch.autograd.grad(expected.square().sum(), z, retain_graph=True)[0]
    torch.testing.assert_close(grad, oracle_grad, atol=1e-5, rtol=2e-5)
    x0 = z - sigma.view(2, 1, 1, 1) * output
    decoded = adapter.decode(x0)
    expected_pixels = adapter.vae.decode(x0 / 2 + 0.25, return_dict=False)[0]
    torch.testing.assert_close(decoded, expected_pixels)
    assert_policy_gradient(adapter, decoded)


@pytest.mark.parametrize("guidance", [None, 3.5])
def test_real_flux_packing_forward_and_refl_gradient(guidance):
    model = diffusers.FluxTransformer2DModel(
        patch_size=1,
        in_channels=8,
        out_channels=8,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=12,
        pooled_projection_dim=8,
        guidance_embeds=guidance is not None,
        axes_dims_rope=(4, 6, 6),
    ).eval()
    adapter = FluxAdapter(
        model,
        image_vae(),
        FluxConfig(2.0, 0.25, latent_channels=2, embedded_guidance_scale=guidance),
    )
    z = torch.randn(2, 2, 4, 6, requires_grad=True)
    sigma = torch.tensor([0.27, 0.83])
    c = FluxConditioning(torch.randn(2, 3, 12), torch.randn(2, 8), torch.zeros(3, 3))
    output = adapter(z, sigma, c)
    packed = diffusers.FluxPipeline._pack_latents(z, 2, 2, 4, 6)
    torch.testing.assert_close(pack_latents(z), packed)
    direct = model(
        hidden_states=packed,
        timestep=sigma,
        guidance=None if guidance is None else torch.full((2,), guidance),
        encoder_hidden_states=c.prompt_embeds,
        pooled_projections=c.pooled_prompt_embeds,
        txt_ids=c.text_ids,
        img_ids=image_ids(z),
        return_dict=False,
    )[0]
    oracle = diffusers.FluxPipeline._unpack_latents(direct, 32, 48, 8)
    torch.testing.assert_close(output, oracle)
    grad = torch.autograd.grad(output.square().sum(), z, retain_graph=True)[0]
    oracle_grad = torch.autograd.grad(oracle.square().sum(), z, retain_graph=True)[0]
    torch.testing.assert_close(grad, oracle_grad)
    assert_policy_gradient(adapter, adapter.decode(z - sigma.view(2, 1, 1, 1) * output))


def wan_vae():
    return (
        diffusers.AutoencoderKLWan(
            base_dim=4,
            z_dim=2,
            dim_mult=[1, 2],
            num_res_blocks=1,
            temperal_downsample=[True],
            latents_mean=[0.1, -0.2],
            latents_std=[2.0, 3.0],
            scale_factor_temporal=2,
            scale_factor_spatial=2,
        )
        .float()
        .eval()
    )


def test_real_wan_cfg_matches_independent_branches_and_keeps_image_gradients():
    model = diffusers.WanTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=6,
        out_channels=2,
        text_dim=12,
        freq_dim=8,
        ffn_dim=32,
        num_layers=1,
        image_dim=8,
        added_kv_proj_dim=16,
        rope_max_seq_len=16,
    ).eval()
    adapter = WanI2VAdapter(
        model, wan_vae(), WanI2VConfig((0.1, -0.2), (2.0, 3.0), temporal_compression=2)
    )
    z = torch.randn(2, 2, 2, 4, 4, requires_grad=True)
    image_latents = torch.randn_like(z, requires_grad=True)
    image_embeds = torch.randn(2, 2, 8, requires_grad=True)
    c = WanI2VConditioning(
        torch.randn(2, 3, 12),
        image_embeds,
        first_frame_condition(image_latents, 2),
        torch.randn(2, 3, 12),
    )
    sigma = torch.tensor([0.1234, 0.7891])
    result = adapter(z, sigma, c)

    def direct(prompt):
        return model(
            hidden_states=torch.cat([z, c.image_condition], 1),
            timestep=sigma * 1000,
            encoder_hidden_states=prompt,
            encoder_hidden_states_image=image_embeds,
            return_dict=False,
        )[0]

    negative, positive = direct(c.negative_prompt_embeds), direct(c.prompt_embeds)
    expected = negative + 5 * (positive - negative)
    torch.testing.assert_close(result, expected, atol=2e-6, rtol=2e-5)
    grad = torch.autograd.grad(
        result.square().sum(), (z, image_latents, image_embeds), retain_graph=True
    )
    oracle_grad = torch.autograd.grad(
        expected.square().sum(), (z, image_latents, image_embeds), retain_graph=True
    )
    for actual, oracle in zip(grad, oracle_grad):
        torch.testing.assert_close(actual, oracle, atol=2e-5, rtol=2e-4)
        assert actual.abs().sum() > 0
    assert_policy_gradient(adapter, adapter.decode(z - sigma.view(2, 1, 1, 1, 1) * result))


def test_real_wan_temporal_decode_native_parity_vjp_and_cache_isolation():
    vae = wan_vae().requires_grad_(False)
    adapter = WanI2VAdapter(
        torch.nn.Identity(), vae, WanI2VConfig((0.1, -0.2), (2.0, 3.0), temporal_compression=2)
    )
    z = (torch.randn(2, 2, 3, 2, 2) * 0.1).requires_grad_()
    raw = z * torch.tensor([2.0, 3.0]).view(1, 2, 1, 1, 1) + torch.tensor([0.1, -0.2]).view(
        1, 2, 1, 1, 1
    )
    expected = vae.decode(raw, return_dict=False)[0]
    actual = adapter.decode(z)
    assert actual.shape == (2, 3, 5, 4, 4)
    torch.testing.assert_close(actual, expected)
    probe = torch.randn_like(actual)
    actual_grad = torch.autograd.grad(actual, z, probe, retain_graph=True)[0]
    oracle_grad = torch.autograd.grad(expected, z, probe, retain_graph=True)[0]
    torch.testing.assert_close(actual_grad, oracle_grad)
    assert torch.isfinite(actual_grad).all() and actual_grad.abs().sum() > 0
    # Last decoded frame must still depend on earlier latent frames.
    cross_time_grad = torch.autograd.grad(actual[:, :, -1].sum(), z)[0]
    assert cross_time_grad[:, :, :-1].abs().sum() > 0
    adapter.decode(torch.randn_like(z))
    torch.testing.assert_close(adapter.decode(z), expected)
    vae.enable_tiling()
    with pytest.raises(ValueError, match="untiled"):
        decode_wan_differentiable(vae, raw)


def test_wan_first_frame_helper_matches_real_pipeline_preparation():
    vae = wan_vae().requires_grad_(False)
    pipe = SimpleNamespace(
        vae=vae,
        vae_scale_factor_temporal=2,
        vae_scale_factor_spatial=2,
        config=SimpleNamespace(expand_timesteps=False),
    )
    image = torch.randn(1, 3, 4, 4)
    noise = torch.randn(1, 2, 3, 2, 2)
    prepared_noise, condition = diffusers.WanImageToVideoPipeline.prepare_latents(
        pipe,
        image=image,
        batch_size=1,
        num_channels_latents=2,
        height=4,
        width=4,
        num_frames=5,
        dtype=torch.float32,
        device=torch.device("cpu"),
        latents=noise,
    )
    video = torch.cat([image.unsqueeze(2), torch.zeros(1, 3, 4, 4, 4)], dim=2)
    raw = vae.encode(video).latent_dist.mode()
    normalized = (raw - torch.tensor([0.1, -0.2]).view(1, 2, 1, 1, 1)) / torch.tensor(
        [2.0, 3.0]
    ).view(1, 2, 1, 1, 1)
    torch.testing.assert_close(condition, first_frame_condition(normalized, 2))
    torch.testing.assert_close(prepared_noise, noise)
    assert torch.count_nonzero(condition[:, 2:, 1:]) > 0


def test_real_hunyuan_vae_decode_once_and_input_gradient():
    vae = diffusers.AutoencoderKLHunyuanVideo(
        in_channels=3,
        out_channels=3,
        latent_channels=2,
        down_block_types=("HunyuanVideoDownBlock3D",) * 3,
        up_block_types=("HunyuanVideoUpBlock3D",) * 3,
        block_out_channels=(4, 4, 4),
        layers_per_block=1,
        norm_num_groups=2,
        spatial_compression_ratio=4,
        temporal_compression_ratio=4,
        mid_block_add_attention=False,
    ).float()
    adapter = HunyuanVideoAdapter(torch.nn.Identity(), vae, HunyuanVideoConfig(latent_channels=2))
    z = (torch.randn(1, 2, 2, 2, 2) * 0.1).requires_grad_()
    actual = adapter.decode(z)
    expected = vae.decode(z / 0.476986, return_dict=False)[0]
    torch.testing.assert_close(actual, expected)
    probe = torch.randn_like(actual)
    actual_grad = torch.autograd.grad(actual, z, probe)[0]
    expected_grad = torch.autograd.grad(expected, z, probe)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    assert actual_grad.abs().sum() > 0 and torch.isfinite(actual_grad).all()
    assert all(p.grad is None for p in vae.parameters())


@pytest.mark.parametrize(
    "adapter_cls, config, model_cls",
    [
        (SD3Adapter, SD3Config(2.0, 0.25, latent_channels=2), diffusers.SD3Transformer2DModel),
        (FluxAdapter, FluxConfig(2.0, 0.25, latent_channels=2), diffusers.FluxTransformer2DModel),
    ],
)
def test_lazy_factories_use_offline_default_and_forward_revision(
    monkeypatch, adapter_cls, config, model_cls
):
    calls = []

    def load_model(path, **kwargs):
        calls.append((path, kwargs))
        return torch.nn.Linear(2, 2)

    def load_vae(path, **kwargs):
        calls.append((path, kwargs))
        return image_vae()

    monkeypatch.setattr(model_cls, "from_pretrained", load_model)
    monkeypatch.setattr(diffusers.AutoencoderKL, "from_pretrained", load_vae)
    adapter_cls.from_pretrained(
        "public/example", config=config, revision="exact-revision", torch_dtype=torch.float32
    )
    assert [c[1]["subfolder"] for c in calls] == ["transformer", "vae"]
    assert all(c[0] == "public/example" and c[1]["local_files_only"] is True for c in calls)
    assert all(c[1]["revision"] == "exact-revision" and "token" not in c[1] for c in calls)
