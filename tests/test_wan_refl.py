"""Native tiny-Wan CPU oracles, not pretrained 14B GPU memory acceptance."""

import inspect

import pytest
import torch

from visual_rl.models import wan_refl
from visual_rl.models.wan import decode_wan_differentiable

native = pytest.importorskip("diffusers.models.autoencoders.autoencoder_kl_wan")


@pytest.fixture
def vae():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(41)
            # The strict production pin (0.33.1) predates these config fields.
            parameters = inspect.signature(native.AutoencoderKLWan).parameters
            scales = {k: 4 for k in ("scale_factor_temporal", "scale_factor_spatial")
                      if k in parameters}
            model = native.AutoencoderKLWan(
                base_dim=4, z_dim=2, dim_mult=[1, 2, 2], num_res_blocks=1,
                temperal_downsample=[True, True], latents_mean=[0.7, -1.3],
                latents_std=[2.0, 3.0], **scales,
            ).float()
        yield model
    finally:
        torch.set_num_threads(threads)


def latents(batch=1):
    return torch.randn(batch, 2, 3, 2, 2, generator=torch.Generator().manual_seed(17))


def assert_frozen(vae):
    assert all(not m.training for m in vae.modules())
    assert all(not p.requires_grad and p.grad is None for p in vae.parameters())


def test_configuration_is_idempotent_and_preserves_reporting(vae):
    if hasattr(vae, "enable_tiling"):
        vae.enable_tiling()
    next(vae.parameters()).grad = torch.ones_like(next(vae.parameters()))
    tiles = {k: v for k, v in vars(vae).items() if "tile" in k or k == "use_tiling"}
    state = {k: v.clone() for k, v in vae.state_dict().items()}
    decode, forward = vae.decode, vae.decoder.forward
    parameters = tuple(vae.parameters())
    regions = sum(isinstance(m, native.WanCausalConv3d) for m in vae.decoder.modules())
    assert regions > 0
    assert wan_refl.configure_wan_vae(vae) == regions
    assert wan_refl.configure_wan_vae(vae) == regions
    assert_frozen(vae)
    assert not vae.is_gradient_checkpointing
    assert vae.decode == decode and vae.decoder.forward == forward
    assert all(a is b for a, b in zip(parameters, vae.parameters()))
    assert tiles == {k: getattr(vae, k) for k in tiles}
    assert state.keys() == vae.state_dict().keys()
    for key, value in state.items():
        torch.testing.assert_close(value, vae.state_dict()[key], atol=0, rtol=0)


@pytest.mark.parametrize("last_frame_only", [False, True])
def test_native_forward_and_temporal_input_gradient_parity(vae, last_frame_only, monkeypatch):
    wan_refl.configure_wan_vae(vae)
    x = latents(batch=2).requires_grad_()
    expected = decode_wan_differentiable(vae, x)
    cotangent = torch.randn(expected.shape, generator=torch.Generator().manual_seed(23))
    if last_frame_only:
        cotangent[:, :, :-1] = 0
    expected_grad, = torch.autograd.grad(expected, x, cotangent)
    calls = []
    checkpoint = wan_refl.checkpoint

    def observe(fn, frame, *cache, **kwargs):
        assert kwargs == {"use_reentrant": False}
        assert isinstance(cache, tuple)
        assert all(t.device == frame.device for t in cache if isinstance(t, torch.Tensor))
        if calls:
            assert any(t.requires_grad for t in cache if isinstance(t, torch.Tensor))
        calls.append(fn)
        return checkpoint(fn, frame, *cache, **kwargs)

    monkeypatch.setattr(wan_refl, "checkpoint", observe)
    decoder_calls = []
    hook = vae.decoder.register_forward_pre_hook(lambda *args: decoder_calls.append(True))
    try:
        actual = wan_refl.decode_wan_checkpointed(vae, x)
        assert len(decoder_calls) == 3
        actual_grad, = torch.autograd.grad(actual, x, cotangent)
        assert len(decoder_calls) == 6
    finally:
        hook.remove()
    assert len(calls) == 3
    assert actual.shape == (2, 3, 9, 8, 8)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-6, rtol=1e-5)
    assert torch.isfinite(actual_grad).all()
    # A loss only on the last output frame must reach both earlier latent frames.
    assert all(actual_grad[:, :, i].abs().sum() > 0 for i in range(3))
    assert_frozen(vae)


def test_native_no_grad_parity_raw_affine_and_untiled_callback(vae, monkeypatch):
    wan_refl.configure_wan_vae(vae)
    x = latents()
    with torch.no_grad():
        expected = vae.decode(x, return_dict=False)[0]
    has_tiling = hasattr(vae, "enable_tiling")
    if has_tiling:
        vae.enable_tiling()
        vae.tile_sample_min_height = vae.tile_sample_min_width = 4

    def forbidden(*args, **kwargs):
        pytest.fail("callback must not use checkpoint in no-grad, reporting decode, or tiles")

    monkeypatch.setattr(wan_refl, "checkpoint", forbidden)
    monkeypatch.setattr(vae, "decode", forbidden)
    if has_tiling:
        monkeypatch.setattr(vae, "tiled_decode", forbidden)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        actual = wan_refl.decode_wan_checkpointed(vae, x)
    assert actual.dtype == torch.float32 and not actual.requires_grad
    assert actual.min() >= -1 and actual.max() <= 1
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    if has_tiling:
        assert vae.use_tiling and vae.tile_sample_min_height == 4


def test_repeated_decode_reverse_backward_and_native_cache_isolation(vae):
    wan_refl.configure_wan_vae(vae)
    vae.clear_cache()
    inputs = [latents().requires_grad_(), (latents() * 0.3 + 0.2).requires_grad_()]
    expected = []
    for x in inputs:
        output = decode_wan_differentiable(vae, x)
        expected.append(torch.autograd.grad(output[:, :, -1].square().sum(), x)[0])
    native_cache = vae._feat_map
    outputs = [wan_refl.decode_wan_checkpointed(vae, x) for x in inputs]
    assert vae._feat_map is native_cache and all(t is None for t in native_cache)
    # Native reporting can clear its own cache while callback graphs are live.
    with torch.no_grad():
        vae.decode(latents(), return_dict=False)
    for i in (1, 0):
        outputs[i][:, :, -1].square().sum().backward()
        torch.testing.assert_close(inputs[i].grad, expected[i], atol=1e-6, rtol=1e-5)
    assert all(t is None for t in vae._feat_map)
    assert_frozen(vae)


def test_fp32_under_autocast_with_lower_precision_input(vae):
    wan_refl.configure_wan_vae(vae)
    x = latents().bfloat16().requires_grad_()
    expected = decode_wan_differentiable(vae, x.float())
    grad, = torch.autograd.grad(expected.square().sum(), x)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = wan_refl.decode_wan_checkpointed(vae, x)
        actual_grad, = torch.autograd.grad(actual.square().sum(), x)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual_grad, grad, atol=0, rtol=0)


def test_decoder_without_first_chunk_keyword(vae, monkeypatch):
    wan_refl.configure_wan_vae(vae)
    original = vae.decoder.forward
    expected = decode_wan_differentiable(vae, latents().requires_grad_())

    # Exercise the 0.33 signature with genuine native decoder computations.
    def legacy_forward(x, feat_cache=None, feat_idx=None):
        return original(x, feat_cache=feat_cache, feat_idx=feat_idx)

    monkeypatch.setattr(vae.decoder, "forward", legacy_forward)
    assert "first_chunk" not in inspect.signature(vae.decoder.forward).parameters
    x = latents().requires_grad_()
    actual = wan_refl.decode_wan_checkpointed(vae, x)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual[:, :, -1].sum().backward()
    assert x.grad[:, :, 0].abs().sum() > 0


@pytest.mark.parametrize("bad", [None, torch.zeros(1, 2, 2, 2), torch.zeros(1, 3, 3, 2, 2),
                                 torch.zeros(1, 2, 0, 2, 2), torch.ones(1, 2, 3, 2, 2).long()])
def test_invalid_latents(vae, bad):
    wan_refl.configure_wan_vae(vae)
    with pytest.raises(ValueError, match="nonempty floating BCTHW"):
        wan_refl.decode_wan_checkpointed(vae, bad)


def test_configuration_errors_do_not_freeze_invalid_models(vae):
    with pytest.raises(ValueError, match="native Wan"):
        wan_refl.configure_wan_vae(torch.nn.Linear(2, 2))
    vae.register_to_config(patch_size=2)
    with pytest.raises(ValueError, match="unpatched Wan2.1"):
        wan_refl.configure_wan_vae(vae)
    assert next(vae.parameters()).requires_grad
    vae.register_to_config(patch_size=None)
    vae.decoder.conv_out.bfloat16()
    with pytest.raises(ValueError, match="FP32"):
        wan_refl.configure_wan_vae(vae)
    assert next(vae.parameters()).requires_grad


def test_decode_requires_frozen_eval_and_matching_device(vae):
    with pytest.raises(ValueError, match="configure_wan_vae"):
        wan_refl.decode_wan_checkpointed(vae, latents())
    wan_refl.configure_wan_vae(vae)
    with pytest.raises(ValueError, match="same device"):
        wan_refl.decode_wan_checkpointed(vae, latents().to("meta"))
