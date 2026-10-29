"""Reject invalid native pixels before clipping hides decoder failures."""

from types import SimpleNamespace

import pytest
import torch

from visual_rl.backends.hunyuan import score_videos
from visual_rl.models.base import checked_vae_decode


class ClippingVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = torch.nn.Identity()

    def decode(self, latents, return_dict=False):
        # Wan's native decode clamps decoder output internally, before callers
        # can inspect it. Keep that boundary in this small regression fixture.
        return (self.decoder(latents).clamp(-1, 1),)


@pytest.mark.parametrize("broken", [float("nan"), float("inf"), float("-inf")])
def test_native_decoder_nonfinite_check_precedes_clip_and_removes_hook(broken):
    vae = ClippingVAE()
    corrupt = vae.decoder.register_forward_hook(lambda *args: torch.full((1, 3, 2, 2), broken))
    hooks_before = tuple(vae.decoder._forward_hooks)
    try:
        with pytest.raises(ValueError, match="decoder.*nonfinite"):
            checked_vae_decode(vae, torch.zeros(1, 3, 2, 2))
        assert tuple(vae.decoder._forward_hooks) == hooks_before
    finally:
        corrupt.remove()
    assert torch.isfinite(checked_vae_decode(vae, torch.zeros(1, 3, 2, 2))).all()


def test_checked_native_decode_preserves_finite_values_gradients_and_hook_cleanup():
    vae = ClippingVAE()
    x = torch.tensor([[[[-2., -.3, .7, 2.]]]], requires_grad=True)
    expected = vae.decode(x, return_dict=False)[0]
    actual = checked_vae_decode(vae, x)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual_grad, = torch.autograd.grad(actual.square().sum(), x)
    expected_grad, = torch.autograd.grad(expected.square().sum(), x)
    torch.testing.assert_close(actual_grad, expected_grad, atol=0, rtol=0)
    assert not vae.decoder._forward_hooks


@pytest.mark.parametrize("broken", [float("inf"), float("-inf")])
def test_native_video_scoring_rejects_invalid_pixels_before_codec(tmp_path, monkeypatch, broken):
    vae = ClippingVAE()
    corrupt = vae.decoder.register_forward_hook(lambda *args: torch.full((1, 3, 1, 2, 2), broken))
    parts = SimpleNamespace(adapter=SimpleNamespace(vae=vae, to_vae_latents=lambda x: x))
    config = SimpleNamespace(training=SimpleNamespace(precision="fp32"))
    from visual_rl.backends import hunyuan

    monkeypatch.setattr(hunyuan, "encode_video", lambda *args, **kwargs: pytest.fail("invalid pixels reached codec"))
    try:
        with pytest.raises(ValueError, match="decoder.*nonfinite"):
            score_videos(config, parts, torch.zeros(1, 3, 1, 2, 2), ["scene"],
                         tmp_path / "videos", torch.device("cpu"))
        assert len(vae.decoder._forward_hooks) == 1
    finally:
        corrupt.remove()
