"""CPU oracles for the row-bounded ReFL sidecar (no model downloads)."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from visual_rl.rewards.inputs import ContinuousRGBInput
from visual_rl.rewards.scalar import FixedNormalization, weighted_score
from visual_rl.training import refl_vjp


class TinyVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(.7))
        self.calls = []
        self.moves = []
        self.drift = False

    def to(self, device):
        self.moves.append(str(device))
        return super().to(device)

    def decode(self, latent, return_dict=False):
        self.calls.append((torch.is_grad_enabled(), latent.detach().clone()))
        offset = .1 if self.drift and torch.is_grad_enabled() else 0.
        return (latent * self.gain + offset,)


class TinyAdapter:
    def __init__(self):
        self.vae = TinyVAE().requires_grad_(False).eval()
        self.affine_calls = 0

    def to_vae_latents(self, latent):
        self.affine_calls += 1
        return latent / 2 + .125


class TinyReward(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(1.3))
        self.moves = []
        self.calls = []
        self.fail = False
        self.requires_grad_(False).eval()

    def to(self, device):
        self.moves.append(str(device))
        return super().to(device)

    def forward(self, inputs):
        inputs.validate()
        assert inputs.frames.dtype == torch.float32
        assert inputs.value_range == "minus_one_one"
        self.calls.append((inputs.frames.is_leaf, tuple(inputs.prompts)))
        if self.fail:
            raise RuntimeError("injected reward failure")
        pixels = (inputs.frames.float() + 1) * .5
        mean = pixels.mean((1, 2, 3, 4))
        raw = {"VQ": mean * self.gain, "MQ": pixels.square().mean((1, 2, 3, 4)),
               "TA": mean.sin()}
        norm = {key: FixedNormalization(.2, std, "test-fixed")
                for key, std in zip(raw, (2., .5, 1.5))}
        # Deliberately raw selected space: the sidecar must use fixed_normalized.
        return weighted_score(raw, {"VQ": .2, "MQ": .3, "TA": .5},
                              space="raw", normalization=norm)


def run(endpoints, adapter, scorer, **kwargs):
    return refl_vjp.staged_latent_vjp(
        endpoints, adapter, scorer, [f"row {i}" for i in range(len(endpoints))],
        device="cpu", fps=8., precision="fp32", **kwargs,
    )


def assert_clean(*modules):
    for module in modules:
        assert not any(part.training for part in module.modules())
        for parameter in module.parameters():
            assert not parameter.requires_grad
            assert parameter.grad is None
            assert parameter.device.type == "cpu"


def test_staged_matches_direct_without_population_division():
    adapter, scorer = TinyAdapter(), TinyReward()
    endpoints = torch.linspace(-.8, .8, 2 * 3 * 4 * 2 * 2).reshape(2, 3, 4, 2, 2)
    endpoints[1].fill_(8.)  # All pixels clipped: a valid zero-gradient row.
    endpoints.requires_grad_(True)
    scale = 2.7
    gradients, losses, metrics = run(endpoints, adapter, scorer, reward_loss_scale=scale)
    assert adapter.affine_calls == 4
    assert [enabled for enabled, _ in adapter.vae.calls] == [False, True, False, True]
    for row in range(2):
        for _, latent in adapter.vae.calls[2 * row:2 * row + 2]:
            torch.testing.assert_close(latent, endpoints[row:row + 1].detach() / 2 + .125)
    assert scorer.calls == [(True, ("row 0",)), (True, ("row 1",))]

    latent = endpoints.detach().clone().requires_grad_(True)
    pixels = adapter.vae.decode(adapter.to_vae_latents(latent))[0]
    pixels = pixels.clamp(-1, 1).float().permute(0, 2, 1, 3, 4)
    scores = scorer(ContinuousRGBInput(pixels, ["row 0", "row 1"], 8., "minus_one_one"))
    expected_loss = -scale * sum(scores.weights[k] * v for k, v in scores.fixed_normalized.items())
    expected_grad, = torch.autograd.grad(expected_loss.sum(), latent)
    torch.testing.assert_close(gradients, expected_grad, atol=1e-8, rtol=1e-6)
    torch.testing.assert_close(losses, expected_loss)
    assert gradients[0].abs().sum() > 0
    assert torch.count_nonzero(gradients[1]) == 0
    assert endpoints.grad is None
    assert gradients.device.type == losses.device.type == "cpu"
    assert not gradients.requires_grad and not losses.requires_grad
    assert losses.shape == (2,)
    assert metrics["train/refl/loss"] == pytest.approx(expected_loss.mean().item())
    assert set(metrics) == {"train/refl/loss"} | {
        f"train/refl/{space}_{key}" for space in ("raw", "normalized") for key in ("VQ", "MQ", "TA")
    }
    for key in ("VQ", "MQ", "TA"):
        assert metrics[f"train/refl/raw_{key}"] == pytest.approx(scores.raw[key].mean().item())
        assert metrics[f"train/refl/normalized_{key}"] == pytest.approx(
            scores.fixed_normalized[key].mean().item())
    assert all(isinstance(value, float) for value in metrics.values())
    assert_clean(adapter.vae, scorer)


def test_failure_cleanup_and_stateful_rematerialization():
    adapter, scorer = TinyAdapter(), TinyReward()
    endpoints = torch.zeros(1, 3, 4, 2, 2)
    scorer.fail = True
    with pytest.raises(RuntimeError, match="injected reward"):
        run(endpoints, adapter, scorer)
    assert_clean(adapter.vae, scorer)
    assert adapter.vae.moves[-1] == scorer.moves[-1] == "cpu"

    scorer.fail = False
    adapter.vae.drift = True
    with pytest.raises(RuntimeError, match="rematerialization differs"):
        run(endpoints, adapter, scorer)
    assert_clean(adapter.vae, scorer)
    assert adapter.vae.moves[-1] == scorer.moves[-1] == "cpu"
    adapter.vae.drift = False
    gradients, _, _ = run(endpoints, adapter, scorer)
    assert torch.isfinite(gradients).all()


def test_bf16_decode_promotes_pixels_before_reward_denormalization():
    class AutocastVAE(TinyVAE):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv3d(3, 3, 1)
            self.output_dtypes = []

        def decode(self, latent, return_dict=False):
            output = self.conv(latent)
            self.output_dtypes.append(output.dtype)
            return (output,)

    torch.manual_seed(23)
    adapter, scorer = TinyAdapter(), TinyReward()
    adapter.vae = AutocastVAE().requires_grad_(False).eval()
    endpoint = torch.randn(1, 3, 4, 2, 2)
    gradients, losses, _ = refl_vjp.staged_latent_vjp(
        endpoint, adapter, scorer, ["row"], device="cpu", fps=8., precision="bf16",
    )
    assert adapter.vae.output_dtypes == [torch.bfloat16, torch.bfloat16]
    latent = endpoint.clone().requires_grad_(True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        decoded = adapter.vae.decode(adapter.to_vae_latents(latent))[0]
        pixels = decoded.clamp(-1, 1).float().permute(0, 2, 1, 3, 4)
        scores = scorer(ContinuousRGBInput(pixels, ["row"], 8., "minus_one_one"))
        expected_loss = -sum(scores.weights[k] * v for k, v in scores.fixed_normalized.items())
    expected_grad, = torch.autograd.grad(expected_loss.sum(), latent)
    torch.testing.assert_close(gradients, expected_grad, atol=0, rtol=0)
    torch.testing.assert_close(losses, expected_loss, atol=0, rtol=0)
    assert_clean(adapter.vae, scorer)


def test_frozen_contract_and_nonfinite_vjp():
    adapter, scorer = TinyAdapter(), TinyReward()
    endpoints = torch.zeros(1, 3, 4, 2, 2)
    scorer.gain.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen weights"):
        run(endpoints, adapter, scorer)
    scorer.gain.requires_grad_(False)
    adapter.vae.train()
    with pytest.raises(ValueError, match="eval mode"):
        run(endpoints, adapter, scorer)
    adapter.vae.eval()

    class BadGradient(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value.clone()

        @staticmethod
        def backward(ctx, grad):
            return torch.full_like(grad, float("nan"))

    original = adapter.to_vae_latents
    adapter.to_vae_latents = lambda value: BadGradient.apply(original(value))
    with pytest.raises(ValueError, match="nonfinite ReFL latent gradient"):
        run(endpoints, adapter, scorer)
    assert_clean(adapter.vae, scorer)


def test_explicit_decoder_receives_one_affine_fp32_with_bf16_reward():
    adapter, scorer = TinyAdapter(), TinyReward()
    endpoints = torch.full((1, 3, 4, 2, 2), .3)
    decoder_calls, reward_autocast = [], []
    native_reward = scorer.forward
    def reward(inputs):
        reward_autocast.append(torch.is_autocast_enabled("cpu"))
        return native_reward(inputs)
    scorer.forward = reward
    def decoder(raw):
        assert raw.dtype == torch.float32
        assert not torch.is_autocast_enabled("cpu")
        decoder_calls.append((torch.is_grad_enabled(), raw.detach().clone()))
        return adapter.vae.decode(raw)[0]
    gradients, losses, _ = refl_vjp.staged_latent_vjp(endpoints, adapter, scorer, ["row"],
        device="cpu", fps=8., precision="bf16", vae_precision="fp32", decode_fn=decoder)
    assert adapter.affine_calls == 2
    assert [grad for grad, _ in decoder_calls] == [False, True]
    assert reward_autocast == [True]
    for _, raw in decoder_calls:
        torch.testing.assert_close(raw, endpoints / 2 + .125, rtol=0, atol=0)
    latent = endpoints.detach().requires_grad_()
    pixels = adapter.vae.decode(latent / 2 + .125)[0].clamp(-1, 1).permute(0, 2, 1, 3, 4)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        scores = native_reward(ContinuousRGBInput(pixels, ["row"], 8., "minus_one_one"))
        expected_loss = -sum(scores.weights[k] * v for k, v in scores.fixed_normalized.items())
    expected_grad, = torch.autograd.grad(expected_loss.sum(), latent)
    torch.testing.assert_close(gradients, expected_grad, rtol=0, atol=0)
    torch.testing.assert_close(losses, expected_loss, rtol=0, atol=0)


def test_staged_native_wan_causal_vjp_equals_direct_chain():
    import inspect
    from functools import partial

    from visual_rl.models.wan import decode_wan_differentiable
    from visual_rl.models.wan_refl import configure_wan_vae, decode_wan_checkpointed

    native = pytest.importorskip("diffusers.models.autoencoders.autoencoder_kl_wan")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(53)
            options = dict(base_dim=4, z_dim=2, dim_mult=[1, 2, 2], num_res_blocks=1,
                temperal_downsample=[True, True], latents_mean=[.7, -1.3], latents_std=[2., 3.])
            if "scale_factor_temporal" in inspect.signature(native.AutoencoderKLWan).parameters:
                options.update(scale_factor_temporal=4, scale_factor_spatial=4)
            vae = native.AutoencoderKLWan(**options).float()
        configure_wan_vae(vae)
        mean = torch.tensor([.7, -1.3]).view(1, 2, 1, 1, 1)
        std = torch.tensor([2., 3.]).view(1, 2, 1, 1, 1)
        adapter = SimpleNamespace(vae=vae, to_vae_latents=lambda z: z.float() * std + mean)
        scorer = TinyReward()
        endpoint = torch.randn(1, 2, 3, 2, 2, generator=torch.Generator().manual_seed(81)) * .2
        gradients, losses, _ = refl_vjp.staged_latent_vjp(endpoint, adapter, scorer, ["scene"],
            device="cpu", fps=8., precision="bf16", vae_precision="fp32",
            decode_fn=partial(decode_wan_checkpointed, vae))
        latent = endpoint.clone().requires_grad_(True)
        pixels = decode_wan_differentiable(vae, adapter.to_vae_latents(latent)).permute(0, 2, 1, 3, 4)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            scores = scorer(ContinuousRGBInput(pixels, ["scene"], 8., "minus_one_one"))
            expected_loss = -sum(scores.weights[k] * v for k, v in scores.fixed_normalized.items())
        expected_gradient, = torch.autograd.grad(expected_loss.sum(), latent)
        torch.testing.assert_close(losses, expected_loss, rtol=0, atol=0)
        torch.testing.assert_close(gradients, expected_gradient, rtol=1e-5, atol=1e-6)
        assert all(gradients[:, :, i].norm() > 0 for i in range(3))
        assert_clean(vae, scorer)
    finally:
        torch.set_num_threads(previous_threads)


def test_native_hunyuan_checkpoint_equivalence_and_no_grad_preservation(monkeypatch):
    native = pytest.importorskip("diffusers.models.autoencoders.autoencoder_kl_hunyuan_video")
    torch.manual_seed(17)
    vae = native.AutoencoderKLHunyuanVideo(
        latent_channels=3, block_out_channels=(4,), layers_per_block=1,
        norm_num_groups=1, spatial_compression_ratio=1,
        down_block_types=("HunyuanVideoDownBlock3D",),
        up_block_types=("HunyuanVideoUpBlock3D",),
    ).requires_grad_(False).eval()
    vae.enable_tiling()
    tiles = {k: v for k, v in vars(vae).items() if "tile" in k or k == "use_tiling"}
    state = {k: v.clone() for k, v in vae.state_dict().items()}
    x = torch.randn(1, 3, 2, 3, 3, requires_grad=True)
    expected = vae.decode(x, return_dict=False)[0]
    cotangent = torch.randn_like(expected)
    expected_grad, = torch.autograd.grad(expected, x, cotangent)
    original_forward = vae.decoder.forward
    assert refl_vjp.configure_hunyuan_vae(vae) == 4
    assert refl_vjp.configure_hunyuan_vae(vae) == 4
    assert vae.decoder._refl_native_forward == original_forward
    assert tiles == {k: getattr(vae, k) for k in tiles}
    assert state.keys() == vae.state_dict().keys()
    for key, value in state.items():
        torch.testing.assert_close(value, vae.state_dict()[key], atol=0, rtol=0)
    calls = []
    original_checkpoint = refl_vjp.checkpoint

    def record(*args, **kwargs):
        assert kwargs["use_reentrant"] is False
        calls.append(args[0])
        return original_checkpoint(*args, **kwargs)

    monkeypatch.setattr(refl_vjp, "checkpoint", record)
    with torch.no_grad():
        preview = vae.decode(x, return_dict=False)[0]
    assert not calls
    actual = vae.decode(x, return_dict=False)[0]
    actual_grad, = torch.autograd.grad(actual, x, cotangent)
    assert len(calls) == 4
    torch.testing.assert_close(preview, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-6, rtol=1e-5)
    assert_clean(vae)


def test_native_reward_wrappers_are_frozen_nonreentrant_and_idempotent():
    import inspect

    native = pytest.importorskip("transformers.models.qwen2_vl.modeling_qwen2_vl")
    config = SimpleNamespace(embed_dim=8, num_heads=2, mlp_ratio=2,
                             hidden_act="gelu", _attn_implementation="eager",
                             attention_dropout=0.)
    scorer = nn.Module()
    scorer.model = nn.ModuleList([native.Qwen2VLVisionBlock(config)])
    scorer.requires_grad_(False).eval()
    x = torch.randn(4, 8, requires_grad=True)
    kwargs = {"cu_seqlens": torch.tensor([0, 4], dtype=torch.int32)}
    if "position_embeddings" in inspect.signature(scorer.model[0].forward).parameters:
        kwargs["position_embeddings"] = (torch.ones(4, 4), torch.zeros(4, 4))
    else:
        kwargs["rotary_pos_emb"] = torch.zeros(4, 2)
    expected = scorer.model[0](x, **kwargs)
    expected_grad, = torch.autograd.grad(expected.square().sum(), x)
    state_keys = set(scorer.state_dict())
    assert refl_vjp.configure_video_reward(scorer) == 1
    wrapper = scorer.model[0]
    assert refl_vjp.configure_video_reward(scorer) == 1
    assert scorer.model[0] is wrapper
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl
    assert wrapper.checkpoint_impl == CheckpointImpl.NO_REENTRANT
    actual = scorer.model[0](x, **kwargs)
    actual_grad, = torch.autograd.grad(actual.square().sum(), x)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_grad, expected_grad)
    assert state_keys == set(scorer.state_dict())
    assert_clean(scorer)
