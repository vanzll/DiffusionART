"""Real native tiny-H3 CPU tests; no pretrained weights or GPUs are used."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from visual_rl.models.minimax_h3 import (
    MODEL_REVISION,
    MiniMaxH3Adapter,
    MiniMaxH3Conditioning,
    native_h3_helpers,
    unpack_video_rows,
)
from visual_rl.models.minimax_h3_assets import resolve_minimax_h3_assets
from visual_rl.training.audio_video_types import AVState


@pytest.fixture
def native_model():
    pytest.importorskip("diffusers.models.transformers.transformer_minimax_h3")
    from diffusers import MiniMaxH3Transformer3DModel
    torch.manual_seed(23)
    return MiniMaxH3Transformer3DModel(
        num_attention_heads=2, attention_head_dim=12, hidden_size=24, num_layers=1,
        num_refiner_layers=1, ffn_dim=32, in_channels=2, audio_in_channels=3,
        patch_size=(1, 2, 2), text_dim=8, freq_dim=8,
        time_embed_hidden_dim=16, time_embed_dim=12, rope_freq_dim=2,
    )


def inputs():
    state = AVState(torch.randn(2, 2, 2, 4, 4), torch.randn(2, 2, 3, 3))
    c = MiniMaxH3Conditioning(torch.randn(2, 4, 8), torch.tensor(
        [[True, True, True, True], [True, True, False, False]],
    ))
    return state, c


def test_native_patch_layout_roundtrip():
    pytest.importorskip("diffusers.models.transformers.transformer_minimax_h3")
    build, pack = native_h3_helpers()
    video = torch.arange(2 * 2 * 3 * 4 * 6).reshape(2, 2, 3, 4, 6).float()
    rows = pack(video, (1, 2, 2))
    assert torch.equal(unpack_video_rows(rows, video.shape, (1, 2, 2)), video)
    pos, tags, vi, ai, ti, cv, ca = build(torch.ones(3).long(), 3, 4, 6, 5,
                                         (1, 2, 2), 2, 2, 0)
    assert pos.dtype == torch.float64 and (cv, ca) == (0, 0)
    assert torch.equal(tags[vi], torch.zeros(18).long())
    assert torch.equal(tags[ai], torch.full((10,), 2))
    assert torch.equal(tags[ti], torch.ones(3).long())
    assert torch.equal(ai, torch.arange(3, 13))
    assert torch.equal(vi, torch.arange(13, 31))


def test_actual_tiny_output_and_parameter_gradient_match_direct_native(native_model):
    reference = copy.deepcopy(native_model)
    state, c = inputs()
    sv, sa = torch.tensor([0.8, 0.4]), torch.tensor([0.3, 0.9])
    actual = MiniMaxH3Adapter(native_model).predict_velocity(state, sv, sa, c)
    build, patchify = native_h3_helpers()
    videos, audios = [], []
    for i, length in enumerate([4, 2]):
        pos, tags, vi, ai, ti, _, _ = build(torch.ones(length).long(), 2, 4, 4, 3,
                                           (1, 2, 2), 2, 2, 0)
        # Independent reference construction: H3 text uses video time, and audio
        # rows use their own time; compare every output and every parameter grad.
        row_t = torch.ones(len(pos)) * (1 - sv[i])
        row_t[ai] = 1 - sa[i]
        unique, inverse = row_t.unique(sorted=True, return_inverse=True)
        vr, ar = reference(
            hidden_states=patchify(state.video[i:i+1], (1, 2, 2))[None],
            audio_hidden_states=state.audio[i:i+1].permute(0, 1, 3, 2).reshape(1, 6, 3),
            encoder_hidden_states=c.prompt_embeds[i:i+1, :length], timestep=unique,
            timestep_indices=inverse, token_tags=tags, position_ids=pos,
            video_indices=vi, audio_indices=ai, text_indices=ti, return_dict=False,
        )
        videos.append(-unpack_video_rows(vr, (1, 2, 2, 4, 4), (1, 2, 2)))
        audios.append(-ar.reshape(1, 2, 3, 3).permute(0, 1, 3, 2))
    ev, ea = torch.cat(videos), torch.cat(audios)
    torch.testing.assert_close(actual.video, ev, rtol=0, atol=0)
    torch.testing.assert_close(actual.audio, ea, rtol=0, atol=0)
    loss = 0.5 * actual.video.square().mean() + 0.5 * actual.audio.square().mean()
    loss.backward()
    (0.5 * ev.square().mean() + 0.5 * ea.square().mean()).backward()
    for (name, p), (other_name, q) in zip(native_model.named_parameters(), reference.named_parameters()):
        assert name == other_name and p.grad is not None
        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
    assert native_model.audio_proj_out.weight.grad.norm() > 0
    assert native_model.proj_out.weight.grad.norm() > 0


def test_distinct_sigma_and_prompt_identity_survive_select(native_model):
    state, c = inputs()
    rows = torch.tensor([1, 0, 1])
    c2 = c.select(rows).to("cpu")
    assert torch.equal(c2.prompt_embeds, c.prompt_embeds[rows])
    adapter = MiniMaxH3Adapter(native_model)
    sv, sa = torch.tensor([0.8, 0.4]), torch.tensor([0.3, 0.9])
    expected = adapter.predict_velocity(state, sv, sa, c)
    actual = adapter.predict_velocity(state.select(rows), sv[rows], sa[rows], c2)
    torch.testing.assert_close(actual.video, expected.video[rows])
    torch.testing.assert_close(actual.audio, expected.audio[rows])
    changed = adapter.predict_velocity(state, sv, sv, c)
    assert not torch.allclose(changed.audio, expected.audio)


@pytest.mark.parametrize("bad", [float("nan"), -0.1, 1.1, [0.4, 0.5, 0.6]])
def test_bad_sigma_rejected(native_model, bad):
    state, c = inputs()
    with pytest.raises(ValueError, match="sigma"):
        MiniMaxH3Adapter(native_model).predict_velocity(state, bad, 0.5, c)


def test_holey_mask_rejected(native_model):
    state, c = inputs()
    mask = c.attention_mask.clone()
    mask[0] = torch.tensor([True, False, True, True])
    with pytest.raises(ValueError, match="right-padded"):
        MiniMaxH3Adapter(native_model).predict_velocity(
            state, 0.5, 0.2, MiniMaxH3Conditioning(c.prompt_embeds, mask))


def test_latent_geometry_from_native_component_metadata(native_model):
    # Geometry only: tiny transformer channel metadata need not match real VAE.
    vae = torch.nn.Linear(1, 1)
    vae.config = SimpleNamespace(latent_channels=24, clip_length=17)
    vae.tokens_chunk_size, vae.spatial_compression_ratio = 5, 16
    audio = torch.nn.Linear(1, 1)
    audio.config = SimpleNamespace(latent_channels=32, sampling_rate=32000)
    audio.hop_length = 800
    adapter = MiniMaxH3Adapter(native_model, vae, audio)
    assert adapter.latent_shapes(2, 768, 1344, 124) == (
        (2, 24, 37, 48, 84), (2, 2, 32, 207))
    with pytest.raises(ValueError):
        adapter.latent_shapes(2, 768, 1344, 123)


def test_assets_reject_floating_identity_and_missing_local_provenance(tmp_path):
    with pytest.raises(ValueError, match="immutable"):
        resolve_minimax_h3_assets(revision="main")
    with pytest.raises(ValueError, match="provenance"):
        resolve_minimax_h3_assets(str(tmp_path))
    (tmp_path / "h3_asset_manifest.json").write_text(json.dumps({"revision": MODEL_REVISION}))
    with pytest.raises(FileNotFoundError, match="model_index"):
        resolve_minimax_h3_assets(str(tmp_path))


@pytest.mark.parametrize("mode", ["full", "lora"])
def test_native_finetune_optimizer_updates_both_modalities(native_model, mode):
    from visual_rl.models.finetune import FineTuneConfig, prepare_policy
    if mode == "lora":
        pytest.importorskip("peft")
        config = FineTuneConfig("lora", rank=2, alpha=2,
                                target_modules=("proj_out", "audio_proj_out"))
    else:
        config = FineTuneConfig("full")
    adapter = MiniMaxH3Adapter(native_model)
    adapter.set_trainable_module(prepare_policy(native_model, config))
    trainable = [p for p in adapter.trainable_module.parameters() if p.requires_grad]
    assert len({id(p) for p in trainable}) == len(trainable)
    before = [p.detach().clone() for p in trainable]
    optimizer = torch.optim.SGD(trainable, lr=0.01)
    state, c = inputs()
    output = adapter.predict_velocity(state, 0.8, 0.3, c)
    (output.video.square().mean() + output.audio.square().mean()).backward()
    optimizer.step()
    assert any(not torch.equal(p, old) for p, old in zip(trainable, before))
    for name, p in adapter.trainable_module.named_parameters():
        if not p.requires_grad:
            assert p.grad is None


def test_decode_denormalization_and_stereo_order_with_explicit_test_decoders(native_model):
    # These intentionally tiny decoder doubles check adapter transforms, not VAE
    # quality. The transformer tests above execute actual upstream H3 modules.
    class Video(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.config = SimpleNamespace(latents_mean=[0.1, 0.2, 0.3], latents_std=[2, 3, 4])

        def decode(self, z, return_dict=False):
            self.seen = z
            return (z,)

    class Audio(Video):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(latents_mean=[1, 2, 3], latents_std=[2, 3, 4],
                                          sampling_rate=32000)

        def decode(self, z, return_dict=False):
            self.seen = z
            return (z.sum(1, keepdim=True),)

    v, a = Video(), Audio()
    adapter = MiniMaxH3Adapter(native_model, v, a)
    state = AVState(torch.zeros(2, 3, 1, 2, 2),
                    torch.arange(2 * 2 * 3 * 4).float().reshape(2, 2, 3, 4))
    media = adapter.decode(state)
    expected = (state.audio * torch.tensor([2, 3, 4])[None, None, :, None]
                + torch.tensor([1, 2, 3])[None, None, :, None]).sum(2)
    torch.testing.assert_close(media.audio, expected)
    torch.testing.assert_close(v.seen, torch.tensor([0.1, 0.2, 0.3])[None, :, None, None, None]
                               .expand_as(state.video))
    assert media.fps == 24 and media.sample_rate == 32000
    assert not media.video.requires_grad and not media.audio.requires_grad
    assert all(not p.requires_grad for p in list(v.parameters()) + list(a.parameters()))


def test_actual_tiny_native_vaes_decode_and_checkpoint_roundtrip(native_model, tmp_path):
    from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio
    vae = AutoencoderKLMiniMaxH3(
        latent_channels=2, block_out_channels=(8, 8), layers_per_block=1,
        spatial_downsample_factors=(2, 2), temporal_downsample_factors=(2, 2),
        norm_num_groups=2, decoder_num_layers=1, decoder_num_attention_heads=2,
        decoder_attention_head_dim=12, decoder_rope_dim_ratio=0.5,
        latents_mean=(0.1, -0.2), latents_std=(1.1, 0.8),
    )
    audio = AutoencoderKLMiniMaxH3Audio(
        encoder_dim=8, encoder_rates=(2, 2), latent_dim=8, latent_channels=2,
        num_attention_heads=2, decoder_dim=16, decoder_rates=(2, 2),
        decoder_kernel_sizes=(4, 4), resblock_kernel_sizes=(3,),
        resblock_dilation_sizes=((1, 3, 5),), latents_mean=[0.2, -0.1],
        latents_std=[0.9, 1.2],
    )
    adapter = MiniMaxH3Adapter(native_model, vae, audio)
    state = AVState(torch.randn(1, 2, 7, 2, 2), torch.randn(1, 2, 2, 4))
    media = adapter.decode(state)
    assert media.video.shape[:3] == (1, 3, 22)
    assert media.audio.shape == (1, 2, 16)
    assert torch.isfinite(media.video).all() and torch.isfinite(media.audio).all()
    assert media.video.min() >= 0 and media.video.max() <= 1
    for name, component in [("vae", vae), ("audio_vae", audio)]:
        component.save_pretrained(tmp_path / name)
    loaded = MiniMaxH3Adapter(
        native_model,
        AutoencoderKLMiniMaxH3.from_pretrained(tmp_path / "vae", local_files_only=True),
        AutoencoderKLMiniMaxH3Audio.from_pretrained(tmp_path / "audio_vae", local_files_only=True),
    ).decode(state)
    torch.testing.assert_close(media.video, loaded.video, rtol=0, atol=0)
    torch.testing.assert_close(media.audio, loaded.audio, rtol=0, atol=0)
