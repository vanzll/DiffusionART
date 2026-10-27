import copy

import pytest
import torch


@pytest.mark.parametrize("padded", [False, True])
def test_native_mask_elision_matches_old_masked_outputs_and_gradients(monkeypatch, padded):
    from diffusers import MiniMaxH3Transformer3DModel

    from visual_rl.models import minimax_h3_batch as batch
    from visual_rl.models.minimax_h3 import MiniMaxH3Adapter, MiniMaxH3Conditioning
    from visual_rl.training.audio_video_types import AVState

    torch.set_num_threads(1)
    torch.manual_seed(23)
    native = MiniMaxH3Transformer3DModel(
        num_attention_heads=2, attention_head_dim=12, hidden_size=24, num_layers=1,
        num_refiner_layers=1, ffn_dim=32, in_channels=2, audio_in_channels=3,
        patch_size=(1, 2, 2), text_dim=8, freq_dim=8, time_embed_hidden_dim=16,
        time_embed_dim=12, rope_freq_dim=2)
    old = copy.deepcopy(native)
    actual_adapter, old_adapter = MiniMaxH3Adapter(native), MiniMaxH3Adapter(old)
    generator = torch.Generator().manual_seed(90)
    lengths = torch.tensor([4, 2, 3, 1] if padded else [4]*4)
    mask = torch.arange(4)[None] < lengths[:, None]
    embeds = torch.randn(4, 4, 8, generator=generator)
    embeds[~mask] = 10000
    conditioning = MiniMaxH3Conditioning(embeds, mask)
    state = AVState(torch.randn(4, 2, 2, 4, 4, generator=generator),
                    torch.randn(4, 2, 3, 3, generator=generator))
    sv, sa = torch.linspace(.2, .9, 4), torch.linspace(.8, .1, 4)
    before = torch.get_rng_state()
    with monkeypatch.context() as context:
        # The pre-fix implementation always supplied this mask.
        context.setattr(batch, "_padding_attention_mask", lambda valid: valid[:, None, None, :])
        expected = old_adapter.predict_velocity(state, sv, sa, conditioning)
        (expected.video.square().mean()+expected.audio.square().mean()).backward()
    masks = []
    original = batch.dispatch_attention_fn
    def observe(*args, **kwargs):
        masks.append(kwargs["attn_mask"])
        return original(*args, **kwargs)
    monkeypatch.setattr(batch, "dispatch_attention_fn", observe)
    actual = actual_adapter.predict_velocity(state, sv, sa, conditioning)
    (actual.video.square().mean()+actual.audio.square().mean()).backward()
    assert masks and all((value is None) == (not padded) for value in masks)
    torch.testing.assert_close(actual.video, expected.video, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(actual.audio, expected.audio, rtol=3e-5, atol=3e-6)
    for parameter, reference in zip(native.parameters(), old.parameters()):
        assert (parameter.grad is None) == (reference.grad is None)
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, reference.grad, rtol=5e-5, atol=3e-6)
    assert torch.equal(before, torch.get_rng_state())
    assert not torch.cuda.is_initialized()
