from dataclasses import fields, is_dataclass

import pytest
import torch

from visual_rl.backends.minimax_h3_forward_chunks import inference_forward_cap, predict_in_chunks
from visual_rl.config import ConfigError
from visual_rl.training.audio_video_rollout import collect_av_rollout
from visual_rl.training.audio_video_types import AVState


def same(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif is_dataclass(left):
        for field in fields(left):
            same(getattr(left, field.name), getattr(right, field.name))
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            same(a, b)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            same(left[key], right[key])
    else:
        assert left == right


def predictor(calls):
    def predict(state, sv, sa, rows):
        calls.append(len(state))
        def value(x, sigma):
            shape = (-1,) + (1,) * (x.ndim - 1)
            return x * .125 + rows.to(x).reshape(shape) * .001 + sigma.to(x).reshape(shape) * .01
        return AVState(value(state.video, sv), value(state.audio, sa), state.video_mask, state.audio_mask)
    return predict


@pytest.mark.parametrize("sigma_kind", ["scalar", "broadcast", "per_row"])
def test_no_grad_chunks_preserve_order_sigmas_masks_and_rng(sigma_kind):
    state = AVState(torch.arange(30.).reshape(5, 2, 3), torch.arange(40.).reshape(5, 4, 2),
                    torch.ones(5, 2, dtype=torch.bool), torch.ones(5, 4, dtype=torch.bool))
    sigma = {"scalar": torch.tensor(.5), "broadcast": torch.tensor([.5]),
             "per_row": torch.linspace(.2, .8, 5)}[sigma_kind]
    rows = torch.tensor([7, 2, 9, 1, 3])
    full_calls, chunk_calls = [], []
    before = torch.get_rng_state().clone()
    with torch.no_grad():
        expected = predictor(full_calls)(state, sigma, sigma, rows)
        actual = predict_in_chunks(predictor(chunk_calls), state, sigma, sigma, rows, cap=2)
    same(actual, expected)
    assert full_calls == [5] and chunk_calls == [2, 2, 1]
    assert torch.equal(before, torch.get_rng_state())


def test_training_forward_is_not_split_and_keeps_gradients():
    state = AVState(torch.ones(5, 2, requires_grad=True), torch.ones(5, 3, requires_grad=True))
    calls = []
    out = predict_in_chunks(predictor(calls), state, torch.tensor(.5), torch.tensor(.4),
                            torch.arange(5), cap=2)
    (out.video.sum() + out.audio.sum()).backward()
    assert calls == [5]
    assert torch.equal(state.video.grad, torch.full_like(state.video, .125))
    assert torch.equal(state.audio.grad, torch.full_like(state.audio, .125))


def test_full49_routed_rollout_cache_and_generator_identity():
    generator = torch.Generator().manual_seed(42)
    noise = AVState(torch.randn(8, 2, 3, generator=generator), torch.randn(8, 4, 2, generator=generator))
    selected = {"video": (1, 8), "audio": (2, 7)}
    positions = (1, 2, 7, 8)
    innovations = {p: AVState(torch.randn(noise.video.shape, generator=generator),
                             torch.randn(noise.audio.shape, generator=generator)) for p in positions}
    base = torch.linspace(1., 0., 50)
    schedules = [shift * base / (1 + (shift - 1) * base) for shift in (12., 3.)]
    state_before = generator.get_state().clone()
    global_before = torch.get_rng_state().clone()
    full_calls, chunk_calls = [], []
    full = predictor(full_calls)
    direct = predictor(chunk_calls)
    def chunked(state, sv, sa, rows):
        return predict_in_chunks(direct, state, sv, sa, rows, cap=4)
    options = dict(selected=positions, stochastic=positions, innovations=innovations,
                   modality_selected=selected, modality_stochastic=selected,
                   microbatch=8, unselected_kind="cps_zero_noise", noise_level=.8)
    with torch.no_grad():
        expected = collect_av_rollout(full, noise, *schedules, **options)
        actual = collect_av_rollout(chunked, noise, *schedules, **options)
    same(actual, expected)
    assert full_calls == [8] * 49
    assert chunk_calls == [4] * 98  # Each trajectory still has exactly49 native denoising steps.
    assert torch.equal(state_before, generator.get_state())
    assert torch.equal(global_before, torch.get_rng_state())


@pytest.mark.parametrize("raw", ["-1", "no", "1.5"])
def test_cap_rejects_invalid_values(monkeypatch, raw):
    monkeypatch.setenv("H3_INFERENCE_FORWARD_MICROBATCH", raw)
    with pytest.raises(ConfigError):
        inference_forward_cap()
