"""Offline H3 reference-recipe and native scheduler equivalence checks."""

from pathlib import Path

import pytest
import torch

from visual_rl.backends.minimax_h3_runtime import sigma_schedules
from visual_rl.config import load_config
from visual_rl.registry import get_backend
from visual_rl.training.audio_video_rollout import collect_av_rollout
from visual_rl.training.audio_video_types import AVState

ROOT = Path(__file__).resolve().parents[1]
# The explicit 480p train-only protocol has its own runtime/offline-eval tests.
RECIPES = sorted(path for path in (ROOT / 'configs/examples').glob('minimax_h3_*.yaml')
                 if path.name not in {'minimax_h3_art_posttrain_full_32gpu.yaml',
                                      'minimax_h3_art_posttrain_full_64gpu_cp2.yaml',
                                      'minimax_h3_art_posttrain_lora_8gpu.yaml',
                                      'minimax_h3_art_three_reward_full_64gpu_cp2.yaml',
                                      'minimax_h3_fast_routed_480p_full_32gpu.yaml'})


@pytest.mark.parametrize('path', RECIPES, ids=lambda path: path.stem)
def test_public_recipes_share_rollout_and_eval_inference(path, monkeypatch):
    for key in ('MODEL', 'TRAIN_MANIFEST', 'REWARD_MANIFEST', 'EVAL_MANIFEST', 'OUTPUT_DIR'):
        monkeypatch.setenv('H3_' + key, '/explicit/offline/' + key.lower())
    for key in ('TRAIN_SHA256', 'EVAL_SHA256'):
        monkeypatch.setenv('H3_' + key, 'a' * 64)
    c = load_config(path)
    assert get_backend(c).name == 'minimax_h3'
    assert (c.sampling.steps, c.sampling.height, c.sampling.width) == (50, 768, 1344)
    assert (c.sampling.frames, c.sampling.fps, c.sampling.guidance_scale) == (124, 24., 1.)
    assert c.sampling.options['video_shift'] == 12.
    assert c.sampling.options['audio_shift'] == 3.
    assert c.evaluation.sampling_steps is None
    assert c.evaluation.height is c.evaluation.width is None
    # Eval follows rollout overrides instead of freezing a second grid.
    tuned = load_config(path, {'sampling': {'steps': 30, 'height': 544, 'width': 960}})
    assert (tuned.evaluation.sampling_steps or tuned.sampling.steps) == 30
    assert (tuned.evaluation.height or tuned.sampling.height) == 544
    assert (tuned.evaluation.width or tuned.sampling.width) == 960


@pytest.mark.parametrize('boundaries', [10, 11, 50])
def test_complete_ode_trajectory_matches_native_paired_schedulers(boundaries):
    from diffusers import MiniMaxH3Scheduler

    schedulers = [MiniMaxH3Scheduler(shift=shift) for shift in (12., 3.)]
    for scheduler in schedulers:
        scheduler.set_timesteps(boundaries)
    video, audio = sigma_schedules(boundaries)
    for sigmas, scheduler in zip((video, audio), schedulers):
        torch.testing.assert_close(sigmas, scheduler.sigmas, rtol=0, atol=0)
    generator = torch.Generator().manual_seed(42)
    noise = AVState(torch.randn(2, 4, generator=generator), torch.randn(2, 3, generator=generator))
    calls = []

    def predict(state, sv, sa, rows):
        calls.append((sv.clone(), sa.clone()))
        # Native output is dz/d(1-sigma); the adapter exposes its negative.
        return AVState(-(.07 * state.video + sv[:, None]),
                       -(.11 * state.audio + sa[:, None]))

    actual = collect_av_rollout(predict, noise, video, audio, microbatch=2, selected=())
    expected = [noise.video.clone(), noise.audio.clone()]
    for i in range(boundaries - 1):
        for j, (scale, scheduler) in enumerate(zip((.07, .11), schedulers)):
            output = scale * expected[j] + scheduler.sigmas[i]
            expected[j] = scheduler.step(output, scheduler.timesteps[i], expected[j], return_dict=False)[0]
    assert len(calls) == boundaries - 1
    torch.testing.assert_close(actual.endpoints.video, expected[0])
    torch.testing.assert_close(actual.endpoints.audio, expected[1])
