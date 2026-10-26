"""Tiny real-runtime duration/reference resume; no pretrained or GPU evidence."""
import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_minimax_h3_runtime import TinyAdapter, TinyConditioner, TinyReward
from test_minimax_h3_runtime import config_factory as make_config_factory

from visual_rl.backends import minimax_h3 as backend
from visual_rl.backends import minimax_h3_runtime as runtime
from visual_rl.backends.minimax_h3_posttraining import POST_TRAINING
from visual_rl.training.audio_video_types import AVMedia, AVState


class DurationAdapter(TinyAdapter):
    def latent_shapes(self, count, height, width, frames, fps):
        self.frames = frames
        # Distinct temporal sizes exercise reference caches of different shapes.
        return (count, frames // 17, 2), (count, frames // 17 + 1, 3)

    def predict_velocity(self, state, video_sigma, audio_sigma, conditioning):
        self.predictions += 1
        values = self.transformer(AVState(state.video.float(), state.audio.float()))
        prompt = conditioning.prompt_embeds[:, 0, 0].reshape(-1, 1, 1) * .001
        return AVState(values.video + prompt, values.audio + prompt)

    def decode(self, state):
        # Keep native rates/duration but do not materialize repeated audio samples.
        frames = self.frames
        samples = round(frames / 24 * 32000)
        video = state.video.float().flatten(1).mean(1).sigmoid().reshape(-1, 1, 1, 1, 1)
        audio = state.audio.float().flatten(1).mean(1).tanh().reshape(-1, 1, 1)
        return AVMedia(video.expand(-1, 3, frames, 4, 4),
                       audio.expand(-1, 2, samples), 24., 32000)


class SixReward(TinyReward):
    def score(self, *args):
        result = super().score(*args)
        result.components['imagebind'] = result.components['video_align'] + result.components['clap']
        return result


def api(created):
    def load(config, device):
        parts = SimpleNamespace(adapter=DurationAdapter('full'), conditioner=TinyConditioner(),
            reward=SixReward(), manifest={'kind': 'explicit_cpu_mixed_reference_fixture'})
        parts.initial_model = copy.deepcopy(parts.adapter.transformer.state_dict())
        created.append(parts)
        return parts
    return SimpleNamespace(validate_config=backend.validate_config,
        load_datasets=backend.load_datasets, load_components=load,
        updates_per_rollout=backend.updates_per_rollout)


def mixed_config(factory, name):
    config = factory('art', 'full', name=name)
    options = copy.deepcopy(config.algorithm.options)
    options['post_training'] = {**POST_TRAINING, 'duration_frames': [124, 243, 345],
                                'full_duration_rewards': True}
    options['loss']['reference_mse_beta'] = 1e-3
    return replace(config, algorithm=replace(config.algorithm, options=options),
        reward=replace(config.reward, weights={**config.reward.weights, 'imagebind': .5}),
        sampling=replace(config.sampling, frames=124, steps=6, samples_per_prompt=8),
        training=replace(config.training, max_updates=8))


def checkpoint(root):
    root = Path(root) / 'checkpoints'
    latest = json.loads((root / 'latest.json').read_text())
    receipt = json.loads((root / latest['receipt']).read_text())
    return torch.load(root / receipt['file'], weights_only=True)


def metric_frames(root):
    return [row['av/rollout_frames'] for row in
        (json.loads(line) for line in (Path(root) / 'metrics.jsonl').read_text().splitlines())]


def equal_nested(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            equal_nested(actual[key], value)
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            equal_nested(left, right)
    else:
        assert actual == expected


def test_mixed_length_reference_runtime_resume_is_exact(tmp_path, monkeypatch):
    config_factory = make_config_factory.__wrapped__(tmp_path, monkeypatch)
    monkeypatch.setenv('H3_ATTEMPT_NONCE', 'mixed-reference-uninterrupted')
    original = mixed_config(config_factory, 'mixed-uninterrupted')
    created = []
    result = runtime.run(original, backend_api=api(created))
    assert result['optimizer_steps'] == 8 and result['completed_rollouts'] == 4
    assert metric_frames(original.runtime.output_dir) == [124, 243, 345, 124]
    expected = checkpoint(original.runtime.output_dir)
    frozen = expected['stateful']['h3_reference_policy']['parameters']
    equal_nested(frozen, created[0].initial_model)
    assert any(not torch.equal(value, expected['model'][name]) for name, value in frozen.items())
    rows = [json.loads(line) for line in
            (Path(original.runtime.output_dir) / 'metrics.jsonl').read_text().splitlines()]
    assert all(row['train/optimizer_steps_per_rollout'] == 2 for row in rows)
    assert rows[0]['av/window_0/video_reference_mse'] < 1e-12
    assert rows[0]['av/window_0/audio_reference_mse'] < 1e-12
    assert any(row['av/video_reference_mse'] > 0 or row['av/audio_reference_mse'] > 0 for row in rows[1:])
    assert json.loads((Path(original.runtime.output_dir) / 'run.json').read_text())['pretrained_reference']

    interrupted = mixed_config(config_factory, 'mixed-interrupted')
    save = runtime.save_checkpoint
    receipts = []
    class StopAfterCheckpoint(Exception):
        pass
    def stop(*args, **kwargs):
        receipt = save(*args, **kwargs)
        receipts.append(receipt)
        raise StopAfterCheckpoint
    monkeypatch.setattr(runtime, 'save_checkpoint', stop)
    monkeypatch.setenv('H3_ATTEMPT_NONCE', 'mixed-reference-interrupted')
    with pytest.raises(StopAfterCheckpoint):
        runtime.run(interrupted, backend_api=api([]))
    assert metric_frames(interrupted.runtime.output_dir) == [124]
    monkeypatch.setattr(runtime, 'save_checkpoint', save)
    monkeypatch.setenv('H3_ATTEMPT_NONCE', 'mixed-reference-resumed')
    result = runtime.run(interrupted, resume=receipts[0], backend_api=api([]))
    assert result['optimizer_steps'] == 8 and result['completed_rollouts'] == 4
    assert metric_frames(interrupted.runtime.output_dir) == [124, 243, 345, 124]
    actual = checkpoint(interrupted.runtime.output_dir)
    for key in ('model', 'optimizer', 'rng', 'generators', 'stateful'):
        equal_nested(actual[key], expected[key])
