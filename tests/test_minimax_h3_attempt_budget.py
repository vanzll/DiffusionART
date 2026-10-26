"""Tiny-runtime boundary continuation; not actual H3 GPU evidence."""
from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from test_h3_mixed_reference_runtime import (
    api, checkpoint, equal_nested, metric_frames, mixed_config, make_config_factory,
)
from visual_rl.backends import minimax_h3_runtime as runtime
from visual_rl.backends.minimax_h3_attempt_budget import AttemptBudget
from visual_rl.backends.minimax_h3_posttraining import duration_frames, evaluation_frames
from visual_rl.training.distributed import DistributedSession, RankInfo


def test_real_runtime_boundary_chain_keeps_original_budget_and_exact_state(tmp_path, monkeypatch):
    factory = make_config_factory.__wrapped__(tmp_path, monkeypatch)
    monkeypatch.delenv('H3_ATTEMPT_STOP_UPDATE', raising=False)
    original = mixed_config(factory, 'continuous')
    monkeypatch.setenv('H3_ATTEMPT_NONCE', 'boundary-continuous')
    runtime.run(original, backend_api=api([]))
    expected = checkpoint(original.runtime.output_dir)
    config = mixed_config(factory, 'boundary-chain')
    root = Path(config.runtime.output_dir)
    resume = None
    saved = []
    save = runtime.save_checkpoint
    def count_save(*args, **kwargs):
        saved.append(kwargs['progress'].optimizer_steps)
        return save(*args, **kwargs)
    monkeypatch.setattr(runtime, 'save_checkpoint', count_save)
    for stop in (2, 4, 6, 8):
        monkeypatch.setenv('H3_ATTEMPT_STOP_UPDATE', str(stop))
        monkeypatch.setenv('H3_ATTEMPT_NONCE', f'boundary-{stop}')
        result = runtime.run(config, resume=resume, backend_api=api([]))
        assert result['optimizer_steps'] == stop
        if stop < 8:
            record = json.loads((root / f'attempt-boundaries/boundary-{stop}.json').read_text())
            assert record['target_optimizer_updates'] == 8
            assert record['remaining_optimizer_updates'] == 8 - stop
            assert record['progress']['completed_rollouts'] == stop // 2
            assert not (root / 'status.json').exists()
            resume = record['checkpoint']
    assert metric_frames(root) == [124, 243, 345, 124]
    assert saved == [2, 4, 6, 8]
    assert json.loads((root / 'status.json').read_text())['status'] == 'completed'
    actual = checkpoint(root)
    for key in ('model', 'optimizer', 'rng', 'generators', 'stateful'):
        equal_nested(actual[key], expected[key])


def test_attempt_stop_cannot_repeat_or_split_an_update_window(tmp_path, monkeypatch):
    factory = make_config_factory.__wrapped__(tmp_path, monkeypatch)
    config = mixed_config(factory, 'reject')
    session = DistributedSession(RankInfo(), torch.device('cpu'), False)
    for value in ('2', '3'):
        monkeypatch.setenv('H3_ATTEMPT_STOP_UPDATE', value)
        with pytest.raises(ValueError, match='must advance'):
            AttemptBudget(session, config, tmp_path, 2, start_optimizer_steps=2)

