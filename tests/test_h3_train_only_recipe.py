"""The ten-transition recipe skips eval but retains resumable offline evaluation."""
import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from test_h3_mixed_reference_runtime import api, mixed_config
from test_minimax_h3_runtime import config_factory as make_config_factory

from visual_rl.backends import minimax_h3 as backend
from visual_rl.backends import minimax_h3_runtime as runtime
from visual_rl.config import ConfigError, load_config, resume_signature

ROOT = Path(__file__).resolve().parents[1]


def test_public_train_only_protocol_and_offline_resume_signature(tmp_path, monkeypatch):
    factory = make_config_factory.__wrapped__(tmp_path, monkeypatch)
    factory('art', 'full')
    validation = ROOT / 'configs/examples/data/minimax_h3_eval32.jsonl'
    monkeypatch.setenv('H3_EVAL_MANIFEST', str(validation))
    monkeypatch.setenv('H3_EVAL_SHA256', hashlib.sha256(validation.read_bytes()).hexdigest())
    monkeypatch.setenv('H3_OUTPUT_DIR', str(tmp_path / 'run'))
    path = ROOT / 'configs/examples/minimax_h3_art_posttrain_full_32gpu.yaml'
    c = load_config(path)
    backend.validate_config(c)
    assert (c.sampling.width, c.sampling.height, c.sampling.frames, c.sampling.fps) == (864, 480, 124, 24)
    assert c.sampling.steps == 11 and c.sampling.guidance_scale == 1
    assert c.sampling.microbatch == c.training.microbatch == 8
    assert c.sampling.prompts_per_rollout == 32 and c.sampling.samples_per_prompt == 8
    assert c.training.max_updates == 400 and backend.updates_per_rollout(c) == 2
    post = c.algorithm.options['post_training']
    assert (post['video_pool'], post['audio_pool'], post['duration_frames']) == (4, 3, [124])
    sv, sa = runtime.sigma_schedules(c.sampling.steps)
    torch.testing.assert_close(sv[:4], torch.tensor([1., .990825688, .979591837, .965517241]))
    torch.testing.assert_close(sa[:3], torch.tensor([1., .964285714, .923076923]))
    assert len(sv) - 1 == len(sa) - 1 == 10
    assert not c.evaluation.enabled
    assert not any(c.evaluation.due(s, 400, completed_rollouts=s // 2) for s in range(0, 401, 2))
    assert (c.evaluation.width, c.evaluation.height, c.evaluation.sampling_steps) == (1344, 768, 50)
    assert c.runtime.checkpoint_every == 50 and c.runtime.checkpoint_keep == 2
    offline = replace(c, evaluation=replace(c.evaluation, at_updates=(2,)))
    assert offline.evaluation.enabled
    assert resume_signature(c.to_dict()) == resume_signature(offline.to_dict())
    changed = replace(offline, sampling=replace(offline.sampling, width=832))
    assert resume_signature(c.to_dict()) != resume_signature(changed.to_dict())


def test_public_cp2_recipe_preserves_training_contract_and_rejects_eval(tmp_path, monkeypatch):
    # Compare every scientific setting, including P/K and the optimizer windows.
    test_public_train_only_protocol_and_offline_resume_signature(tmp_path, monkeypatch)
    full32 = load_config(ROOT / 'configs/examples/minimax_h3_art_posttrain_full_32gpu.yaml')
    cp2 = load_config(ROOT / 'configs/examples/minimax_h3_art_posttrain_full_64gpu_cp2.yaml')
    backend.validate_config(cp2)
    assert cp2.model.options['context_parallel_size'] == 2
    assert cp2.runtime.wandb['name'].endswith('full64-cp2')
    actual, expected = cp2.to_dict(), full32.to_dict()
    actual['model']['options'].pop('context_parallel_size')
    for value in (actual, expected):
        value.pop('name')
        value.pop('provenance')
        value['runtime']['wandb'].pop('name')
    assert actual == expected
    offline = replace(cp2, evaluation=replace(cp2.evaluation, at_updates=(2,)))
    with pytest.raises(ConfigError, match='training only'):
        backend.validate_config(offline)
    # Direct callers must also fail before distributed setup or asset loading.
    with pytest.raises(ConfigError, match='training only'):
        runtime.run(cp2, evaluate_only=True)


def test_training_skips_all_eval_then_offline_restores_without_optimizer_updates(tmp_path, monkeypatch):
    factory = make_config_factory.__wrapped__(tmp_path, monkeypatch)
    base = mixed_config(factory, 'train-only')
    options = copy.deepcopy(base.algorithm.options)
    options['post_training']['duration_frames'] = [124]
    validation = tmp_path / 'heldout32.jsonl'
    validation.write_text(''.join(json.dumps({'sample_id': f'eval{i:03d}', 'prompt': f'heldout {i}',
        'metadata': {'reward_text': {'video': f'visual eval{i}', 'audio': f'sound eval{i}'}}}) + '\n'
        for i in range(32)))
    c = replace(base, algorithm=replace(base.algorithm, options=options),
        sampling=replace(base.sampling, steps=11, microbatch=8),
        training=replace(base.training, max_updates=6, microbatch=8),
        runtime=replace(base.runtime, checkpoint_every=2, checkpoint_keep=2),
        evaluation=replace(base.evaluation, every_updates=0, every_rollouts=0, at_updates=(),
            manifest=str(validation), manifest_sha256=hashlib.sha256(validation.read_bytes()).hexdigest(),
            height=768, width=1344, sampling_steps=50, seed=42))
    monkeypatch.setenv('H3_ATTEMPT_NONCE', 'train-only-runtime-fixture')
    original_evaluate = runtime.evaluate

    def forbidden(*args, **kwargs):
        raise AssertionError('disabled evaluation executed')

    monkeypatch.setattr(runtime, 'evaluate', forbidden)
    result = runtime.run(c, backend_api=api([]))
    assert result['optimizer_steps'] == 6 and result['completed_rollouts'] == 3
    root = Path(c.runtime.output_dir)
    assert not (root / 'evaluation').exists()
    checkpoints = root / 'checkpoints'
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoints.iterdir() if p.is_file()}
    # A committed latest pointer plus the two retained complete CPU snapshots.
    assert len(list(checkpoints.glob('*.pt'))) == 2
    rows = [json.loads(s) for s in (root / 'metrics.jsonl').read_text().splitlines()]
    assert all(x['train/optimizer_steps_per_rollout'] == 2 for x in rows)

    def save_fixture(media, target, **kwargs):
        target = Path(target)
        target.parent.mkdir(parents=True)
        target.write_bytes(b'muxed AV fixture')
        target.with_suffix('.wav').write_bytes(b'PCM fixture')
        return target

    monkeypatch.setattr(runtime, 'evaluate', original_evaluate)
    monkeypatch.setattr(runtime, 'save_media', save_fixture)
    offline = replace(c, evaluation=replace(c.evaluation, at_updates=(2,)))
    runtime.run(offline, resume='latest', evaluate_only=True, backend_api=api([]))
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoints.iterdir() if p.is_file()}
    assert before == after
    manifests = list((root / 'evaluation').glob('*/*/manifest.json'))
    assert len(manifests) == 1
    samples = json.loads(manifests[0].read_text())['samples']
    assert len(samples) == 32 and {x['optimizer_step'] for x in samples} == {6}
    assert [x['seed'] for x in samples] == list(range(42, 74))
    monkeypatch.setattr(runtime, 'evaluate', forbidden)
    resumed = runtime.run(c, resume='latest', backend_api=api([]))
    assert resumed['optimizer_steps'] == 6
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in checkpoints.iterdir() if p.is_file()}
