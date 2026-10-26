import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from visual_rl.backends import minimax_h3 as backend
from visual_rl.backends import minimax_h3_runtime as runtime
from visual_rl.backends.minimax_h3_eval_reuse import evaluation_protocol, reuse_pretrained_eval0
from visual_rl.backends.minimax_h3_posttraining import (
    POST_TRAINING, evaluation_frames, rollout_config, validate_post_training,
)
from visual_rl.config import ConfigError
from visual_rl.rewards.audio_video import AVScores
from visual_rl.training.distributed import distributed_session
from visual_rl.wandb_export import MediaRecords, Records


def test_eval124_subset_leaves_all_training_durations_and_legacy_defaults(tmp_path, monkeypatch):
    from test_minimax_h3_runtime import config_factory
    base = config_factory.__wrapped__(tmp_path, monkeypatch)('art', 'full')
    post = {**POST_TRAINING, 'duration_frames': [124,243,345], 'full_duration_rewards': True}
    legacy = replace(base, algorithm=replace(base.algorithm, options={**base.algorithm.options,
                     'post_training': post}), sampling=replace(base.sampling, frames=124))
    c = replace(legacy, algorithm=replace(legacy.algorithm, options={**legacy.algorithm.options,
                'post_training': {**post, 'eval_frames': [124]}}))
    validate_post_training(c.algorithm.options['post_training'], 50)
    assert evaluation_frames(legacy) == (124,243,345)
    assert evaluation_frames(c) == (124,)
    assert [rollout_config(c, n).sampling.frames for n in range(6)] == [124,243,345,124,243,345]
    for invalid in ([], [243,124], [124,124], [107], [True], [[124]]):
        with pytest.raises(ConfigError):
            validate_post_training({**post, 'eval_frames': invalid}, 50)


def test_reuse_original_media_only_checks_protocol_and_does_not_regenerate(tmp_path, monkeypatch):
    from test_minimax_h3_runtime import TinyReward, config_factory, fixture_api
    base = config_factory.__wrapped__(tmp_path, monkeypatch)('art', 'full', evaluation=True)
    dataset, _ = backend.load_datasets(base)
    post = {**POST_TRAINING, 'duration_frames': [124,243,345],
            'full_duration_rewards': True, 'eval_frames': [124]}
    c = replace(base, algorithm=replace(base.algorithm, options={**base.algorithm.options,
        'post_training': post}), sampling=replace(base.sampling, frames=124),
        reward=replace(base.reward, weights={**base.reward.weights, 'imagebind': .5}))
    parts = fixture_api([]).load_components(c, torch.device('cpu'))
    class SixReward(TinyReward):
        def score(self, *args):
            scores = super().score(*args)
            return AVScores({**scores.components, 'imagebind': scores.components['video_align'] * .3},
                            scores.diagnostics)
    parts.reward = SixReward()
    generated = []
    def media_fixture(media, target, **_):
        target = Path(target); target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b'original pretrained AV fixture')
        target.with_suffix('.wav').write_bytes(b'original float audio fixture')
        generated.append(media.video.shape[2])
        return target
    monkeypatch.setattr(runtime, 'save_media', media_fixture)
    source, target = tmp_path/'original', tmp_path/'formal'
    source.mkdir(); target.mkdir()
    with distributed_session('none', device='cpu', timeout_seconds=30) as session:
        runtime.evaluate(session, c, parts, dataset, 0, source)
    assert generated == [124] * len(dataset.rows)
    assert runtime._completed_evaluation(c, dataset, 0, source)
    identity = dict(config_sha256=c.fingerprint(), parameters={'trainable': {'float32': 3}},
                    asset_manifest={'original': 'fixed'}, dp_world_size=1, pretrained_reference=True)
    (source/'resolved-config.json').write_text(json.dumps(c.to_dict()))
    (source/'run.json').write_text(json.dumps(identity))
    (target/'run.json').write_text(json.dumps(identity))
    original_hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()}
    formal = replace(c, training=replace(c.training, max_updates=400),
                     runtime=replace(c.runtime, output_dir=str(target)))
    receipt = reuse_pretrained_eval0(formal, dataset, target, source)
    assert receipt['policy_weights_loaded_from_source'] is False
    assert receipt['samples'] == len(dataset.rows)
    assert receipt['source_config_sha256'] == c.fingerprint()
    assert runtime._completed_evaluation(formal, dataset, 0, target)
    with distributed_session('none', device='cpu', timeout_seconds=30) as session:
        assert runtime.evaluate(session, formal, parts, dataset, 0, target) == {}
    assert generated == [124] * len(dataset.rows)
    assert len(list(Records(target).read())) == len(list(MediaRecords(target).read())) == 1
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest for p,digest in original_hashes.items())
    for section, key, value in [('seed', None, 99), ('sampling', 'steps', 25),
                                ('reward', 'weights', {'wrong': 1})]:
        document = copy.deepcopy(c.to_dict())
        if key is None: document[section] = value
        else: document[section][key] = value
        assert evaluation_protocol(document) != evaluation_protocol(c.to_dict())
    bad = replace(formal, seed=99)
    with pytest.raises(ConfigError, match='protocol'):
        reuse_pretrained_eval0(bad, dataset, target, source)
    next(source.rglob('*.wav')).write_bytes(b'corrupted')
    with pytest.raises(ConfigError, match='incomplete'):
        reuse_pretrained_eval0(formal, dataset, target, source)
