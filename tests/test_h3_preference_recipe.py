"""Three-reward semantics and real CPU update/resume; no pretrained GPU claim."""

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_h3_mixed_reference_runtime import (
    DurationAdapter,
    TinyConditioner,
    checkpoint,
    equal_nested,
    mixed_config,
)
from test_minimax_h3_runtime import config_factory as make_config_factory

from visual_rl.backends import minimax_h3 as backend
from visual_rl.backends import minimax_h3_runtime as runtime
from visual_rl.config import ConfigError, load_config, resume_signature
from visual_rl.models.prepare_h3_preference import prepare_manifest
from visual_rl.models.prepare_minimax_h3 import build_reward_manifest
from visual_rl.rewards.audio_video import AVScores
from visual_rl.rewards.av_profiles import (
    PICKSCORE_PROCESSOR_REPOSITORY,
    PICKSCORE_PROCESSOR_REVISION,
    PICKSCORE_REPOSITORY,
    PICKSCORE_REVISION,
    THREE_REWARD_ASSETS,
    THREE_REWARD_PROFILE,
    THREE_REWARD_WEIGHTS,
)
from visual_rl.rewards.h3_preference import (
    H3PreferenceReward,
    VideoPickScore,
    pickscore_frame_indices,
)
from visual_rl.training.audio_video_objectives import route_rewards
from visual_rl.training.audio_video_types import AVMedia

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / 'configs/examples/minimax_h3_art_three_reward_full_64gpu_cp2.yaml'


def test_only_reward_data_bindings_and_run_names_change(tmp_path, monkeypatch):
    factory = make_config_factory.__wrapped__(tmp_path, monkeypatch)
    factory('art', 'full')
    monkeypatch.setenv('H3_EVAL_MANIFEST', str(tmp_path / 'unused-eval.jsonl'))
    monkeypatch.setenv('H3_OUTPUT_DIR', str(tmp_path / 'run'))
    monkeypatch.setenv('H3_OMNINFT_DATA_ROOT', str(tmp_path / 'omninft-data'))
    current = load_config(RECIPE)
    baseline = load_config(RECIPE.with_name('minimax_h3_art_posttrain_full_64gpu_cp2.yaml'))
    backend.validate_config(current)
    assert current.reward.weights == THREE_REWARD_WEIGHTS
    assert current.reward.options['profile'] == THREE_REWARD_PROFILE
    assert current.data.manifest == str(tmp_path / 'omninft-data/train.jsonl')
    assert current.data.sha256 == '29c004c3927c34663ee8f95ad1b34b1bce7c91c95f86af18724627324e82699d'
    assert current.evaluation.manifest == str(tmp_path / 'omninft-data/test.jsonl')
    assert current.evaluation.manifest_sha256 == 'e0e6b9f2451e2089582121436f6a8b7b539920b671e0ae9ba1587f86c83785a4'
    assert resume_signature(current.to_dict()) != resume_signature(baseline.to_dict())
    actual, expected = current.to_dict(), baseline.to_dict()
    for value in (actual, expected):
        value.pop('name')
        value.pop('reward')
        value['data'].pop('manifest')
        value['data'].pop('sha256')
        value['evaluation'].pop('manifest')
        value['evaluation'].pop('manifest_sha256')
        value['runtime']['wandb'].pop('name')
    assert actual == expected
    for bad in (
        replace(current, reward=replace(current.reward, options={'manifest': 'unused'})),
        replace(current, reward=replace(current.reward, weights={**current.reward.weights, 'hpsv3': 0.})),
        replace(current, reward=replace(current.reward, weights={**current.reward.weights, 'imagebind': .5})),
        replace(current, reward=replace(current.reward, options={**current.reward.options, 'profile': 'typo'})),
    ):
        with pytest.raises(ConfigError):
            backend.validate_config(bad)


def test_group_statistics_precede_routing_and_imagebind_is_shared_once():
    groups = torch.arange(16) // 8
    x = torch.arange(16).float()
    values = {'pickscore': x.square(), 'clap': x.sin(), 'imagebind': x.cos()}
    def route(values):
        return route_rewards(values, groups=groups, weights=THREE_REWARD_WEIGHTS, prompt_group_std=True)
    actual = route(values)
    normalized = {}
    for name, value in values.items():
        normalized[name] = torch.cat([(v - v.mean()) / (v.std(correction=1) + 1e-4)
                                      for v in value.split(8)])
    torch.testing.assert_close(actual['video'], normalized['pickscore'] + normalized['imagebind'])
    torch.testing.assert_close(actual['audio'], normalized['clap'] + normalized['imagebind'])
    torch.testing.assert_close(actual['shared'], sum(normalized.values()))
    assert torch.equal(route({**values, 'clap': x.exp()})['video'], actual['video'])
    assert torch.equal(route({**values, 'pickscore': x.exp()})['audio'], actual['audio'])
    constant = route({**values, 'imagebind': torch.ones(16)})
    torch.testing.assert_close(constant['video'], normalized['pickscore'])
    with pytest.raises(ValueError, match='K8'):
        route_rewards({k: v[:4] for k, v in values.items()}, groups=groups[:4],
                      weights=THREE_REWARD_WEIGHTS, prompt_group_std=True)


def test_pickscore_uses_all_eight_uniform_frames_raw_mean_and_video_text():
    observed = {}
    class Scorer(torch.nn.Module):
        def forward(self, images, prompts, *, microbatch):
            observed.update(images=images.clone(), prompts=prompts, microbatch=microbatch)
            return SimpleNamespace(raw={'pickscore': torch.tensor([1., 2., 3., 4., 5., 6., 7., 100.])})
    video = torch.arange(124, dtype=torch.uint8)[:, None, None, None].expand(-1, 3, 2, 2)
    result = VideoPickScore(Scorer()).infer(video, 'visual subject')
    indices = pickscore_frame_indices(124)
    assert indices.tolist() == [0, 18, 35, 53, 70, 88, 105, 123]
    torch.testing.assert_close(observed['images'][:, 0, 0, 0], indices.float() / 255)
    assert observed['prompts'] == ['visual subject'] * 8 and observed['microbatch'] == 8
    assert result['score'] == 16. and result['frame_min_raw'] == 1. and result['frame_max_raw'] == 100.
    with pytest.raises(ValueError, match='eight'):
        pickscore_frame_indices(7)


def test_three_scorers_keep_prompt_audio_rate_and_restore_residency(monkeypatch):
    from visual_rl.rewards._native_av import clap, imagebind
    class Native:
        def __init__(self):
            self.model = torch.nn.Linear(1, 1)
            self.device = 'cpu'
        def infer(self, video, prompt):
            assert video.shape == (124, 3, 2, 2) and prompt == 'visual text'
            return {'score': 22.}
    async def cl(*args, **kwargs):
        assert args[2] == 'sound text' and args[3]['audio_sample_rate'] == 32000
        assert args[3]['audio'].shape[0] == 2
        assert kwargs['score_scale'] == .5 and kwargs['score_offset'] == .5
        return {'score': .8}
    async def ib(**kwargs):
        assert kwargs['extra_info']['fps'] == 24.
        return {'score': .3}
    monkeypatch.setattr(clap, 'compute_score', cl)
    monkeypatch.setattr(imagebind, 'compute_score', ib)
    models = {k: Native() for k in THREE_REWARD_WEIGHTS}
    reward = H3PreferenceReward(models)
    media = AVMedia(torch.zeros(1, 3, 124, 2, 2), torch.zeros(1, 2, round(124 / 24 * 32000)), 24., 32000)
    result = reward.score(media, ['visual text'], ['sound text'])
    assert set(result.components) == set(THREE_REWARD_WEIGHTS)
    assert result.components['pickscore'].item() == 22.
    assert all(model.device == torch.device('cpu') for model in models.values())


def preference_assets(tmp_path, monkeypatch):
    from visual_rl.rewards._native_av import imagebind
    monkeypatch.setattr(imagebind, 'validate_source', lambda p: p)
    paths, provenance = {}, {}
    for name in THREE_REWARD_ASSETS:
        p = tmp_path / name
        if name == 'imagebind':
            p.write_bytes(b'test-weights')
        else:
            p.mkdir()
            (p / 'config.json').write_text('{}')
        paths[name] = str(p)
        provenance[name] = {'repo': 'fixture/repo', 'revision': 'a' * 40, 'license': 'test-only'}
    provenance['imagebind_source'] = {'repo': imagebind.IMAGEBIND_REPOSITORY,
        'revision': imagebind.IMAGEBIND_REVISION, 'license': imagebind.IMAGEBIND_LICENSE}
    for name, repo, revision in (
        ('pickscore', PICKSCORE_REPOSITORY, PICKSCORE_REVISION),
        ('pickscore_processor', PICKSCORE_PROCESSOR_REPOSITORY, PICKSCORE_PROCESSOR_REVISION),
    ):
        provenance[name].update(repo=repo, revision=revision)
    return paths, provenance


def test_three_asset_manifest_roundtrip_and_rejects_wrong_model(tmp_path, monkeypatch):
    paths, provenance = preference_assets(tmp_path, monkeypatch)
    manifest = build_reward_manifest(paths, provenance)
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    observed = {}
    def load(cls, paths, **kwargs):
        observed.update(paths=paths, **kwargs)
        return SimpleNamespace()
    monkeypatch.setattr(H3PreferenceReward, 'from_paths', classmethod(load))
    c = SimpleNamespace(reward=SimpleNamespace(weights=THREE_REWARD_WEIGHTS,
        options={'manifest': str(path), 'profile': THREE_REWARD_PROFILE}),
        algorithm=SimpleNamespace(options={'post_training': {'full_duration_rewards': True}}))
    reward = backend.load_reward(c, 'cpu')
    assert set(observed['paths']) == THREE_REWARD_ASSETS and observed['full_duration'] is True
    assert reward.asset_identity['manifest_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    provenance['pickscore']['revision'] = 'b' * 40
    with pytest.raises(ValueError, match='pinned official'):
        build_reward_manifest(paths, provenance)
    manifest['provenance'] = provenance
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='pinned official'):
        backend.load_reward(c, 'cpu')


def test_three_loader_only_loads_requested_frozen_offline_models(tmp_path, monkeypatch):
    from visual_rl.rewards._native_av import clap, imagebind
    from visual_rl.rewards.pickscore import PickScore
    paths, provenance = preference_assets(tmp_path, monkeypatch)
    manifest = build_reward_manifest(paths, provenance)
    observed = {}
    class Native:
        def __init__(self, *args, **kwargs):
            self.model = torch.nn.Linear(1, 1)
            self.device = torch.device('cpu')
        def close(self):
            pass
    def cl(*args, **kwargs):
        observed['clap'] = kwargs
        return Native()
    def ps(cls, **kwargs):
        observed['pickscore'] = kwargs
        return torch.nn.Linear(1, 1)
    monkeypatch.setattr(clap, 'CLAPModel', cl)
    monkeypatch.setattr(imagebind, 'ImageBindModel', Native)
    monkeypatch.setattr(PickScore, 'from_pretrained', classmethod(ps))
    reward = H3PreferenceReward.from_paths(paths, device='cpu', expected_sha256=manifest['sha256'])
    assert set(reward.models) == set(THREE_REWARD_WEIGHTS)
    assert observed['clap']['model_kwargs'] == {'local_files_only': True}
    assert observed['pickscore']['local_files_only'] is True
    assert observed['pickscore']['model_revision'] == PICKSCORE_REVISION
    assert observed['pickscore']['processor_revision'] == PICKSCORE_PROCESSOR_REVISION
    assert observed['pickscore']['space'] == 'raw' and observed['pickscore']['dtype'] == torch.float32
    assert all(not p.requires_grad for p in reward.models['clap'].model.parameters())
    # Inventory failures happen before a model is loaded.
    (Path(paths['pickscore']) / 'config.json').write_text('changed')
    with pytest.raises(ValueError, match='checksum'):
        H3PreferenceReward.from_paths(paths, device='cpu', expected_sha256=manifest['sha256'])


def test_prepare_reuses_successful_scorer_bytes_and_rejects_drift(tmp_path, monkeypatch):
    paths, provenance = preference_assets(tmp_path, monkeypatch)
    old = build_reward_manifest(paths, provenance)
    path = tmp_path / 'successful-manifest.json'
    path.write_text(json.dumps(old))
    result = prepare_manifest(path, paths['pickscore'], paths['pickscore_processor'])
    for key in ('clap', 'imagebind', 'imagebind_source'):
        assert result['sha256'][key] == old['sha256'][key]
    (Path(paths['clap']) / 'config.json').write_text('changed after old run')
    with pytest.raises(ValueError, match='successful run inventory'):
        prepare_manifest(path, paths['pickscore'], paths['pickscore_processor'])


def test_real_runtime_u2_reference_checkpoint_and_resume(tmp_path, monkeypatch):
    factory = make_config_factory.__wrapped__(tmp_path, monkeypatch)
    c = mixed_config(factory, 'three-continuous')
    options = copy.deepcopy(c.algorithm.options)
    options['post_training']['duration_frames'] = [124]
    c = replace(c, algorithm=replace(c.algorithm, options=options),
        reward=replace(c.reward, weights=THREE_REWARD_WEIGHTS,
            options={**c.reward.options, 'profile': THREE_REWARD_PROFILE}))
    class Reward:
        def score(self, media, video_prompts, audio_prompts):
            v, a = media.video.flatten(1).mean(1), media.audio.flatten(1).mean(1)
            return AVScores({'pickscore': v, 'clap': a, 'imagebind': v + a}, {})
        def close(self):
            pass
    def load(config, device):
        return SimpleNamespace(adapter=DurationAdapter('full'), conditioner=TinyConditioner(),
            reward=Reward(), manifest={'kind': 'three-reward-cpu-fixture'})
    api = SimpleNamespace(validate_config=backend.validate_config, load_datasets=backend.load_datasets,
        load_components=load, updates_per_rollout=backend.updates_per_rollout)
    monkeypatch.setenv('H3_ATTEMPT_NONCE', 'three-reward-uninterrupted')
    result = runtime.run(c, backend_api=api)
    assert result['optimizer_steps'] == 8 and result['completed_rollouts'] == 4
    expected = checkpoint(c.runtime.output_dir)
    rows = [json.loads(s) for s in (Path(c.runtime.output_dir) / 'metrics.jsonl').read_text().splitlines()]
    assert all(row['train/optimizer_steps_per_rollout'] == 2 for row in rows)
    assert all('train/reward/pickscore' in row and 'train/reward/hpsv3' not in row for row in rows)
    split = replace(c, runtime=replace(c.runtime, output_dir=str(tmp_path / 'three-split')),
                    training=replace(c.training, max_updates=4))
    runtime.run(split, backend_api=api)
    resumed = replace(split, training=replace(split.training, max_updates=8))
    runtime.run(resumed, resume=Path(split.runtime.output_dir) / 'checkpoints/latest.json', backend_api=api)
    actual = checkpoint(split.runtime.output_dir)
    for key in ('model', 'optimizer', 'generators', 'stateful'):
        equal_nested(actual[key], expected[key])
    for key in ('optimizer_steps', 'completed_rollouts'):
        assert actual['progress'][key] == expected['progress'][key]
