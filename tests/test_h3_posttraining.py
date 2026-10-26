"""H3 modality selection, K8 normalization and exact mean-loss update semantics."""

import copy
import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from visual_rl.algorithms import art_loss
from visual_rl.backends.minimax_h3 import validate_config
from visual_rl.backends.minimax_h3_posttraining import selected_positions
from visual_rl.config import ConfigError, Evaluation, load_config
from visual_rl.training.audio_video import prompt_disjoint_windows, run_av_updates
from visual_rl.training.audio_video_objectives import prepare_art_objective, route_rewards
from visual_rl.training.audio_video_rollout import collect_av_rollout
from visual_rl.training.audio_video_types import AVState


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.video = torch.nn.Parameter(torch.tensor(.1))
        self.audio = torch.nn.Parameter(torch.tensor(.2))

    def forward(self, state, sv, sa, rows):
        return AVState(state.video * self.video * (1 + sv[:, None]),
                       state.audio * self.audio * (1 + sa[:, None]))


def setup(selected):
    model = Policy()
    g = torch.Generator().manual_seed(9)
    noise = AVState(torch.randn(16, 4, generator=g), torch.randn(16, 3, generator=g))
    sv = torch.tensor([1., .99, .98, .96, .8, 0.])
    sa = torch.tensor([1., .96, .91, .85, .6, 0.])
    union = tuple(sorted(set(selected['video']) | set(selected['audio'])))
    cache = collect_av_rollout(model, noise, sv, sa, selected=union,
                              modality_selected=selected, microbatch=4)
    groups = torch.arange(16) // 8
    reward = torch.arange(16).float().remainder(8)
    objective = prepare_art_objective(cache, model, reward, reward.square(), groups, 'cpu',
                                      loss_options={'corrective_mask_enabled': False})
    return model, groups, objective


@pytest.mark.parametrize('video_pool,audio_pool', [(4, 3), (15, 10)])
def test_pairs_are_stateless_shared_and_do_not_advance_noise_rng(video_pool, audio_pool):
    state = random.getstate()
    torch_state = torch.get_rng_state().clone()
    pools = {'video_pool': video_pool, 'audio_pool': audio_pool}
    draws = [selected_positions(42, step, **pools) for step in range(1, 21)]
    assert draws == [selected_positions(42, step, **pools) for step in range(1, 21)]
    assert random.getstate() == state
    assert torch.equal(torch_state, torch.get_rng_state())
    assert len({d['video'] for d in draws}) > 1
    assert len({d['audio'] for d in draws}) > 1
    for draw in draws:
        for name, pool in [('video', video_pool), ('audio', audio_pool)]:
            assert len(draw[name]) == 2
            assert tuple(sorted(set(draw[name]))) == draw[name]
            assert set(draw[name]) <= set(range(pool))
    assert selected_positions(42, 1) == {'video': (1, 3), 'audio': (1, 2)}


def test_runtime_uses_configured_pools_and_eval_preserves_complete_grid():
    from visual_rl.backends.minimax_h3_posttraining import POST_TRAINING
    from visual_rl.backends.minimax_h3_runtime import collect, sigma_schedules

    post = {**POST_TRAINING, 'video_pool': 15, 'audio_pool': 10}
    c = SimpleNamespace(seed=42, algorithm=SimpleNamespace(name='diffusionart',
        options={'post_training': post}), sampling=SimpleNamespace(microbatch=2))
    session = SimpleNamespace(distributed=True, device=torch.device('cpu'))
    g = torch.Generator().manual_seed(42)
    noise = AVState(torch.randn(2, 4, generator=g), torch.randn(2, 3, generator=g))
    sigmas = sigma_schedules(50)
    model = Policy()
    cache = collect(None, c, model, noise, sigmas, session, g, rollout_index=3)
    expected = selected_positions(42, 4, video_pool=15, audio_pool=10)
    assert cache.video_selected == expected['video']
    assert cache.audio_selected == expected['audio']
    assert cache.selected == tuple(sorted(set(expected['video']) | set(expected['audio'])))
    evaluation = collect(None, c, model, noise, sigmas, session, g,
                         rollout_index=3, evaluation=True)
    assert evaluation.selected == ()
    torch.testing.assert_close(cache.endpoints.video, evaluation.endpoints.video, rtol=0, atol=0)
    torch.testing.assert_close(cache.endpoints.audio, evaluation.endpoints.audio, rtol=0, atol=0)


def test_six_rewards_normalize_complete_k8_before_routing():
    names = ['video_align', 'hpsv3', 'audiobox', 'clap', 'desync', 'imagebind']
    components = {name: torch.arange(16).float() * (i + 1) for i, name in enumerate(names)}
    components['desync'] = torch.full((16,), .7)
    groups = torch.arange(16) // 8
    weights = dict.fromkeys(names, 1.)
    result = route_rewards(components, groups=groups, weights=weights, prompt_group_std=True)
    for route, keys in [('video', ['video_align', 'hpsv3', 'desync', 'imagebind']),
                        ('audio', ['audiobox', 'clap', 'desync', 'imagebind'])]:
        expected = torch.zeros(16)
        for group in range(2):
            mask = groups == group
            for name in keys:
                value = components[name][mask]
                if name != 'desync':
                    expected[mask] += (value - value.mean()) / (value.std(correction=1) + 1e-4)
        torch.testing.assert_close(result[route], expected)
    changed = {**components, 'hpsv3': components['hpsv3'].square()}
    alternate = route_rewards(changed, groups=groups, weights=weights, prompt_group_std=True)
    assert not torch.equal(alternate['video'], result['video'])
    assert torch.equal(alternate['audio'], result['audio'])
    with pytest.raises(ValueError, match='K8'):
        route_rewards({k: v[:4] for k, v in components.items()}, groups=groups[:4],
                      weights=weights, prompt_group_std=True)


def manual_window(model, objective, items):
    cache = objective.cache
    rows = sorted({row for row, _ in items})
    losses = []
    for name in ('video', 'audio'):
        selected = getattr(cache, name + '_selected')
        branch = []
        for row in rows:
            for position in selected:
                slot = cache.selected.index(position)
                state = cache.states[slot].select(slice(row, row + 1))
                old = cache.old_velocities[slot].select(slice(row, row + 1))
                sv, sa = [s[position:position + 1] for s in (cache.video_sigmas, cache.audio_sigmas)]
                prediction = model(state, sv, sa, torch.tensor([row]))
                schedule = getattr(cache, name + '_sigmas')
                dt = schedule[position + 1:position + 2] - schedule[position:position + 1]
                value = art_loss(getattr(prediction, name), getattr(old, name),
                    sv if name == 'video' else sa,
                    getattr(objective.directions, name)[row:row + 1],
                    getattr(objective, name + '_magnitudes')[row:row + 1],
                    state=getattr(state, name), ode_velocity_coefficient=dt,
                    current_ode_mean=getattr(state, name) + dt[:, None] * getattr(prediction, name),
                    old_ode_mean=getattr(state, name) + dt[:, None] * getattr(old, name),
                    corrective_mask_enabled=False)
                branch.append(value['per_item_loss'].squeeze())
        losses.append(torch.stack(branch).mean())
    return .5 * losses[0] + .5 * losses[1]


@pytest.mark.parametrize('selection', [
    {'video': (0, 1), 'audio': (0, 1)},
    {'video': (0, 3), 'audio': (0, 1)},
    {'video': (0, 3), 'audio': (1, 2)},
])
@pytest.mark.parametrize('microbatch', [1, 3, 8])
def test_two_adam_updates_match_separate_modality_means(selection, microbatch):
    model, groups, objective = setup(selection)
    reference, reference_groups, reference_objective = setup(selection)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3)
    calls = []
    optimizer.register_step_post_hook(lambda *args, **kwargs: calls.append(1))
    metrics = run_av_updates(model, optimizer, objective, groups, algorithm='art',
                            microbatch=microbatch, max_grad_norm=1.)
    for items in prompt_disjoint_windows(reference_groups, len(reference_objective.cache.selected)):
        reference_optimizer.zero_grad(set_to_none=True)
        manual_window(reference, reference_objective, items).backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.)
        reference_optimizer.step()
    assert len(calls) == metrics['train/optimizer_steps_per_rollout'] == 2
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
        for key in ['step', 'exp_avg', 'exp_avg_sq']:
            torch.testing.assert_close(optimizer.state[actual][key], reference_optimizer.state[expected][key],
                                       rtol=2e-5, atol=2e-8)


def test_rollout_eval_cadence_counts_completed_rollouts_and_final():
    evaluation = Evaluation(every_updates=0, every_rollouts=50)
    assert evaluation.enabled
    assert not evaluation.due(98, 400, completed_rollouts=49)
    assert evaluation.due(100, 400, completed_rollouts=50)
    assert not evaluation.due(102, 400, completed_rollouts=51)
    assert evaluation.due(400, 400, completed_rollouts=200)
    assert evaluation.due(120, 120, completed_rollouts=60)
    with pytest.raises(ConfigError):
        Evaluation(every_rollouts=50)


def test_public_recipe_contract_and_default_config_identity(tmp_path, monkeypatch):
    for name in ['H3_MODEL', 'H3_REWARD_MANIFEST', 'H3_OUTPUT_DIR', 'H3_EVAL_MANIFEST']:
        monkeypatch.setenv(name, str(tmp_path / name))
    train = tmp_path / 'train.jsonl'
    train.write_text(json.dumps({'sample_id': 'train', 'prompt': 'sound and scene'}) + '\n')
    monkeypatch.setenv('H3_TRAIN_MANIFEST', str(train))
    monkeypatch.setenv('H3_TRAIN_SHA256', hashlib.sha256(train.read_bytes()).hexdigest())
    monkeypatch.setenv('H3_EVAL_SHA256', 'a' * 64)
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / 'configs/examples/minimax_h3_art_posttrain_full_32gpu.yaml')
    validate_config(config)
    assert config.evaluation.every_rollouts == 50
    assert config.runtime.checkpoint_every == 50 and config.runtime.checkpoint_keep == 2
    assert config.sampling.microbatch == config.training.microbatch == 8
    assert len(config.reward.weights) == 6 and config.training.max_updates == 400
    assert config.algorithm.options['post_training']['video_pool'] == 15
    assert config.algorithm.options['post_training']['audio_pool'] == 10
    legacy = load_config(root / 'configs/examples/minimax_h3_art_full_32gpu.yaml')
    validate_config(legacy)
    assert 'every_rollouts' not in legacy.to_dict()['evaluation']
    assert legacy.fingerprint() == replace(legacy, evaluation=replace(legacy.evaluation, every_rollouts=0)).fingerprint()
    path = root / 'configs/examples/minimax_h3_art_posttrain_full_32gpu.yaml'
    for invalid in (0, 1, 49, True):
        bad = copy.deepcopy(config.to_dict())
        bad['algorithm']['options']['post_training']['audio_pool'] = invalid
        with pytest.raises(ConfigError, match='nonfinal'):
            validate_config(load_config(path, bad))
    with pytest.raises(ConfigError, match='nonfinal'):
        validate_config(load_config(path, {'sampling': {'steps': 10}}))
    # Frozen legacy configs retain their valid pools and unchanged selection sequence.
    validate_config(load_config(path, {'sampling': {'steps': 10}, 'algorithm': {'options': {
        'post_training': {'video_pool': 4, 'audio_pool': 3}}}}))


def test_resume_repairs_final_eval_before_return_without_repeating_complete_eval(tmp_path, monkeypatch):
    from test_minimax_h3_runtime import TinyReward, config_factory, fixture_api

    from visual_rl.backends import minimax_h3_runtime as runtime
    from visual_rl.backends.minimax_h3_posttraining import POST_TRAINING

    factory = config_factory.__wrapped__(tmp_path, monkeypatch)
    base = factory('art', 'full', name='post')
    validation = tmp_path / 'eval.jsonl'
    validation.write_text(''.join(json.dumps({'sample_id': f'eval-{i:02d}', 'prompt': f'held out {i}',
        'metadata': {'reward_text': {'video': f'visual eval {i}', 'audio': f'sound eval {i}'}}}) + '\n'
        for i in range(32)))
    config = replace(base,
        algorithm=replace(base.algorithm, options={**base.algorithm.options, 'post_training': POST_TRAINING}),
        reward=replace(base.reward, weights={**base.reward.weights, 'imagebind': .5}),
        sampling=replace(base.sampling, steps=6, samples_per_prompt=8),
        training=replace(base.training, max_updates=2),
        evaluation=Evaluation(every_updates=0, every_rollouts=50, manifest=str(validation),
            manifest_sha256=hashlib.sha256(validation.read_bytes()).hexdigest(), fps=24., reward_fps=24.))
    original = TinyReward.score
    def six(self, *args):
        result = original(self, *args)
        result.components['imagebind'] = result.components['video_align'] + result.components['clap']
        return result
    monkeypatch.setattr(TinyReward, 'score', six)
    calls, complete = [], set()
    class EvalInterrupted(Exception):
        pass
    fail = [True]
    def evaluate(session, c, parts, dataset, step, root):
        calls.append(step)
        if step == 2 and fail[0]:
            fail[0] = False
            raise EvalInterrupted
        complete.add(step)
    monkeypatch.setattr(runtime, 'evaluate', evaluate)
    monkeypatch.setattr(runtime, '_completed_evaluation', lambda c, dataset, step, root: step in complete)
    with pytest.raises(EvalInterrupted):
        runtime.run(config, backend_api=fixture_api([]))
    root = Path(config.runtime.output_dir)
    latest = json.loads((root / 'checkpoints/latest.json').read_text())
    receipt = root / 'checkpoints' / latest['receipt']
    # The final model was saved before the interrupted final evaluation.
    result = runtime.run(config, resume=receipt, backend_api=fixture_api([]))
    assert result['optimizer_steps'] == 2 and result['completed_rollouts'] == 1
    assert calls == [0, 2, 2]
    runtime.run(config, resume=receipt, backend_api=fixture_api([]))
    assert calls == [0, 2, 2]


def test_completed_eval_audits_manifest_order_seeds_and_media(tmp_path):
    from types import SimpleNamespace

    from visual_rl.backends.minimax_h3_runtime import _completed_evaluation
    from visual_rl.evaluation import EvaluationManifest, SampleRecord
    from visual_rl.evaluation.audio_video_gallery import publish_av_gallery
    from visual_rl.rewards.scalar import weighted_score

    directory = tmp_path / 'evaluation/step-000000002/id'
    directory.mkdir(parents=True)
    names = ['video_align', 'hpsv3', 'audiobox', 'clap', 'desync', 'imagebind']
    scores = weighted_score(dict.fromkeys(names, .5), dict.fromkeys(names, 1.), space='raw')
    rows = [SimpleNamespace(sample_id=name, text=f'prompt {name}') for name in ('z', 'a')]
    records = []
    for i, row in enumerate(rows):
        path = directory / f'{row.sample_id}.mp4'
        path.write_bytes(b'muxed av fixture')
        path.with_suffix('.wav').write_bytes(b'original audio fixture')
        records.append(SampleRecord.from_file(media_root=tmp_path, path=path.relative_to(tmp_path).as_posix(),
            sample_id=row.sample_id, prompt=row.text, seed=42 + i, optimizer_step=2,
            framecount=4, fps=24., scores=scores))
    manifest = EvaluationManifest(2, tuple(row.sample_id for row in rows), tuple(records))
    (directory / 'manifest.json').write_text(manifest.to_json())
    (directory / 'summary.json').write_text('{"eval/optimizer_step":2}')
    publish_av_gallery(tmp_path, directory, records, 2)
    config = SimpleNamespace(seed=42, evaluation=SimpleNamespace(seed=None),
                             reward=SimpleNamespace(weights=dict.fromkeys(names, 1.)))
    dataset = SimpleNamespace(rows=rows)
    assert _completed_evaluation(config, dataset, 2, tmp_path)
    (directory / 'a.wav').unlink()
    assert not _completed_evaluation(config, dataset, 2, tmp_path)
