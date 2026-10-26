"""Frozen H3 reference semantics, independent of ART targets and modal sizes."""
import copy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from visual_rl.backends.minimax_h3 import validate_config
from visual_rl.backends.minimax_h3_reference import ReferenceState, prepare_reference_predictions
from visual_rl.config import ConfigError, load_config
from visual_rl.training.audio_video_objectives import prepare_art_objective
from visual_rl.training.audio_video_rollout import collect_av_rollout, feature_mean
from visual_rl.training.audio_video_types import AVState


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.video = torch.nn.Parameter(torch.tensor(.1))
        self.audio = torch.nn.Parameter(torch.tensor(.2))

    def forward(self, state, sv, sa, rows):
        return AVState(state.video * self.video * (1 + sv[:, None]),
                       state.audio * self.audio * (1 + sa[:, None]))


def setup(selection, *, padding=0):
    model = Policy()
    rng = torch.Generator().manual_seed(19)
    video, audio = torch.randn(8, 4, generator=rng), torch.randn(8, 3, generator=rng)
    vm, am = None, None
    if padding:
        video = torch.cat((video, torch.full((8, padding), 1000.)), 1)
        audio = torch.cat((audio, torch.full((8, padding), -1000.)), 1)
        vm = torch.arange(video.shape[1]).expand(8, -1) < 4
        am = torch.arange(audio.shape[1]).expand(8, -1) < 3
    noise = AVState(video, audio, vm, am)
    sv = torch.tensor([1., .99, .98, .96, .8, 0.])
    sa = torch.tensor([1., .96, .91, .85, .6, 0.])
    union = tuple(sorted(set(selection['video']) | set(selection['audio'])))
    cache = collect_av_rollout(model, noise, sv, sa, selected=union,
        modality_selected=selection, microbatch=3)
    reference = ReferenceState(model)
    with reference.use_reference():
        predictions = prepare_reference_predictions(cache, model, 'cpu', 3)
    with torch.no_grad():
        model.video.add_(.3)
        model.audio.sub_(.4)
    return model, cache, reference, predictions


def objective(model, cache, predictions=None, beta=0., *, reward=None, **options):
    reward = torch.arange(8).float() if reward is None else reward
    loss = {**options, 'reference_mse_beta': beta}
    return prepare_art_objective(cache, model, reward, reward.square(),
        torch.zeros(8, dtype=torch.long), 'cpu', loss_options=loss,
        reference_predictions=predictions)


def expected_mse(model, cache, predictions):
    losses = []
    for name in ('video', 'audio'):
        values = []
        for step in getattr(cache, name + '_selected'):
            slot = cache.selected.index(step)
            state = cache.states[slot]
            predict = model(state, cache.video_sigmas[step].expand(len(state)),
                cache.audio_sigmas[step].expand(len(state)), torch.arange(len(state)))
            values.append(feature_mean((getattr(predict, name) - getattr(predictions[slot], name)).square(),
                getattr(state, name + '_mask')))
        losses.append(torch.stack(values).mean())
    return .5 * losses[0] + .5 * losses[1]


@pytest.mark.parametrize('selection', [
    {'video': (0, 1), 'audio': (0, 1)},
    {'video': (0, 3), 'audio': (0, 1)},
    {'video': (0, 3), 'audio': (1, 2)},
])
@pytest.mark.parametrize('padding', [0, 7])
def test_reference_loss_and_gradient_equal_separate_modal_means(selection, padding):
    model, cache, _, predictions = setup(selection, padding=padding)
    obj = objective(model, cache, predictions, 1e-3)
    policy = objective(model, cache)
    actual = obj(obj.items).mean() - policy(policy.items).mean()
    expected = 1e-3 * expected_mse(model, cache, predictions)
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=1e-7)
    actual_grad = torch.autograd.grad(actual, tuple(model.parameters()), retain_graph=True)
    expected_grad = torch.autograd.grad(expected, tuple(model.parameters()))
    for left, right in zip(actual_grad, expected_grad):
        torch.testing.assert_close(left, right, rtol=2e-4, atol=1e-7)
    assert obj.has_reference
    assert all(not p.video.requires_grad and not p.audio.requires_grad for p in predictions)


def test_reference_is_not_suppressed_by_zero_reward_or_corrective_mask():
    model, cache, _, predictions = setup({'video': (0, 3), 'audio': (1, 2)})
    obj = objective(model, cache, predictions, 1e-3, reward=torch.zeros(8),
        velocity_mask_metric='fixed_mse', movement_mask_threshold=0.)
    assert not obj.has_guidance and obj.has_reference
    torch.testing.assert_close(obj(obj.items).mean(), 1e-3 * expected_mse(model, cache, predictions))
    obj(obj.items).mean().backward()
    assert all(p.grad is not None and p.grad.abs() > 0 for p in model.parameters())


def test_reference_is_added_when_policy_terms_are_correctively_masked():
    model, cache, _, predictions = setup({'video': (0, 1), 'audio': (0, 1)})
    obj = objective(model, cache, predictions, 1e-3,
        velocity_mask_metric='fixed_mse', movement_mask_threshold=0.)
    # Force a desired displacement in the same direction as the current-old
    # movement, making the policy movement mask active for every row.
    obj.directions = AVState(-cache.states[0].video, cache.states[0].audio)
    obj.video_magnitudes.fill_(1000.)
    obj.audio_magnitudes.fill_(1000.)
    value = obj(obj.items).mean()
    assert obj.diagnostics['av/video_mask_fraction'] == 1.
    assert obj.diagnostics['av/audio_mask_fraction'] == 1.
    torch.testing.assert_close(value, 1e-3 * expected_mse(model, cache, predictions))


def test_reference_state_remains_initial_and_restores_across_current_checkpoint():
    model = Policy()
    initial = copy.deepcopy(model.state_dict())
    reference = ReferenceState(model)
    with torch.no_grad():
        model.video.add_(5.)
        model.audio.add_(6.)
    current = copy.deepcopy(model.state_dict())
    saved = copy.deepcopy(reference.state_dict())
    with reference.use_reference():
        assert not torch.is_grad_enabled()
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, initial[name], atol=0, rtol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, current[name], atol=0, rtol=0)
    resumed = Policy()
    restored_reference = ReferenceState(resumed)
    resumed.load_state_dict(current)
    restored_reference.load_state_dict(saved)
    with restored_reference.use_reference():
        for name, value in resumed.state_dict().items():
            torch.testing.assert_close(value, initial[name], atol=0, rtol=0)
    for name, value in resumed.state_dict().items():
        torch.testing.assert_close(value, current[name], atol=0, rtol=0)
    assert restored_reference.stateful() == {'h3_reference_policy': restored_reference}
    with pytest.raises(ValueError, match='schema/identity'):
        restored_reference.load_state_dict({**saved, 'kind': 'current_policy'})


def test_reference_inference_preserves_rows_sigmas_and_parameter_state_on_failure():
    model, cache, reference, _ = setup({'video': (0, 3), 'audio': (1, 2)})
    current = copy.deepcopy(model.state_dict())
    calls = []
    def predict(state, sv, sa, rows):
        calls.append((rows.tolist(), sv.tolist(), sa.tolist(), torch.is_grad_enabled()))
        if len(calls) == 4:
            raise RuntimeError('reference inference failed')
        return model(state, sv, sa, rows)
    with pytest.raises(RuntimeError, match='inference failed'):
        with reference.use_reference():
            prepare_reference_predictions(cache, predict, 'cpu', 3)
    assert calls[0] == ([0, 1, 2], [1.] * 3, [1.] * 3, False)
    assert calls[1][0] == [3, 4, 5]
    assert calls[2][0] == [6, 7]
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, current[name], atol=0, rtol=0)


def test_objective_rejects_missing_misaligned_or_attached_reference():
    model, cache, _, predictions = setup({'video': (0, 1), 'audio': (0, 1)})
    for beta in (True, -1., float('nan'), float('inf')):
        with pytest.raises(ValueError, match='finite and nonnegative'):
            objective(model, cache, predictions, beta)
    with pytest.raises(ValueError, match='requires cached'):
        objective(model, cache, beta=1e-3)
    with pytest.raises(ValueError, match='requires cached'):
        objective(model, cache, predictions)
    with pytest.raises(ValueError, match='selected cached positions'):
        objective(model, cache, predictions[:-1], 1e-3)
    attached = (replace(predictions[0], video=predictions[0].video.requires_grad_()), *predictions[1:])
    with pytest.raises(ValueError, match='detached'):
        objective(model, cache, attached, 1e-3)


def test_beta_zero_is_identical_to_legacy_objective():
    model, cache, _, _ = setup({'video': (0, 3), 'audio': (1, 2)})
    legacy = prepare_art_objective(cache, model, torch.arange(8).float(), torch.arange(8).float().square(),
        torch.zeros(8, dtype=torch.long), 'cpu')
    zero = objective(model, cache)
    assert not zero.has_reference
    torch.testing.assert_close(legacy(legacy.items), zero(zero.items), atol=0, rtol=0)


def test_h3_config_accepts_explicit_beta_and_rejects_invalid_reference_options(monkeypatch, tmp_path):
    for name in ('H3_MODEL', 'H3_REWARD_MANIFEST', 'H3_TRAIN_MANIFEST'):
        monkeypatch.setenv(name, str(tmp_path / name))
    monkeypatch.setenv('H3_TRAIN_SHA256', 'a' * 64)
    config = load_config(Path(__file__).parents[1] / 'configs/examples/minimax_h3_art_full.yaml')
    for beta in (0., 1e-3):
        options = copy.deepcopy(config.algorithm.options)
        options['loss']['reference_mse_beta'] = beta
        validate_config(replace(config, algorithm=replace(config.algorithm, options=options)))
    for beta in (True, -1., float('nan'), float('inf')):
        options = copy.deepcopy(config.algorithm.options)
        options['loss']['reference_mse_beta'] = beta
        with pytest.raises(ConfigError, match='reference_mse_beta'):
            validate_config(replace(config, algorithm=replace(config.algorithm, options=options)))
