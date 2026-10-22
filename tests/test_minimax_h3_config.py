"""Public recipe validation is offline and rejects hidden scientific changes."""
import hashlib
import json
from pathlib import Path

import pytest

from visual_rl.backends.minimax_h3 import load_datasets, reward_text, updates_per_rollout
from visual_rl.config import ConfigError, load_config
from visual_rl.registry import get_backend


@pytest.fixture
def bindings(tmp_path, monkeypatch):
    path = tmp_path / 'train.jsonl'
    row = {'sample_id':'a', 'prompt':'generation',
           'metadata':{'reward_text':{'video':'visual event', 'audio':'sound event'}}}
    path.write_text(json.dumps(row) + '\n')
    for name, value in {'H3_MODEL':str(tmp_path), 'H3_TRAIN_MANIFEST':str(path),
                        'H3_TRAIN_SHA256':hashlib.sha256(path.read_bytes()).hexdigest(),
                        'H3_REWARD_MANIFEST':str(tmp_path / 'rewards.json')}.items():
        monkeypatch.setenv(name, value)


def recipe(algo='nft', mode='lora', overrides=None):
    root = Path(__file__).resolve().parents[1]
    return load_config(root / f'configs/examples/minimax_h3_{algo}_{mode}.yaml', overrides)


@pytest.mark.parametrize('algo', ['art','nft','fast'])
@pytest.mark.parametrize('mode', ['lora','full'])
def test_six_public_recipes_are_offline_valid(bindings, algo, mode):
    c = recipe(algo, mode)
    assert get_backend(c).name == 'minimax_h3'
    assert updates_per_rollout(c) == (1 if algo == 'nft' else 2)
    assert c.sampling.steps - 1 == 9
    assert c.model.options['finetune']['mode'] == mode
    data, evaluation = load_datasets(c)
    assert evaluation is None
    assert reward_text(data.rows, 2) == (['visual event']*2, ['sound event']*2)


@pytest.mark.parametrize('overrides', [
    {'sampling':{'guidance_scale':6.}},
    {'model':{'options':{'local_files_only':False}}},
    {'algorithm':{'options':{'epoch':2}}},
    {'algorithm':{'options':{'old_policy':{'rate':.1}}}},
    {'reward':{'weights':{'desync':2.}}},
])
def test_reject_silent_semantics_changes(bindings, overrides):
    with pytest.raises(ConfigError):
        get_backend(recipe(overrides=overrides))


def test_art_no_reference_or_forward_override(bindings):
    for option in ['reference_mse_beta','reference_prediction','state','restore_anchor_mask']:
        with pytest.raises(ConfigError):
            get_backend(recipe('art', overrides={'algorithm':{'options':{'loss':{option:1.}}}}))


def test_reward_text_is_required_not_implicitly_shared(bindings, monkeypatch, tmp_path):
    path = tmp_path / 'bare.jsonl'
    path.write_text(json.dumps({'sample_id':'a', 'prompt':'no sound text'})+'\n')
    monkeypatch.setenv('H3_TRAIN_MANIFEST', str(path))
    monkeypatch.setenv('H3_TRAIN_SHA256', hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ConfigError, match='reward_text'):
        load_datasets(recipe())


def test_geometry_uses_pinned_h3_vae_not_wan_rule(bindings):
    assert recipe().sampling.frames == 124
    with pytest.raises(ConfigError, match='17n'):
        get_backend(recipe(overrides={'sampling':{'frames':121}}))
