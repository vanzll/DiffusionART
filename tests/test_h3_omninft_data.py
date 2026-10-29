"""Lossless OmniNFT prompt/split mapping and opt-in 16-prompt test protocol."""

import hashlib
import json
from dataclasses import replace

import pytest
from test_h3_mixed_reference_runtime import mixed_config
from test_minimax_h3_runtime import config_factory as make_config_factory

from visual_rl.backends import minimax_h3 as backend
from visual_rl.backends.minimax_h3 import load_datasets
from visual_rl.config import ConfigError, Evaluation
from visual_rl.data import PromptDataset
from visual_rl.models import prepare_h3_omninft_data as preparation
from visual_rl.rewards.av_profiles import THREE_REWARD_PROFILE, THREE_REWARD_WEIGHTS


def raw(rows):
    return ''.join(json.dumps(row) + '\n' for row in rows).encode()


def item(text='full AV prompt'):
    return {'prompt_av': text, 'prompt_v': ' visual text with spaces ', 'prompt_a': 'audio text'}


def test_prompt_content_order_duplicates_and_native_ids_are_preserved():
    records = [{**item(), 'idx': 19, 'category': 'human_speech'},
               {**item(), 'idx': 2, 'category': 'human_speech'}]
    rows = preparation.convert_records(raw(records), 'train')
    assert [r['sample_id'] for r in rows] == ['omninft-train-19', 'omninft-train-2']
    for output, source in zip(rows, records):
        assert output['prompt'] == source['prompt_av']
        assert output['metadata']['reward_text'] == {'video': source['prompt_v'], 'audio': source['prompt_a']}
        assert output['metadata']['index'] == source['idx']
        assert output['metadata']['category'] == source['category']
    test = preparation.convert_records(raw([item('heldout'), item('heldout 2')]), 'test')
    assert [r['sample_id'] for r in test] == ['omninft-test-0', 'omninft-test-1']
    assert all(r['metadata']['category'] == 'validation' for r in test)


@pytest.mark.parametrize('records', [
    [{**item(), 'idx': 2}], [{**item(), 'idx': True, 'category': 'bad'}],
    [{**item(), 'prompt_a': ''}],
    [{**item(), 'idx': 1, 'category': 'x'}, {**item(), 'idx': 1, 'category': 'x'}],
])
def test_bad_source_rows_fail_instead_of_silently_filtering(records):
    with pytest.raises(ValueError):
        preparation.convert_records(raw(records), 'train')


def test_prepare_pins_bytes_is_idempotent_and_never_overwrites_other_data(tmp_path, monkeypatch):
    source, output = tmp_path / 'source', tmp_path / 'output'
    source.mkdir()
    splits = {}
    for split, records in [('train', [item('train1'), item('train2')]), ('test', [item('heldout')])]:
        data = raw(records)
        name = f'{split}-source.jsonl'
        (source / name).write_bytes(data)
        splits[split] = (name, hashlib.sha256(data).hexdigest(), len(records))
    monkeypatch.setattr(preparation, 'SPLITS', splits)
    report = preparation.prepare(output, source_dir=source)
    train = PromptDataset.load(output / 'train.jsonl', expected_sha256=report['splits']['train']['sha256'], task='t2av')
    assert [r.text for r in train.rows] == ['train1', 'train2']
    assert report['splits']['test']['count'] == 1 and report['normalized_prompt_overlap'] == 0
    assert preparation.prepare(output, source_dir=source) == report
    env = (output / 'data.env').read_text()
    assert 'H3_EVAL_MANIFEST=' in env and str(output / 'test.jsonl') in env
    (output / 'train.jsonl').write_text('unrelated existing artifact')
    with pytest.raises(ValueError, match='refusing to replace'):
        preparation.prepare(output, source_dir=source)
    (source / 'train-source.jsonl').write_text('modified')
    with pytest.raises(ValueError, match='pinned upstream bytes'):
        preparation.prepare(tmp_path / 'new-output', source_dir=source)


def test_omninft_split_binds_sixteen_tests_independently_of_reward_profile(tmp_path, monkeypatch):
    factory = make_config_factory.__wrapped__(tmp_path, monkeypatch)
    config = mixed_config(factory, 'omninft-test-protocol')
    heldout = tmp_path / 'heldout.jsonl'
    heldout.write_text(''.join(json.dumps({'sample_id': f'test-{i}', 'prompt': f'heldout prompt {i}',
        'metadata': {'reward_text': {'video': f'visual {i}', 'audio': f'sound {i}'}}}) + '\n' for i in range(16)))
    config = replace(config, evaluation=Evaluation(every_updates=2, manifest=str(heldout),
        manifest_sha256=hashlib.sha256(heldout.read_bytes()).hexdigest()))
    with pytest.raises(ConfigError, match='32 fixed prompts'):
        load_datasets(config)
    preference = replace(config, reward=replace(config.reward, weights=THREE_REWARD_WEIGHTS,
        options={**config.reward.options, 'profile': THREE_REWARD_PROFILE}))
    assert len(load_datasets(preference)[1].rows) == 16
    # A verified OmniNFT binding selects the held-out split for six rewards too.
    monkeypatch.setattr(backend, 'OMNINFT_TEST_SHA256', config.evaluation.manifest_sha256)
    train, test = load_datasets(config)
    assert len(test.rows) == 16 and len(train.rows) == 3


@pytest.mark.parametrize('count,overlap,message', [(15, False, '16 fixed prompts'),
                                                (16, True, 'held out')])
def test_omninft_protocol_preserves_count_and_heldout_guards(tmp_path, monkeypatch, count, overlap, message):
    factory = make_config_factory.__wrapped__(tmp_path, monkeypatch)
    config = mixed_config(factory, 'omninft-invalid-test')
    records = [{'sample_id': f'test-{i}', 'prompt': f'heldout prompt {i}',
        'metadata': {'reward_text': {'video': f'visual {i}', 'audio': f'sound {i}'}}} for i in range(count)]
    if overlap:
        records[0]['prompt'] = json.loads(open(config.data.manifest).readline())['prompt']
    path = tmp_path / 'heldout.jsonl'
    path.write_bytes(raw(records))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(backend, 'OMNINFT_TEST_SHA256', digest)
    config = replace(config, evaluation=Evaluation(every_updates=2, manifest=str(path), manifest_sha256=digest))
    with pytest.raises(ConfigError, match=message):
        load_datasets(config)
