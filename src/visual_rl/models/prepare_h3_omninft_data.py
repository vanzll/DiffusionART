"""Prepare the exact verl-omni OmniNFT VGGSound train/test prompt splits.

Only the storage schema changes: prompt_av -> prompt, prompt_v/prompt_a ->
reward_text.video/audio. No truncation, filtering, deduplication or reshuffling.
This coordinator CLI uses only the standard library and never starts a model.
"""

import argparse
import hashlib
import json
import os
import shlex
import tempfile
import urllib.request
from pathlib import Path

REVISION = 'fb9237f6e74edf0d0f2a683f4d975b79fde588fe'
BASE_URL = f'https://raw.githubusercontent.com/zghhui/OmniNFT/{REVISION}/dataset/vggsound'
SPLITS = {
    'train': ('train_metadata_20k.jsonl', 'e2b3629e4aa729bb91b780c5933f49e2044e603f38ea3c730e5ee99313d3db66', 19487),
    'test': ('test_metadata.jsonl', '4c5c47856856e8896c90ea0edb3656cee051f9e93f0f78867350a2bba1d63fb8', 16),
}


def convert_records(raw, split):
    if split not in SPLITS:
        raise ValueError('unknown OmniNFT split')
    rows, seen = [], set()
    for line in raw.decode('utf-8').splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if (not isinstance(item, dict) or any(not isinstance(item.get(k), str) or not item[k].strip()
                                             for k in ('prompt_av', 'prompt_v', 'prompt_a'))):
            raise ValueError('OmniNFT rows require all three nonempty prompt strings')
        if ('idx' in item) != ('category' in item):
            raise ValueError('OmniNFT idx/category must appear together')
        idx = item.get('idx', len(rows))
        category = item.get('category', 'validation' if split == 'test' else split)
        if type(idx) is not int or not isinstance(category, str) or not category.strip() or idx in seen:
            raise ValueError('OmniNFT idx/category must be valid with unique prompt identities')
        seen.add(idx)
        rows.append({'sample_id': f'omninft-{split}-{idx}', 'prompt': item['prompt_av'],
                     'metadata': {'index': idx, 'category': category,
                                  'reward_text': {'video': item['prompt_v'], 'audio': item['prompt_a']}}})
    if not rows:
        raise ValueError('empty OmniNFT split')
    return rows


def _publish(path, data):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f'refusing to replace a different existing dataset artifact: {path}')
        return
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def prepare(output_dir, *, source_dir=None):
    root = Path(output_dir).absolute()
    root.mkdir(parents=True, exist_ok=True)
    payloads, rows, report = {}, {}, {'source_revision': REVISION, 'splits': {}}
    for split, (filename, expected, count) in SPLITS.items():
        local = Path(source_dir) / filename if source_dir else root / filename
        if local.exists():
            if local.stat().st_size > 64 * 1024 * 1024:
                raise ValueError('OmniNFT source exceeds size limit')
            raw = local.read_bytes()
        elif source_dir:
            raise FileNotFoundError(local)
        else:
            with urllib.request.urlopen(f'{BASE_URL}/{filename}', timeout=30) as response:
                raw = response.read(64 * 1024 * 1024 + 1)
        if len(raw) > 64 * 1024 * 1024 or hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f'OmniNFT source differs from pinned upstream bytes: {filename}')
        rows[split] = convert_records(raw, split)
        if len(rows[split]) != count:
            raise ValueError(f'OmniNFT split count differs: {split}')
        converted = ''.join(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n'
                            for row in rows[split]).encode('utf-8')
        payloads[filename], payloads[f'{split}.jsonl'] = raw, converted
        report['splits'][split] = {'count': count, 'source_sha256': expected,
                                   'manifest': str(root / f'{split}.jsonl'),
                                   'sha256': hashlib.sha256(converted).hexdigest()}
    train = {' '.join(row['prompt'].lower().split()) for row in rows['train']}
    test = {' '.join(row['prompt'].lower().split()) for row in rows['test']}
    if train & test:
        raise ValueError('OmniNFT train/test prompt overlap')
    report['normalized_prompt_overlap'] = 0
    report['train_duplicate_prompt_rows'] = len(rows['train']) - len({r['prompt'] for r in rows['train']})
    env = {'H3_OMNINFT_DATA_ROOT': str(root),
           'H3_TRAIN_MANIFEST': report['splits']['train']['manifest'],
           'H3_TRAIN_SHA256': report['splits']['train']['sha256'],
           'H3_EVAL_MANIFEST': report['splits']['test']['manifest'],
           'H3_EVAL_SHA256': report['splits']['test']['sha256']}
    payloads['data.env'] = ''.join(f'export {k}={shlex.quote(v)}\n' for k, v in env.items()).encode()
    payloads['dataset-info.json'] = (json.dumps(report, indent=2) + '\n').encode()
    # Validate both sources completely before publishing a ready environment.
    for name, data in payloads.items():
        _publish(root / name, data)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--source-dir', help='optional directory containing the two pinned original JSONL files')
    args = parser.parse_args(argv)
    report = prepare(args.output_dir, source_dir=args.source_dir)
    print(json.dumps({'root': str(Path(args.output_dir).absolute()),
                      'train_rows': report['splits']['train']['count'],
                      'test_rows': report['splits']['test']['count'],
                      'overlap': report['normalized_prompt_overlap']}))


if __name__ == '__main__':
    main()
