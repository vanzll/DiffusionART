"""Reuse an identified pretrained eval0 inside the bounded stdlib I/O worker."""
import hashlib
import json
import math
import os
import time
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
        allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def protocol(config):
    sampling, evaluation = config['sampling'], config['evaluation']
    return dict(model=config['model'], reward=config['reward'],
        width=evaluation.get('width') or sampling['width'],
        height=evaluation.get('height') or sampling['height'],
        steps=evaluation.get('sampling_steps') or sampling['steps'],
        fps=sampling['fps'], guidance_scale=sampling['guidance_scale'],
        sampling_options=sampling['options'], seed=(config['seed'] if evaluation.get('seed') is None else evaluation['seed']),
        eval_manifest_sha256=evaluation['manifest_sha256'],
        full_duration_rewards=config['algorithm']['options'].get('post_training', {}).get('full_duration_rewards'))


def reuse_eval0(request, atomic_json):
    root, source = Path(request['root']).resolve(), Path(request['source']).resolve()
    config = request['config']
    if config['evaluation'].get('frames') != 124:
        raise ValueError('Pretrained eval0 reuse requires explicit evaluation.frames=124')
    original = json.loads((source / 'resolved-config.json').read_text())
    run = json.loads((source / 'run.json').read_text())
    if protocol(original) != protocol(config) or run['asset_manifest'] != request['expected_assets']:
        raise ValueError('Pretrained eval0 asset or inference/reward protocol differs')
    identity = dict(original)
    identity.pop('assets', None)
    identity['runtime'] = dict(identity['runtime'])
    identity['runtime'].pop('checkpoint_keep', None)
    identity['runtime'].pop('max_consecutive_no_signal_rollouts', None)
    if run['config_sha256'] != digest(identity):
        raise ValueError('Pretrained eval0 source run/config identity differs')
    dataset = Path(config['evaluation']['manifest']).read_bytes()
    if hashlib.sha256(dataset).hexdigest() != config['evaluation']['manifest_sha256']:
        raise ValueError('Pretrained eval0 prompt manifest bytes differ')
    prompts = [json.loads(line) for line in dataset.splitlines() if line.strip()]
    if len(prompts) != 32 or len({p['sample_id'] for p in prompts}) != 32:
        raise ValueError('Pretrained eval0 requires the fixed32 independent prompts')
    candidates = list((source / 'evaluation/step-000000000-frames-124').glob('*/manifest.json'))
    if len(candidates) != 1:
        raise ValueError('Pretrained eval0 source publication is ambiguous or missing')
    manifest_path = candidates[0]
    manifest = json.loads(manifest_path.read_text())
    payload = dict(manifest)
    saved_hash = payload.pop('manifest_sha256')
    if digest(payload) != saved_hash or manifest['optimizer_step'] != 0:
        raise ValueError('Pretrained eval0 canonical manifest identity differs')
    samples = {p['sample_id']: p for p in manifest['samples']}
    if set(samples) != {p['sample_id'] for p in prompts}:
        raise ValueError('Pretrained eval0 sample identities differ')
    for index, prompt in enumerate(prompts):
        sample = samples[prompt['sample_id']]
        if (sample['optimizer_step'] != 0 or sample['framecount'] != 124 or sample['fps'] != 24
                or sample['prompt'] != prompt['prompt'] or sample['seed'] != protocol(config)['seed'] + index
                or sample['scores']['weights'] != config['reward']['weights']
                or set(sample['scores']['raw']) != set(config['reward']['weights'])
                or not all(math.isfinite(v) for v in sample['scores']['raw'].values())):
            raise ValueError('Pretrained eval0 prompt/seed/media/reward identity differs')
    media = json.loads((manifest_path.parent / 'wandb-media.json').read_text())
    if len(media['samples']) != 32 or media['optimizer_step'] != 0:
        raise ValueError('Pretrained eval0 audio-video manifest differs')
    verified = []
    for row in media['samples']:
        sample = samples[row['sample_id']]
        if row['raw_rewards'] != sample['scores']['raw']:
            raise ValueError('Pretrained eval0 media raw scores differ')
        for name in ('video', 'audio_master'):
            asset = row[name]
            path = (source / asset['path']).resolve()
            if not path.is_relative_to(manifest_path.parent) or path.is_symlink():
                raise ValueError('Pretrained eval0 media escaped its source')
            if path.stat().st_size != asset['bytes']:
                raise ValueError('Pretrained eval0 media size differs')
            with path.open('rb') as stream:
                checksum = hashlib.sha256()
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    checksum.update(chunk)
                actual = checksum.hexdigest()
            if actual != asset['sha256']:
                raise ValueError('Pretrained eval0 media bytes differ')
            verified.append(asset)
    destination = root / manifest_path.parent.relative_to(source)
    if destination.exists():
        raise ValueError('Pretrained eval0 target already exists')
    for path in manifest_path.parent.rglob('*'):
        if path.is_symlink():
            raise ValueError('Pretrained eval0 source contains a symlink')
        target = destination / path.relative_to(manifest_path.parent)
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix in ('.mp4', '.wav'):
                os.link(path, target)
            else:
                target.write_bytes(path.read_bytes())
    summary = json.loads((destination / 'summary.json').read_text())
    summary.update(manifest=str(destination / 'manifest.json'), **{'eval/reused_pretrained_baseline': 1})
    atomic_json(destination / 'summary.json', summary)
    atomic_json(root / 'evaluation/latest.json', dict(optimizer_step=0,
        manifest=str((destination / 'manifest.json').relative_to(root)),
        summary=str((destination / 'summary.json').relative_to(root))))
    provenance = dict(status='verified_pretrained_eval0_reused', source_output=str(source),
        source_manifest=str(manifest_path), manifest_sha256=saved_hash,
        source_run_config_sha256=run['config_sha256'], protocol_sha256=digest(protocol(config)),
        asset_identity_sha256=digest(request['expected_assets']), optimizer_step=0,
        frames=124, samples=32, media_assets_verified=verified, unix=time.time())
    atomic_json(root / 'eval0-reuse.json', provenance)
    return dict(samples=32, frames=124, source=str(source), manifest=str(destination / 'manifest.json'))
