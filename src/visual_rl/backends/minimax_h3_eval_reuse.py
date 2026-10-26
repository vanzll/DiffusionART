"""Explicit import of an audited original pretrained eval0, never policy weights."""
import hashlib
import json
import os
from pathlib import Path
import shutil

from ..config import ConfigError
from ..evaluation import EvaluationManifest
from ..evaluation.audio_video_gallery import publish_av_gallery
from ..training.checkpoint import _atomic_json
from .minimax_h3_posttraining import duration_config, evaluation_frames, evaluation_step_directory


def evaluation_protocol(document):
    """Match effective generation/scoring inputs, allowing operational run edits."""
    sampling = dict(document['sampling'])
    sampling.pop('microbatch', None)
    evaluation = dict(document['evaluation'])
    evaluation.pop('manifest', None)  # Dataset bytes/ordered rows checked independently.
    for key in ('every_updates', 'every_rollouts', 'at_updates', 'retain_all'):
        evaluation.pop(key, None)
    post = document['algorithm']['options'].get('post_training') or {}
    return dict(model=document['model'], reward=document['reward'], sampling=sampling,
                evaluation=evaluation, seed=document['seed'],
                full_duration_rewards=post.get('full_duration_rewards', False))


def reuse_pretrained_eval0(c, dataset, root, source):
    """Called under the fresh run owner lock, after original H3 initialization.

    Source is an explicit operator-selected original output. Its policy payload
    is never opened. Canonical media hashes, every prompt/seed and model/scorer
    identity must agree. New metadata records the imported origin explicitly.
    """
    from .minimax_h3_runtime import _completed_duration_evaluation
    from .sd3 import _publish_evaluation
    if dataset is None or evaluation_frames(c) != (124,):
        raise ConfigError('Pretrained eval0 reuse requires the explicit124-only validation recipe')
    source, root = Path(source).resolve(strict=True), Path(root).resolve(strict=True)
    if source == root:
        raise ConfigError('Pretrained eval0 source must be a different original output')
    old_config = json.loads((source / 'resolved-config.json').read_text())
    if evaluation_protocol(old_config) != evaluation_protocol(c.to_dict()):
        raise ConfigError('Pretrained eval0 generation/scoring protocol differs')
    original = json.loads((source / 'run.json').read_text())
    current = json.loads((root / 'run.json').read_text())
    for name in ('asset_manifest', 'parameters', 'dp_world_size', 'pretrained_reference'):
        if original[name] != current[name]:
            raise ConfigError(f'Pretrained eval0 {name} differs')
    if old_config['model']['options']['finetune']['mode'] != 'full':
        raise ConfigError('Pretrained eval0 reuse is restricted to original full-policy initialization')
    view = duration_config(c, 124)
    if not _completed_duration_evaluation(view, dataset, 0, source):
        raise ConfigError('Original pretrained eval0 media/prompt/seed evidence is incomplete')
    candidates = list((source / 'evaluation' / evaluation_step_directory(view, 0)).glob('*/manifest.json'))
    if len(candidates) != 1:
        raise ConfigError('Pretrained eval0 source publication is ambiguous')
    manifest_path = candidates[0]
    manifest = EvaluationManifest.from_json(manifest_path.read_text())
    destination = root / manifest_path.parent.relative_to(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    def copy_asset(src, dst):
        if Path(src).is_symlink():
            raise ConfigError('Pretrained eval0 import rejects symlinks')
        if Path(src).suffix in ('.mp4', '.wav'):
            try:
                os.link(src, dst)
                return dst
            except OSError:
                pass
        return shutil.copyfile(src, dst)
    shutil.copytree(manifest_path.parent, destination, copy_function=copy_asset)
    # Metadata is copied, never hardlinked: republishing cannot alter source.
    shutil.copyfile(manifest_path, destination / 'source-manifest.json')
    _publish_evaluation(view, dataset, 0, root, destination, list(manifest.samples),
                        metric_prefix='eval/frames124')
    publish_av_gallery(root, destination, list(manifest.samples), 0, metric_namespace='eval/frames124')
    receipt = dict(kind='reused_original_pretrained_eval0_media_only', optimizer_step=0,
        source_output=str(source), source_manifest=str(manifest_path),
        source_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        source_run_sha256=hashlib.sha256((source / 'run.json').read_bytes()).hexdigest(),
        source_config_sha256=original['config_sha256'], target_config_sha256=c.fingerprint(),
        target_manifest=str(destination / 'manifest.json'), samples=len(manifest.samples),
        policy_weights_loaded_from_source=False, protocol=evaluation_protocol(c.to_dict()))
    _atomic_json(root / 'pretrained-eval0-reuse.json', receipt)
    if not _completed_duration_evaluation(view, dataset, 0, root):
        raise ConfigError('Imported pretrained eval0 failed destination integrity audit')
    return receipt
