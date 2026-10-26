import json
import os
from pathlib import Path
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from visual_rl.backends.minimax_h3_checkpoint_admission import (
    _try_lock, capacity_bytes, checkpoint_admission,
)
from visual_rl.training.distributed import DistributedSession, RankInfo


def options(root, **changes):
    value = dict(schema_version=1, lock_path=str(root / 'writer.lock'),
                 allowed_output_roots=[str(root)], quota_paths=[],
                 checkpoint_bytes=1, headroom_bytes=1,
                 timeout_seconds=.3, poll_seconds=.01)
    value.update(changes)
    path = root / 'admission.json'
    path.write_text(json.dumps(value))
    return path


def test_mutual_exclusion_uses_same_inode_and_releases(tmp_path):
    first = _try_lock(tmp_path / 'writer.lock')
    assert first is not None
    try:
        assert _try_lock(tmp_path / 'writer.lock') is None
    finally:
        os.close(first)
    second = _try_lock(tmp_path / 'writer.lock')
    assert second is not None
    os.close(second)


def test_capacity_includes_parent_quota_even_if_statvfs_is_higher(tmp_path, monkeypatch):
    def get(path, key):
        return b'100' if key == 'ceph.quota.max_bytes' else b'93'
    monkeypatch.setattr(os, 'getxattr', get)
    assert capacity_bytes(tmp_path, [str(tmp_path)]) == 7


def test_failed_writer_marker_survives_and_next_writer_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv('H3_CHECKPOINT_ADMISSION_CONFIG', str(options(tmp_path)))
    session = DistributedSession(RankInfo(), torch.device('cpu'), False)
    with pytest.raises(ValueError, match='native write failed'):
        with checkpoint_admission(session, tmp_path, 2):
            raise ValueError('native write failed')
    marker = json.loads((tmp_path / 'writer.lock.state.json').read_text())
    assert marker['status'] == 'failed_drain_unproven'
    with pytest.raises(RuntimeError, match='no committed/drained'):
        with checkpoint_admission(session, tmp_path, 2):
            pytest.fail('must not write over a failed writer')


def test_insufficient_space_times_out_before_creating_writer_marker(tmp_path, monkeypatch):
    monkeypatch.setenv('H3_CHECKPOINT_ADMISSION_CONFIG', str(options(tmp_path, checkpoint_bytes=10**20)))
    session = DistributedSession(RankInfo(), torch.device('cpu'), False)
    with pytest.raises(TimeoutError):
        with checkpoint_admission(session, tmp_path, 2):
            pytest.fail('must not enter body without space')
    assert not (tmp_path / 'writer.lock.state.json').exists()


def _worker(rank, directory, mode):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from visual_rl.training.fsdp_checkpoint import save_fsdp_checkpoint, load_fsdp_checkpoint
    from visual_rl.training.state import Progress
    root = Path(directory)
    torch.set_num_threads(1)
    torch.manual_seed(19)
    os.environ['H3_CHECKPOINT_ADMISSION_CONFIG'] = str(root / 'admission.json')
    if mode == 'env_mismatch' and rank:
        os.environ.pop('H3_CHECKPOINT_ADMISSION_CONFIG')
    dist.init_process_group('gloo', init_method=f'file://{root}/rendezvous', rank=rank,
                            world_size=2, timeout=timedelta(seconds=10))
    try:
        session = DistributedSession(RankInfo(rank, 2, rank, 2), torch.device('cpu'), True)
        if mode == 'env_mismatch':
            with pytest.raises(Exception, match='identity differs'):
                with checkpoint_admission(session, root, 2):
                    raise AssertionError('mismatched configuration entered write')
            return
        if mode == 'quota_failure':
            import visual_rl.backends.minimax_h3_checkpoint_admission as admission
            def probe(*args):
                if rank:raise OSError('rank-local quota failure')
                return 100
            admission.capacity_bytes = probe
            with pytest.raises(RuntimeError, match='fresh quota failed'):
                with checkpoint_admission(session, root, 2):
                    raise AssertionError('quota failure entered write')
            return
        model = FSDP(torch.nn.Linear(4, 4), device_id=torch.device('cpu'))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        model(torch.ones(2, 4)).square().mean().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        before = next(model.parameters()).detach().clone()
        generator = torch.Generator().manual_seed(42 + rank)
        sampling_before = generator.get_state().clone()
        rng_before = torch.get_rng_state().clone()
        saved = dict(model=model, optimizer=optimizer, config_sha256='a' * 64,
                     generators={'sampling': generator}, stateful={})
        with checkpoint_admission(session, root, 2):
            receipt = save_fsdp_checkpoint(root / 'checkpoints', progress=Progress(
                optimizer_steps=2, completed_rollouts=1, attempts=1), keep_last=2, **saved)
        assert torch.equal(torch.get_rng_state(), rng_before)
        assert torch.equal(generator.get_state(), sampling_before)
        with torch.no_grad():
            next(model.parameters()).add_(10)
        progress = load_fsdp_checkpoint(receipt, **saved)
        assert progress.optimizer_steps == 2 and progress.completed_rollouts == 1
        assert torch.equal(next(model.parameters()), before)
        assert json.loads((root / 'writer.lock.state.json').read_text())['status'] == 'committed'
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('mode', ['fsdp_roundtrip', 'env_mismatch', 'quota_failure'])
def test_real_two_rank_collective_order_and_error_propagation(tmp_path, mode):
    options(tmp_path)
    mp.spawn(_worker, args=(str(tmp_path), mode), nprocs=2, join=True)
