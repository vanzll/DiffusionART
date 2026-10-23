import copy
import json
import os
import time
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from visual_rl.backends.minimax_h3_checkpoint_audit import fingerprint


def test_restore_fingerprint_preserves_values_rng_and_dictionary_order():
    tensor = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    value = {'model': tensor, 'optimizer': {0: {'step': torch.tensor(4.)}}, 'metadata': ('fixed', 8)}
    expected = copy.deepcopy(value)
    before = torch.get_rng_state().clone()
    actual = fingerprint(value)
    assert actual == fingerprint(dict(reversed(list(value.items()))))
    torch.testing.assert_close(tensor, expected['model'], rtol=0, atol=0)
    assert torch.equal(before, torch.get_rng_state())
    value['model'][0, 0] = 1
    assert fingerprint(value) != actual


def test_restore_fingerprint_covers_dtype_shape_scalar_and_generator_state():
    tensor = torch.tensor([1., 2.])
    assert fingerprint(tensor) != fingerprint(tensor.double())
    assert fingerprint(tensor) != fingerprint(tensor.reshape(1, 2))
    assert fingerprint(torch.tensor(4.)) != fingerprint(torch.tensor(5.))
    generator = torch.Generator().manual_seed(42)
    before = generator.get_state().clone()
    fingerprint(before)
    assert torch.equal(before, generator.get_state())


def test_restore_fingerprint_rejects_nonfinite_tensor():
    with pytest.raises(ValueError, match='Nonfinite'):
        fingerprint({'exp_avg': torch.tensor([float('nan')])})


def _readback_worker(rank, directory, mode):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    from visual_rl.backends.minimax_h3_checkpoint_audit import audit_restored_checkpoint
    from visual_rl.training.checkpoint import rng_state
    from visual_rl.training.distributed import DistributedSession, RankInfo
    from visual_rl.training.fsdp_checkpoint import load_fsdp_checkpoint, save_fsdp_checkpoint
    from visual_rl.training.state import Progress

    torch.set_num_threads(1)
    os.environ['H3_ATTEMPT_NONCE'] = 'cpu-readback-' + mode
    os.environ['RANK'] = str(rank)
    root = Path(directory)
    device = torch.device('cpu')
    dist.init_process_group('gloo', init_method=f'file://{root}/rendezvous', rank=rank,
                            world_size=2, timeout=timedelta(seconds=40))
    try:
        torch.manual_seed(23)
        model = FSDP(torch.nn.Linear(4, 4), device_id=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model(torch.ones(2, 4)).square().mean().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        generator = torch.Generator().manual_seed(42 + rank)
        progress = Progress(optimizer_steps=1, completed_rollouts=1, attempts=1)
        options = dict(model=model, optimizer=optimizer, config_sha256='a' * 64,
                       stateful={}, generators={'av_sampling': generator})
        receipt = save_fsdp_checkpoint(root / 'checkpoints', progress=progress, **options)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(1)
        progress = load_fsdp_checkpoint(receipt, **options)
        before = fingerprint(rng_state(cuda_device=device))
        if mode == 'mismatch' and rank == 0:
            next(iter(optimizer.state.values()))['exp_avg'].add_(1)
        session = DistributedSession(RankInfo(rank, 2, rank, 2), device, True)
        if mode == 'mismatch':
            with pytest.raises(RuntimeError, match='H3 exact checkpoint readback audit'):
                audit_restored_checkpoint(session, receipt, model, optimizer, {},
                                          {'av_sampling': generator}, progress, root)
        else:
            audit_restored_checkpoint(session, receipt, model, optimizer, {},
                                      {'av_sampling': generator}, progress, root)
            saved = json.loads((root / 'restore-audits' / ('cpu-readback-' + mode) /
                                f'rank-{rank:05d}.json').read_text())
            assert saved['visible_rng_preserved'] and saved['optimizer_steps'] == 1
            assert set(saved['hashes']) == {'model', 'optimizer', 'buffers', 'stateful',
                                           'generators', 'rng', 'progress'}
        assert before == fingerprint(rng_state(cuda_device=device))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('mode', ['exact', 'mismatch'])
def test_actual_fsdp_restore_audit_and_failure_preserve_rng(tmp_path, mode):
    context = mp.spawn(_readback_worker, args=(str(tmp_path), mode), nprocs=2, join=False)
    deadline = time.monotonic() + 80
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail('H3 checkpoint audit regression exceeded80s')
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
