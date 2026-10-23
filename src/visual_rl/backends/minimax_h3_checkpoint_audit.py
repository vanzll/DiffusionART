"""Optional H3-only exact readback of a restored local FSDP checkpoint."""

import hashlib
import json
import os
import re
import time
from pathlib import Path

import torch

from ..training.checkpoint import _atomic_json, restore_rng, rng_state
from ..training.fsdp_checkpoint import _local_state, _pack_model
from .minimax_h3_observability import event


def fingerprint(value):
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().to('cpu').contiguous()
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                raise ValueError('Nonfinite tensor in H3 restored checkpoint audit')
            digest.update(json.dumps(['tensor', str(tensor.dtype), list(tensor.shape)]).encode())
            data = memoryview(tensor.reshape(-1).view(torch.uint8).numpy()).cast('B')
            for offset in range(0, len(data), 8 * 1024 * 1024):
                digest.update(data[offset:offset + 8 * 1024 * 1024])
        elif isinstance(item, dict):
            digest.update(b'dict')
            for key in sorted(item, key=lambda entry: (type(entry).__name__, str(entry))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            digest.update(str(len(item)).encode())
            for child in item:
                visit(child)
        else:
            digest.update(json.dumps([type(item).__name__, item], allow_nan=False).encode())
    visit(value)
    return digest.hexdigest()


def audit_restored_checkpoint(session, receipt, model, optimizer, stateful, generators, progress, root):
    nonce = os.environ.get('H3_ATTEMPT_NONCE', '')
    if not re.fullmatch(r'[A-Za-z0-9_.-]{1,128}', nonce):
        raise ValueError('H3 restore audit requires a safe unique attempt nonce')
    rank = session.identity.rank
    receipt = Path(receipt).absolute()
    begin = time.monotonic()
    before_rng = rng_state(cuda_device=session.device)
    event('restore_readback_audit_begin', optimizer_steps=progress.optimizer_steps)
    try:
        live = _local_state(model)
        def compare():
            manifest = json.loads(receipt.read_text())
            record = manifest['ranks'][rank]
            if record['file'] != f'rank-{rank:05d}.pt':
                raise ValueError('H3 audit checkpoint rank identity differs')
            payload = torch.load(receipt.parent / record['file'], map_location='cpu',
                                 weights_only=True, mmap=True)
            actual = dict(model=_pack_model(live), optimizer=optimizer.state_dict(),
                          buffers=dict(model.named_buffers()),
                          stateful={name: component.state_dict() for name, component in stateful.items()},
                          generators={name: generator.get_state() for name, generator in generators.items()},
                          rng=before_rng, progress=progress.state_dict())
            hashes = {}
            for name, value in actual.items():
                expected_hash, actual_hash = fingerprint(payload[name]), fingerprint(value)
                if expected_hash != actual_hash:
                    raise ValueError(f'H3 restored {name} bytes differ from the saved local shard')
                hashes[name] = actual_hash
            return hashes
        hashes = session.local('H3 exact checkpoint readback audit', compare)
    finally:
        restore_rng(before_rng, cuda_device=session.device)
    def publish():
        if fingerprint(rng_state(cuda_device=session.device)) != fingerprint(before_rng):
            raise ValueError('H3 checkpoint audit changed the visible RNG state')
        destination = Path(root) / 'restore-audits' / nonce / f'rank-{rank:05d}.json'
        destination.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(destination, dict(status='exact_in_memory_restore_verified', rank=rank,
                     world_size=session.identity.world_size, checkpoint=str(receipt),
                     optimizer_steps=progress.optimizer_steps, hashes=hashes,
                     visible_rng_preserved=True, seconds=time.monotonic() - begin))
    session.local('H3 checkpoint readback receipt', publish)
    event('restore_readback_audit_complete', optimizer_steps=progress.optimizer_steps,
          seconds=time.monotonic() - begin, visible_rng_preserved=True)
