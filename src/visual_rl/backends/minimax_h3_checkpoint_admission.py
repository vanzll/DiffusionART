"""Opt-in shared checkpoint writer exclusion and fresh capacity admission.

All ranks call at the same rollout boundary. The body contains native FSDP
collectives: never put this context or its body inside session.local(). A dead
writer leaves a durable uncommitted marker and requires verified drain before
another writer can proceed. This is not a filesystem or allocation supervisor.
"""

from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import socket
import time

import torch
import torch.distributed as dist

from ..training.checkpoint import _atomic_json


def capacity_bytes(root, quota_paths):
    """Read quota counters before statvfs; include every configured ancestor."""
    available = []
    for path in quota_paths:
        limit = int(os.getxattr(path, 'ceph.quota.max_bytes'))
        used = int(os.getxattr(path, 'ceph.dir.rbytes'))
        if limit <= 0 or used < 0:
            raise ValueError('Invalid shared checkpoint quota counters')
        available.append(max(0, limit - used))
    status = os.statvfs(root)
    available.append(status.f_bavail * status.f_frsize)
    return min(available)


def _options(path, root):
    if Path(path).stat().st_size > 65536:
        raise ValueError('Checkpoint admission config is too large')
    value = json.loads(Path(path).read_text())
    if value.get('schema_version') != 1:
        raise ValueError('Unknown checkpoint admission schema')
    for key in ('checkpoint_bytes', 'headroom_bytes', 'timeout_seconds', 'poll_seconds'):
        number = value[key]
        if (isinstance(number, bool) or not isinstance(number, (int, float))
                or not math.isfinite(number) or number <= 0):
            raise ValueError('Checkpoint admission budgets must be positive')
    if not (value['poll_seconds'] <= value['timeout_seconds'] <= 900):
        raise ValueError('Checkpoint admission deadline must be bounded to <=900s')
    if not any(root.is_relative_to(Path(p).resolve()) for p in value['allowed_output_roots']):
        raise ValueError('Output is outside shared checkpoint admission scope')
    lock = Path(value['lock_path'])
    if not lock.is_absolute() or lock.is_symlink() or not lock.parent.is_dir():
        raise ValueError('Checkpoint admission requires an existing shared lock directory')
    for path in value['quota_paths']:
        if not root.is_relative_to(Path(path).resolve()):
            raise ValueError('Quota path must be an output ancestor')
    return value


def _try_lock(path):
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    except BaseException:
        os.close(fd)
        raise
    return fd


@contextmanager
def checkpoint_admission(session, root, optimizer_steps):
    """Serialize writes across jobs, without changing checkpoint state/retention."""
    requested = os.environ.get('H3_CHECKPOINT_ADMISSION_CONFIG')
    session.agree({'checkpoint_admission_config': requested})
    if not requested:
        yield
        return
    root = Path(root).resolve()
    options = session.local('checkpoint admission configuration', lambda: _options(requested, root))
    session.agree(options)
    lock = Path(options['lock_path'])
    marker = lock.with_suffix(lock.suffix + '.state.json')
    fd = None
    begun = False
    last_notice = 0.
    started = time.monotonic()

    def release():
        nonlocal fd
        if fd is not None:
            os.close(fd)
            fd = None

    def acquire():
        nonlocal fd
        fd = _try_lock(lock)
        if fd is not None and marker.exists():
            previous = json.loads(marker.read_text())
            if previous.get('status') not in ('committed', 'drain_verified'):
                raise RuntimeError('Previous checkpoint writer has no committed/drained receipt')

    def publish(status, **extra):
        owner_start = Path('/proc/self/stat').read_text().rsplit(') ', 1)[1].split()[19]
        _atomic_json(marker, dict(status=status, output=str(root), optimizer_steps=optimizer_steps,
            attempt=os.environ.get('H3_ATTEMPT_NONCE'), host=socket.gethostname(), pid=os.getpid(),
            start_ticks=owner_start, unix=time.time(), **extra))

    try:
        while True:
            session.primary('checkpoint writer mutex', acquire)
            acquired = torch.tensor(int(fd is not None), device=session.device, dtype=torch.int32)
            if session.distributed:
                dist.broadcast(acquired, src=0)
            remaining = 0
            if bool(acquired.item()):
                remaining = session.local('checkpoint fresh quota', lambda: capacity_bytes(root, options['quota_paths']))
                free = torch.tensor(remaining, device=session.device, dtype=torch.int64)
                if session.distributed:
                    dist.all_reduce(free, op=dist.ReduceOp.MIN)
                remaining = int(free.item())
                if remaining >= options['checkpoint_bytes'] + options['headroom_bytes']:
                    session.primary('checkpoint writer receipt', lambda: publish('writing',
                        observed_free_bytes=remaining, checkpoint_budget_bytes=options['checkpoint_bytes'],
                        headroom_bytes=options['headroom_bytes']))
                    begun = True
                    break
                release()
            elapsed = time.monotonic() - started
            expired = torch.tensor(int(elapsed >= options['timeout_seconds']), device=session.device)
            if session.distributed:
                dist.all_reduce(expired, op=dist.ReduceOp.MAX)
            if bool(expired.item()):
                raise TimeoutError('Shared checkpoint admission deadline reached; previous checkpoints preserved')
            if session.identity.rank == 0 and elapsed - last_notice >= 30:
                print('H3_CHECKPOINT_ADMISSION ' + json.dumps(dict(status='waiting', seconds=elapsed,
                      free_bytes=remaining, optimizer_steps=optimizer_steps)), flush=True)
                last_notice = elapsed
            time.sleep(options['poll_seconds'])
        yield
        def committed():
            pointer = json.loads((root / 'checkpoints/latest.json').read_text())
            receipt = (root / 'checkpoints' / pointer['manifest']).resolve()
            if not receipt.is_relative_to(root / 'checkpoints'):
                raise ValueError('Checkpoint receipt escaped its output')
            manifest = json.loads(receipt.read_text())
            if (manifest['counts']['optimizer_steps'] != optimizer_steps
                    or len(manifest['ranks']) != session.identity.world_size):
                raise ValueError('Checkpoint admission completion receipt differs')
            publish('committed', checkpoint=str(receipt),
                    payload_bytes=sum(x['bytes'] for x in manifest['ranks']))
        session.primary('checkpoint writer committed receipt', committed)
    except BaseException:
        if begun and session.identity.rank == 0:
            # Local only: a native collective may already have failed elsewhere.
            # Keep the original writing marker if even this metadata IO fails.
            try:
                publish('failed_drain_unproven')
            except Exception:
                pass
        raise
    finally:
        release()
