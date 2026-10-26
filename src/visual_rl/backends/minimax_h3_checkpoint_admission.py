"""Opt-in cross-job checkpoint admission with isolated, bounded metadata IO.

Only a persistent rank0 I/O child owns the flock FD. It holds it through the
entire all-rank native save and acknowledges marker commit before release.
Training ranks never resolve/stat/open/read quota/config/marker paths here.
Do not wrap this context or its native collective body in session.local().
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist

from .minimax_h3_checkpoint_io import CheckpointIO


@contextmanager
def checkpoint_admission(session, root, optimizer_steps):
    requested = os.environ.get('H3_CHECKPOINT_ADMISSION_CONFIG')
    session.agree({'checkpoint_admission_config': requested})
    if not requested:
        yield
        return
    # Lexical only. Every filesystem-dependent resolution happens in the child.
    root = Path(root)
    io = None
    begun = False
    started = time.monotonic()
    last_notice = 0.

    def spawn():
        nonlocal io
        if not root.is_absolute():
            raise ValueError('Checkpoint output must be absolute')
        io = CheckpointIO()

    try:
        session.local('checkpoint I/O worker start', spawn)
        options = session.local('checkpoint admission configuration', lambda: io.request(
            'init', config=requested, root=str(root), parent={'pid': os.getpid()}))
        session.agree(options)
        while True:
            expired = torch.tensor(int(time.monotonic() - started >= options['timeout_seconds']),
                                   device=session.device, dtype=torch.int32)
            if session.distributed:
                dist.all_reduce(expired, op=dist.ReduceOp.MAX)
            if bool(expired.item()):
                raise TimeoutError('Shared checkpoint admission deadline reached; previous checkpoints preserved')
            acquired = session.primary('checkpoint writer mutex', lambda: io.request('acquire'))
            flag = torch.tensor(int(bool(acquired)), device=session.device, dtype=torch.int32)
            if session.distributed:
                dist.broadcast(flag, src=0)
            remaining = 0
            if bool(flag.item()):
                remaining = session.local('checkpoint fresh quota', lambda: io.request('quota'))
                free = torch.tensor(remaining, device=session.device, dtype=torch.int64)
                if session.distributed:
                    dist.all_reduce(free, op=dist.ReduceOp.MIN)
                remaining = int(free.item())
                if remaining >= options['checkpoint_bytes'] + options['headroom_bytes']:
                    session.primary('checkpoint writer receipt', lambda: io.request('writing', values=dict(
                        optimizer_steps=optimizer_steps, attempt=os.environ.get('H3_ATTEMPT_NONCE'),
                        observed_free_bytes=remaining, checkpoint_budget_bytes=options['checkpoint_bytes'],
                        headroom_bytes=options['headroom_bytes'])))
                    begun = True
                    break
                session.primary('checkpoint writer mutex release', lambda: io.request('release'))
            elapsed = time.monotonic() - started
            if session.identity.rank == 0 and elapsed - last_notice >= 30:
                print('H3_CHECKPOINT_ADMISSION ' + json.dumps(dict(status='waiting',
                    seconds=elapsed, free_bytes=remaining, optimizer_steps=optimizer_steps)), flush=True)
                last_notice = elapsed
            time.sleep(options['poll_seconds'])
        yield
        session.primary('checkpoint writer committed receipt', lambda: io.request('commit',
            optimizer_steps=optimizer_steps, world_size=session.identity.world_size))
        session.primary('checkpoint writer mutex release', lambda: io.request('release'))
    except BaseException:
        # No collectives after a possibly failed native collective. Abort only
        # when the child has acknowledged its previous request; else preserve
        # the writing marker and let the fenced owner prove drain/quarantine.
        if begun and session.identity.rank == 0 and io is not None and not io.inflight:
            try:
                io.request('abort', timeout=.2)
            except Exception:
                pass
        raise
    finally:
        if io is not None:
            io.close()
