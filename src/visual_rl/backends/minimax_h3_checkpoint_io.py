"""Bounded pipe RPC to a persistent, standard-library checkpoint I/O child."""
import json
import os
import select
import subprocess
import time
from pathlib import Path


class CheckpointIO:
    def __init__(self, helper=None, timeout=10):
        # Use the local system interpreter: reading the shared helper script is
        # done by the child after exec, never by the training main process.
        helper = helper or Path(__file__).with_name('minimax_h3_checkpoint_io_worker.py')
        self.process = subprocess.Popen(['/usr/bin/python3', '-I', '-S', '-u', str(helper)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=0, close_fds=True)
        self.timeout, self.sequence, self.pending = timeout, 0, b''
        self.inflight = False
        os.set_blocking(self.process.stdin.fileno(), False)
        os.set_blocking(self.process.stdout.fileno(), False)

    def request(self, operation, *, timeout=None, **values):
        self.sequence += 1
        self.inflight = True
        remaining = json.dumps(dict(sequence=self.sequence, operation=operation, **values)).encode() + b'\n'
        if len(remaining) > 65536:
            raise ValueError('Checkpoint I/O request exceeds bounded protocol')
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError(f'Checkpoint I/O helper pid={self.process.pid} timed out: {operation}; drain required')
            if self.process.poll() is not None:
                raise RuntimeError(f'Checkpoint I/O helper exited during {operation}')
            reads, writes, _ = select.select([self.process.stdout],
                [self.process.stdin] if remaining else [], [], min(left, .05))
            if writes:
                try:
                    count = os.write(self.process.stdin.fileno(), remaining)
                    remaining = remaining[count:]
                except BlockingIOError:
                    pass
            if reads:
                data = os.read(self.process.stdout.fileno(), 65536)
                if not data:
                    raise RuntimeError('Checkpoint I/O helper closed its response pipe')
                self.pending += data
                if len(self.pending) > 1024 * 1024:
                    raise ValueError('Checkpoint I/O response exceeds bounded protocol')
            if b'\n' in self.pending:
                line, self.pending = self.pending.split(b'\n', 1)
                value = json.loads(line)
                if value['sequence'] != self.sequence:
                    raise RuntimeError('Checkpoint I/O response identity differs')
                self.inflight = False
                if 'error' in value:
                    raise RuntimeError(value['error'])
                return value['result']

    def close(self):
        # Never wait indefinitely for a child stuck in uninterruptible FS IO.
        # The existing nonce-fenced owner must prove drain (or quarantine it).
        try:
            self.process.kill()
        except ProcessLookupError:
            pass
        try:
            self.process.wait(timeout=.1)
        except subprocess.TimeoutExpired:
            pass
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            stream.close()
