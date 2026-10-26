"""Standard-library-only persistent checkpoint I/O worker; owns its lock FD."""
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

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


def atomic_json(path, value):
    fd, temporary = tempfile.mkstemp(prefix='.checkpoint-admission-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class Worker:
    def __init__(self):
        self.fd = None
        self.phase = None

    def publish(self, status, **values):
        owner = dict(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().rsplit(') ', 1)[1].split()[19])
        atomic_json(self.marker, dict(status=status, output=str(self.root), unix=time.time(),
                    lock_owner=owner, rank0_owner=self.parent, **values))
        self.phase = status

    def release(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def dispatch(self, request):
        operation = request['operation']
        if operation == 'init':
            # Every resolve/stat/read is deliberately inside this child.
            self.root = Path(request['root']).resolve()
            self.options = _options(request['config'], self.root)
            self.lock = Path(self.options['lock_path'])
            self.marker = self.lock.with_suffix(self.lock.suffix + '.state.json')
            self.parent = request['parent']
            if self.parent['pid'] != os.getppid():
                raise ValueError('Checkpoint I/O parent identity differs')
            self.parent['start_ticks'] = Path('/proc', str(os.getppid()), 'stat').read_text().rsplit(') ', 1)[1].split()[19]
            return self.options
        if operation == 'acquire':
            if self.fd is not None:
                raise RuntimeError('Checkpoint writer already owns the lock')
            self.fd = _try_lock(self.lock)
            if self.fd is not None and self.marker.exists():
                if json.loads(self.marker.read_text()).get('status') not in ('committed', 'drain_verified'):
                    raise RuntimeError('Previous checkpoint writer has no committed/drained receipt')
            return self.fd is not None
        if operation == 'quota':
            return capacity_bytes(self.root, self.options['quota_paths'])
        if operation == 'writing':
            if self.fd is None:
                raise RuntimeError('Checkpoint I/O child does not own the lock')
            self.write_identity = request['values']
            self.publish('writing', **self.write_identity)
            return {'held': True, 'lock_owner_pid': os.getpid()}
        if operation == 'commit':
            pointer = json.loads((self.root / 'checkpoints/latest.json').read_text())
            receipt = (self.root / 'checkpoints' / pointer['manifest']).resolve()
            if not receipt.is_relative_to(self.root / 'checkpoints'):
                raise ValueError('Checkpoint receipt escaped its output')
            manifest = json.loads(receipt.read_text())
            if (manifest['counts']['optimizer_steps'] != request['optimizer_steps']
                    or len(manifest['ranks']) != request['world_size']):
                raise ValueError('Checkpoint completion receipt differs')
            self.publish('committed', **self.write_identity, checkpoint=str(receipt),
                         payload_bytes=sum(x['bytes'] for x in manifest['ranks']))
            return True
        if operation == 'abort':
            if self.phase == 'writing':
                self.publish('failed_drain_unproven', **self.write_identity)
            self.release()
            return True
        if operation == 'release':
            if self.phase == 'writing':
                raise RuntimeError('Cannot release an uncommitted checkpoint writer')
            self.release()
            return True
        raise ValueError('Unknown checkpoint I/O operation')


def main():
    worker = Worker()
    try:
        for line in sys.stdin.buffer:
            request = json.loads(line)
            try:
                response = dict(sequence=request['sequence'], result=worker.dispatch(request))
            except Exception as error:
                response = dict(sequence=request['sequence'], error=f'{type(error).__name__}: {error}')
            print(json.dumps(response), flush=True)
    finally:
        # EOF is not a successful commit; leave any writing marker intact.
        worker.release()


if __name__ == '__main__':
    main()
