import os
import sys
import time
from types import SimpleNamespace

import pytest

from visual_rl.rewards.process import JsonWorker, _cleanup


def test_rapid_ready_then_exit_cleanup_race():
    code = (
        'import os,sys; f=os.fdopen(int(sys.argv[1]),"wb"); '
        'f.write(b\'{"status":"ready"}\\n\'); f.flush(); os._exit(0)'
    )
    for _ in range(40):
        worker = JsonWorker([sys.executable, "-c", code], [],
                            startup_timeout=3, request_timeout=1)
        worker.close()
        assert worker.process.poll() is not None


def test_persistent_group_permission_failure_is_not_silently_ignored(monkeypatch):
    read_fd, write_fd = os.pipe()
    os.close(write_fd)
    process = SimpleNamespace(stdin=None, pid=-1, poll=lambda: 0)

    def denied(*args):
        raise PermissionError("fixture group permission denied")

    monkeypatch.setattr(os, "killpg", denied)
    start = time.monotonic()
    with pytest.raises(PermissionError, match="fixture group permission denied"):
        _cleanup(process, read_fd)
    assert time.monotonic() - start < 1
