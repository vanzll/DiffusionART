import json
import os
import time
from contextlib import contextmanager

import pytest

from visual_rl.backends.minimax_h3_checkpoint_io import CheckpointIO
from visual_rl.backends.minimax_h3_failure_fence import native_failure_fence


@pytest.fixture
def owned(tmp_path, monkeypatch):
    attempt = tmp_path / "attempts" / "test-native-owned"
    attempt.mkdir(parents=True)
    for key, value in {"H3_ACCEPTANCE_ROOT": str(tmp_path), "H3_ATTEMPT_DIR": str(attempt),
                       "H3_ATTEMPT_NONCE": attempt.name, "RANK": "3", "H3_RUNTIME_COMMIT": "a" * 40}.items():
        monkeypatch.setenv(key, value)
    return tmp_path, attempt


def test_owned_failure_is_published_before_outer_pg_teardown(owned, capsys):
    _, attempt = owned
    marker = attempt / "failure.runtime-rank-00003"
    observed = []
    @contextmanager
    def pg():
        try:
            yield
        finally:
            observed.append(marker.exists())
    with pytest.raises(RuntimeError, match="not-a-real-secret"):
        with pg(), native_failure_fence():
            raise RuntimeError("not-a-real-secret")
    assert observed == [True]
    record = json.loads(marker.read_text())
    assert record["exception_type"] == "RuntimeError" and record["pid"] == os.getpid()
    assert "not-a-real-secret" not in marker.read_text()
    assert "not-a-real-secret" not in capsys.readouterr().err


def test_stalled_io_is_bounded_and_never_masks_original_failure(owned, tmp_path, capsys):
    helper = tmp_path / "blocked.py"
    helper.write_text("import sys,time\nsys.stdin.buffer.readline()\ntime.sleep(30)\n")
    clients = []
    def factory():
        client = CheckpointIO(helper=helper, timeout=.1)
        clients.append(client)
        return client
    begin = time.monotonic()
    with pytest.raises(ValueError, match="original"):
        with native_failure_fence(io_factory=factory):
            raise ValueError("original")
    assert time.monotonic() - begin < 2
    assert clients[0].process.poll() is not None
    assert "REPORT_ERROR TimeoutError" in capsys.readouterr().err


def test_worker_rejects_foreign_attempt(owned, tmp_path):
    root, attempt = owned
    client = CheckpointIO()
    try:
        with pytest.raises(RuntimeError, match="escaped"):
            client.request("runtime_failure", root=str(root), attempt=str(tmp_path / "foreign"),
                           nonce=attempt.name, record=dict(pid=os.getpid(), rank=3, attempt=attempt.name))
    finally:
        client.close()
    assert not (tmp_path / "foreign" / "failure.runtime-rank-00003").exists()


def test_unowned_failure_does_not_start_an_io_helper(monkeypatch):
    monkeypatch.delenv("H3_ATTEMPT_NONCE", raising=False)
    def forbidden():
        raise AssertionError("unowned helper")
    with pytest.raises(ValueError, match="original"):
        with native_failure_fence(io_factory=forbidden):
            raise ValueError("original")
