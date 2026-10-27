import json
import threading
import time

import pytest

from visual_rl.training import checkpoint
from visual_rl.training import checkpoint_retention as retention
from visual_rl.training.distributed_checkpoint import _write


@pytest.fixture
def worker(monkeypatch):
    worker = retention._CleanupWorker()
    monkeypatch.setattr(retention, "_WORKER", worker)
    yield worker
    assert worker.wait_idle(timeout=10)


def publish(root, token):
    directory = root / ("step-000000002-" + token * 32)
    directory.mkdir()
    ranks = []
    for rank in range(2):
        record = _write(directory / f"rank-{rank:05d}.pt", {"rank": rank})
        checkpoint._atomic_json(directory / f"rank-{rank:05d}.record.json", record)
        ranks.append(record)
    receipt = directory / "manifest.json"
    checkpoint._atomic_json(receipt, {"schema_version": 1, "kind": "fsdp-local",
        "config_sha256": "a" * 64, "counts": {"optimizer_steps": 2},
        "layout": {"world_size": 2, "members": [0, 1]}, "ranks": ranks})
    checkpoint._atomic_json(root / "latest.json", {"manifest": f"{directory.name}/manifest.json"})
    return receipt


def enqueue(root, receipt, **kwargs):
    options = dict(expected_config_sha256="a" * 64, expected_world_size=2, keep_last=2)
    options.update(kwargs)
    return retention.enqueue_resumed_retention(root, receipt, **options)


def test_resume_recovers_missing_registration_without_adopting_legacy(tmp_path, worker):
    legacy = publish(tmp_path, "f")
    receipts = []
    for token in "123":
        current = publish(tmp_path, token)
        receipts.append(current)
        assert enqueue(tmp_path, current)
        assert worker.wait_idle(timeout=10)
    assert legacy.exists()  # Unregistered legacy is never silently adopted/pruned.
    assert not receipts[0].parent.exists()
    assert all(path.exists() for path in receipts[1:])
    state = json.loads((tmp_path / ".checkpoint-retention.json").read_text())
    assert len(state["entries"]) == 2


def test_foreign_or_relative_resume_is_not_queued(tmp_path, worker):
    foreign = tmp_path / "other" / "checkpoints" / "step-000000002-abc" / "manifest.json"
    assert not enqueue(tmp_path, foreign)
    assert not enqueue(tmp_path, "relative/manifest.json")
    assert worker._thread is None
    assert not (tmp_path / ".checkpoint-retention.json").exists()


def test_resume_metadata_stall_does_not_block_caller_or_spawn_workers(tmp_path, worker, monkeypatch):
    receipt = publish(tmp_path, "1")
    started, release = threading.Event(), threading.Event()
    original = retention._json

    def stalled(path):
        started.set()
        assert release.wait(timeout=10)
        return original(path)

    monkeypatch.setattr(retention, "_json", stalled)
    try:
        begin = time.monotonic()
        assert enqueue(tmp_path, receipt)
        assert time.monotonic() - begin < 1
        assert started.wait(timeout=3)
        thread = worker._thread
        for _ in range(20):
            assert enqueue(tmp_path, receipt)
        assert worker._thread is thread
        assert not worker.wait_idle(timeout=.02)
    finally:
        release.set()
    assert worker.wait_idle(timeout=10)
    assert (tmp_path / ".checkpoint-retention.json").exists()


@pytest.mark.parametrize("options", [{"expected_config_sha256": "b" * 64}, {"expected_world_size": 4}])
def test_resume_rejects_wrong_identity_before_registration(tmp_path, worker, options):
    receipt = publish(tmp_path, "1")
    with pytest.warns(RuntimeWarning, match="resumed retention identity differs"):
        assert enqueue(tmp_path, receipt, **options)
        assert worker.wait_idle(timeout=10)
    assert receipt.exists()
    assert not (tmp_path / ".checkpoint-retention.json").exists()


def test_resume_still_validates_payload_after_native_load(tmp_path, worker):
    receipt = publish(tmp_path, "1")
    (receipt.parent / "rank-00001.pt").write_bytes(b"changed after load")
    with pytest.warns(RuntimeWarning, match="payload integrity differs"):
        assert enqueue(tmp_path, receipt)
        assert worker.wait_idle(timeout=10)
    assert receipt.exists()
    assert not (tmp_path / ".checkpoint-retention.json").exists()


def test_resume_queue_failure_is_nonfatal(tmp_path, worker, monkeypatch):
    receipt = publish(tmp_path, "1")

    def unavailable(job):
        raise OSError("queue unavailable")

    monkeypatch.setattr(worker, "enqueue", unavailable)
    with pytest.warns(RuntimeWarning, match="queue unavailable"):
        assert not enqueue(tmp_path, receipt)
    assert receipt.exists()
