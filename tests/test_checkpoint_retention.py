import json
import shutil
from contextlib import nullcontext

import pytest
import torch

from visual_rl.training import checkpoint
from visual_rl.training import checkpoint_retention as retention
from visual_rl.training.state import Progress


def _hold_retention_lock(root, ready, release):
    from pathlib import Path

    with retention._retention_lock(Path(root)):
        ready.set()
        assert release.wait(timeout=15)


@pytest.fixture
def writer(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())

    def save(step, *, config="a" * 64, keep_last=2):
        return checkpoint.save_checkpoint(
            tmp_path, model=model, optimizer=optimizer,
            progress=Progress(optimizer_steps=step), config_sha256=config, keep_last=keep_last)

    return save, model, optimizer


def payload(receipt):
    return receipt.parent / json.loads(receipt.read_text())["file"]


def test_local_retention_publication_order_and_legacy_compatibility(tmp_path, writer):
    save, model, optimizer = writer
    legacy = save(10, keep_last=None)
    assert not (tmp_path / ".checkpoint-retention.json").exists()
    first, second = save(11), save(12)
    current = save(1)
    assert legacy.exists() and payload(legacy).exists()
    assert not first.exists() and not first.with_suffix(".pt").exists()
    assert second.exists() and current.exists()
    assert checkpoint.load_checkpoint(tmp_path / "latest.json", model=model, optimizer=optimizer,
                                      config_sha256="a" * 64).optimizer_steps == 1
    latest = json.loads((tmp_path / "latest.json").read_text())
    assert latest["receipt"] == current.name
    assert set(json.loads(current.read_text())) == {
        "schema_version", "file", "sha256", "bytes", "optimizer_step", "config_sha256"}


@pytest.mark.parametrize("artifact", ["corrupt", "foreign", "unpublished", "symlink_payload",
                                     "symlink_receipt", "missing", "arbitrary"])
def test_local_retention_leaves_unsafe_artifacts(tmp_path, writer, artifact):
    save, _, _ = writer
    first = save(1, config="b" * 64 if artifact == "foreign" else "a" * 64)
    target = payload(first)
    if artifact == "corrupt":
        target.write_bytes(b"corrupt")
    elif artifact == "unpublished":
        # A complete, same-config copy not registered by successful publication.
        stem = "step-000000001-" + "f" * 32
        record = json.loads(first.read_text())
        record["file"] = stem + ".pt"
        copied = tmp_path / (stem + ".json")
        copied.write_text(json.dumps(record))
        shutil.copyfile(target, tmp_path / record["file"])
        first, target = copied, tmp_path / record["file"]
    elif artifact in {"symlink_payload", "symlink_receipt"}:
        link = target if artifact == "symlink_payload" else first
        outside = tmp_path.parent / (artifact + ".data")
        shutil.copyfile(link, outside)
        link.unlink()
        link.symlink_to(outside)
    elif artifact == "missing":
        target.unlink()
    elif artifact == "arbitrary":
        registry = tmp_path / ".checkpoint-retention.json"
        state = json.loads(registry.read_text())
        state["entries"][0]["receipt"] = "../arbitrary.json"
        registry.write_text(json.dumps(state))
    with pytest.warns(RuntimeWarning) if artifact in {
        "corrupt", "symlink_payload", "symlink_receipt", "missing", "arbitrary"
    } else nullcontext():
        current = save(2, keep_last=1)
    assert first.exists()
    assert target.exists() or artifact == "missing"
    assert current.exists()


@pytest.mark.parametrize("failure", ["write", "receipt", "latest"])
def test_failed_save_never_prunes_or_replaces_previous_latest(tmp_path, writer, monkeypatch, failure):
    save, model, optimizer = writer
    first = save(1)
    latest = (tmp_path / "latest.json").read_bytes()
    original = checkpoint._atomic_json

    def fail_json(path, value):
        if failure == "receipt" or path.name == "latest.json":
            raise OSError("publication failed")
        return original(path, value)

    def fail_write(*args):
        raise OSError("write failed")

    if failure == "write":
        monkeypatch.setattr(torch, "save", fail_write)
    else:
        monkeypatch.setattr(checkpoint, "_atomic_json", fail_json)
    with pytest.raises(OSError):
        save(2, keep_last=1)
    assert (tmp_path / "latest.json").read_bytes() == latest
    assert first.exists() and payload(first).exists()
    assert checkpoint.load_checkpoint(first, model=model, optimizer=optimizer,
                                      config_sha256="a" * 64).optimizer_steps == 1
    unpublished = set(tmp_path.glob("step-*.json")) - {first}
    monkeypatch.undo()
    save(3, keep_last=1)
    assert not first.exists()
    assert all(path.exists() for path in unpublished)


def test_cleanup_error_is_nonfatal(tmp_path, writer, monkeypatch):
    save, _, _ = writer
    first = save(1)
    original = retention._safe_path

    def reject(path):
        if path == first:
            raise OSError("cleanup denied")
        return original(path)

    monkeypatch.setattr(retention, "_safe_path", reject)
    with pytest.warns(RuntimeWarning, match="cleanup denied"):
        current = save(2, keep_last=1)
    assert first.exists() and current.exists()


def test_unlink_error_is_nonfatal_even_with_warnings_as_errors(tmp_path, writer, monkeypatch):
    import warnings
    from pathlib import Path

    save, _, _ = writer
    first = save(1)
    target = payload(first)
    original = Path.unlink

    def reject(path, *args, **kwargs):
        if path == target:
            raise OSError("unlink denied")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", reject)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        current = save(2, keep_last=1)
    assert first.exists() and target.exists() and current.exists()


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2"])
def test_invalid_keep_last_rejected_before_writing(tmp_path, writer, value):
    save, _, _ = writer
    with pytest.raises(ValueError, match="keep_last"):
        save(1, keep_last=value)
    assert not list(tmp_path.iterdir())


def test_symlink_root_disables_cleanup(tmp_path, writer):
    save, model, optimizer = writer
    first = save(1)
    link = tmp_path.parent / "linked-checkpoints"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.warns(RuntimeWarning, match="symlink"):
        checkpoint.save_checkpoint(link, model=model, optimizer=optimizer,
                                   progress=Progress(optimizer_steps=2),
                                   config_sha256="a" * 64, keep_last=1)
    assert first.exists() and payload(first).exists()


@pytest.mark.parametrize("problem", ["corrupt", "foreign_root", "symlink"])
def test_untrusted_registry_disables_cleanup(tmp_path, writer, problem):
    save, _, _ = writer
    first = save(1)
    registry = tmp_path / ".checkpoint-retention.json"
    if problem == "corrupt":
        registry.write_text("not json")
    elif problem == "foreign_root":
        state = json.loads(registry.read_text())
        state["root"] = str(tmp_path.parent)
        registry.write_text(json.dumps(state))
    else:
        outside = tmp_path.parent / "foreign-registry.json"
        shutil.copyfile(registry, outside)
        registry.unlink()
        registry.symlink_to(outside)
    with pytest.warns(RuntimeWarning):
        current = save(2, keep_last=1)
    assert first.exists() and current.exists()


@pytest.mark.parametrize("kind", ["ddp", "fsdp-local"])
@pytest.mark.parametrize("unsafe", ["topology", "extra", "corrupt", "symlink", "incomplete"])
def test_distributed_snapshot_safety_without_native_collectives(tmp_path, kind, unsafe):
    from visual_rl.training.distributed_checkpoint import _write

    def publish(token, topology, keep=1):
        directory = tmp_path / ("step-000000001-" + token * 32)
        directory.mkdir()
        ranks = []
        for rank in range(2):
            record = _write(directory / f"rank-{rank:05d}.pt", {"rank": rank})
            checkpoint._atomic_json(directory / f"rank-{rank:05d}.record.json", record)
            ranks.append(record)
        manifest = {"schema_version": 1, "kind": kind, "config_sha256": "a" * 64,
                    "counts": {"optimizer_steps": 1}, "ranks": ranks}
        if kind == "ddp":
            shared = _write(directory / "shared.pt", {"model": {}})
            checkpoint._atomic_json(directory / "shared.record.json", shared)
            manifest.update(shared=shared, world_size=2)
        else:
            manifest["layout"] = topology
        receipt = directory / "manifest.json"
        checkpoint._atomic_json(receipt, manifest)
        checkpoint._atomic_json(tmp_path / "latest.json", {"manifest": f"{directory.name}/manifest.json"})
        retention.retain_checkpoints(tmp_path, receipt, keep_last=keep,
                                    identity={"kind": kind, "config_sha256": "a" * 64,
                                              "topology": topology})
        return receipt

    first = publish("1", {"world_size": 2, "members": [2, 3] if unsafe == "topology" else [0, 1]})
    if unsafe == "extra":
        (first.parent / "foreign.txt").write_text("not owned")
    elif unsafe == "corrupt":
        (first.parent / "rank-00001.pt").write_bytes(b"bad")
    elif unsafe == "symlink":
        target = first.parent / "rank-00001.pt"
        outside = tmp_path / "outside.pt"
        shutil.copyfile(target, outside)
        target.unlink()
        target.symlink_to(outside)
    elif unsafe == "incomplete":
        (first.parent / "rank-00001.pt").unlink()
    with pytest.warns(RuntimeWarning) if unsafe != "topology" else nullcontext():
        current = publish("2", {"world_size": 2, "members": [0, 1]})
    assert first.exists() and current.exists()


def test_worker_is_single_bounded_daemon_and_coalesces_metadata(monkeypatch):
    import threading

    started, release = threading.Event(), threading.Event()
    seen = []

    def delayed(root, receipt, *, identity, keep_last):
        seen.append((receipt, identity))
        if receipt == "first":
            started.set()
            assert release.wait(timeout=5)

    worker = retention._CleanupWorker()
    monkeypatch.setattr(retention, "_WORKER", worker)
    monkeypatch.setattr(retention, "retain_checkpoints", delayed)
    identity = {"kind": "ddp", "topology": {"members": [0, 1]}}
    try:
        retention.enqueue_retention("root", "first", identity=identity, keep_last=1)
        assert started.wait(timeout=1)
        thread = worker._thread
        assert thread.daemon
        for index in range(100):
            retention.enqueue_retention("root", str(index), identity=identity, keep_last=1)
        identity["topology"]["members"].append(2)
        assert worker._thread is thread
        assert worker._pending["receipt"] == "99"
        assert not worker.wait_idle(timeout=0.02)
    finally:
        release.set()
        assert worker.wait_idle(timeout=2)
    assert [receipt for receipt, _ in seen] == ["first", "99"]
    assert all(item["topology"]["members"] == [0, 1] for _, item in seen)


def test_stale_queued_publication_never_adopts_or_prunes(tmp_path, writer):
    save, _, _ = writer
    first = save(1, keep_last=None)
    current = save(2, keep_last=None)
    with pytest.warns(RuntimeWarning, match="not published latest"):
        retention.retain_checkpoints(tmp_path, first, keep_last=1,
                                    identity={"kind": "local", "config_sha256": "a" * 64})
    assert first.exists() and current.exists()
    assert not (tmp_path / ".checkpoint-retention.json").exists()


def test_latest_rechecked_after_slow_integrity_validation(tmp_path, writer, monkeypatch):
    save, _, _ = writer
    first = save(1)
    original = retention._snapshot
    checks = 0

    def change_latest(root, entry):
        nonlocal checks
        result = original(root, entry)
        if entry["receipt"] == first.name:
            checks += 1
            if checks == 2:
                checkpoint._atomic_json(tmp_path / "latest.json", {"receipt": first.name})
        return result

    monkeypatch.setattr(retention, "_snapshot", change_latest)
    current = save(2, keep_last=1)
    assert checks == 2
    assert first.exists() and payload(first).exists() and current.exists()


def test_process_lock_skips_sync_cleanup_without_registry_mutation(tmp_path, writer):
    import multiprocessing
    import time

    save, _, _ = writer
    first = save(1)
    current = save(2, keep_last=None)
    registry = tmp_path / ".checkpoint-retention.json"
    before = registry.read_bytes()
    ctx = multiprocessing.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    process = ctx.Process(target=_hold_retention_lock, args=(str(tmp_path), ready, release))
    process.start()
    try:
        assert ready.wait(timeout=10)
        begin = time.monotonic()
        with pytest.warns(RuntimeWarning):
            retention.retain_checkpoints(tmp_path, current, keep_last=1,
                                        identity={"kind": "local", "config_sha256": "a" * 64})
        assert time.monotonic() - begin < 2
        assert first.exists() and current.exists()
        assert registry.read_bytes() == before
    finally:
        release.set()
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
    assert process.exitcode == 0
    retention.retain_checkpoints(tmp_path, current, keep_last=1,
                                identity={"kind": "local", "config_sha256": "a" * 64})
    assert registry.exists() and not first.exists()
    next_receipt = save(3, keep_last=1)
    assert not current.exists() and next_receipt.exists()


def test_daemon_holds_root_lock_through_slow_validation(tmp_path, writer, monkeypatch):
    import threading

    save, _, _ = writer
    current = save(1, keep_last=None)
    entered, release = threading.Event(), threading.Event()
    worker = retention._CleanupWorker()
    original = retention._snapshot

    def delayed(root, entry):
        entered.set()
        assert release.wait(timeout=5)
        return original(root, entry)

    monkeypatch.setattr(retention, "_WORKER", worker)
    monkeypatch.setattr(retention, "_snapshot", delayed)
    try:
        retention.enqueue_retention(tmp_path, current, keep_last=1,
                                    identity={"kind": "local", "config_sha256": "a" * 64})
        assert entered.wait(timeout=1)
        with pytest.raises(BlockingIOError), retention._retention_lock(tmp_path):
            pytest.fail("daemon's flock was bypassed")
        assert not worker.wait_idle(timeout=0.02)
    finally:
        release.set()
        assert worker.wait_idle(timeout=2)
    with retention._retention_lock(tmp_path):
        assert (tmp_path / ".checkpoint-retention.json").exists()


@pytest.mark.parametrize("unsafe", ["symlink", "hardlink", "fifo"])
def test_retention_lock_rejects_unsafe_inode(tmp_path, writer, unsafe):
    import os

    save, _, _ = writer
    first = save(1, keep_last=None)
    lock = tmp_path / ".checkpoint-retention.lock"
    outside = tmp_path.parent / "foreign-lock"
    outside.write_text("foreign")
    if unsafe == "symlink":
        lock.symlink_to(outside)
    elif unsafe == "hardlink":
        os.link(outside, lock)
    else:
        os.mkfifo(lock)
    with pytest.warns(RuntimeWarning):
        current = save(2, keep_last=1)
    assert first.exists() and current.exists()
    assert outside.read_text() == "foreign"
    assert not (tmp_path / ".checkpoint-retention.json").exists()
