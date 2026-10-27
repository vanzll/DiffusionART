"""Conservative retention of explicitly registered, published checkpoints.

Legacy and unregistered artifacts are never adopted. Cleanup uses a separate
per-root lock, not the trainer's output lease or a hostile-filesystem lock.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import stat
import threading
import warnings
from contextlib import contextmanager
from pathlib import Path

_STEM = re.compile(r"step-[0-9]{9,}-[0-9a-f]{32}\Z")
_REGISTRY = ".checkpoint-retention.json"
_LOCK = ".checkpoint-retention.lock"


class _CleanupWorker:
    """One permanent daemon and one replaceable metadata-only pending job.

    A stalled filesystem operation pins this worker, never the training ranks.
    No replacement thread or unbounded shutdown join is permitted.
    """

    def __init__(self):
        self._condition = threading.Condition()
        self._thread = None
        self._pending = None
        self._busy = False

    def enqueue(self, job):
        with self._condition:
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="checkpoint-retention", daemon=True)
                self._thread.start()
            self._pending = job
            self._condition.notify()

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._pending is not None)
                job, self._pending = self._pending, None
                self._busy = True
            try:
                identity = job["identity"]
                if identity is None:
                    manifest = _json(Path(job["receipt"]))
                    if (manifest.get("kind") != "fsdp-local"
                            or manifest.get("config_sha256") != job["expected_config_sha256"]
                            or manifest.get("layout", {}).get("world_size") != job["expected_world_size"]):
                        raise ValueError("resumed retention identity differs")
                    identity = {"kind": "fsdp-local", "config_sha256": manifest["config_sha256"],
                                "topology": manifest["layout"]}
                retain_checkpoints(job["root"], job["receipt"], identity=identity,
                                   keep_last=job["keep_last"])
            except Exception as exc:
                _warn(exc)
            finally:
                with self._condition:
                    self._busy = False
                    self._condition.notify_all()

    def wait_idle(self, timeout):
        """Bounded observation for tests only; save/exit never waits for IO."""
        with self._condition:
            return self._condition.wait_for(lambda: not self._busy and self._pending is None,
                                            timeout=timeout)


_WORKER = _CleanupWorker()


def enqueue_retention(root, receipt, *, identity, keep_last):
    """Best-effort enqueue with no IO, tensor/model ownership or collective."""
    if keep_last is None:
        return
    try:
        # JSON roundtrip accepts only metadata and detaches mutable layout lists.
        job = json.loads(json.dumps({"root": str(root), "receipt": str(receipt),
                                    "identity": identity, "keep_last": keep_last}, allow_nan=False))
        _WORKER.enqueue(job)
    except Exception as exc:
        _warn(exc)


def validate_keep_last(keep_last):
    if keep_last is not None and (type(keep_last) is not int or keep_last <= 0):
        raise ValueError("keep_last must be None or a positive integer")


def enqueue_resumed_retention(root, receipt, *, expected_config_sha256, expected_world_size, keep_last):
    """Recover an interrupted cleanup after a successful native FSDP load.

    Only an absolute manifest in this output's checkpoint root is eligible.
    All filesystem/identity/payload checks remain on the existing single daemon;
    its conservative latest/registration/integrity/deletion policy is unchanged.
    A submitted job is not evidence that registration or cleanup has completed.
    """
    root, receipt = Path(root), Path(receipt)
    if (keep_last is None or not root.is_absolute() or not receipt.is_absolute()
            or receipt.name != "manifest.json" or receipt.parent.parent != root):
        return False
    validate_keep_last(keep_last)
    job = {"root": str(root), "receipt": str(receipt), "identity": None,
           "expected_config_sha256": expected_config_sha256,
           "expected_world_size": expected_world_size, "keep_last": keep_last}
    try:
        _WORKER.enqueue(json.loads(json.dumps(job, allow_nan=False)))
        return True
    except Exception as exc:
        _warn(exc)
        return False


def _safe_path(path):
    path = path.absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("retention rejects symlink paths")
    return path


def _json(path):
    _safe_path(path)
    if not path.is_file() or path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("retention metadata is not a bounded regular file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("retention metadata must be an object")
    return value


def _checked_file(path, record):
    from .checkpoint import _digest

    _safe_path(path)
    if (not isinstance(record, dict) or record.get("file") != path.name
            or not path.is_file() or path.stat().st_size != record.get("bytes")
            or _digest(path) != record.get("sha256")):
        raise ValueError("retention payload integrity differs")


def _snapshot(root, entry):
    """Return only exact owned files, after validating the entire snapshot."""
    from .checkpoint import _digest

    identity = entry["identity"]
    name = entry["receipt"]
    if not isinstance(name, str):
        raise ValueError("invalid retention receipt")
    if identity["kind"] == "local":
        stem = name.removesuffix(".json")
        if name != stem + ".json" or not _STEM.fullmatch(stem):
            raise ValueError("invalid local retention receipt")
        receipt = root / name
    else:
        stem = name.removesuffix("/manifest.json")
        if name != stem + "/manifest.json" or not _STEM.fullmatch(stem):
            raise ValueError("invalid distributed retention receipt")
        receipt = root / stem / "manifest.json"
    record = _json(receipt)
    if (_digest(receipt) != entry["sha256"] or record.get("schema_version") != 1
            or record.get("config_sha256") != identity["config_sha256"]):
        raise ValueError("retention receipt identity differs")
    if identity["kind"] == "local":
        payload = root / (stem + ".pt")
        if record.get("optimizer_step") != int(stem.split("-")[1]):
            raise ValueError("retention step differs")
        _checked_file(payload, record)
        return receipt, [payload, receipt]
    kind = identity["kind"]
    if kind not in {"ddp", "fsdp-local"} or record.get("kind") != kind:
        raise ValueError("retention kind differs")
    topology = identity["topology"]
    world = topology["world_size"]
    if type(world) is not int or world <= 0:
        raise ValueError("invalid retention topology")
    if kind == "ddp" and record.get("world_size") != world:
        raise ValueError("retention topology differs")
    if kind == "fsdp-local" and record.get("layout") != topology:
        raise ValueError("retention layout differs")
    if (record["counts"]["optimizer_steps"] != int(stem.split("-")[1])
            or len(record["ranks"]) != world):
        raise ValueError("retention rank/step differs")
    pairs = [(f"rank-{rank:05d}", value) for rank, value in enumerate(record["ranks"])]
    if kind == "ddp":
        pairs.append(("shared", record["shared"]))
    files = [receipt]
    for prefix, value in pairs:
        payload = receipt.parent / (prefix + ".pt")
        metadata = receipt.parent / (prefix + ".record.json")
        _checked_file(payload, value)
        if _json(metadata) != value:
            raise ValueError("retention rank receipt differs")
        files.extend([payload, metadata])
    if {p.name for p in receipt.parent.iterdir()} != {p.name for p in files}:
        raise ValueError("retention directory contains unowned artifacts")
    return receipt, files


def _warn(exc):
    # Cleanup is advisory even when a caller promotes warnings to exceptions.
    try:
        warnings.warn(f"checkpoint retention skipped: {exc}", RuntimeWarning, stacklevel=3)
    except Exception:
        pass


@contextmanager
def _retention_lock(root):
    """The cleanup caller owns this fd until all registry/pruning IO completes.

    Keep the lock inode in place across runs: unlink/recreate would let a new
    process bypass a stalled worker that still holds the original flock.
    """
    path = _safe_path(root / _LOCK)
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("retention lock requires a regular, non-hardlinked file")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


def retain_checkpoints(root, receipt, *, identity, keep_last):
    """Advisory cleanup under a nonblocking per-root, cross-process flock.

    Both synchronous callers and the daemon participate. Busy roots are skipped;
    a stalled owner is never replaced, even after the trainer releases its lease.
    """
    if keep_last is None:
        return
    try:
        root = _safe_path(Path(root).absolute())
        with _retention_lock(root):
            _retain_locked(root, receipt, identity=identity, keep_last=keep_last)
    except Exception as exc:
        _warn(exc)


def _retain_locked(root, receipt, *, identity, keep_last):
    """Run only after successful latest publication; all errors are nonfatal.

    Newest means publication order, not largest step (a resumed run can rewind).
    The current receipt and whatever latest presently names are always protected.
    """
    if keep_last is None:
        return
    try:
        from .checkpoint import _atomic_json, _digest

        root = _safe_path(Path(root).absolute())
        receipt = _safe_path(Path(receipt).absolute())
        registry = root / _REGISTRY
        if registry.exists() or registry.is_symlink():
            state = _json(registry)
            if state.get("schema_version") != 1 or state.get("root") != str(root):
                raise ValueError("foreign retention registry")
            entries = state["entries"]
            if not isinstance(entries, list):
                raise ValueError("invalid retention entries")
        else:
            entries = []
        latest = _json(root / "latest.json")
        current = receipt.relative_to(root).as_posix()
        latest_name = latest.get("receipt", latest.get("manifest"))
        if latest_name != current:
            raise ValueError("current checkpoint is not published latest")
        entry = {"receipt": current, "sha256": _digest(receipt), "identity": identity}
        _snapshot(root, entry)
        entries = [item for item in entries if item["receipt"] != current] + [entry]
        state = {"schema_version": 1, "root": str(root), "entries": entries}
        _atomic_json(registry, state)
        eligible = []
        for item in entries:
            if item["identity"] != identity:
                continue
            try:
                candidate, files = _snapshot(root, item)
                eligible.append((item, candidate, files))
            except Exception as exc:
                _warn(exc)
        for item, candidate, files in eligible[:-keep_last]:
            latest = _json(root / "latest.json")
            if item["receipt"] in {current, latest.get("receipt"), latest.get("manifest")}:
                continue
            try:
                # Revalidate immediately before exact-file unlink; never recurse.
                candidate, files = _snapshot(root, item)
                # Hashing can be slow while training publishes a newer latest.
                latest = _json(root / "latest.json")
                if item["receipt"] in {current, latest.get("receipt"), latest.get("manifest")}:
                    continue
                for path in files:
                    _safe_path(path).unlink()
                if identity["kind"] != "local":
                    candidate.parent.rmdir()
                entries.remove(item)
            except Exception as exc:
                _warn(exc)
        _atomic_json(registry, state)
    except Exception as exc:
        _warn(exc)
