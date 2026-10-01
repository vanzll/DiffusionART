"""Conservative retention of explicitly registered, published checkpoints.

Legacy and unregistered artifacts are never adopted. The registry is local to
one absolute root; this is not a concurrent-writer or hostile-filesystem lock.
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

_STEM = re.compile(r"step-[0-9]{9,}-[0-9a-f]{32}\Z")
_REGISTRY = ".checkpoint-retention.json"


def validate_keep_last(keep_last):
    if keep_last is not None and (type(keep_last) is not int or keep_last <= 0):
        raise ValueError("keep_last must be None or a positive integer")


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
    except Warning:
        pass


def retain_checkpoints(root, receipt, *, identity, keep_last):
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
