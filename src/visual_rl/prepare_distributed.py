"""Coordinate one asset preparation on a shared cache before torchrun."""

import fcntl
import hashlib
import json
import os
import time
from pathlib import Path

from .config import ConfigError, _read
from .data import file_sha256
from .training.checkpoint import _atomic_json


def _bindings(spec):
    names = set(spec.get("defaults", {}))

    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if (key == "env" or key.endswith("_env")) and isinstance(item, str):
                    names.add(item)
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(spec)
    return names


def prepare_shared_assets(config_path, plan, prepare, *, timeout=7200, run_id=None):
    """Serialize preparation, then share its generated environment bindings.

    Nodes must use the same shared cache, model/data bindings and recipe. The
    lock wait is bounded; this is not a hard deadline for filesystem calls.
    Standard Hub machinery still owns model download/cache integrity.

    A fresh run_id shared by all nodes makes the receipt specific to this
    launch. The first node fully prepares and validates the assets; peers reuse
    that completed validation on the same shared filesystem. Without run_id,
    existing receipts are revalidated as before.
    """
    spec = _read(Path(config_path)).get("assets", {})
    if not spec:
        prepare(plan)
        return
    root = Path(os.environ.get("VISUAL_RL_ASSET_ROOT", "~/.cache/visual-rl")).expanduser().absolute()
    names = _bindings(spec)
    inputs = {name: os.environ.get(name) for name in sorted(names)}
    generated = {str(destination) for entries in plan for entry, destination in entries
                 if entry.get("action") != "manifest_binding"}
    source_names = {entry["source_env"] for entry in spec.get("transformations", [])
                    if "source_env" in entry}
    source_names.update(entry["env"] for entry in spec.get("prompts", []))
    manifests = {name: file_sha256(value) for name, value in inputs.items()
                 if value and value not in generated and name in source_names
                 and Path(value).is_file()}
    identity = hashlib.sha256(json.dumps({"assets": spec, "bindings": inputs, "manifests": manifests},
                                        sort_keys=True).encode()).hexdigest()
    if run_id is not None:
        if not isinstance(run_id, str) or not run_id.strip() or len(run_id) > 128:
            raise ConfigError("shared preparation run_id must be a nonempty string of at most 128 characters")
        identity = hashlib.sha256(json.dumps({"asset_identity": identity, "run_id": run_id},
                                            sort_keys=True).encode()).hexdigest()
    directory = root / "prepared"
    directory.mkdir(parents=True, exist_ok=True)
    receipt = directory / f"{identity}.json"
    deadline = time.monotonic() + timeout
    with (directory / f"{identity}.lock").open("a") as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise ConfigError("timed out waiting for shared asset preparation") from None
                time.sleep(.5)
        if receipt.exists():
            record = json.loads(receipt.read_text())
            values = record.get("bindings", {})
            if (record.get("identity") == identity and set(values) <= names
                    and all(Path(path).exists() for path in record.get("paths", []))):
                os.environ.update(values)
                if run_id is not None:
                    return
                # Reuse conversion/conditioning, but still validate its existing
                # receipts and inventory. Never redownload model snapshots here.
                prepare(([], plan[1]))
                return
        prepare(plan)
        values = {name: os.environ[name] for name in names if name in os.environ}
        paths = [value for value in values.values() if Path(value).is_absolute() and Path(value).exists()]
        _atomic_json(receipt, {"identity": identity, "bindings": values, "paths": paths})
