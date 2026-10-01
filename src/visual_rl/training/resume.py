"""Explicit history rollback to a complete checkpoint, under the run owner lock."""

import json
import math
import os
import shutil
import uuid
from pathlib import Path

from ..config import ConfigError
from .checkpoint import _atomic_json
from .state import Progress


def validate_resume_request(resume, rewind, evaluate_only):
    if type(rewind) is not bool or type(evaluate_only) is not bool:
        raise ConfigError("resume modes must be boolean")
    if (rewind or evaluate_only) and not resume:
        raise ConfigError("resume modes require an explicit checkpoint")
    if rewind and evaluate_only:
        raise ConfigError("evaluation must not rewind training history")


def prepare_resume_history(c, root, progress, *, receipt, rewind=False):
    """Archive discarded records/media before replacing the active history.

    Model, optimizer, EMA, RNG and data cursors come ONLY from the checkpoint.
    Physical cost includes recorded work discarded by the rollback. Rank zero
    calls this function once; all ranks restore the resulting cost afterwards.
    Interrupted rollback can be retried with the same explicit rewind option;
    each archive is immutable evidence, never a replacement training checkpoint.
    """
    root = Path(root)
    step = progress.optimizer_steps
    journal = root / ".resume-rewind.json"
    pending = None
    if journal.exists():
        pending = json.loads(journal.read_text())
        if (pending.get("config_sha256") != c.fingerprint()
                or pending.get("optimizer_step") != step
                or pending.get("checkpoint") != str(Path(receipt).absolute())
                or pending.get("state") != "prepared"
                or len(pending.get("archive", "")) != 32
                or any(char not in "0123456789abcdef" for char in pending["archive"])):
            raise ConfigError("finish the pending rewind using its original checkpoint/config")
    history = root / "metrics.jsonl"
    kept = bytearray()
    newer = pending is not None
    elapsed, training = progress.elapsed_seconds, progress.training_seconds
    if pending:
        elapsed, training = max(elapsed, pending["elapsed_seconds"]), max(training, pending["training_seconds"])
        if not math.isfinite(elapsed) or not math.isfinite(training):
            raise ConfigError("pending rewind has invalid physical cost")
    if history.exists():
        lines = history.read_bytes().splitlines(keepends=True)
        for index, line in enumerate(lines):
            try:
                row = json.loads(line)
            except ValueError as exc:
                if index != len(lines) - 1 or line.endswith(b"\n"):
                    raise ConfigError("invalid metrics history; cannot safely rewind") from exc
                newer = True
                continue
            if not isinstance(row, dict):
                raise ConfigError("metrics history rows must be objects")
            newer |= not line.endswith(b"\n")
            row_step = row.get("train/optimizer_step")
            if type(row_step) is not int or row_step < 0:
                raise ConfigError("metrics history has invalid optimizer step")
            post_checkpoint = row_step > step
            for key, saved in (("system/rollout_attempts", progress.attempts),
                               ("system/completed_rollouts", progress.completed_rollouts)):
                count = row.get(key, saved)
                if type(count) is not int or count < 0:
                    raise ConfigError("metrics history has invalid rollout counter")
                post_checkpoint |= count > saved
            if post_checkpoint:
                newer = True
            else:
                kept.extend(line if line.endswith(b"\n") else line + b"\n")
            for key, previous in (("cost/elapsed_seconds", elapsed), ("cost/training_seconds", training)):
                value = row.get(key, previous)
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ConfigError("metrics history has invalid physical cost")
                if key == "cost/elapsed_seconds":
                    elapsed = max(elapsed, value)
                else:
                    training = max(training, value)

    for name in ("cost.json", "status.json"):
        path = root / name
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        fingerprint = record.get("config_sha256")
        if fingerprint is None and name == "status.json" and record.get("kind") == "software_fixture_not_paper_result":
            fingerprint = json.loads((root / "run.json").read_text()).get("config_sha256")
        if fingerprint != c.fingerprint():
            raise ConfigError("stored cost/status belongs to a different run/config")
        saved = Progress.from_state_dict(record["progress"] if name == "cost.json" else
                                        {key: record[key] for key in Progress.__dataclass_fields__})
        newer |= (saved.optimizer_steps > step or saved.attempts > progress.attempts
                  or saved.completed_rollouts > progress.completed_rollouts)
        elapsed, training = max(elapsed, saved.elapsed_seconds), max(training, saved.training_seconds)

    evaluations = []
    evaluation_root = root / "evaluation"
    if evaluation_root.exists():
        for directory in sorted(evaluation_root.glob("step-*")):
            suffix = directory.name.removeprefix("step-")
            if suffix.isdigit() and int(suffix) > step:
                if directory.is_symlink() or not directory.is_dir():
                    raise ConfigError("evaluation rollback requires regular directories")
                evaluations.append(directory)
    newer |= bool(evaluations)
    if not newer:
        progress.elapsed_seconds, progress.training_seconds = elapsed, training
        return
    if not rewind:
        raise ConfigError("history newer than checkpoint; use --resume-rewind to archive discarded work")
    if training > elapsed:
        raise ConfigError("stored training cost exceeds elapsed cost")

    archive_id = pending["archive"] if pending else uuid.uuid4().hex
    archive = root / "resume-history" / archive_id
    if pending:
        if archive.is_symlink() or not archive.is_dir():
            raise ConfigError("pending rewind archive is missing or unsafe")
    else:
        archive.mkdir(parents=True, exist_ok=False)
    # Finish evidence copies before changing any active records or media.
    for name in ("metrics.jsonl", "cost.json", "status.json", "evaluation/latest.json"):
        source = root / name
        if source.exists():
            if source.is_symlink() or not source.is_file():
                raise ConfigError("resume history must use regular files")
            target = archive / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copyfile(source, target)
                with target.open("rb") as file:
                    os.fsync(file.fileno())
    record = {"checkpoint": str(Path(receipt).absolute()), "optimizer_step": step,
              "config_sha256": c.fingerprint(), "elapsed_seconds": elapsed,
              "training_seconds": training, "state": "prepared", "archive": archive_id}
    _atomic_json(archive / "resume.json", record)
    _atomic_json(journal, record)
    for directory in evaluations:
        target = archive / "evaluation" / directory.name
        target.parent.mkdir(parents=True, exist_ok=True)
        directory.rename(target)
    if history.exists():
        temporary = archive / "active-metrics.jsonl"
        with temporary.open("wb") as file:
            file.write(kept)
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(history)
    state = progress.state_dict()
    state.update(elapsed_seconds=elapsed, training_seconds=training)
    _atomic_json(root / "cost.json", {"config_sha256": c.fingerprint(), "progress": state})
    (root / "status.json").unlink(missing_ok=True)
    latest = evaluation_root / "latest.json"
    if latest.exists() and json.loads(latest.read_text())["optimizer_step"] > step:
        latest.unlink()
    record["state"] = "completed"
    _atomic_json(archive / "resume.json", record)
    journal.unlink()
    progress.elapsed_seconds, progress.training_seconds = elapsed, training
