"""Isolated scalar W&B export; training JSONL and eval summaries stay authoritative."""

import argparse
import importlib.util
import json
import math
import netrc
import os
import select
import signal
import subprocess
import sys
import threading
import uuid
import warnings
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

from .config import ConfigError


def validate_credentials(*, mode=None):
    """Check local login before GPU work; never prompt or persist a token."""
    if os.environ.get("WANDB_MODE", mode or "online") in {"offline", "disabled", "dryrun"}:
        return
    if os.environ.get("WANDB_API_KEY"):
        return
    host = urlsplit(os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai")).hostname
    try:
        login = netrc.netrc(os.environ.get("NETRC")).authenticators(host)
        if login and login[2]:
            return
    except (OSError, netrc.NetrcParseError):
        pass
    raise ConfigError("W&B online logging requires WANDB_API_KEY or an existing wandb login; "
                      "set WANDB_MODE=offline or runtime.log=jsonl to opt out")


def _publish(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _owner_status(path, value):
    try:
        _publish(path, value)
    except OSError:
        warnings.warn("W&B owner status unavailable; local training metrics are authoritative",
                      RuntimeWarning)


def _deadline_guard(done, *, follow_fd, timeout):
    if follow_fd is not None:
        while not done.is_set():
            if select.select([follow_fd], [], [], .5)[0]:
                break
        else:
            return
    if not done.wait(timeout):
        # The exporter uses its own process group. A manual CLI invocation
        # must never kill its caller's shell group.
        if os.getpgrp() == os.getpid():
            os.killpg(os.getpgrp(), signal.SIGKILL)
        os._exit(124)


class Records:
    """Tail complete JSONL records and atomically published evaluation summaries."""

    def __init__(self, root):
        self.root = Path(root)
        self.offset = 0
        self.evaluations = set()

    def read(self):
        source = self.root / "metrics.jsonl"
        if source.exists():
            with source.open("rb") as stream:
                if source.stat().st_size < self.offset:
                    raise ValueError("metrics history was truncated during export")
                stream.seek(self.offset)
                while line := stream.readline():
                    if not line.endswith(b"\n"):
                        break
                    row = json.loads(line)
                    yield self._scalars(row)
                    self.offset = stream.tell()
        for path in sorted((self.root / "evaluation").glob("*/*/summary.json")):
            if path in self.evaluations:
                continue
            row = json.loads(path.read_text())
            if "eval/optimizer_step" not in row:
                raise ValueError("evaluation summary has no optimizer step")
            yield self._scalars(row)
            self.evaluations.add(path)

    @staticmethod
    def _scalars(row):
        if not isinstance(row, dict):
            raise ValueError("metrics must be an object")
        values = {key: value for key, value in row.items()
                  if type(value) in (int, float, bool)}
        if not values or any(not math.isfinite(value) for value in values.values()):
            raise ValueError("metrics must contain finite scalars")
        return values


def export(root, destination, *, follow_fd=None):
    """One SDK owner, one fresh run per export, no retry of unknown SDK writes."""
    root, destination = Path(root), Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    # Configure SDK storage before import/init. Do not inherit stale run IDs.
    for variable, directory in (
        ("WANDB_DIR", "run"), ("WANDB_CACHE_DIR", "cache"),
        ("WANDB_DATA_DIR", "data"), ("WANDB_CONFIG_DIR", "config"), ("TMPDIR", "tmp"),
    ):
        path = destination / directory
        path.mkdir()
        os.environ[variable] = str(path.resolve())
    for variable in ("WANDB_RUN_ID", "WANDB_RESUME", "WANDB_RUN_GROUP"):
        os.environ.pop(variable, None)
    import wandb

    config_path = root / "resume-config.json"
    if not config_path.exists():
        config_path = root / "resolved-config.json"
    config = json.loads(config_path.read_text())
    logging = config.get("runtime", {}).get("wandb", {})
    mode = os.environ.get("WANDB_MODE", logging.get("mode", "online"))
    record = {"state": "initializing", "source": str(root.resolve()), "records": 0}
    status = destination / "status.json"
    _publish(status, record)
    try:
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", logging.get("project", "visual-diffusion-rl")),
            entity=os.environ.get("WANDB_ENTITY", logging.get("entity")),
            name=os.environ.get("WANDB_NAME", logging.get("name", config["name"])),
            group=config["name"], id=uuid.uuid4().hex, resume="never",
            mode=mode, config=config,
            settings=wandb.Settings(init_timeout=30, console="off", disable_git=True,
                                    save_code=False),
        )
        run.define_metric("train/optimizer_step")
        run.define_metric("eval/optimizer_step")
        run.define_metric("train/*", step_metric="train/optimizer_step", step_sync=False)
        run.define_metric("cost/*", step_metric="train/optimizer_step", step_sync=False)
        run.define_metric("eval/*", step_metric="eval/optimizer_step", step_sync=False)
        record.update(state="running", run_id=run.id, url=run.url,
                      sdk_dir=run.dir, mode=mode)
        _publish(status, record)
        reader = Records(root)
        while True:
            # EOF is sent only after the parent's final metric has been flushed.
            done = follow_fd is None or bool(select.select([follow_fd], [], [], 0)[0])
            for row in reader.read():
                run.log(row)
                record["records"] += 1
            _publish(status, record)
            if done:
                break
            select.select([follow_fd], [], [], 0.5)
        record["state"] = "finishing"
        _publish(status, record)
        run.finish()
        record["state"] = "finished"
        _publish(status, record)
    except Exception as exc:
        # Exception text can contain credentials/URLs; keep only its class here.
        record.update(state="failed", error_type=type(exc).__name__)
        _publish(status, record)
        raise


@contextmanager
def open_metrics(config, root):
    """Keep the trainer's existing file API; only rank zero calls this factory."""
    root = Path(root)
    with (root / "metrics.jsonl").open("a", encoding="utf-8") as stream:
        if config.runtime.log == "jsonl":
            yield stream
            return
        if importlib.util.find_spec("wandb") is None:
            raise RuntimeError("W&B logging requires installing the 'wandb' extra")
        identity = uuid.uuid4().hex
        destination = root / "wandb-exports" / identity
        owner = destination.parent / f"{identity}.owner.json"
        command = [sys.executable, "-m", "visual_rl.wandb_export", "--root", str(root.resolve()),
                   "--destination", str(destination.resolve()), "--follow-stdin"]
        process = None
        try:
            destination.parent.mkdir(exist_ok=True)
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL, start_new_session=True,
                                       env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
            _owner_status(owner, {"state": "running", "pid": process.pid,
                                  "destination": str(destination.resolve())})
        except OSError:
            warnings.warn("W&B exporter could not start; local metrics remain available", RuntimeWarning)
        try:
            yield stream
        finally:
            # File.__exit__ still reports canonical-log failure, but it cannot
            # prevent the independent SDK owner from receiving its stop signal.
            if process is not None:
                try:
                    stream.flush()
                except OSError:
                    pass
                process.stdin.close()
                try:
                    code = process.wait(timeout=10)
                    state = "exited" if code == 0 else "pending_or_unknown" if code < 0 else "failed"
                except subprocess.TimeoutExpired:
                    state, code = "pending_or_unknown", None
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        pass
                _owner_status(owner, {"state": state, "exit_code": code,
                                      "destination": str(destination.resolve())})
                if state != "exited":
                    warnings.warn("W&B export incomplete; replay local metrics using visual_rl.wandb_export",
                                  RuntimeWarning)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--follow-stdin", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=120,
                        help="standalone export wall-clock deadline; live mode drains for 10s after EOF")
    args = parser.parse_args()
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        parser.error("timeout must be positive and finite")
    done = threading.Event()
    follow_fd = 0 if args.follow_stdin else None
    watcher = threading.Thread(target=_deadline_guard, kwargs={"done": done,
        "follow_fd": follow_fd, "timeout": 10 if args.follow_stdin else args.timeout_seconds}, daemon=True)
    watcher.start()
    # Leave the daemon armed through interpreter/SDK atexit cleanup as well.
    export(args.root, args.destination, follow_fd=follow_fd)


if __name__ == "__main__":
    main()
