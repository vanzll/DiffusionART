"""One-shot, same-GPU canonical VideoReward phases in a pinned interpreter."""

import json
import math
import os
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

import torch

from .scalar import FixedNormalization, weighted_score

REWARD_REVISION = "4f26600130683e6f1de9f5d463887f28e8ef995c"
REWARD_SOURCE_REVISION = "15cc71d53cc2e6e18a68ee607d5fb6ba9a99e344"
WORKER_SCRIPT = Path(__file__).with_name("small_video_reward_worker.py")
COMPONENTS = ("VQ", "MQ", "TA")
WEIGHTS = {"VQ": 0.5, "MQ": 0.5, "TA": 0.0}
JSON_LIMIT = 2 * 1024 * 1024
LOG_LIMIT = 8 * 1024 * 1024
_BOOTSTRAP = {
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "GROUP_RANK",
    "GROUP_WORLD_SIZE",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "ROLE_NAME",
    "NODE_RANK",
}
_PREFIXES = (
    "TORCHELASTIC_",
    "ACCELERATE_",
    "TORCH_DISTRIBUTED_",
    "PMI_",
    "PMIX_",
    "OMPI_COMM_WORLD_",
    "MV2_",
    "MPI_",
)


class RewardProcessError(RuntimeError):
    pass


def atomic_json(path, value):
    data = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    if len(data) > JSON_LIMIT:
        raise ValueError("reward JSON exceeds its bounded transport limit")
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def process_identity(pid):
    # /proc command names may contain spaces and parentheses.
    stat = Path(f"/proc/{pid}/stat").read_text()
    fields = stat[stat.rfind(")") + 2 :].split()
    return {
        "pid": pid,
        "ppid": int(fields[1]),
        "pgid": int(fields[2]),
        "start_ticks": int(fields[19]),
    }


def physical_visible_device(device, *, environ=None):
    environ = os.environ if environ is None else environ
    device = torch.device(device)
    if device.type == "cpu":
        return ""
    if device.type != "cuda":
        raise ValueError("phased VideoReward supports a parent CPU or CUDA device only")
    index = device.index if device.index is not None else torch.cuda.current_device()
    visible = environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return str(index)
    devices = [part.strip() for part in visible.split(",")]
    if index < 0 or index >= len(devices) or devices[index] in ("", "-1"):
        raise ValueError("parent CUDA rank has no declared visible physical GPU")
    return devices[index]


def worker_environment(*, visible_device, phase, environ=None):
    source = os.environ if environ is None else environ
    env = {
        key: value
        for key, value in source.items()
        if key not in _BOOTSTRAP and not key.startswith(_PREFIXES)
    }
    phase = Path(phase)
    env.update(
        CUDA_VISIBLE_DEVICES=visible_device,
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        TOKENIZERS_PARALLELISM="false",
        PYTHONUNBUFFERED="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[2]),
        TMPDIR=str(phase),
        HF_HOME=str(phase / "hf-cache"),
        XDG_CACHE_HOME=str(phase / "cache"),
        TORCH_HOME=str(phase / "torch-cache"),
        CUDA_CACHE_PATH=str(phase / "cuda-cache"),
        TORCHINDUCTOR_CACHE_DIR=str(phase / "torchinductor"),
        TRITON_CACHE_DIR=str(phase / "triton"),
        HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
    )
    return env


def _drain(pipe, logfile, record):
    kept = 0
    with pipe, logfile.open("wb") as stream:
        while True:
            chunk = pipe.read(65536)
            if not chunk:
                return
            remaining = max(0, LOG_LIMIT - kept)
            stream.write(chunk[:remaining])
            kept += min(len(chunk), remaining)
            if len(chunk) > remaining:
                record["log_truncated"] = True


def _stop_owned_child(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)
    else:
        process.wait()


class PhasedVideoReward:
    """No persistent model/service. Return scores only after this phase exits.

    The parent policy/FSDP shards may remain resident; the worker is restricted
    to this rank's existing physical GPU, never an additional scorer device.
    """

    def __init__(
        self,
        *,
        python,
        native_root,
        checkpoint,
        backbone,
        revision,
        video_fps,
        reward_fps=4.0,
        ffprobe="ffprobe",
        scratch,
        timeout=900.0,
        bashrc=None,
    ):
        if revision != REWARD_REVISION:
            raise ValueError("phased scorer requires the immutable canonical VideoReward revision")
        if (
            type(timeout) not in (int, float)
            or not math.isfinite(timeout)
            or not 0 < timeout <= 900
            or type(video_fps) not in (float, int)
            or not math.isfinite(video_fps)
            or video_fps <= 0
            or reward_fps != 4.0
        ):
            raise ValueError(
                "phased scorer requires bounded timeout, finite native FPS and reward4fps"
            )
        self.python = Path(python).expanduser().absolute()
        if not self.python.is_file() or not os.access(self.python, os.X_OK):
            raise ValueError("canonical reward interpreter must be an existing executable")
        self.native_root, self.checkpoint, self.backbone = [
            Path(path).resolve(strict=True) for path in (native_root, checkpoint, backbone)
        ]
        if not all(path.is_dir() for path in (self.native_root, self.checkpoint, self.backbone)):
            raise ValueError(
                "canonical reward source/checkpoint/backbone must be existing directories"
            )
        self.scratch = Path(scratch).absolute()
        if any(path.is_symlink() for path in (self.scratch, *self.scratch.parents)):
            raise ValueError("reward phase scratch must be an owned nonsymlink directory")
        self.scratch.mkdir(parents=True, exist_ok=True)
        self.timeout, self.video_fps, self.reward_fps = float(timeout), float(video_fps), 4.0
        self.revision, self.ffprobe = revision, str(ffprobe)
        self.bashrc = None if bashrc is None else str(Path(bashrc).resolve(strict=True))
        self.device, self.weights, self.space = torch.device("cpu"), dict(WEIGHTS), "raw"
        self.normalization, self.last_phase, self.last_metrics = None, None, {}
        self._lock = threading.Lock()

    def to(self, device):
        with self._lock:
            target = torch.device(device)
            if target.type not in ("cpu", "cuda"):
                raise ValueError("phased scorer supports CPU/CUDA placement only")
            self.device = target
        return self

    def eval(self):
        return self

    def score_paths(self, paths, prompts):
        with self._lock:
            return self._score_paths(paths, prompts)

    def _score_paths(self, paths, prompts):
        if (
            not paths
            or len(paths) != len(prompts)
            or len(paths) > 1024
            or any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts)
        ):
            raise ValueError("phased reward requires matching nonempty ordered paths/prompts")
        paths = [str(Path(path).resolve(strict=True)) for path in paths]
        if any(not Path(path).is_file() for path in paths):
            raise ValueError("reward videos must be regular local files")
        token = uuid.uuid4().hex
        phase = self.scratch / f"phase-{token}"
        phase.mkdir(exist_ok=False)
        self.last_phase = phase
        visible = physical_visible_device(self.device)
        request = {
            "schema_version": 1,
            "request_id": token,
            "paths": paths,
            "prompts": list(prompts),
            "native_root": str(self.native_root),
            "checkpoint": str(self.checkpoint),
            "backbone": str(self.backbone),
            "revision": self.revision,
            "source_revision": REWARD_SOURCE_REVISION,
            "weights": WEIGHTS,
            "video_fps": self.video_fps,
            "reward_fps": 4.0,
            "ffprobe": self.ffprobe,
            "scratch": str(phase),
            "visible_device": visible,
            "device": "cuda:0" if self.device.type == "cuda" else "cpu",
        }
        atomic_json(phase / "request.json", request)
        command = [
            str(self.python),
            "-u",
            str(WORKER_SCRIPT),
            "--input",
            str(phase / "request.json"),
            "--output",
            str(phase / "response.json"),
            "--parent-pid",
            str(os.getpid()),
        ]
        if self.bashrc is not None:
            command.extend(["--bashrc", self.bashrc])
        audit = {
            "schema_version": 1,
            "request_id": token,
            "status": "starting",
            "parent_pid": os.getpid(),
            "parent_pgid": os.getpgrp(),
            "parent_logical_device": str(self.device),
            "visible_device": visible,
            "timeout_seconds": self.timeout,
            "log_truncated": False,
        }
        process, reader = None, None
        start = time.monotonic()
        resident = {
            "memory/scorer_parent_resident_allocated_bytes": (
                torch.cuda.memory_allocated(self.device) if self.device.type == "cuda" else 0
            ),
            "memory/scorer_parent_resident_reserved_bytes": (
                torch.cuda.memory_reserved(self.device) if self.device.type == "cuda" else 0
            ),
        }
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=worker_environment(visible_device=visible, phase=phase),
                start_new_session=False,
            )
            try:
                audit["child"] = process_identity(process.pid)
            except FileNotFoundError:
                audit["child"] = {"pid": process.pid, "start_ticks": None}
            atomic_json(phase / "audit.json", audit)
            reader = threading.Thread(
                target=_drain,
                args=(process.stdout, phase / "worker.log", audit),
                name="small-video-reward-log",
                daemon=True,
            )
            reader.start()
            code = process.wait(timeout=self.timeout)
            reader.join(timeout=3)
            if code or reader.is_alive():
                raise RewardProcessError("canonical reward phase failed")
            output = phase / "response.json"
            if output.is_symlink() or not output.is_file() or output.stat().st_size > JSON_LIMIT:
                raise RewardProcessError("canonical reward phase response is invalid")
            result = json.loads(output.read_text())
            if (
                result.get("schema_version") != 1
                or result.get("request_id") != token
                or set(result.get("normalization", {})) != set(COMPONENTS)
                or not isinstance(result.get("rows"), list)
                or len(result["rows"]) != len(paths)
            ):
                raise RewardProcessError("canonical reward phase response identity differs")
            child = result.get("process", {})
            if (
                child.get("pid") != process.pid
                or child.get("ppid") != os.getpid()
                or child.get("pgid") != os.getpgrp()
                or child.get("visible_device") != visible
                or type(child.get("start_ticks")) is not int
                or audit["child"].get("start_ticks") not in (None, child["start_ticks"])
            ):
                raise RewardProcessError("canonical reward phase child ownership differs")
            audit["child"] = {key: child[key] for key in ("pid", "ppid", "pgid", "start_ticks")}
            for index, row in enumerate(result["rows"]):
                if set(row) != {"index", *COMPONENTS} or row["index"] != index:
                    raise RewardProcessError("canonical reward phase row order differs")
                for key in COMPONENTS:
                    if type(row[key]) not in (int, float) or not math.isfinite(row[key]):
                        raise RewardProcessError(
                            "canonical reward phase returned a nonfinite/missing component"
                        )
            normalization = {
                key: FixedNormalization(**value) for key, value in result["normalization"].items()
            }
            if any(
                value.source != f"VideoReward@{self.revision}" for value in normalization.values()
            ):
                raise RewardProcessError("canonical reward normalization provenance differs")
            metrics, runtime = result.get("metrics"), result.get("runtime")
            required_metrics = {
                "reward/load_seconds",
                "reward/scoring_seconds",
                "reward/worker_total_seconds",
                "memory/scorer_peak_allocated_bytes",
                "memory/scorer_peak_reserved_bytes",
            }
            if (
                not isinstance(metrics, dict)
                or not required_metrics <= set(metrics)
                or any(
                    not key.startswith(("reward/", "memory/scorer_"))
                    or type(value) not in (int, float)
                    or not math.isfinite(value)
                    or value < 0
                    for key, value in metrics.items()
                )
                or not isinstance(runtime, dict)
                or runtime.get("native_source_revision") != REWARD_SOURCE_REVISION
                or runtime.get("reward_revision") != self.revision
                or runtime.get("transformers") != "4.46.1"
            ):
                raise RewardProcessError(
                    "canonical reward environment/telemetry provenance differs"
                )
            raw = {
                key: torch.tensor([row[key] for row in result["rows"]], dtype=torch.float32)
                for key in COMPONENTS
            }
            scores = weighted_score(raw, WEIGHTS, space="raw", normalization=normalization)
            self.normalization = normalization
            self.last_metrics = {
                "reward/phase_seconds": time.monotonic() - start,
                **resident,
                **metrics,
            }
            audit.update(
                status="completed",
                exit_code=code,
                elapsed_seconds=time.monotonic() - start,
                worker_process=result.get("process"),
                runtime=result.get("runtime"),
                metrics=self.last_metrics,
            )
            atomic_json(phase / "audit.json", audit)
            return scores
        except BaseException as exc:
            if process is not None:
                _stop_owned_child(process)
            if reader is not None:
                reader.join(timeout=3)
            audit.update(
                status="failed",
                error_type=type(exc).__name__,
                elapsed_seconds=time.monotonic() - start,
                exit_code=None if process is None else process.returncode,
            )
            atomic_json(phase / "audit.json", audit)
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise RewardProcessError(
                "canonical VideoReward phase failed; inspect its owned phase audit"
            ) from None
