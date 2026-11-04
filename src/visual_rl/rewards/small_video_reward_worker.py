"""Standalone canonical-reward interpreter entrypoint; stdlib-only bootstrap."""

import argparse
import ctypes
import importlib.metadata
import json
import math
import os
import re
import signal
import sys
import tempfile
import time
import traceback
from pathlib import Path

JSON_LIMIT = 2 * 1024 * 1024
REVISION = "4f26600130683e6f1de9f5d463887f28e8ef995c"
SOURCE_REVISION = "15cc71d53cc2e6e18a68ee607d5fb6ba9a99e344"
COMPONENTS = ("VQ", "MQ", "TA")


def parent_death_guard(expected_parent):
    """Before any CUDA/third-party import, including across optional bashrc exec."""
    if not sys.platform.startswith("linux"):
        raise RuntimeError("canonical phased reward requires Linux parent-death fencing")
    library = ctypes.CDLL(None, use_errno=True)
    if library.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "cannot install parent-death guard")
    if os.getppid() != expected_parent:
        os._exit(125)


def atomic_json(path, value):
    encoded = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    if len(encoded) > JSON_LIMIT:
        raise ValueError("worker JSON exceeds byte bound")
    fd, pending = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, path)
    finally:
        Path(pending).unlink(missing_ok=True)


def execute(request):
    start = time.monotonic()
    # Only now may canonical environment Torch/Transformers/CUDA be imported.
    import torch

    from visual_rl.models.source_pin import import_pinned_source
    from visual_rl.rewards.video_reward import VideoReward

    if importlib.metadata.version("transformers") != "4.46.1":
        raise RuntimeError("canonical reward interpreter must pin Transformers4.46.1")
    import_pinned_source(request["native_root"], package="fastvideo", revision=SOURCE_REVISION)
    from fastvideo.models.videoalign.inference import VideoVLMRewardInference
    from fastvideo.models.videoalign.prompt_template import build_prompt

    device = torch.device(request["device"])
    if device.type == "cuda":
        if torch.cuda.device_count() != 1:
            raise RuntimeError("reward worker must see exactly its parent rank GPU")
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(0)
    phase = Path(request["scratch"])
    previous = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="native-load-", dir=phase) as temporary:
        (Path(temporary) / "Qwen2-VL-2B-Instruct").symlink_to(
            Path(request["backbone"]).resolve(strict=True), target_is_directory=True
        )
        try:
            os.chdir(temporary)
            inferencer = VideoVLMRewardInference(
                str(Path(request["checkpoint"]).resolve(strict=True)),
                device="cpu",
                dtype=torch.bfloat16,
            )
        finally:
            os.chdir(previous)
    scorer = VideoReward(
        inferencer,
        build_prompt,
        weights={"VQ": 0.5, "MQ": 0.5, "TA": 0.0},
        normalization_source=f"VideoReward@{REVISION}",
        space="raw",
        video_fps=request["video_fps"],
        reward_fps=4.0,
        ffprobe=request["ffprobe"],
    )
    scorer.to(device).eval()
    if device.type == "cuda":
        torch.cuda.synchronize(0)
    load_complete = time.monotonic()
    nvml_load = _nvml_used(torch, device)
    score_start = time.monotonic()
    with torch.no_grad():
        scores = scorer.score_paths(request["paths"], request["prompts"])
    rows = []
    for index in range(len(request["paths"])):
        row = {"index": index}
        for key in COMPONENTS:
            value = float(scores.raw[key][index])
            if not math.isfinite(value):
                raise ValueError("canonical VideoReward returned a nonfinite component")
            row[key] = value
        rows.append(row)
    metrics = {"reward/worker_peak_allocated_bytes": 0, "reward/worker_peak_reserved_bytes": 0}
    if device.type == "cuda":
        torch.cuda.synchronize(0)
        metrics.update(
            {
                "reward/worker_peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                "reward/worker_peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
            }
        )
    score_complete = time.monotonic()
    nvml_score = _nvml_used(torch, device)
    metrics.update(
        {
            "reward/load_seconds": load_complete - start,
            "reward/scoring_seconds": score_complete - score_start,
            "reward/worker_total_seconds": time.monotonic() - start,
            "memory/scorer_peak_allocated_bytes": metrics.pop("reward/worker_peak_allocated_bytes"),
            "memory/scorer_peak_reserved_bytes": metrics.pop("reward/worker_peak_reserved_bytes"),
        }
    )
    if nvml_load is not None:
        metrics["memory/scorer_nvml_total_used_after_load_bytes"] = nvml_load
    if nvml_score is not None:
        metrics["memory/scorer_nvml_total_used_after_score_bytes"] = nvml_score
    runtime = {"python": sys.version.split()[0]}
    for distribution in ("torch", "transformers", "flash-attn", "accelerate", "diffusers"):
        runtime[distribution] = importlib.metadata.version(distribution)
    runtime["native_source_revision"], runtime["reward_revision"] = SOURCE_REVISION, REVISION
    return {
        "schema_version": 1,
        "request_id": request["request_id"],
        "rows": rows,
        "normalization": {
            key: {"mean": value.mean, "std": value.std, "source": value.source}
            for key, value in scores.normalization.items()
        },
        "metrics": metrics,
        "runtime": runtime,
    }


def _nvml_used(torch, device):
    """Actual total-device snapshots, explicitly not a sampled NVML peak."""
    if device.type != "cuda":
        return None
    try:
        identifier = str(torch.cuda.get_device_properties(0).uuid)
        if not identifier.startswith(("GPU-", "MIG-")):
            return None
        library = ctypes.CDLL("libnvidia-ml.so.1")
        if library.nvmlInit_v2() != 0:
            return None
        try:
            handle = ctypes.c_void_p()
            if library.nvmlDeviceGetHandleByUUID(identifier.encode(), ctypes.byref(handle)) != 0:
                return None

            class Memory(ctypes.Structure):
                _fields_ = [(name, ctypes.c_ulonglong) for name in ("total", "free", "used")]

            memory = Memory()
            if library.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)) != 0:
                return None
            return int(memory.used)
        finally:
            library.nvmlShutdown()
    except (OSError, AttributeError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--bashrc")
    args = parser.parse_args()
    parent_death_guard(args.parent_pid)
    if args.bashrc is not None:
        # Positional arguments remain literal; no shell interpolation of paths.
        os.execv(
            "/bin/bash",
            [
                "bash",
                "--noprofile",
                "--norc",
                "-c",
                'set +x; source "$1" >/dev/null 2>&1 && set +x && exec "$2" -u "$3" --input "$4" --output "$5" --parent-pid "$6"',
                "small-video-reward",
                args.bashrc,
                sys.executable,
                str(Path(__file__).resolve()),
                args.input,
                args.output,
                str(args.parent_pid),
            ],
        )
    path, output = Path(args.input).resolve(strict=True), Path(args.output).absolute()
    if (
        path.is_symlink()
        or path.stat().st_size > JSON_LIMIT
        or output.parent.resolve() != path.parent
    ):
        raise ValueError("worker requires bounded colocated request/response files")
    request = json.loads(path.read_text())
    expected = {
        "schema_version",
        "request_id",
        "paths",
        "prompts",
        "native_root",
        "checkpoint",
        "backbone",
        "revision",
        "source_revision",
        "weights",
        "video_fps",
        "reward_fps",
        "ffprobe",
        "scratch",
        "visible_device",
        "device",
    }
    if (
        not isinstance(request, dict)
        or set(request) != expected
        or request.get("schema_version") != 1
        or request.get("revision") != REVISION
        or request.get("source_revision") != SOURCE_REVISION
        or not isinstance(request.get("request_id"), str)
        or not re.fullmatch(r"[0-9a-f]{32}", request["request_id"])
        or request.get("weights") != {"VQ": 0.5, "MQ": 0.5, "TA": 0.0}
        or request.get("reward_fps") != 4.0
        or request.get("device") not in ("cpu", "cuda:0")
        or not isinstance(request.get("paths"), list)
        or not 0 < len(request["paths"]) <= 1024
        or not isinstance(request.get("prompts"), list)
        or len(request["paths"]) != len(request["prompts"])
        or any(not isinstance(prompt, str) or not prompt.strip() for prompt in request["prompts"])
        or any(
            not isinstance(video, str) or not Path(video).is_file() for video in request["paths"]
        )
        or any(
            not isinstance(request.get(key), str) or not Path(request[key]).is_dir()
            for key in ("native_root", "checkpoint", "backbone")
        )
        or type(request.get("video_fps")) not in (int, float)
        or not math.isfinite(request["video_fps"])
        or request["video_fps"] <= 0
        or not isinstance(request.get("ffprobe"), str)
        or not request["ffprobe"]
        or not isinstance(request.get("visible_device"), str)
        or "," in request["visible_device"]
        or (request["device"] == "cpu" and request["visible_device"] != "")
        or (request["device"] == "cuda:0" and request["visible_device"] in ("", "-1"))
        or Path(request["scratch"]).resolve() != path.parent
    ):
        raise ValueError("canonical worker request contract differs")
    # Optional bashrc cannot reintroduce torchrun bootstrap or widen device visibility.
    for key in tuple(os.environ):
        if key in {
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
        } or key.startswith(
            (
                "TORCHELASTIC_",
                "ACCELERATE_",
                "TORCH_DISTRIBUTED_",
                "PMI_",
                "PMIX_",
                "OMPI_COMM_WORLD_",
                "MV2_",
                "MPI_",
            )
        ):
            os.environ.pop(key)
    os.environ.update(
        CUDA_VISIBLE_DEVICES=request["visible_device"],
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
    )
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    stat = Path(f"/proc/{os.getpid()}/stat").read_text()
    fields = stat[stat.rfind(")") + 2 :].split()
    process = {
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "pgid": os.getpgrp(),
        "start_ticks": int(fields[19]),
        "visible_device": os.environ["CUDA_VISIBLE_DEVICES"],
    }
    atomic_json(path.parent / "worker-start.json", process)
    result = execute(request)
    result["process"] = process
    atomic_json(output, result)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
