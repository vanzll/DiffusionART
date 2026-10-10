import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

from visual_rl.rewards.geneval import GenEvalScore
from visual_rl.rewards.geneval_process import GenEvalInfrastructureError, GenEvalWorker


@pytest.fixture
def native_worker(tmp_path, monkeypatch):
    """Actual isolated subprocess/FD transport, with only the native loader replaced."""
    from visual_rl.rewards import geneval_worker, process

    script = tmp_path / "worker.py"
    script.write_text(f'''
import sys, time, os, runpy, subprocess
from pathlib import Path
sys.path.insert(0, {str(Path(geneval_worker.__file__).resolve().parents[2])!r})
from visual_rl.rewards import geneval_loader
def load(manifest, digest, **kwargs):
    if manifest == "fail":
        raise ValueError("initialization failure")
    print("native stdout is not the protocol", flush=True)
    def score(images, metadata, only_strict):
        assert all(image.mode == "RGB" for image in images)
        pixel = images[0].getpixel((0, 0))[0]
        if pixel in (250, 251, 252):
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
            Path(manifest).write_text(str(child.pid))
            if pixel == 250: time.sleep(30)
            if pixel == 251: os._exit(7)
            raise ValueError("parent error with active child")
        if pixel == 255: time.sleep(30)
        if pixel == 254: raise ValueError("native failure")
        if pixel == 253: os._exit(7)
        n = len(images)
        scores, ordinary, strict = [float(pixel)] * n, [0. if only_strict else 1.] * n, [1.] * n
        tags = ("single_object", "two_object", "counting", "colors", "position", "color_attr")
        grouped = {{tag: [x if row["tag"] == tag else -10. for x,row in zip(ordinary,metadata)] for tag in tags}}
        grouped_strict = {{tag: [x if row["tag"] == tag else -10. for x,row in zip(strict,metadata)] for tag in tags}}
        return scores,ordinary,strict,grouped,grouped_strict
    return score
geneval_loader._load_native_engine = load
runpy.run_path({geneval_worker.__file__!r}, run_name="__main__")
''')
    original = process.subprocess.Popen

    def launch(command, **kwargs):
        if command[:3] != [sys.executable, "-I", "-u"]:
            return original(command, **kwargs)
        assert command[:3] == [sys.executable, "-I", "-u"]
        return original([*command[:3], str(script), *command[4:]], **kwargs)

    monkeypatch.setattr(process.subprocess, "Popen", launch)


def worker(manifest="ok", **kwargs):
    return GenEvalWorker(manifest, "a" * 64, python=sys.executable, device="cpu",
                         startup_timeout=10, **kwargs)


def call(engine, pixel=0, evaluation=False):
    return GenEvalScore(engine)(torch.full((1, 3, 8, 8), pixel / 255.), ["one"],
                               microbatch=1, evaluation=evaluation,
                               metadata=[{"prompt": "one", "tag": "single_object",
                                          "include": [{"class": "cat", "count": 1}]}])


def test_persistent_isolated_worker_scores_jpeg_and_closes(native_worker):
    engine = worker()
    pid = engine.process.pid
    try:
        assert call(engine, 20).weighted.item() == 20
        result = call(engine, 40, evaluation=True)
        assert result.weighted.item() == 40
        assert result.diagnostics["geneval/accuracy"].item() == 1
        assert engine.process.pid == pid
    finally:
        engine.close()
    assert engine.process.poll() is not None
    engine.close()
    with pytest.raises(GenEvalInfrastructureError):
        call(engine)


@pytest.mark.parametrize("pixel", [253, 254, 255])
def test_failure_does_not_become_zero_or_replace_worker(native_worker, pixel):
    engine = worker(request_timeout=.2)
    begin = time.monotonic()
    with pytest.raises(GenEvalInfrastructureError):
        call(engine, pixel)
    assert time.monotonic() - begin < 3
    assert engine.process.poll() is not None


def test_startup_failure_is_reported(native_worker):
    with pytest.raises(GenEvalInfrastructureError, match="initialization failed: ValueError: initialization failure"):
        worker("fail")


@pytest.mark.parametrize("pixel", [250, 251, 252])
def test_worker_cleanup_includes_descendants_after_hang_exit_or_error(native_worker, tmp_path, pixel):
    marker = tmp_path / "child-pid"
    engine = worker(str(marker), request_timeout=.4)
    assert os.getpgid(engine.process.pid) == engine.process.pid
    assert engine.process.pid != os.getpgrp()
    with pytest.raises(GenEvalInfrastructureError):
        call(engine, pixel)
    pid = int(marker.read_text())
    try:
        deadline = time.monotonic() + 2
        while True:
            state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)],
                                   capture_output=True, text=True, timeout=1).stdout.strip()
            if not state or state.startswith("Z"):
                break
            if time.monotonic() > deadline:
                pytest.fail("reward worker descendant survived cleanup")
            time.sleep(.02)
    finally:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
