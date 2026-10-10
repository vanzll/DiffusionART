"""Actual process/pipe tests using a fake native module, never real Paddle."""

import os
import sys
import time
from types import ModuleType

import numpy as np
import pytest
from test_paddle_loader import bundle as bundle

from visual_rl.rewards.ocr_process import OCRInfrastructureError, PaddleWorker
from visual_rl.rewards.paddle_loader import VERSIONS


@pytest.fixture
def native_module(tmp_path, monkeypatch):
    directory = tmp_path / "native"
    directory.mkdir()
    for name, version in VERSIONS.items():
        metadata = directory / f"{name}-{version}.dist-info"
        metadata.mkdir()
        (metadata / "METADATA").write_text(f"Name: {name}\nVersion: {version}\n")
    (directory / "paddleocr.py").write_text("""
import os, sys, time, types
import numpy as np
sys.modules["tools"] = types.ModuleType("tools")
sys.path.insert(0, "/fake-paddle-import-path")
print("native library stdout must not corrupt the protocol", flush=True)
class PaddleOCR:
    def __init__(self, **kwargs):
        if os.environ.get("OCR_TEST_INIT_FAIL"):
            raise ValueError("fixture init failure")
        assert kwargs["use_gpu"] is False
        self.count = 0
    def ocr(self, image, cls):
        assert cls is False and image.dtype == np.uint8
        if image[0, 0, 0] == 255:
            time.sleep(30)
        if image[0, 0, 0] == 254:
            raise ValueError("fixture recognition failure")
        if image[0, 0, 0] == 253:
            os._exit(7)
        if image[0, 0, 0] == 252:
            return [[[[0, 0], ("ignored", np.float32("nan"))], [[0, 0], ("ok", 1.0)]]]
        self.count += 1
        return [[[[0, 0], (str(self.count), np.float32(0.9))]]]
""")
    monkeypatch.setenv("PYTHONPATH", str(directory) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    return directory


def test_persistent_worker_preserves_parent_imports_rgb_and_order(
    bundle, native_module, monkeypatch
):
    previous = ModuleType("tools")
    monkeypatch.setitem(sys.modules, "tools", previous)
    paths = list(sys.path)
    worker = PaddleWorker(str(bundle[0]), bundle[1], startup_timeout=10)
    process = worker.process
    try:
        for count in (1, 2, 3):
            value = worker.ocr(np.zeros((2, 3, 3), dtype=np.uint8), cls=False)
            assert value[0][0][1][0] == str(count)
        assert sys.modules["tools"] is previous and sys.path == paths
    finally:
        worker.close()
    assert process.poll() is not None
    worker.close()
    with pytest.raises(OCRInfrastructureError):
        worker.ocr(np.zeros((2, 2, 3), dtype=np.uint8), cls=False)


@pytest.mark.parametrize("pixel", [253, 255])
def test_dead_or_hung_worker_is_bounded_fatal_and_not_restarted(bundle, native_module, pixel):
    worker = PaddleWorker(str(bundle[0]), bundle[1], startup_timeout=10, request_timeout=0.2)
    start = time.monotonic()
    with pytest.raises(OCRInfrastructureError):
        worker.ocr(np.full((2, 2, 3), pixel, dtype=np.uint8), cls=False)
    assert time.monotonic() - start < 3
    assert worker.process.poll() is not None


def test_engine_exception_allows_next_request_without_worker_replacement(bundle, native_module):
    worker = PaddleWorker(str(bundle[0]), bundle[1], startup_timeout=10)
    pid = worker.process.pid
    try:
        with pytest.raises(RuntimeError, match="native engine error"):
            worker.ocr(np.full((2, 2, 3), 254, dtype=np.uint8), cls=False)
        assert worker.ocr(np.zeros((2, 2, 3), dtype=np.uint8), cls=False)[0][0][1][0] == "1"
        assert worker.process.pid == pid
    finally:
        worker.close()


def test_initialization_failure_is_not_accepted(bundle, native_module, monkeypatch):
    monkeypatch.setenv("OCR_TEST_INIT_FAIL", "1")
    with pytest.raises(OCRInfrastructureError, match="initialization failed"):
        PaddleWorker(str(bundle[0]), bundle[1], startup_timeout=10)


def test_zero_error_policy_does_not_swallow_worker_failure():
    pytest.importorskip("rapidfuzz")
    import torch

    from visual_rl.rewards.ocr import OCRScore

    class Dead:
        def ocr(self, *a, **kw):
            raise OCRInfrastructureError("dead")

    scorer = OCRScore(Dead(), error_policy="zero")
    with pytest.raises(OCRInfrastructureError, match="dead"):
        scorer(torch.zeros(1, 3, 2, 2), ['"word"'], microbatch=1)


def test_nan_confidence_keeps_historical_score_and_worker_reusable(bundle, native_module):
    pytest.importorskip("rapidfuzz")
    import torch

    from visual_rl.rewards.ocr import OCRScore

    worker = PaddleWorker(str(bundle[0]), bundle[1], startup_timeout=10)
    scorer = OCRScore(worker, error_policy="raise")
    try:
        result = scorer(torch.full((1, 3, 2, 2), 252 / 255), ['"ok"'], microbatch=1)
        assert result.raw["ocr"].item() == 1
        assert worker.ocr(np.zeros((2, 2, 3), dtype=np.uint8), cls=False)[0][0][1][0] == "1"
    finally:
        scorer.close()
