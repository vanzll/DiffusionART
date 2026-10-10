"""Persistent CPU OCR interpreter with bounded, single-request pipe transport."""

import base64
import json
import sys

import numpy as np

from .process import JsonWorker

MAX_RGB_BYTES = 64 * 1024 * 1024
MAX_RESPONSE_BYTES = 2 * 1024 * 1024


class OCRInfrastructureError(RuntimeError):
    """Worker/transport failures must not become ordinary zero OCR rewards."""


class PaddleWorker(JsonWorker):
    def __init__(
        self, manifest_path, manifest_sha256, *, startup_timeout=120.0, request_timeout=60.0
    ):
        super().__init__(
            [sys.executable, "-m", "visual_rl.rewards.paddle_worker"],
            [manifest_path, manifest_sha256], startup_timeout=startup_timeout,
            request_timeout=request_timeout, error_type=OCRInfrastructureError,
            label="OCR", response_limit=MAX_RESPONSE_BYTES,
        )

    def ocr(self, image, *, cls):
        if (
            cls is not False
            or not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
            or min(image.shape) < 1
            or image.nbytes > MAX_RGB_BYTES
        ):
            raise ValueError("worker requires bounded uint8 HWC RGB with cls=False")
        with self._lock:
            payload = (
                json.dumps(
                    {"shape": image.shape, "rgb": base64.b64encode(image.tobytes()).decode("ascii")}
                ).encode()
                + b"\n"
            )
            try:
                reply = self._exchange(payload, self.timeout)
                status = reply.get("status")
                if status not in ("ok", "engine_error"):
                    raise OCRInfrastructureError("OCR worker reported a protocol failure")
                if (status == "ok" and set(reply) != {"status", "result"}) or (
                    status == "engine_error"
                    and (
                        set(reply) != {"status", "error_type"}
                        or not isinstance(reply["error_type"], str)
                    )
                ):
                    raise OCRInfrastructureError("OCR worker returned an invalid result envelope")
            except Exception as exc:
                self.close()
                if isinstance(exc, OCRInfrastructureError):
                    raise
                raise OCRInfrastructureError("OCR worker transport failed") from exc
            if status == "engine_error":
                raise RuntimeError(f"OCR native engine error: {reply.get('error_type', 'unknown')}")
            return reply["result"]
