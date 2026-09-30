"""Isolated GenEval interpreter; legacy CUDA dependencies never enter the trainer."""

import base64
import json
from pathlib import Path

from .process import JsonWorker

MAX_JPEG_BYTES = 64 * 1024 * 1024
MAX_REQUEST_BYTES = (MAX_JPEG_BYTES + 2) // 3 * 4 + 4 * 1024 * 1024
MAX_RESPONSE_BYTES = 4 * 1024 * 1024


class GenEvalInfrastructureError(RuntimeError):
    pass


class GenEvalWorker(JsonWorker):
    def __init__(self, manifest_path, manifest_sha256, *, python, device,
                 detector_microbatch=1, color_workers=0, startup_timeout=300, request_timeout=300):
        if not isinstance(python, str) or not Path(python).is_absolute() or not Path(python).is_file():
            raise ValueError("GenEval requires an absolute worker Python executable")
        if not isinstance(device, str) or (device != "cpu" and not device.startswith("cuda:")):
            raise ValueError("GenEval worker device must be cpu or explicit cuda index")
        if (type(detector_microbatch) is not int or detector_microbatch < 1
                or type(color_workers) is not int or not 0 <= color_workers <= 16):
            raise ValueError("invalid GenEval detector batch or color worker count")
        super().__init__(
            [python, "-I", "-u", str(Path(__file__).with_name("geneval_worker.py"))],
            [manifest_path, manifest_sha256, device, str(detector_microbatch), str(color_workers)],
            startup_timeout=startup_timeout, request_timeout=request_timeout,
            error_type=GenEvalInfrastructureError, label="GenEval", response_limit=MAX_RESPONSE_BYTES,
        )

    def __call__(self, images, metadata, *, only_strict):
        if (not isinstance(images, list) or not 0 < len(images) <= 256
                or any(not isinstance(image, bytes) or not image for image in images)
                or sum(map(len, images)) > MAX_JPEG_BYTES or type(only_strict) is not bool
                or not isinstance(metadata, (list, tuple)) or len(metadata) != len(images)):
            raise ValueError("invalid or oversized GenEval request")
        payload = (json.dumps({"images": [base64.b64encode(image).decode("ascii") for image in images],
                              "metadata": metadata, "only_strict": only_strict},
                             allow_nan=False).encode() + b"\n")
        if len(payload) > MAX_REQUEST_BYTES:
            raise ValueError("GenEval request exceeds byte limit")
        with self._lock:
            try:
                reply = self._exchange(payload, self.timeout)
                if set(reply) != {"status", "result"} or reply["status"] != "ok":
                    raise GenEvalInfrastructureError(
                        f"GenEval engine failed: {reply.get('error_type', 'protocol')}"
                        + (f": {reply['message']}" if isinstance(reply.get("message"), str) else "")
                    )
                return reply["result"]
            except Exception as exc:
                self.close()
                if isinstance(exc, GenEvalInfrastructureError):
                    raise
                raise GenEvalInfrastructureError("GenEval worker transport failed") from exc
