"""Private subprocess entrypoint: Paddle imports never enter the trainer."""

import base64
import json
import os
import sys

import numpy as np

from .ocr_process import MAX_RESPONSE_BYTES, MAX_RGB_BYTES
from .paddle_loader import _load_native_engine


def serve(output, engine, source):
    def send(value):
        def native(item):
            if isinstance(item, np.generic):
                return item.item()
            if isinstance(item, np.ndarray):
                return item.tolist()
            raise TypeError("unsupported OCR result type")

        # This private Python-to-Python protocol preserves native confidences:
        # the historical scorer ignores NaN via confidence > 0, not fatal I/O.
        message = json.dumps(value, allow_nan=True, default=native).encode() + b"\n"
        if len(message) > MAX_RESPONSE_BYTES:
            raise ValueError("OCR response exceeds byte limit")
        output.write(message)
        output.flush()

    send({"status": "ready"})
    limit = (MAX_RGB_BYTES + 2) // 3 * 4 + 1024
    while line := source.readline(limit + 1):
        if len(line) > limit or not line.endswith(b"\n"):
            raise ValueError("OCR request exceeds byte limit or is incomplete")
        request = json.loads(line)
        shape = request["shape"]
        if (
            not isinstance(shape, list)
            or len(shape) != 3
            or shape[2] != 3
            or any(type(n) is not int or n < 1 for n in shape)
        ):
            raise ValueError("invalid RGB shape")
        raw = base64.b64decode(request["rgb"], validate=True)
        if len(raw) > MAX_RGB_BYTES or len(raw) != shape[0] * shape[1] * 3:
            raise ValueError("RGB shape/byte count mismatch")
        image = np.frombuffer(raw, dtype=np.uint8).reshape(shape).copy()
        try:
            result = engine.ocr(image, cls=False)
        except Exception as exc:
            send({"status": "engine_error", "error_type": type(exc).__name__})
        else:
            send({"status": "ok", "result": result})


def main():
    fd, manifest, digest = sys.argv[1:]
    with os.fdopen(int(fd), "wb") as output:
        try:
            engine = _load_native_engine(manifest, digest)
        except Exception as exc:
            output.write(
                json.dumps({"status": "init_error", "error_type": type(exc).__name__}).encode()
                + b"\n"
            )
            output.flush()
            return 1
        serve(output, engine, sys.stdin.buffer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
