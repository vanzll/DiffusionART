"""Private GenEval worker, executable with a separately installed legacy Python."""

import base64
import json
import os
import sys
from io import BytesIO
from pathlib import Path

# -I ignores PYTHONPATH; import this exact checkout, not another installed release.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from visual_rl.rewards.geneval_loader import _load_native_engine
from visual_rl.rewards.geneval_process import (
    MAX_JPEG_BYTES,
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
)


def serve(output, engine, source):
    from PIL import Image

    def send(value):
        payload = json.dumps(value, allow_nan=False).encode() + b"\n"
        if len(payload) > MAX_RESPONSE_BYTES:
            raise ValueError("GenEval response exceeds byte limit")
        output.write(payload)
        output.flush()

    send({"status": "ready"})
    while line := source.readline(MAX_REQUEST_BYTES + 1):
        if len(line) > MAX_REQUEST_BYTES or not line.endswith(b"\n"):
            raise ValueError("invalid GenEval request framing")
        request = json.loads(line)
        if (not isinstance(request, dict) or set(request) != {"images", "metadata", "only_strict"}
                or not isinstance(request["images"], list) or not 0 < len(request["images"]) <= 256
                or not isinstance(request["metadata"], list)
                or len(request["images"]) != len(request["metadata"])
                or type(request["only_strict"]) is not bool):
            raise ValueError("invalid GenEval request")
        images, byte_count = [], 0
        try:
            for encoded in request["images"]:
                raw = base64.b64decode(encoded, validate=True)
                byte_count += len(raw)
                if byte_count > MAX_JPEG_BYTES:
                    raise ValueError("GenEval JPEG batch exceeds byte limit")
                with Image.open(BytesIO(raw), formats=["JPEG"]) as image:
                    if image.mode != "RGB" or image.width * image.height > 16 * 1024**2:
                        raise ValueError("GenEval requires bounded RGB JPEGs")
                    image.load()
                    images.append(image.copy())
            result = engine(images, request["metadata"], only_strict=request["only_strict"])
            send({"status": "ok", "result": result})
        except Exception as exc:
            send({"status": "engine_error", "error_type": type(exc).__name__,
                  "message": str(exc)[:1000]})
            return 1
        finally:
            for image in images:
                image.close()
    return 0


def main():
    fd, manifest, digest, device, batch, workers = sys.argv[1:]
    with os.fdopen(int(fd), "wb") as output:
        try:
            engine = _load_native_engine(manifest, digest, device=device,
                                         detector_microbatch=int(batch), color_workers=int(workers))
        except Exception as exc:
            output.write((json.dumps({"status": "init_error", "error_type": type(exc).__name__,
                                     "message": str(exc)[:1000]})
                          + "\n").encode())
            output.flush()
            return 1
        return serve(output, engine, sys.stdin.buffer)


if __name__ == "__main__":
    raise SystemExit(main())
