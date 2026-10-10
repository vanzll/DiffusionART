"""File-based FFmpeg encoding with exit-code and full-decode validation."""

import json
import math
import os
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path

import torch


def encode_video(frames, target, *, fps, ffmpeg="ffmpeg", ffprobe="ffprobe", timeout=120):
    """Encode TCHW RGB in [-1,1], publish only a complete MP4.

    Transport uses a complete raw file, not stdin. Uses H.264 CRF25/yuv420p,
    matching the reporting codec recipe. No re-generation or lossy fallback.
    Filesystem operations require a responsive output directory.
    """
    if (not isinstance(frames, torch.Tensor) or frames.ndim != 4 or frames.shape[1] != 3
            or not frames.is_floating_point() or not torch.isfinite(frames).all()
            or min(frames.shape) < 1 or any(n % 2 for n in frames.shape[-2:])
            or (frames < -1).any() or (frames > 1).any()):
        raise ValueError("video frames must be finite TCHW RGB in [-1,1] with even spatial dimensions")
    if not math.isfinite(fps) or fps <= 0 or not 0 < timeout <= 600:
        raise ValueError("positive FPS and bounded encoder timeout required")
    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    count, _, height, width = frames.shape
    rgb = quantize_rgb(frames)
    with tempfile.TemporaryDirectory(prefix=".encoding-", dir=target.parent) as temp:
        raw, encoded = Path(temp, "input.rgb"), Path(temp, "output.mp4")
        with raw.open("wb") as stream:
            data = memoryview(rgb.numpy()).cast("B")
            written = stream.write(data)
            if written != data.nbytes:
                raise OSError("incomplete raw video write")
        if raw.stat().st_size != count * height * width * 3:
            raise OSError("raw video size differs from frame dimensions")
        subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                        "-video_size", f"{width}x{height}", "-framerate", str(fps), "-i", str(raw),
                        "-frames:v", str(count), "-an", "-c:v", "libx264", "-crf", "25",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(encoded)],
                       check=True, capture_output=True, timeout=timeout)
        # Full decode, not merely a successful encoder exit or nonempty file.
        command = [ffmpeg, "-nostdin", "-v", "error", "-xerror", "-i", str(encoded), "-map", "0:v:0"]
        try:
            check = subprocess.run(command + ["-fps_mode", "passthrough", "-f", "framemd5", "-"],
                                   check=True, capture_output=True, timeout=timeout)
        except subprocess.CalledProcessError as error:
            if b"Unrecognized option 'fps_mode'" not in (error.stderr or b""):
                raise
            # FFmpeg 4.x uses vsync; recent versions removed that older spelling.
            check = subprocess.run(command + ["-vsync", "0", "-f", "framemd5", "-"],
                                   check=True, capture_output=True, timeout=timeout)
        decoded = [line for line in check.stdout.splitlines() if line and not line.startswith(b"#")]
        if len(decoded) != count:
            raise ValueError("encoded MP4 failed full frame-count validation")
        result = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                                 "stream=width,height,avg_frame_rate", "-of", "json", str(encoded)],
                                check=True, capture_output=True, timeout=timeout)
        streams = json.loads(result.stdout).get("streams", [])
        if (len(streams) != 1 or streams[0]["width"] != width or streams[0]["height"] != height
                or not math.isclose(float(Fraction(streams[0]["avg_frame_rate"])), fps, rel_tol=1e-6)):
            raise ValueError("encoded MP4 geometry/FPS differs")
        # The caller owns its output directory; never silently overwrite a file.
        os.link(encoded, target)
    return target


def quantize_rgb(frames):
    """Match VideoProcessor denormalize -> FP32 NumPy -> uint8 export order."""
    normalized = (frames.detach() / 2 + .5).clamp(0, 1)
    return (normalized.cpu().permute(0, 2, 3, 1).float() * 255).to(torch.uint8).contiguous()
