"""Frozen, injected OCR inference with the historical image reward semantics."""

import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch

from .ocr_process import OCRInfrastructureError
from .pickscore import quantized_rgb
from .scalar import weighted_score


def extract_target(prompt: str) -> str:
    """Return the first double-quoted target, lowercased without ASCII spaces."""
    if not isinstance(prompt, str):
        raise ValueError("OCR prompt must be a string with a double-quoted target")
    parts = prompt.split('"', 2)
    if len(parts) != 3:
        raise ValueError("OCR prompt must contain a complete double-quoted target")
    target = parts[1].replace(" ", "").lower()
    if not target:
        raise ValueError("OCR target must be nonempty after normalization")
    return target


class OCRScore(torch.nn.Module):
    """One RGB image per engine call; microbatch only bounds pixel conversion.

    The engine is supplied by the caller and must expose ``ocr(image, cls=False)``.
    Moving this module moves its output device, not a non-torch native engine.
    """

    def __init__(self, engine, *, error_policy: Literal["raise", "zero"], device="cpu"):
        super().__init__()
        if error_policy not in ("raise", "zero"):
            raise ValueError("OCRScore error_policy must be raise or zero")
        if not callable(getattr(engine, "ocr", None)):
            raise ValueError("OCRScore engine must expose a callable ocr method")
        self.engine = engine
        if isinstance(engine, torch.nn.Module):
            engine.requires_grad_(False)
        self.error_policy = error_policy
        self.register_buffer("_device_anchor", torch.empty(0, device=device), persistent=False)
        self.train(False)

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, images, prompts: Sequence[str], *, microbatch: int):
        from PIL import Image
        from rapidfuzz.distance import Levenshtein

        if type(microbatch) is not int or microbatch < 1:
            raise ValueError("OCRScore microbatch must be a positive integer")
        if isinstance(prompts, (str, bytes)) or not isinstance(prompts, Sequence) or not prompts:
            raise ValueError("one paired OCR prompt per image is required")
        targets = []
        for index, prompt in enumerate(prompts):
            try:
                targets.append(extract_target(prompt))
            except ValueError:
                raise ValueError(
                    f"OCR prompt at sample {index} requires a nonempty double-quoted target"
                ) from None

        tensor_input = isinstance(images, torch.Tensor)
        if tensor_input:
            if (
                images.ndim != 4
                or images.shape[1] != 3
                or min(images.shape) < 1
                or not images.is_floating_point()
            ):
                raise ValueError("OCRScore images must be nonempty floating [B,3,H,W] RGB")
        elif not isinstance(images, Sequence) or any(
            not isinstance(im, Image.Image) or im.mode != "RGB" for im in images
        ):
            raise ValueError("provide paired RGB PIL images or a [0,1] RGB tensor")
        if len(images) != len(targets):
            raise ValueError("provide paired images and OCR prompts with equal lengths")
        # Validate the whole input before inference, without a full-batch pixel allocation.
        if tensor_input:
            for start in range(0, len(images), microbatch):
                batch = images[start : start + microbatch]
                if not torch.isfinite(batch).all() or (batch < 0).any() or (batch > 1).any():
                    raise ValueError("OCRScore images must be finite RGB in [0,1]")

        self.train(False)
        scores = []
        for start in range(0, len(images), microbatch):
            batch = images[start : start + microbatch]
            if tensor_input:
                batch = quantized_rgb(batch)
            for offset, image in enumerate(batch):
                index = start + offset
                target = targets[index]
                try:
                    result = self.engine.ocr(np.array(image), cls=False)
                    recognized = (
                        "".join(res[1][0] if res[1][1] > 0 else "" for res in result[0])
                        if result[0]
                        else ""
                    )
                    recognized = recognized.replace(" ", "").lower()
                    score = (
                        1.0
                        if target in recognized
                        else (
                            1
                            - min(Levenshtein.distance(recognized, target), len(target))
                            / len(target)
                        )
                    )
                except OCRInfrastructureError:
                    raise
                except Exception:
                    if self.error_policy == "raise":
                        raise RuntimeError(f"OCR processing failed at sample {index}") from None
                    warnings.warn(
                        f"OCR processing failed at sample {index}; assigning zero reward",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    score = 0.0
                scores.append(score)
        raw = torch.tensor(scores, dtype=torch.float32, device=self._device_anchor.device)
        return weighted_score({"ocr": raw}, {"ocr": 1}, space="raw")

    def close(self):
        close = getattr(self.engine, "close", None)
        if close is not None:
            close()
