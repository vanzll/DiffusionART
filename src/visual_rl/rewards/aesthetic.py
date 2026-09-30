"""Frozen native aesthetic scores; no implicit scaling or head downloads.

Head architecture: improved-aesthetic-predictor simple_inference.py at
fe88a163f4661b4ddabba0751ff645e2e620746e. Checkpoints are supplied by the caller.
"""

import hashlib
import io
from collections.abc import Sequence
from pathlib import Path

import torch

from .pickscore import _projected_features, quantized_rgb
from .scalar import weighted_score


class AestheticMLP(torch.nn.Module):
    """Production 768-dimensional head with upstream-compatible state keys."""

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(768, 1024),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1024, 128),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 16),
            torch.nn.Linear(16, 1),
        )

    def forward(self, features):
        return self.layers(features)


class AestheticScore(torch.nn.Module):
    """Image-only reward with prompts validated for the paired scorer API.

    Float RGB tensors follow round-to-uint8 then slow PIL preprocessing.
    The model and head retain their supplied precision (historically fp32).
    """

    def __init__(self, model, processor, head):
        super().__init__()
        self.model = model.requires_grad_(False).eval()
        self.processor = processor
        self.head = head.requires_grad_(False).eval()
        self.train(False)

    def train(self, mode=True):
        # Parent trainers must not enable dropout in this frozen reward model.
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, images, prompts: Sequence[str], *, microbatch: int):
        from PIL import Image

        if (
            isinstance(prompts, (str, bytes))
            or not isinstance(prompts, Sequence)
            or not prompts
            or any(not isinstance(p, str) or not p.strip() for p in prompts)
        ):
            raise ValueError("one nonempty prompt per image is required")
        if type(microbatch) is not int or microbatch < 1:
            raise ValueError("AestheticScore microbatch must be a positive integer")
        if isinstance(images, torch.Tensor):
            images = quantized_rgb(images)
        if (
            not isinstance(images, Sequence)
            or len(images) != len(prompts)
            or any(not isinstance(im, Image.Image) or im.mode != "RGB"
                   or min(im.size) < 1 for im in images)
        ):
            raise ValueError("provide paired RGB PIL images or a [0,1] RGB tensor")
        parameter = next(self.model.parameters())
        if not parameter.is_floating_point():
            raise ValueError("AestheticScore requires floating model parameters")
        device, dtype = parameter.device, parameter.dtype
        if any(p.device != device or p.dtype != dtype for p in self.head.parameters()):
            raise ValueError("AestheticScore model and head must share device and dtype")
        scores = []
        self.eval()
        with torch.autocast(device_type=device.type, enabled=False):
            for start in range(0, len(images), microbatch):
                batch = list(images[start:start + microbatch])
                inputs = self.processor(images=batch, return_tensors="pt")
                inputs = {
                    key: value.to(device=device, dtype=dtype if value.is_floating_point()
                                  else value.dtype)
                    for key, value in inputs.items()
                }
                features = _projected_features(self.model.get_image_features(**inputs), len(batch))
                if features.shape[1] != 768:
                    raise ValueError("AestheticScore requires 768 projected image features")
                value = self.head(features)
                if (
                    not isinstance(value, torch.Tensor)
                    or value.shape != (len(batch), 1)
                    or not value.is_floating_point()
                    or not torch.isfinite(value).all()
                ):
                    raise ValueError("AestheticScore head must return finite floating [B,1] scores")
                scores.append(value.squeeze(1))
        return weighted_score(
            {"aesthetic": torch.cat(scores)}, {"aesthetic": 1.0},
            space="raw", normalization=None,
        )

    @classmethod
    def from_pretrained(
        cls, *, model_path: str, model_revision: str, processor_path: str,
        processor_revision: str, head_path: str, head_sha256: str,
        device="cpu", dtype=torch.float32, local_files_only=True,
    ):
        """Load explicit CLIP revisions and a SHA256-verified local MLP state.

        Only CLIP loading can opt into network access. The head is always local,
        loaded with weights_only=True and strict production architecture keys.
        """
        if any(not isinstance(x, str) or not x.strip() for x in (
            model_path, model_revision, processor_path, processor_revision, head_path,
        )):
            raise ValueError("explicit model/processor paths, revisions and local head path required")
        if (
            not isinstance(head_sha256, str)
            or len(head_sha256) != 64
            or any(c not in "0123456789abcdefABCDEF" for c in head_sha256)
        ):
            raise ValueError("head_sha256 must be a 64-character SHA256 hex digest")
        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("AestheticScore dtype must be fp32, fp16 or bf16")
        if type(local_files_only) is not bool:
            raise ValueError("local_files_only must be boolean")
        path = Path(head_path)
        if not path.is_file():
            raise ValueError("head_path must be an existing local file")
        # Deserialize exactly the bytes verified, even if the path is replaced.
        max_bytes = 16 * 1024 * 1024
        with path.open("rb") as stream:
            payload = stream.read(max_bytes + 1)
        if len(payload) > max_bytes:
            raise ValueError("aesthetic head file exceeds 16 MiB")
        if hashlib.sha256(payload).hexdigest() != head_sha256.lower():
            raise ValueError("aesthetic head SHA256 mismatch")
        state = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
        head = AestheticMLP()
        head.load_state_dict(state, strict=True)
        head = head.to(device=device, dtype=dtype)
        if any(not torch.isfinite(p).all() for p in head.parameters()):
            raise ValueError("aesthetic head parameters must be finite")

        from transformers import CLIPModel, CLIPProcessor

        try:
            from transformers import CLIPImageProcessorPil as ImageProcessor
        except ImportError:  # Transformers 4's CLIPImageProcessor uses PIL.
            from transformers import CLIPImageProcessor as ImageProcessor

        options = dict(revision=processor_revision, local_files_only=local_files_only)
        processor = CLIPProcessor.from_pretrained(processor_path, **options)
        processor.image_processor = ImageProcessor.from_pretrained(processor_path, **options)
        model = CLIPModel.from_pretrained(
            model_path, revision=model_revision, local_files_only=local_files_only
        ).to(device=device, dtype=dtype)
        return cls(model, processor, head)
