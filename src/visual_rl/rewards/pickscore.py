"""Frozen native PickScore inference with explicit historical score scaling.

This is the quantized-image scalar scorer, not a differentiable ReFL adapter.
The official model's logits and the research recipe's logits / 26 are both
retained; neither is a softmax probability or a batch-normalized reward.
"""

from collections.abc import Sequence

import torch

from .scalar import FixedNormalization, weighted_score


def quantized_rgb(images: torch.Tensor):
    """Match the historical tensor -> uint8 RGB -> PIL path without an STE."""
    from PIL import Image

    if (
        not isinstance(images, torch.Tensor)
        or images.ndim != 4
        or images.shape[1] != 3
        or min(images.shape) < 1
        or not images.is_floating_point()
        or not torch.isfinite(images).all()
        or (images < 0).any()
        or (images > 1).any()
    ):
        raise ValueError("PickScore images must be finite nonempty [B,3,H,W] RGB in [0,1]")
    pixels = (images.detach() * 255).round().clamp(0, 255).to(torch.uint8)
    pixels = pixels.permute(0, 2, 3, 1).cpu().numpy()
    return [Image.fromarray(value) for value in pixels]


def _projected_features(result, count):
    # Transformers 4 returns tensors; 5 returns a model output whose pooler
    # field already includes the CLIP projection. Never project that twice.
    value = result if isinstance(result, torch.Tensor) else getattr(result, "pooler_output", None)
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 2
        or value.shape[0] != count
        or value.shape[1] < 1
        or not value.is_floating_point()
        or not torch.isfinite(value).all()
    ):
        raise ValueError("PickScore model returned invalid projected features")
    norm = value.norm(p=2, dim=-1, keepdim=True)
    if (norm <= 0).any() or not torch.isfinite(norm).all():
        raise ValueError("PickScore feature norm must be finite and positive")
    return value / norm


class PickScore(torch.nn.Module):
    """Injected CLIP model and processor; default precision is the model's dtype.

    ``space='fixed_normalized'`` selects the historical raw/26 training score.
    ``space='raw'`` selects the native logit. Both calls return both components.
    Image and prompt lists are paired by index, never sorted or cross-ranked.
    """

    def __init__(self, model, processor, *, space: str):
        super().__init__()
        if space not in ("raw", "fixed_normalized"):
            raise ValueError("PickScore space must be raw or fixed_normalized")
        self.model = model.requires_grad_(False).eval()
        self.processor = processor
        self.space = space
        self.train(False)

    def train(self, mode=True):
        # A trainer calling .train() on its entire graph must not enable RM dropout.
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
            raise ValueError("PickScore microbatch must be a positive integer")
        if isinstance(images, torch.Tensor):
            images = quantized_rgb(images)
        if (
            not isinstance(images, Sequence)
            or len(images) != len(prompts)
            or any(not isinstance(im, Image.Image) or im.mode != "RGB" for im in images)
        ):
            raise ValueError("provide paired RGB PIL images or a [0,1] RGB tensor")
        parameter = next(self.model.parameters())
        if not parameter.is_floating_point():
            raise ValueError("PickScore requires floating model parameters")
        device, dtype = parameter.device, parameter.dtype
        scores = []
        self.model.eval()
        with torch.autocast(device_type=device.type, enabled=False):
            for start in range(0, len(images), microbatch):
                stop = min(start + microbatch, len(images))
                options = dict(padding=True, truncation=True, max_length=77, return_tensors="pt")
                image_inputs = self.processor(images=list(images[start:stop]), **options)
                text_inputs = self.processor(text=list(prompts[start:stop]), **options)

                def move(inputs):
                    return {
                        key: tensor.to(device=device, dtype=dtype if tensor.is_floating_point()
                                       else tensor.dtype)
                        for key, tensor in inputs.items()
                    }

                image_features = _projected_features(
                    self.model.get_image_features(**move(image_inputs)), stop - start
                )
                text_features = _projected_features(
                    self.model.get_text_features(**move(text_inputs)), stop - start
                )
                if image_features.shape != text_features.shape:
                    raise ValueError("PickScore image/text feature shape mismatch")
                # Retain the historical matrix/diagonal arithmetic order.
                scores.append(
                    (self.model.logit_scale.exp() * (text_features @ image_features.T)).diag()
                )
        raw = torch.cat(scores)
        return weighted_score(
            {"pickscore": raw}, {"pickscore": 1.0}, space=self.space,
            normalization={"pickscore": FixedNormalization(
                mean=0.0, std=26.0, source="research-pickscore-logit-divisor-26"
            )},
        )

    @classmethod
    def from_pretrained(
        cls, *, model_path: str, processor_path: str, model_revision: str,
        processor_revision: str, space: str, device="cpu", dtype=torch.float32,
        local_files_only=True,
    ):
        """Load explicit revisions. Network is opt-in; HF handles credentials.

        Pin the slow PIL image processor even when torchvision is installed.
        Selecting a fast tensor preprocessor can change scores and is not an
        incidental environment optimization for a reproduction recipe.
        """
        if space not in ("raw", "fixed_normalized"):
            raise ValueError("PickScore space must be raw or fixed_normalized")
        if any(not isinstance(x, str) or not x.strip() for x in (
            model_path, processor_path, model_revision, processor_revision
        )):
            raise ValueError("explicit model/processor paths and revisions are required")
        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("PickScore dtype must be fp32, fp16 or bf16")
        if type(local_files_only) is not bool:
            raise ValueError("local_files_only must be boolean")
        from transformers import CLIPModel, CLIPProcessor

        try:
            from transformers import CLIPImageProcessorPil as ImageProcessor
        except ImportError:  # Transformers 4's CLIPImageProcessor is the PIL backend.
            from transformers import CLIPImageProcessor as ImageProcessor

        options = dict(revision=processor_revision, local_files_only=local_files_only)
        processor = CLIPProcessor.from_pretrained(processor_path, **options)
        processor.image_processor = ImageProcessor.from_pretrained(processor_path, **options)
        model = CLIPModel.from_pretrained(
            model_path, revision=model_revision, local_files_only=local_files_only
        ).to(device=device, dtype=dtype)
        return cls(model, processor, space=space)
