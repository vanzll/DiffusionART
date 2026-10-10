"""Frozen CLIP logits with the canonical, unquantized tensor image transform."""

from collections.abc import Mapping, Sequence

import torch

from .scalar import FixedNormalization, ScoreResult, weighted_score


def _get_size(size):
    # Canonical integer sizes mean (height, width), not shortest-edge resize.
    if type(size) is int and size > 0:
        return (size, size)
    if isinstance(size, Mapping):
        if "height" in size and "width" in size:
            values = (size["height"], size["width"])
            if all(type(value) is int and value > 0 for value in values):
                return values
        elif type(size.get("shortest_edge")) is int and size["shortest_edge"] > 0:
            return size["shortest_edge"]
    raise ValueError(f"invalid CLIPScore image size: {size}")


class CLIPScore(torch.nn.Module):
    """Paired raw logits and fixed raw/30 scores, never batch-normalized.

    Floating RGB tensors retain their precision through resize/crop/normalize;
    only the final model input is cast to the model's dtype. PIL RGB sequences
    are converted to floating tensors before the same transform.
    """

    def __init__(self, model, processor, *, space: str, antialias: bool = True):
        super().__init__()
        if space not in ("raw", "fixed_normalized"):
            raise ValueError("CLIPScore space must be raw or fixed_normalized")
        if type(antialias) is not bool:
            raise ValueError("CLIPScore antialias must be boolean")
        from torchvision import transforms as T

        image_processor = processor.image_processor
        config = image_processor.to_dict()
        # Ignore HF resample/do_rescale: the reference uses tensor transforms.
        self.transform = T.Compose(
            [
                T.Resize(
                    _get_size(config.get("size")),
                    interpolation=T.InterpolationMode.BILINEAR,
                    antialias=antialias,
                )
                if config.get("do_resize")
                else torch.nn.Identity(),
                T.CenterCrop(_get_size(config.get("crop_size")))
                if config.get("do_center_crop")
                else torch.nn.Identity(),
                T.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
                if config.get("do_normalize")
                else torch.nn.Identity(),
            ]
        )
        self.model = model.requires_grad_(False).eval()
        self.processor = processor
        self.space = space
        self.antialias = antialias
        self.train(False)

    def train(self, mode=True):
        # Parent training mode must never enable reward-model dropout.
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, images, prompts: Sequence[str], *, microbatch: int) -> ScoreResult:
        from PIL import Image
        from torchvision.transforms.functional import to_tensor

        if (
            isinstance(prompts, (str, bytes))
            or not isinstance(prompts, Sequence)
            or not prompts
            or any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts)
        ):
            raise ValueError("one nonempty prompt per image is required")
        if type(microbatch) is not int or microbatch < 1:
            raise ValueError("CLIPScore microbatch must be a positive integer")
        tensor_input = isinstance(images, torch.Tensor)
        if tensor_input:
            if (
                images.ndim != 4
                or images.shape[1] != 3
                or min(images.shape) < 1
                or not images.is_floating_point()
                or not torch.isfinite(images).all()
                or (images < 0).any()
                or (images > 1).any()
            ):
                raise ValueError("CLIPScore images must be finite nonempty [B,3,H,W] RGB in [0,1]")
        elif (
            not isinstance(images, Sequence)
            or not images
            or any(
                not isinstance(im, Image.Image) or im.mode != "RGB" or min(im.size) < 1
                for im in images
            )
        ):
            raise ValueError("provide paired RGB PIL images or a [0,1] RGB tensor")
        if len(images) != len(prompts):
            raise ValueError("provide paired images and prompts")
        parameter = next(self.model.parameters())
        if not parameter.is_floating_point():
            raise ValueError("CLIPScore requires floating model parameters")
        device, dtype = parameter.device, parameter.dtype
        scores = []
        self.model.eval()
        with torch.autocast(device_type=device.type, enabled=False):
            for start in range(0, len(images), microbatch):
                stop = min(start + microbatch, len(images))
                if tensor_input:
                    pixels = self.transform(images[start:stop])
                else:
                    # Transform separately to support differently sized PIL images.
                    pixels = torch.stack(
                        [self.transform(to_tensor(im)) for im in images[start:stop]]
                    )
                text = self.processor(
                    text=list(prompts[start:stop]),
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                text = {
                    key: value.to(
                        device=device, dtype=dtype if value.is_floating_point() else value.dtype
                    )
                    for key, value in text.items()
                }
                logits = self.model(
                    **text, pixel_values=pixels.to(device=device, dtype=dtype)
                ).logits_per_image
                if (
                    not isinstance(logits, torch.Tensor)
                    or logits.shape != (stop - start, stop - start)
                    or not logits.is_floating_point()
                    or not torch.isfinite(logits).all()
                ):
                    raise ValueError("CLIPScore model returned invalid or nonfinite logits")
                scores.append(logits.diagonal())
        raw = torch.cat(scores)
        return weighted_score(
            {"clipscore": raw},
            {"clipscore": 1.0},
            space=self.space,
            normalization={
                "clipscore": FixedNormalization(
                    mean=0.0, std=30.0, source="research-clipscore-logit-divisor-30"
                )
            },
        )

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_path: str,
        processor_path: str,
        model_revision: str,
        processor_revision: str,
        space: str,
        device="cpu",
        dtype=torch.float32,
        local_files_only=True,
        antialias: bool = True,
    ):
        """Load explicit assets/revisions, offline by default; no bundled weights.

        Pin PIL image-processor configuration independently of installed HF
        backends. Its image preprocessing is never called during scoring.
        """
        if space not in ("raw", "fixed_normalized"):
            raise ValueError("CLIPScore space must be raw or fixed_normalized")
        if type(antialias) is not bool:
            raise ValueError("CLIPScore antialias must be boolean")
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (model_path, processor_path, model_revision, processor_revision)
        ):
            raise ValueError("explicit model/processor paths and revisions are required")
        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("CLIPScore dtype must be fp32, fp16 or bf16")
        if type(local_files_only) is not bool:
            raise ValueError("local_files_only must be boolean")
        from transformers import CLIPModel, CLIPProcessor

        try:
            from transformers import CLIPImageProcessorPil as ImageProcessor
        except ImportError:  # Transformers 4 uses the PIL backend under this name.
            from transformers import CLIPImageProcessor as ImageProcessor

        options = dict(revision=processor_revision, local_files_only=local_files_only)
        processor = CLIPProcessor.from_pretrained(processor_path, **options)
        processor.image_processor = ImageProcessor.from_pretrained(processor_path, **options)
        model = CLIPModel.from_pretrained(
            model_path, revision=model_revision, local_files_only=local_files_only
        ).to(device=device, dtype=dtype)
        return cls(model, processor, space=space, antialias=antialias)
