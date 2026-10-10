"""Frozen HPSv2.1 inference using the vendor ViT-H-14 model and transforms."""

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from .image_dispatch import validate_image_metadata
from .pickscore import quantized_rgb
from .scalar import ScoreResult, weighted_score


class HPSv2Score(torch.nn.Module):
    """Paired feature dot products, without logit scale or extra normalization.

    The vendor forward already normalizes both features. CUDA uses historical
    AMP; returned raw scores are always detached CPU FP32. Composite rewards
    own weighting and any fixed normalization.
    """

    def __init__(self, model, preprocess, tokenizer):
        super().__init__()
        self.model = model.requires_grad_(False).eval()
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.train(False)

    def train(self, mode=True):
        # Parent training mode must never enable reward-model dropout.
        super().train(False)
        return self

    @torch.no_grad()
    def forward(
        self, images, prompts: Sequence[str], metadata=None, *, microbatch: int
    ) -> ScoreResult:
        from PIL import Image

        if (
            isinstance(prompts, (str, bytes))
            or not isinstance(prompts, Sequence)
            or not prompts
            or any(not isinstance(p, str) or not p.strip() for p in prompts)
        ):
            raise ValueError("one nonempty prompt per image is required")
        if type(microbatch) is not int or microbatch < 1:
            raise ValueError("HPSv2 microbatch must be a positive integer")
        validate_image_metadata(self, prompts, metadata)
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
            raise ValueError("HPSv2 requires floating model parameters")
        device = parameter.device
        scores = []
        self.eval()
        for start in range(0, len(images), microbatch):
            batch = images[start:start + microbatch]
            pixels = [self.preprocess(im.convert("RGB")) for im in batch]
            if any(
                not isinstance(value, torch.Tensor) or value.ndim != 3
                or value.shape[0] != 3 or min(value.shape) < 1
                or not value.is_floating_point() or not torch.isfinite(value).all()
                or value.shape != pixels[0].shape
                for value in pixels
            ):
                raise ValueError("HPSv2 preprocess must return finite matching [3,H,W] tensors")
            pixels = torch.stack(pixels).to(device=device, non_blocking=True)
            tokens = self.tokenizer(list(prompts[start:start + microbatch]))
            if (
                not isinstance(tokens, torch.Tensor) or tokens.ndim != 2
                or tokens.shape[0] != len(batch) or tokens.shape[1] < 1
                or tokens.dtype not in (torch.int32, torch.int64)
            ):
                raise ValueError("HPSv2 tokenizer must return paired integer [B,L] tokens")
            tokens = tokens.to(device=device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                outputs = self.model(pixels, tokens)
                if not isinstance(outputs, Mapping):
                    raise ValueError("HPSv2 model must return image/text features")
                features = [outputs.get(key) for key in ("image_features", "text_features")]
                if any(
                    not isinstance(value, torch.Tensor) or value.ndim != 2
                    or value.shape[0] != len(batch) or value.shape[1] < 1
                    or not value.is_floating_point() or not torch.isfinite(value).all()
                    or value.device != device
                    for value in features
                ):
                    raise ValueError("HPSv2 model returned invalid or nonfinite features")
                image_features, text_features = features
                if image_features.shape != text_features.shape:
                    raise ValueError("HPSv2 image/text feature shape mismatch")
                # Preserve c81b12ba's paired reduction, including AMP arithmetic.
                value = (image_features * text_features).sum(dim=-1)
            scores.append(value.float().cpu())
        return weighted_score(
            {"hpsv2": torch.cat(scores)}, {"hpsv2": 1.0}, space="raw"
        )

    @classmethod
    def from_pretrained(
        cls, *, model_path: str, model_revision: str, device="cpu", checkpoint_sha256: str
    ):
        """Load a SHA256-verified local weight FILE; never resolve/download assets.

        The revision is caller-supplied provenance, not a network lookup. A full
        vendor state_dict is required; partial states never trigger LAION loading.
        """
        if any(not isinstance(value, str) or not value.strip()
               for value in (model_path, model_revision)):
            raise ValueError("explicit local model file and model revision are required")
        if (
            not isinstance(checkpoint_sha256, str) or len(checkpoint_sha256) != 64
            or any(c not in "0123456789abcdefABCDEF" for c in checkpoint_sha256)
        ):
            raise ValueError("checkpoint_sha256 must be a 64-character SHA256 hex digest")
        path = Path(model_path)
        if not path.is_file():
            raise ValueError("model_path must be an existing local weight file")
        target = torch.device(device)
        # Check the installation before reading the multi-GiB weight file.
        try:
            from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
            from hpsv2.src.open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
        except FileNotFoundError as error:
            raise ImportError(
                "HPSv2 installation is missing vendor assets; install an asset-complete "
                "hpsv2==1.2.0 wheel containing bpe_simple_vocab_16e6.txt.gz"
            ) from error
        except ImportError as error:
            raise ImportError("HPSv2 rewards require the optional hpsv2==1.2.0 dependency") from error
        # Stream the large checkpoint, then deserialize through the same open
        # descriptor so replacing the path cannot substitute unverified weights.
        with path.open("rb") as stream:
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
            if digest.hexdigest() != checkpoint_sha256.lower():
                raise ValueError("HPSv2 checkpoint SHA256 mismatch")
            stream.seek(0)
            checkpoint = torch.load(stream, map_location="cpu", weights_only=True)
        state = checkpoint.get("state_dict") if isinstance(checkpoint, Mapping) else None
        if (
            not isinstance(state, Mapping) or not state
            or any(not isinstance(key, str) or not key
                   or not isinstance(value, torch.Tensor) or value.layout != torch.strided
                   or value.device.type != "cpu" or value.is_complex()
                   or not torch.isfinite(value).all() for key, value in state.items())
        ):
            raise ValueError("HPSv2 checkpoint must contain a finite tensor state_dict")
        model, _, preprocess = create_model_and_transforms(
            "ViT-H-14", pretrained=None, precision="amp", device="cpu",
            jit=False, force_quick_gelu=False, force_custom_text=False,
            force_patch_dropout=False, force_image_size=None, pretrained_image=False,
            image_mean=OPENAI_DATASET_MEAN, image_std=OPENAI_DATASET_STD,
            light_augmentation=True, aug_cfg={}, output_dict=True,
            with_score_predictor=False, with_region_predictor=False,
        )
        model.load_state_dict(state, strict=True)
        model = model.to(device=target).eval()
        scorer = cls(model, preprocess, get_tokenizer("ViT-H-14"))
        scorer.model_path = str(path.resolve())
        scorer.model_revision = model_revision
        scorer.checkpoint_sha256 = checkpoint_sha256.lower()
        return scorer
