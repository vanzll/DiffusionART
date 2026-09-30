"""Scalar reward composition with explicit, once-only normalization."""

from collections.abc import Mapping, Sequence

import torch

from .image_dispatch import score_image_batch, validate_image_metadata
from .scalar import ScoreResult, weighted_score


class CompositeImageReward(torch.nn.Module):
    """Combine component raw scores; never combine already scaled child totals.

    This is the ordinary no-grad image scoring path. Group advantage
    normalization and pairwise conversion belong to the training algorithm.
    Zero-weight components are still scored and validated for reporting.
    """

    def __init__(self, scorers, weights, *, space, normalization=None, microbatches=None):
        super().__init__()
        if (
            not isinstance(scorers, Mapping)
            or not scorers
            or any(not isinstance(k, str) or not k or "." in k for k in scorers)
            or any(not isinstance(v, torch.nn.Module) for v in scorers.values())
        ):
            raise ValueError("named torch module scorers are required")
        # Validate the scalar contract before allocating/loading any image data.
        contract = weighted_score(
            {k: 0.0 for k in scorers}, weights, space=space, normalization=normalization
        )
        limits = {} if microbatches is None else dict(microbatches)
        if set(limits) - set(scorers) or any(type(v) is not int or v < 1 for v in limits.values()):
            raise ValueError("component microbatches must be positive integers for known scorers")
        self.scorers = torch.nn.ModuleDict(dict(sorted(scorers.items())))
        self.weights = dict(contract.weights)
        self.space = contract.space
        self.normalization = contract.normalization
        self.microbatches = limits
        self.requires_metadata = any(getattr(s, "requires_metadata", False) for s in scorers.values())
        self.supports_evaluation_mode = True
        self.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, images, prompts, *, microbatch, metadata=None, evaluation=False):
        if (
            not isinstance(prompts, Sequence)
            or isinstance(prompts, (str, bytes))
            or not prompts
            or any(not isinstance(p, str) or not p.strip() for p in prompts)
        ):
            raise ValueError("one nonempty prompt per image is required")
        if type(microbatch) is not int or microbatch < 1:
            raise ValueError("microbatch must be a positive integer")
        if not isinstance(images, (Sequence, torch.Tensor)) or len(images) != len(prompts):
            raise ValueError("image/prompt count mismatch")
        validate_image_metadata(self, prompts, metadata)
        self.eval()
        raw = {}
        diagnostics = {}
        for name, scorer in self.scorers.items():
            limit = min(microbatch, self.microbatches.get(name, microbatch))
            result = score_image_batch(scorer, images, prompts, microbatch=limit,
                                       metadata=metadata, evaluation=evaluation)
            if not isinstance(result, ScoreResult) or set(result.raw) != {name}:
                raise ValueError(f"scorer {name} must return exactly its named raw component")
            value = result.raw[name]
            if not isinstance(value, torch.Tensor) or value.shape != (len(prompts),):
                raise ValueError(f"scorer {name} must return one tensor score per image")
            raw[name] = value
            if result.diagnostics is not None:
                if diagnostics.keys() & result.diagnostics.keys():
                    raise ValueError("duplicate component diagnostic names")
                diagnostics.update(result.diagnostics)
        return weighted_score(raw, self.weights, space=self.space, normalization=self.normalization,
                              diagnostics=diagnostics or None)
