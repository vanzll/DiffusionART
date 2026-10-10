"""Continuous RGB contract only; no codec surrogate or model-specific preprocessing."""

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from .scalar import ScoreResult, finite_number


@dataclass(frozen=True)
class ContinuousRGBInput:
    frames: Any
    prompts: Sequence[str]
    fps: float
    value_range: str
    source: str = "continuous_rgb"

    def validate(self, *, require_grad=True):
        """Validate BTCHW RGB without copying, clamping, detaching, or encoding.

        Provenance is a caller assertion, not an autograd-graph forensic check.
        Passing quantized/decoded MP4 pixels with a synthetic gradient violates
        this contract even if the tensor has requires_grad=True.
        """
        import torch

        if self.source != "continuous_rgb":
            raise ValueError("only continuous RGB is allowed; no MP4/uint8/PIL STE")
        x = self.frames
        if (
            not isinstance(x, torch.Tensor)
            or not x.is_floating_point()
            or x.ndim != 5
            or x.shape[2] != 3
            or any(n <= 0 for n in x.shape)
        ):
            raise ValueError("frames must be nonempty floating BTCHW RGB")
        if (
            isinstance(self.prompts, str)
            or len(self.prompts) != x.shape[0]
            or any(not isinstance(p, str) or not p.strip() for p in self.prompts)
        ):
            raise ValueError("one nonempty prompt per batch item is required")
        if finite_number(self.fps, "fps") <= 0:
            raise ValueError("fps must be positive")
        ranges = {"zero_one": (0, 1), "minus_one_one": (-1, 1)}
        if self.value_range not in ranges:
            raise ValueError("value_range must be zero_one or minus_one_one")
        low, high = ranges[self.value_range]
        if not torch.isfinite(x).all().item() or (x < low).any().item() or (x > high).any().item():
            raise ValueError("frames are nonfinite or outside the declared range")
        if require_grad and not x.requires_grad:
            raise ValueError("differentiable reward requires input gradients")
        return self


class DifferentiableReward(Protocol):
    """Implementations freeze RM weights, preserve input gradients, return [B] scores.

    Model/tokenizer/preprocessing/checkpoint normalization provenance and native
    decoded-pixel parity belong to the model adapter. Continuous and MP4 scores
    are not claimed equivalent. Pairwise objectives belong to shared kernels,
    not this reward-input layer.
    """

    def __call__(self, inputs: ContinuousRGBInput) -> ScoreResult: ...
