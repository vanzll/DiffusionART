"""Portable reward contracts; importing this package does not load a model."""

from .inputs import ContinuousRGBInput, DifferentiableReward
from .scalar import FixedNormalization, ScoreResult, weighted_score

__all__ = [
    "ContinuousRGBInput",
    "DifferentiableReward",
    "FixedNormalization",
    "ScoreResult",
    "weighted_score",
]
