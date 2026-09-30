"""Explicit metadata capability for annotation-dependent image scorers."""

from collections.abc import Sequence
from copy import deepcopy


def validate_image_metadata(scorer, prompts, metadata):
    if metadata is not None and (
        not isinstance(metadata, Sequence)
        or isinstance(metadata, (str, bytes))
        or len(metadata) != len(prompts)
        or any(row is not None and not isinstance(row, dict) for row in metadata)
    ):
        raise ValueError("metadata must contain one JSON object or None per prompt")
    if getattr(scorer, "requires_metadata", False) and (
        metadata is None or any(row is None for row in metadata)
    ):
        raise ValueError("this image scorer requires metadata for every prompt")


def score_image_batch(scorer, images, prompts, *, microbatch, metadata=None, evaluation=False):
    validate_image_metadata(scorer, prompts, metadata)
    options = {}
    if getattr(scorer, "requires_metadata", False):
        options["metadata"] = deepcopy(metadata)
    if getattr(scorer, "supports_evaluation_mode", False):
        options["evaluation"] = evaluation
    return scorer(images, prompts, microbatch=microbatch, **options)
