"""Explicit fixed-normalization and weighted scores, without model dependencies."""

import math
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Any, Mapping, Optional


def finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


@dataclass(frozen=True)
class FixedNormalization:
    mean: float
    std: float
    source: str

    def __post_init__(self):
        object.__setattr__(self, "mean", finite_number(self.mean, "mean"))
        object.__setattr__(self, "std", finite_number(self.std, "std"))
        if self.std <= 0 or not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("normalization requires positive std and nonempty source")


@dataclass(frozen=True)
class ScoreResult:
    raw: Mapping[str, Any]
    fixed_normalized: Optional[Mapping[str, Any]]
    weights: Mapping[str, float]
    normalization: Optional[Mapping[str, FixedNormalization]]
    space: str
    weighted: Any
    diagnostics: Optional[Mapping[str, Any]] = None

    def to_dict(self):
        """Serialize one scalar sample. Tensor results must be explicitly detached upstream."""
        result = {
            "raw": {k: finite_number(v, k) for k, v in self.raw.items()},
            "fixed_normalized": None
            if self.fixed_normalized is None
            else {k: finite_number(v, k) for k, v in self.fixed_normalized.items()},
            "weights": dict(self.weights),
            "space": self.space,
            "weighted": finite_number(self.weighted, "weighted"),
            "normalization": None
            if self.normalization is None
            else {
                k: {"mean": v.mean, "std": v.std, "source": v.source}
                for k, v in self.normalization.items()
            },
        }
        if self.diagnostics is not None:
            result["diagnostics"] = {k: finite_number(v, k) for k, v in self.diagnostics.items()}
        return result

    @classmethod
    def from_dict(cls, data):
        if set(data) - {"diagnostics"} != {
            "raw",
            "fixed_normalized",
            "weights",
            "normalization",
            "space",
            "weighted",
        }:
            raise ValueError("invalid score schema")
        finite_number(data["weighted"], "weighted")
        if data["fixed_normalized"] is not None:
            for value in data["fixed_normalized"].values():
                finite_number(value, "fixed_normalized")
        norm = data["normalization"]
        result = weighted_score(
            data["raw"],
            data["weights"],
            space=data["space"],
            normalization=None
            if norm is None
            else {k: FixedNormalization(**v) for k, v in norm.items()},
            diagnostics=data.get("diagnostics"),
        )
        if result.to_dict() != data:
            raise ValueError("recorded score does not match its raw values and config")
        return result


def _validate_values(values):
    if all(isinstance(v, Real) and not isinstance(v, bool) for v in values):
        for v in values:
            finite_number(v, "score")
        return
    # Lazy import: ordinary evaluation/scalar scoring does not require torch.
    try:
        import torch
    except ImportError as exc:
        raise ValueError("scores must be finite reals or floating torch tensors") from exc
    first = values[0]
    if not all(isinstance(v, torch.Tensor) and v.is_floating_point() for v in values):
        raise ValueError("do not mix scalar and tensor score components")
    for v in values:
        if (
            v.shape != first.shape
            or v.device != first.device
            or v.dtype != first.dtype
            or v.numel() == 0
            or not torch.isfinite(v).all().item()
        ):
            raise ValueError("tensor scores must be finite, nonempty, and share shape/device/dtype")


def weighted_score(raw, weights, *, space, normalization=None, diagnostics=None):
    """Return sum(weight[k] * score[k]); never rescale weights or center a batch.

    Keys must match exactly, including zero-weight components. Tensor inputs
    retain autograd and shape; no implicit broadcasting or batch reduction.
    """
    raw, weights = dict(raw), dict(weights)
    if not raw or any(not isinstance(k, str) or not k.strip() for k in raw):
        raise ValueError("nonempty named reward components required")
    raw = dict(sorted(raw.items()))
    if set(raw) != set(weights):
        raise ValueError("raw and weight components must match exactly")
    if space not in ("raw", "fixed_normalized"):
        raise ValueError("space must explicitly be raw or fixed_normalized")
    weights = {k: finite_number(v, "weight") for k, v in weights.items()}
    _validate_values(list(raw.values()))
    fixed = None
    if normalization is not None:
        normalization = dict(normalization)
        if set(normalization) != set(raw) or not all(
            isinstance(v, FixedNormalization) for v in normalization.values()
        ):
            raise ValueError("fixed normalization config must cover every component")
        fixed = {k: (v - normalization[k].mean) / normalization[k].std for k, v in raw.items()}
        _validate_values(list(fixed.values()))
    if space == "fixed_normalized" and fixed is None:
        raise ValueError("fixed_normalized scores require explicit fixed statistics")
    selected = raw if space == "raw" else fixed
    total = sum(weights[k] * selected[k] for k in raw)
    _validate_values([total])
    if diagnostics is not None:
        diagnostics = dict(diagnostics)
        if not diagnostics or any(not isinstance(k, str) or not k.strip() for k in diagnostics):
            raise ValueError("diagnostics require nonempty named values")
        _validate_values([total, *diagnostics.values()])
    return ScoreResult(
        MappingProxyType(raw),
        None if fixed is None else MappingProxyType(fixed),
        MappingProxyType(weights),
        None if normalization is None else MappingProxyType(normalization),
        space,
        total,
        None if diagnostics is None else MappingProxyType(diagnostics),
    )
