"""Historical JPEG GenEval reward; dense optimization and binary diagnostics differ."""

from collections.abc import Sequence
from copy import deepcopy
from io import BytesIO
from numbers import Real

import torch

from .pickscore import quantized_rgb
from .scalar import weighted_score

TAGS = ("single_object", "two_object", "counting", "colors", "position", "color_attr")
COLORS = ("red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white")
RELATIONS = ("left of", "right of", "above", "below")


def validate_metadata(prompts, metadata):
    if (not isinstance(prompts, Sequence) or isinstance(prompts, (str, bytes))
            or not prompts or any(not isinstance(p, str) or not p.strip() for p in prompts)
            or not isinstance(metadata, Sequence) or isinstance(metadata, (str, bytes))
            or len(metadata) != len(prompts) or not prompts):
        raise ValueError("GenEval needs one metadata object per prompt")
    for prompt, row in zip(prompts, metadata, strict=True):
        if (not isinstance(row, dict) or row.get("prompt") != prompt
                or row.get("tag") not in TAGS or not isinstance(row.get("include"), list)
                or not row["include"] or not isinstance(row.get("exclude", []), list)):
            raise ValueError("invalid GenEval prompt/tag/include/exclude metadata")
        for kind in ("include", "exclude"):
            for index, clause in enumerate(row.get(kind, [])):
                allowed = {"class", "count", "color", "position"} if kind == "include" else {"class", "count"}
                if (not isinstance(clause, dict) or set(clause) - allowed
                        or not isinstance(clause.get("class"), str) or not clause["class"].strip()
                        or type(clause.get("count")) is not int or clause["count"] < 1):
                    raise ValueError("invalid GenEval object clause")
                if "color" in clause and clause["color"] not in COLORS:
                    raise ValueError("invalid GenEval color")
                if "position" in clause:
                    position = clause["position"]
                    if (not isinstance(position, list) or len(position) != 2
                            or position[0] not in RELATIONS or type(position[1]) is not int
                            or not 0 <= position[1] < index):
                        raise ValueError("GenEval position must reference an earlier include clause")


def jpeg_bytes(images):
    from PIL import Image

    if isinstance(images, torch.Tensor):
        images = quantized_rgb(images)
    if (not isinstance(images, Sequence) or not images
            or any(not isinstance(im, Image.Image) or im.mode != "RGB" for im in images)):
        raise ValueError("GenEval requires RGB PIL images or floating RGB tensors")
    encoded = []
    for image in images:
        output = BytesIO()
        image.save(output, format="JPEG")
        encoded.append(output.getvalue())
    return encoded


class GenEvalScore(torch.nn.Module):
    """The engine accepts JPEG bytes and returns the five historical score arrays.

    No gradient, binary substitution, score clipping, or batch normalization is
    applied. Engine/transport errors are fatal, never fabricated as zero reward.
    """

    requires_metadata = True
    supports_evaluation_mode = True

    def __init__(self, engine, *, device="cpu"):
        super().__init__()
        self.engine, self.device = engine, torch.device(device)
        self.eval()

    @torch.no_grad()
    def forward(self, images, prompts, *, microbatch, metadata, evaluation=False):
        validate_metadata(prompts, metadata)
        if type(microbatch) is not int or microbatch < 1 or type(evaluation) is not bool:
            raise ValueError("invalid GenEval microbatch/evaluation mode")
        if len(images) != len(prompts):
            raise ValueError("GenEval image/prompt count mismatch")
        raw, diagnostics = [], {}
        for start in range(0, len(prompts), microbatch):
            stop = min(start + microbatch, len(prompts))
            rows = metadata[start:stop]
            reply = self.engine(jpeg_bytes(images[start:stop]), deepcopy(rows), only_strict=not evaluation)
            if not isinstance(reply, (tuple, list)) or len(reply) != 5:
                raise ValueError("invalid GenEval result envelope")
            scores, ordinary, strict, grouped, grouped_strict = reply

            def values(data, *, allowed=None):
                if (not isinstance(data, (tuple, list)) or len(data) != len(rows)
                        or any(isinstance(x, bool) or not isinstance(x, Real) for x in data)):
                    raise ValueError("invalid GenEval score array")
                tensor = torch.tensor(data, dtype=torch.float32, device=self.device)
                if not torch.isfinite(tensor).all() or (
                    allowed is not None and any(x not in allowed for x in data)
                ):
                    raise ValueError("invalid GenEval numeric score")
                return tensor

            raw.append(values(scores))
            binary = {"strict_accuracy": values(strict, allowed=(0, 1))}
            values(ordinary, allowed=(0, 1))
            if evaluation:
                binary["accuracy"] = values(ordinary, allowed=(0, 1))
            for groups, accuracy in ((grouped, ordinary), (grouped_strict, strict)):
                if not isinstance(groups, dict) or set(groups) != set(TAGS):
                    raise ValueError("invalid GenEval grouped score keys")
                for tag in TAGS:
                    values(groups[tag], allowed=(-10, 0, 1))
                    expected = [score if row["tag"] == tag else -10.0
                                for score, row in zip(accuracy, rows, strict=True)]
                    if groups[tag] != expected:
                        raise ValueError("GenEval grouped scores do not match metadata")
            for name, tensor in binary.items():
                diagnostics.setdefault(f"geneval/{name}", []).append(tensor)
                groups = grouped_strict if name == "strict_accuracy" else grouped
                for tag in TAGS:
                    diagnostics.setdefault(f"geneval/tag/{tag}/{name}", []).append(values(groups[tag]))
        return weighted_score(
            {"geneval": torch.cat(raw)}, {"geneval": 1.0}, space="raw",
            diagnostics={name: torch.cat(parts) for name, parts in diagnostics.items()},
        )

    def close(self):
        close = getattr(self.engine, "close", None)
        if close is not None:
            close()


def summarize_diagnostics(diagnostics):
    """Reduce valid tag members, not the historical -10 nonmember sentinels."""
    metrics = {}
    for name, values in (diagnostics or {}).items():
        if name.startswith("geneval/tag/"):
            values = values[values != -10]
        if values.numel():
            metrics[name] = float(values.mean())
    for metric in ("accuracy", "strict_accuracy"):
        groups = [metrics[f"geneval/tag/{tag}/{metric}"] for tag in TAGS
                  if f"geneval/tag/{tag}/{metric}" in metrics]
        if groups:
            metrics[f"geneval/{metric}_macro"] = sum(groups) / len(groups)
            metrics[f"geneval/{metric}_tags_present"] = len(groups)
    return metrics
