"""Config validation and lazy loaders shared by native image backends."""

import torch

from ..config import ConfigError, Reward
from .scalar import FixedNormalization

SINGLE_REWARDS = frozenset({"pickscore", "aesthetic", "clipscore", "ocr", "geneval"})
IMAGE_REWARDS = SINGLE_REWARDS | {"composite"}


def validate_reward_prompts(reward, prompts, *, metadata=None):
    if reward.name == "geneval" or (reward.name == "composite" and "geneval" in reward.weights):
        from .geneval import validate_metadata

        try:
            validate_metadata(prompts, metadata)
        except ValueError as exc:
            raise ConfigError(f"invalid GenEval dataset: {exc}") from exc
    if reward.name == "ocr" or (reward.name == "composite" and "ocr" in reward.weights):
        from .ocr import extract_target

        for index, prompt in enumerate(prompts):
            try:
                extract_target(prompt)
            except ValueError as exc:
                raise ConfigError(
                    f"OCR prompt at index {index} has no valid quoted target"
                ) from exc


def _keys(value, allowed, required, label):
    if not isinstance(value, dict) or set(value) - allowed or required - set(value):
        raise ConfigError(f"invalid {label} keys; allowed={sorted(allowed)}")


def _positive(value, label):
    if type(value) is not int or value < 1:
        raise ConfigError(f"{label} must be a positive integer")


def _children(reward):
    children = {}
    for name, spec in reward.options["components"].items():
        if name not in SINGLE_REWARDS:
            raise ConfigError(f"unsupported image reward component: {name}")
        _keys(
            spec,
            {"checkpoint", "revision", "options"},
            {"checkpoint", "revision", "options"},
            f"{name} component",
        )
        children[name] = Reward(name=name, weights={name: 1.0}, normalization="raw", **spec)
    return children


def _normalization(reward):
    raw = reward.options.get("fixed_statistics")
    if raw is None:
        if reward.normalization == "fixed":
            raise ConfigError("fixed composite scores require explicit fixed_statistics")
        return None
    if not isinstance(raw, dict) or set(raw) != set(reward.weights):
        raise ConfigError("fixed_statistics must cover every reward component")
    stats = {}
    for name, spec in raw.items():
        _keys(spec, {"mean", "std", "source"}, {"mean", "std", "source"}, "fixed_statistics")
        try:
            stats[name] = FixedNormalization(**spec)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"invalid {name} normalization: {exc}") from exc
    return stats


def validate_image_reward(reward):
    if reward.name not in IMAGE_REWARDS:
        raise ConfigError(f"unsupported native image reward: {reward.name}")
    r = reward.options
    if reward.name == "composite":
        _keys(
            r,
            {"microbatch", "components", "fixed_statistics"},
            {"microbatch", "components"},
            "composite reward",
        )
        if reward.checkpoint is not None or reward.revision is not None:
            raise ConfigError("composite asset paths/revisions belong to its components")
        if not isinstance(r["components"], dict) or set(r["components"]) != set(reward.weights):
            raise ConfigError("composite components and weights must match exactly")
        _positive(r["microbatch"], "composite microbatch")
        for child in _children(reward).values():
            validate_image_reward(child)
        _normalization(reward)
        return
    if any(
        not isinstance(value, str) or not value.strip()
        for value in (reward.checkpoint, reward.revision)
    ):
        raise ConfigError("reward checkpoint and revision must be explicit")
    if reward.name == "ocr":
        _keys(r, {"microbatch", "error_policy"}, {"microbatch", "error_policy"}, "OCR options")
        _positive(r["microbatch"], "OCR microbatch")
        if (
            reward.normalization != "raw"
            or reward.weights != {"ocr": 1.0}
            or r["error_policy"] not in ("raise", "zero")
            or len(reward.revision) != 64
            or any(c not in "0123456789abcdef" for c in reward.revision)
        ):
            raise ConfigError(
                "OCR requires raw unit weight, manifest SHA256 and explicit error_policy"
            )
        return
    if reward.name == "geneval":
        from pathlib import Path

        _keys(r, {"microbatch", "worker_python", "detector_microbatch", "color_workers",
                  "startup_timeout", "request_timeout"},
              {"microbatch", "worker_python", "detector_microbatch", "color_workers"}, "GenEval options")
        for name in ("microbatch", "detector_microbatch"):
            _positive(r[name], f"GenEval {name}")
        if (r["microbatch"] > 256 or type(r["color_workers"]) is not int
                or not 0 <= r["color_workers"] <= 16
                or not isinstance(r["worker_python"], str) or not Path(r["worker_python"]).is_absolute()
                or reward.normalization != "raw" or reward.weights != {"geneval": 1.0}
                or len(reward.revision) != 64
                or any(c not in "0123456789abcdef" for c in reward.revision)):
            raise ConfigError("GenEval requires raw unit weight, manifest SHA256 and explicit worker settings")
        for name in ("startup_timeout", "request_timeout"):
            value = r.get(name, 300)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not 0 < value <= 600):
                raise ConfigError(f"GenEval {name} must be positive and <=600 seconds")
        return
    head = {"head_path", "head_sha256"} if reward.name == "aesthetic" else set()
    resize = {"antialias"} if reward.name == "clipscore" else set()
    _keys(
        r,
        {"processor", "processor_revision", "microbatch", "local_files_only"} | head | resize,
        {"processor", "processor_revision", "microbatch"} | head | resize,
        "reward.options",
    )
    for key in ("processor", "processor_revision"):
        if not isinstance(r[key], str) or not r[key].strip():
            raise ConfigError(f"reward {key} must be explicit")
    if reward.weights != {reward.name: 1.0}:
        raise ConfigError("single-scorer weight must be 1; use composite for explicit weights")
    _positive(r["microbatch"], "reward microbatch")
    if type(r.get("local_files_only", True)) is not bool:
        raise ConfigError("reward local_files_only must be boolean")
    if reward.name == "clipscore":
        if type(r["antialias"]) is not bool:
            raise ConfigError("CLIPScore antialias must be explicitly boolean")
        import importlib.util

        if importlib.util.find_spec("torchvision") is None:
            raise ConfigError("CLIPScore requires the image-rewards extra (torchvision)")
    if reward.name == "aesthetic":
        digest = r["head_sha256"]
        if (
            reward.normalization != "raw"
            or not isinstance(r["head_path"], str)
            or not r["head_path"].strip()
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ConfigError("Aesthetic requires raw scores and explicit head path/SHA256")


def load_image_reward(reward, *, device):
    validate_image_reward(reward)
    if reward.name == "geneval":
        from .geneval import GenEvalScore
        from .geneval_loader import verify_bundle
        from .geneval_process import GenEvalWorker

        verify_bundle(reward.checkpoint, reward.revision)
        target = torch.device(device)
        if target.type == "cuda" and target.index is None:
            target = torch.device("cuda", torch.cuda.current_device())
        r = reward.options
        engine = GenEvalWorker(
            reward.checkpoint, reward.revision, python=r["worker_python"], device=str(target),
            detector_microbatch=r["detector_microbatch"], color_workers=r["color_workers"],
            startup_timeout=r.get("startup_timeout", 300), request_timeout=r.get("request_timeout", 300),
        )
        try:
            return GenEvalScore(engine, device=target)
        except BaseException:
            engine.close()
            raise
    if reward.name == "ocr":
        from .paddle_loader import load_ocr_reward

        return load_ocr_reward(
            manifest_path=reward.checkpoint,
            manifest_sha256=reward.revision,
            error_policy=reward.options["error_policy"],
            device=device,
        )
    if reward.name == "composite":
        from .composite import CompositeImageReward

        children = _children(reward)
        loaded = {}
        try:
            for name, child in sorted(children.items()):
                loaded[name] = load_image_reward(child, device=device)
            return CompositeImageReward(
                loaded, reward.weights,
                space="fixed_normalized" if reward.normalization == "fixed" else "raw",
                normalization=_normalization(reward),
                microbatches={name: child.options["microbatch"] for name, child in children.items()},
            )
        except BaseException:
            for scorer in loaded.values():
                close = getattr(scorer, "close", None)
                if close is not None:
                    close()
            raise
    r = reward.options
    options = dict(
        model_path=reward.checkpoint,
        model_revision=reward.revision,
        processor_path=r["processor"],
        processor_revision=r["processor_revision"],
        device=device,
        dtype=torch.float32,
        local_files_only=r.get("local_files_only", True),
    )
    if reward.name == "aesthetic":
        from .aesthetic import AestheticScore

        return AestheticScore.from_pretrained(
            **options, head_path=r["head_path"], head_sha256=r["head_sha256"]
        )
    if reward.name == "clipscore":
        from .clipscore import CLIPScore

        return CLIPScore.from_pretrained(
            **options,
            antialias=r["antialias"],
            space="fixed_normalized" if reward.normalization == "fixed" else "raw",
        )
    from .pickscore import PickScore

    return PickScore.from_pretrained(
        **options, space="fixed_normalized" if reward.normalization == "fixed" else "raw"
    )
