"""Strict, composable experiment configuration without executable config files."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


class _UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ConfigError("configuration keys must be strings")
        if key in result:
            raise ConfigError(f"duplicate configuration key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def _merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if key == "weights":
            result[key] = value
            continue
        if (
            key in ("algorithm", "model", "reward")
            and isinstance(value, dict)
            and "name" in value
            and isinstance(result.get(key), dict)
            and result[key].get("name") != value["name"]
        ):
            result[key] = value
            continue
        result[key] = (
            _merge(result[key], value)
            if isinstance(result.get(key), dict) and isinstance(value, dict)
            else value
        )
    return result


def _read(path: Path, stack: tuple[Path, ...] = ()) -> dict:
    path = path.resolve()
    if path in stack or len(stack) >= 16:
        raise ConfigError(f"cyclic or excessive config inheritance: {path.name}")
    if path.stat().st_size > 1024 * 1024:
        raise ConfigError("configuration exceeds 1 MiB")
    try:
        document = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueLoader)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML: {path.name}") from exc
    if not isinstance(document, dict):
        raise ConfigError("configuration must be a mapping")
    parents = document.pop("extends", [])
    if isinstance(parents, str):
        parents = [parents]
    if not isinstance(parents, list) or any(not isinstance(p, str) for p in parents):
        raise ConfigError("extends must be a path or list of paths")
    result: dict = {}
    for parent in parents:
        result = _merge(result, _read(path.parent / parent, (*stack, path)))
    return _merge(result, document)


def _expand(value: Any) -> Any:
    if isinstance(value, str):

        def substitute(match):
            name = match.group(1)
            if name not in os.environ:
                raise ConfigError(f"required environment variable is unset: {name}")
            return os.environ[name]

        return re.sub(r"\$\{([A-Za-z_][A-Za-z_0-9]*)\}", substitute, value)
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            optional_hash = (re.fullmatch(r"\$\{([A-Za-z_][A-Za-z_0-9]*)\}", item)
                             if key in {"sha256", "manifest_sha256"} and isinstance(item, str) else None)
            result[key] = (None if optional_hash and optional_hash[1] not in os.environ else _expand(item))
        return result
    return value


def _positive_int(value: Any, label: str, *, zero: bool = False) -> None:
    if type(value) is not int or value < (0 if zero else 1):
        raise ConfigError(f"{label} must be a {'nonnegative' if zero else 'positive'} integer")


def _finite(value: Any, label: str, *, positive: bool = False) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
        or value < 0
        or (positive and value == 0)
    ):
        raise ConfigError(f"{label} must be finite and {'positive' if positive else 'nonnegative'}")


def _update_steps(value, label):
    if (not isinstance(value, (list, tuple))
            or any(type(step) is not int or step < 1 for step in value)
            or list(value) != sorted(set(value))):
        raise ConfigError(f"{label} must be sorted distinct positive update numbers")
    return tuple(value)


def _artifact_due(step, maximum, every, explicit):
    return bool((every or explicit) and step > 0 and
                (step in explicit or (every and step % every == 0) or step == maximum))


def _parse(cls, data: Any):
    if not isinstance(data, dict):
        raise ConfigError(f"{cls.__name__} must be a mapping")
    extra = set(data) - {f.name for f in fields(cls)}
    if extra:
        raise ConfigError(f"unknown {cls.__name__} keys: {sorted(extra)}")
    try:
        return cls(**data)
    except TypeError as exc:
        raise ConfigError(f"invalid {cls.__name__}: {exc}") from exc


@dataclass(frozen=True)
class Component:
    name: str
    checkpoint: str | None = None
    revision: str | None = None
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ConfigError("component name must be nonempty")
        if not isinstance(self.options, dict):
            raise ConfigError("component options must be a mapping")
        for label in ("checkpoint", "revision"):
            if getattr(self, label) is not None and not isinstance(getattr(self, label), str):
                raise ConfigError(f"component {label} must be a string")


@dataclass(frozen=True)
class Reward:
    name: str
    weights: dict[str, float]
    checkpoint: str | None = None
    revision: str | None = None
    normalization: str = "raw"
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        Component(self.name, self.checkpoint, self.revision, self.options)
        if not isinstance(self.weights, dict) or not self.weights:
            raise ConfigError("reward weights must be a nonempty mapping")
        for key, value in self.weights.items():
            if not isinstance(key, str) or not key:
                raise ConfigError("reward component names must be nonempty strings")
            _finite(value, f"reward weight {key}")
        if not any(self.weights.values()):
            raise ConfigError("at least one reward weight must be positive")
        if self.normalization not in ("raw", "fixed"):
            raise ConfigError("reward normalization must be raw or fixed")


@dataclass(frozen=True)
class Sampling:
    steps: int = 16
    prompts_per_rollout: int = 32
    samples_per_prompt: int = 8
    microbatch: int = 8
    guidance_scale: float = 1.0
    height: int = 480
    width: int = 480
    frames: int = 53
    fps: float = 8.0
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        for key in (
            "steps",
            "prompts_per_rollout",
            "samples_per_prompt",
            "microbatch",
            "height",
            "width",
            "frames",
        ):
            _positive_int(getattr(self, key), f"sampling.{key}")
        _finite(self.fps, "sampling.fps", positive=True)
        _finite(self.guidance_scale, "sampling.guidance_scale")
        if not isinstance(self.options, dict):
            raise ConfigError("sampling.options must be a mapping")


@dataclass(frozen=True)
class Training:
    max_updates: int = 400
    learning_rate: float = 1e-5
    microbatch: int = 8
    grad_clip: float = 1.0
    precision: str = "bf16"
    optimizer: str = "adamw"
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    epsilon: float = 1e-8
    allow_tf32: bool | None = None
    timestep_selection: str = "configured"

    def __post_init__(self):
        if self.timestep_selection not in ("configured", "first", "high_noise"):
            raise ConfigError("training.timestep_selection must be configured, first or high_noise")
        _positive_int(self.max_updates, "training.max_updates")
        _positive_int(self.microbatch, "training.microbatch")
        for key in ("learning_rate", "grad_clip", "epsilon"):
            _finite(getattr(self, key), f"training.{key}", positive=True)
        _finite(self.weight_decay, "training.weight_decay")
        if self.precision not in ("fp32", "bf16", "fp16"):
            raise ConfigError("training.precision must be fp32, bf16 or fp16")
        if self.optimizer != "adamw":
            raise ConfigError("only AdamW is currently implemented")
        if self.allow_tf32 is not None and type(self.allow_tf32) is not bool:
            raise ConfigError("training.allow_tf32 must be boolean or null")
        if not isinstance(self.betas, (list, tuple)) or len(self.betas) != 2:
            raise ConfigError("training.betas must contain two numbers")
        for value in self.betas:
            _finite(value, "training.betas")
            if value >= 1:
                raise ConfigError("AdamW betas must be below one")
        object.__setattr__(self, "betas", tuple(self.betas))


@dataclass(frozen=True)
class Data:
    manifest: str
    sha256: str | None = None
    shuffle: bool = True
    sampler_seed: int | None = None
    order_shards: int = 1
    drop_last: bool = False

    def __post_init__(self):
        if not isinstance(self.manifest, str) or not self.manifest.strip():
            raise ConfigError("data.manifest must be a nonempty path")
        if self.sha256 is not None and (
            not isinstance(self.sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.sha256)
        ):
            raise ConfigError("data.sha256 must be a SHA256 or null")
        if type(self.shuffle) is not bool:
            raise ConfigError("data.shuffle must be boolean")
        if self.sampler_seed is not None and (
            type(self.sampler_seed) is not int or not 0 <= self.sampler_seed < 2**63
        ):
            raise ConfigError("data.sampler_seed must be an integer in [0, 2**63) or null")
        _positive_int(self.order_shards, "data.order_shards")
        if type(self.drop_last) is not bool:
            raise ConfigError("data.drop_last must be a boolean")


@dataclass(frozen=True)
class Evaluation:
    every_updates: int = 10
    every_rollouts: int = 0
    manifest: str | None = None
    retain_all: bool = True
    fps: float = 8.0
    reward_fps: float = 4.0
    manifest_sha256: str | None = None
    sampling_steps: int | None = None
    height: int | None = None
    width: int | None = None
    frames: int | None = None
    seed: int | None = None
    at_updates: tuple[int, ...] = ()
    shift: float | None = None

    def __post_init__(self):
        _positive_int(self.every_updates, "evaluation.every_updates", zero=True)
        _positive_int(self.every_rollouts, "evaluation.every_rollouts", zero=True)
        if self.every_rollouts and (self.every_updates or self.at_updates):
            raise ConfigError("rollout evaluation cadence cannot be mixed with update cadence")
        object.__setattr__(self, "at_updates", _update_steps(self.at_updates, "evaluation.at_updates"))
        _finite(self.fps, "evaluation.fps", positive=True)
        _finite(self.reward_fps, "evaluation.reward_fps", positive=True)
        if type(self.retain_all) is not bool:
            raise ConfigError("evaluation.retain_all must be boolean")
        if self.manifest is not None and not isinstance(self.manifest, str):
            raise ConfigError("evaluation.manifest must be a path")
        if self.manifest_sha256 is not None and (
            not isinstance(self.manifest_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", self.manifest_sha256)
        ):
            raise ConfigError("evaluation.manifest_sha256 must be a SHA256")
        if self.sampling_steps is not None:
            _positive_int(self.sampling_steps, "evaluation.sampling_steps")
        for key in ("height", "width", "frames"):
            if getattr(self, key) is not None:
                _positive_int(getattr(self, key), f"evaluation.{key}")
        if self.seed is not None:
            _positive_int(self.seed, "evaluation.seed", zero=True)
        if self.shift is not None:
            _finite(self.shift, "evaluation.shift", positive=True)

    @property
    def enabled(self):
        return bool(self.every_updates or self.every_rollouts or self.at_updates)

    def due(self, step, maximum, *, completed_rollouts=None):
        if self.every_rollouts:
            if type(completed_rollouts) is not int or completed_rollouts < 0:
                raise ConfigError("rollout evaluation requires the completed-rollout counter")
            return step > 0 and (step == maximum or
                                 completed_rollouts > 0 and completed_rollouts % self.every_rollouts == 0)
        return _artifact_due(step, maximum, self.every_updates, self.at_updates)


@dataclass(frozen=True)
class Runtime:
    output_dir: str = "outputs/experiment"
    device: str = "cuda"
    distributed: str = "none"
    checkpoint_every: int = 200
    log: str = "jsonl"
    checkpoint_at_updates: tuple[int, ...] = ()
    checkpoint_keep: int | None = None
    wandb: dict[str, str] = field(default_factory=dict)
    max_consecutive_no_signal_rollouts: int | None = None

    def __post_init__(self):
        if not isinstance(self.output_dir, str) or not self.output_dir.strip():
            raise ConfigError("runtime.output_dir is required")
        if self.device not in ("cpu", "cuda"):
            raise ConfigError("runtime.device must be cpu or cuda")
        if self.distributed not in ("none", "ddp", "fsdp"):
            raise ConfigError("unknown distributed strategy")
        _positive_int(self.checkpoint_every, "runtime.checkpoint_every", zero=True)
        if self.checkpoint_keep is not None:
            _positive_int(self.checkpoint_keep, "runtime.checkpoint_keep")
        if self.max_consecutive_no_signal_rollouts is not None:
            _positive_int(self.max_consecutive_no_signal_rollouts,
                          "runtime.max_consecutive_no_signal_rollouts")
        object.__setattr__(self, "checkpoint_at_updates",
                           _update_steps(self.checkpoint_at_updates, "runtime.checkpoint_at_updates"))
        if self.log not in ("jsonl", "wandb"):
            raise ConfigError("runtime.log must be jsonl or wandb")
        if (not isinstance(self.wandb, dict)
                or set(self.wandb) - {"project", "entity", "name", "mode"}
                or any(not isinstance(v, str) or not v.strip() for v in self.wandb.values())):
            raise ConfigError("runtime.wandb supports project, entity, name and mode; credentials belong in the environment")
        if self.wandb.get("mode", "online") not in {"online", "offline", "disabled"}:
            raise ConfigError("invalid runtime.wandb.mode")

    def checkpoint_due(self, step, maximum):
        return _artifact_due(step, maximum, self.checkpoint_every, self.checkpoint_at_updates)


@dataclass(frozen=True)
class Experiment:
    name: str
    task: str
    algorithm: Component
    model: Component
    reward: Reward
    sampling: Sampling
    training: Training
    evaluation: Evaluation
    runtime: Runtime
    seed: int = 42
    schema_version: int = 1
    provenance: dict[str, Any] = field(default_factory=dict)
    data: Data | None = None
    assets: dict[str, Any] = field(default_factory=dict)
    _checkpoint_identity: str | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.name, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.name
        ):
            raise ConfigError("name must be a filesystem-safe experiment identifier")
        if self.task not in ("t2i", "t2v", "i2v", "t2av"):
            raise ConfigError("task must be t2i, t2v, i2v or t2av")
        if self.evaluation.every_rollouts and self.model.name != "minimax_h3":
            raise ConfigError("rollout-based evaluation is currently supported by MiniMax-H3 only")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ConfigError("unsupported schema_version")
        _positive_int(self.seed, "seed", zero=True)
        if self.seed >= 2**32:
            raise ConfigError("seed must be smaller than 2**32")
        if not isinstance(self.provenance, dict):
            raise ConfigError("provenance must be a mapping")
        from .prepare_assets import validate_assets

        validate_assets(self.assets)
        if self.algorithm.name == "refl" and self.sampling.samples_per_prompt != 1:
            raise ConfigError("the supported ReFL recipe requires K=1")
        if self.training.timestep_selection != "configured" and self.algorithm.name != "diffusionart":
            raise ConfigError("training.timestep_selection is supported only by DiffusionART (DiffusionART-Flash)")
        if self.training.timestep_selection == "high_noise" and self.sampling.steps < 4:
            raise ConfigError("training.timestep_selection=high_noise requires at least four rollout steps")
        if (self.training.timestep_selection == "high_noise" and self.model.name == "minimax_h3"
                and self.sampling.steps < 6):
            raise ConfigError("H3 high_noise requires at least six sigma boundaries for four nonfinal rollout positions")
        if (self.training.timestep_selection == "high_noise" and self.model.name == "hunyuan_public"
                and self.sampling.steps < 5):
            raise ConfigError("Hunyuan high_noise requires at least five rollout steps for four nonfinal positions")
        if self.task == "t2i" and self.sampling.frames != 1:
            raise ConfigError("T2I requires sampling.frames=1")
        if any(step > self.training.max_updates for step in
               (*self.evaluation.at_updates, *self.runtime.checkpoint_at_updates)):
            raise ConfigError("explicit artifact steps must not exceed training.max_updates")

    def to_dict(self) -> dict:
        document = asdict(self)
        document.pop("_checkpoint_identity")
        # Preserve existing config/checkpoint identities when Flash is disabled.
        if self.training.timestep_selection == "configured":
            document["training"].pop("timestep_selection")
        if not self.evaluation.every_rollouts:
            document["evaluation"].pop("every_rollouts")
        if not self.assets:
            document.pop("assets")
        if self.data is not None:
            if self.data.sampler_seed is None:
                document["data"].pop("sampler_seed")
            if self.data.order_shards == 1:
                document["data"].pop("order_shards")
            if not self.data.drop_last:
                document["data"].pop("drop_last")
        if not self.runtime.wandb:
            document["runtime"].pop("wandb")
        if self.runtime.max_consecutive_no_signal_rollouts is None:
            document["runtime"].pop("max_consecutive_no_signal_rollouts")
        return document

    def fingerprint(self) -> str:
        return self._checkpoint_identity or config_fingerprint(self.to_dict())

    def execution_fingerprint(self) -> str:
        document = self.to_dict()
        document.pop("assets", None)
        return hashlib.sha256(json.dumps(document, sort_keys=True, allow_nan=False).encode()).hexdigest()


def resume_signature(document: dict) -> str:
    """Compare training semantics separately from explicitly operational choices."""
    document = dict(document)
    document.pop("assets", None)
    document["runtime"] = dict(document.get("runtime", {}))
    for key in ("output_dir", "checkpoint_keep", "checkpoint_every", "checkpoint_at_updates", "log", "wandb",
                "max_consecutive_no_signal_rollouts"):
        document["runtime"].pop(key, None)
    document["training"] = dict(document.get("training", {}))
    document["training"].pop("max_updates", None)
    document["evaluation"] = dict(document.get("evaluation", {}))
    for key in ("every_updates", "every_rollouts", "at_updates", "retain_all"):
        document["evaluation"].pop(key, None)
    return config_fingerprint(document)


def config_fingerprint(document: dict) -> str:
    """Exclude retention and liveness budgets, including for legacy checkpoints."""
    document = dict(document)
    document.pop("assets", None)
    document["runtime"] = dict(document.get("runtime", {}))
    document["runtime"].pop("checkpoint_keep", None)
    document["runtime"].pop("max_consecutive_no_signal_rollouts", None)
    return hashlib.sha256(
        json.dumps(document, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    ).hexdigest()


def load_config(path: str | Path, overrides: dict | None = None) -> Experiment:
    data = _expand(_merge(_read(Path(path)), overrides or {}))
    from .data import PromptDataset, file_sha256

    for section, path_key, sha_key in (("data", "manifest", "sha256"),
                                      ("evaluation", "manifest", "manifest_sha256")):
        spec = data.get(section)
        if (isinstance(spec, dict) and spec.get(path_key) and not spec.get(sha_key)
                and Path(spec[path_key]).is_file()):
            spec[sha_key] = (PromptDataset.load(spec[path_key], task="i2v").sha256
                             if section == "evaluation" and data.get("task") == "i2v"
                             else file_sha256(spec[path_key]))
    for name, cls in (("algorithm", Component), ("model", Component), ("reward", Reward)):
        value = data.get(name)
        if isinstance(value, str) and cls is Component:
            value = {"name": value}
        data[name] = _parse(cls, value)
    for name, cls in (
        ("sampling", Sampling),
        ("training", Training),
        ("evaluation", Evaluation),
        ("runtime", Runtime),
    ):
        data[name] = _parse(cls, data.get(name, {}))
    if data.get("data") is not None:
        data["data"] = _parse(Data, data["data"])
    result = _parse(Experiment, data)
    try:
        result.fingerprint()
    except (ValueError, TypeError, RecursionError) as exc:
        raise ConfigError("configuration must contain finite JSON-compatible values") from exc
    return result
