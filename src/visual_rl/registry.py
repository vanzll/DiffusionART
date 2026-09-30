"""Lazy backend selection with explicit capability validation."""

from dataclasses import dataclass
from importlib import import_module
from typing import Callable

from .config import ConfigError, Experiment


@dataclass(frozen=True)
class Backend:
    name: str
    tasks: frozenset[str]
    algorithms: frozenset[str]
    rewards: frozenset[str]
    entrypoint: str
    precisions: frozenset[str] = frozenset({"fp32"})
    distributed: frozenset[str] = frozenset({"none"})
    status: str = "development"
    config_validator: str | None = None

    def validate(self, config: Experiment):
        for value, allowed, label in (
            (config.task, self.tasks, "task"),
            (config.algorithm.name, self.algorithms, "algorithm"),
            (config.reward.name, self.rewards, "reward"),
            (config.training.precision, self.precisions, "precision"),
            (config.runtime.distributed, self.distributed, "distributed strategy"),
        ):
            if value not in allowed:
                raise ConfigError(
                    f"{self.name} does not support {label} {value!r}; allowed: {sorted(allowed)}"
                )
        if self.config_validator:
            module, name = self.config_validator.split(":", 1)
            getattr(import_module(module), name)(config)

    def load(self) -> Callable:
        module, name = self.entrypoint.split(":", 1)
        function = getattr(import_module(module), name)
        if not callable(function):
            raise TypeError("backend entrypoint is not callable")
        return function


_BACKENDS: dict[str, Backend] = {}


def register_backend(backend: Backend):
    if backend.name in _BACKENDS:
        raise ValueError(f"backend already registered: {backend.name}")
    _BACKENDS[backend.name] = backend


def get_backend(config: Experiment) -> Backend:
    try:
        backend = _BACKENDS[config.model.name]
    except KeyError as exc:
        raise ConfigError(f"model backend {config.model.name!r} is not integrated yet") from exc
    backend.validate(config)
    return backend


def list_backends():
    return tuple(_BACKENDS[key] for key in sorted(_BACKENDS))


register_backend(Backend(
    name="wan_i2v_distilled", tasks=frozenset({"i2v"}),
    distributed=frozenset({"none", "fsdp"}),
    algorithms=frozenset({"diffusionart", "diffusionnft", "flow_grpo_fast", "dancegrpo", "refl"}),
    rewards=frozenset({"video_reward"}), precisions=frozenset({"bf16"}),
    entrypoint="visual_rl.backends.wan:run", config_validator="visual_rl.backends.wan:validate_config",
    status="native distilled Wan integration; paper/GPU acceptance pending",
))


register_backend(Backend(
    name="hunyuan_public", tasks=frozenset({"t2v"}),
    distributed=frozenset({"none", "fsdp"}),
    algorithms=frozenset({"diffusionart", "diffusionnft", "refl", "flow_grpo_fast", "dancegrpo"}), rewards=frozenset({"video_reward"}),
    precisions=frozenset({"bf16"}),
    entrypoint="visual_rl.backends.hunyuan:run", config_validator="visual_rl.backends.hunyuan:validate_config",
    status="native single-device/FSDP ART/NFT/ReFL/Fast/Dance integration; paper/GPU acceptance pending",
))


register_backend(Backend(
    name="flux", tasks=frozenset({"t2i"}),
    distributed=frozenset({"none", "ddp"}),
    algorithms=frozenset({"diffusionart", "diffusionnft", "flow_grpo", "flow_grpo_fast", "flow_dppo"}),
    rewards=frozenset({"pickscore", "aesthetic", "clipscore", "ocr", "geneval", "composite"}),
    precisions=frozenset({"fp32", "bf16"}),
    entrypoint="visual_rl.backends.sd3:run", config_validator="visual_rl.backends.sd3:validate_config",
    status="native development runner; image DDP integration, paper/GPU acceptance pending",
))


register_backend(Backend(
    name="sd3", tasks=frozenset({"t2i"}),
    distributed=frozenset({"none", "ddp"}),
    algorithms=frozenset({"diffusionart", "diffusionnft", "flow_grpo", "flow_grpo_fast", "flow_dppo", "dancegrpo"}),
    rewards=frozenset({"pickscore", "aesthetic", "clipscore", "ocr", "geneval", "composite"}), precisions=frozenset({"fp32", "bf16", "fp16"}),
    entrypoint="visual_rl.backends.sd3:run", config_validator="visual_rl.backends.sd3:validate_config",
    status="native development runner; image DDP integration, paper/GPU acceptance pending",
))


register_backend(
    Backend(
        name="tensor_fixture",
        tasks=frozenset({"t2i"}),
        algorithms=frozenset(
            {
                "diffusionart",
                "diffusionnft",
                "flow_grpo",
                "flow_grpo_fast",
                "dancegrpo",
                "flow_dppo",
                "refl",
            }
        ),
        rewards=frozenset({"quadratic"}),
        entrypoint="visual_rl.backends.tensor_fixture:run",
        status="CPU software fixture; not a paper reproduction",
        config_validator="visual_rl.backends.tensor_fixture:validate_config",
    )
)
