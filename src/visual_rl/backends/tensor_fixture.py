"""Small CPU integration fixture, NOT a generative model or paper benchmark."""

import json
import random
from pathlib import Path

import numpy as np
import torch

from ..algorithms import art_scalar_covariance, old_policy_decay, refl_loss
from ..config import ConfigError
from ..sampling import shifted_schedule
from ..training.checkpoint import _atomic_json, load_checkpoint, save_checkpoint
from ..training.objectives import (
    ARTObjective,
    GRPOObjective,
    NFTObjective,
    disjoint_windows,
    group_advantages,
)
from ..training.rollout import collect_rollout, endpoint_renoising
from ..training.state import CostClock, PolicyShadow, Progress
from ..training.update import accumulated_step


class TensorPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(6, 16), torch.nn.Tanh(), torch.nn.Linear(16, 4)
        )

    def forward(self, state, sigma, condition):
        return self.network(torch.cat((state, sigma[:, None], condition[:, None]), dim=1))


def validate_config(config):
    if (
        config.runtime.device != "cpu"
        or config.runtime.log != "jsonl"
        or config.model.options
        or config.model.checkpoint
        or config.reward.options
        or config.reward.weights != {"quadratic": 1.0}
        or config.reward.normalization != "raw"
        or config.sampling.options
        or config.evaluation.enabled
        or config.training.allow_tf32 is not None
        or config.algorithm.options
        or config.data is not None
        or config.evaluation.manifest is not None
        or config.evaluation.manifest_sha256 is not None
        or config.evaluation.sampling_steps is not None
        or config.evaluation.shift is not None
    ):
        raise ConfigError(
            "tensor_fixture only supports its documented CPU software-test configuration"
        )
    if (
        config.sampling.prompts_per_rollout > 16
        or config.sampling.samples_per_prompt > 8
        or config.training.max_updates > 20
    ):
        raise ConfigError("tensor_fixture is bounded to tiny integration tests, not paper training")
    if config.sampling.guidance_scale != 1 or config.sampling.steps < 2:
        raise ConfigError("tensor_fixture uses no CFG and at least two steps")
    algorithm = config.algorithm.name
    k = config.sampling.samples_per_prompt
    if algorithm != "refl" and (
        k < 2 or k % 2 or config.training.max_updates % (1 if algorithm == "diffusionnft" else 2)
    ):
        raise ConfigError("group fixture requires even K and complete update windows")


def run(config, *, resume=None, resume_rewind=False, evaluate_only=False):
    from ..training.resume import prepare_resume_config, validate_resume_request

    config, resume = prepare_resume_config(config, resume)
    validate_resume_request(resume, resume_rewind, evaluate_only)
    if evaluate_only:
        raise ConfigError("CPU fixture has no media evaluation; use a model backend")
    validate_config(config)
    algorithm = config.algorithm.name
    k, p = config.sampling.samples_per_prompt, config.sampling.prompts_per_rollout
    root = Path(config.runtime.output_dir)
    if resume is None:
        root.mkdir(parents=True, exist_ok=False)
    elif not root.is_dir():
        raise ConfigError("fixture resume requires its original output directory")
    # Lock ownership prevents competing local writers. No stale PID takeover.
    import fcntl

    with (root / ".owner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("output directory already has an active writer") from exc
        _run_owned(config, root, resume, algorithm, p, k, resume_rewind=resume_rewind)


def _run_owned(config, root, resume, algorithm, p, k, *, resume_rewind=False):
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    model = TensorPolicy().eval()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        betas=config.training.betas,
        eps=config.training.epsilon,
        weight_decay=config.training.weight_decay,
    )
    shadow = PolicyShadow(model) if algorithm == "diffusionnft" else None
    generator = torch.Generator().manual_seed(config.seed)
    progress = Progress()
    components = {"old_policy": shadow} if shadow else {}
    if resume:
        progress = load_checkpoint(
            resume,
            model=model,
            optimizer=optimizer,
            config_sha256=config.fingerprint(),
            stateful=components,
            generators={"sampling": generator},
        )
        from ..training.resume import prepare_resume_history

        prepare_resume_history(config, root, progress, receipt=resume, rewind=resume_rewind)
    else:
        _atomic_json(root / "resolved-config.json", config.to_dict())
        _atomic_json(
            root / "run.json",
            {
                "kind": "software_fixture_not_paper_result",
                "config_sha256": config.fingerprint(),
                "device": "cpu",
            },
        )
    clock = CostClock(progress, gpu_count=0)
    groups = torch.arange(p).repeat_interleave(k)
    condition = groups.float() / p
    sigmas = shifted_schedule(config.sampling.steps, 1.0)
    selected = tuple(range(config.sampling.steps - 1))

    def predict(state, sigma, rows):
        return model(state, sigma, condition[rows])

    def score(endpoint, rows=None):
        c = condition if rows is None else condition[rows]
        return -(endpoint - c[:, None]).square().mean(1)

    with (root / "metrics.jsonl").open("a", encoding="utf-8") as log:
        while progress.optimizer_steps < config.training.max_updates:
            with clock.training_phase():
                progress.attempts += 1
                noise = torch.randn((p * k, 4), generator=generator)
                policy_gradient = algorithm in (
                    "flow_grpo",
                    "flow_grpo_fast",
                    "dancegrpo",
                    "flow_dppo",
                )
                stochastic = (
                    ((0,) if algorithm == "flow_grpo_fast" else selected) if policy_gradient else ()
                )
                chosen = (
                    stochastic
                    if policy_gradient
                    else ((len(sigmas) - 2,) if algorithm == "refl" else selected)
                )
                innovations = {i: torch.randn(noise.shape, generator=generator) for i in stochastic}
                kwargs = dict(
                    selected=chosen,
                    stochastic=stochastic,
                    innovations=innovations,
                    microbatch=config.sampling.microbatch,
                )
                if shadow:
                    with shadow.use():
                        cache = collect_rollout(predict, noise, sigmas, **kwargs)
                else:
                    cache = collect_rollout(predict, noise, sigmas, **kwargs)
                rewards = score(cache.endpoints).detach()
                if algorithm == "diffusionart":
                    statistics = art_scalar_covariance(cache.endpoints, rewards, groups, p)
                    objective = ARTObjective(
                        cache,
                        predict,
                        statistics["group_directions"][groups],
                        statistics["group_guidance_magnitudes"][groups],
                        torch.zeros(p * k, dtype=torch.bool),
                        torch.device("cpu"),
                    )
                    windows = disjoint_windows(p * k, len(chosen))
                elif policy_gradient:
                    advantages = group_advantages(
                        rewards,
                        groups,
                        mode="group_sample_std"
                        if algorithm == "dancegrpo"
                        else "global_population_std",
                        eps=1e-8 if algorithm == "dancegrpo" else 1e-4,
                    )
                    options = (
                        {}
                        if algorithm == "flow_dppo"
                        else {"ratio_clip": 1e-4 if algorithm == "dancegrpo" else 1e-5}
                    )
                    objective = GRPOObjective(
                        cache, predict, advantages, torch.device("cpu"), algorithm, options
                    )
                    windows = disjoint_windows(p * k, len(chosen))
                elif algorithm == "diffusionnft":
                    rows = torch.arange(p * k).repeat_interleave(len(chosen))
                    sigma = sigmas[list(chosen)].repeat(p * k)
                    endpoints = cache.endpoints[rows]
                    states = endpoint_renoising(
                        endpoints, sigma, torch.randn(endpoints.shape, generator=generator)
                    )
                    with shadow.use(), torch.no_grad():
                        old = predict(states, sigma, rows).detach().clone()
                    advantages = group_advantages(
                        rewards, groups, mode="global_population_std", eps=1e-4
                    )
                    objective = NFTObjective(
                        endpoints,
                        states,
                        sigma,
                        old,
                        rows,
                        advantages[rows],
                        predict,
                        torch.device("cpu"),
                    )
                    windows = (tuple(range(len(rows))),)
                elif algorithm == "refl":
                    # Stop the entire prefix; differentiate only the final Euler step.
                    last = cache

                    def objective(items):
                        rows = torch.tensor(items)
                        state = last.states[rows, 0]
                        t = sigmas[-2].expand(len(rows))
                        endpoint = state + (sigmas[-1] - sigmas[-2]) * predict(state, t, rows)
                        return refl_loss({"quadratic": score(endpoint, rows)}, {"quadratic": 1.0})[
                            "per_item_loss"
                        ]

                    windows = (tuple(range(p)),)
                else:
                    raise ConfigError("unknown fixture algorithm")
                for items in windows:
                    metrics = accumulated_step(
                        model,
                        optimizer,
                        items,
                        objective,
                        microbatch=config.training.microbatch,
                        denominator=len(items),
                        max_grad_norm=config.training.grad_clip,
                    )
                    if shadow:
                        shadow.update(old_policy_decay(progress.optimizer_steps, "slow-ema"))
                    progress.optimizer_steps += 1
                    clock.tick()
                    metrics.update(clock.metrics())
                    metrics["train/reward/quadratic"] = float(rewards.mean())
                    log.write(json.dumps(metrics, sort_keys=True, allow_nan=False) + "\n")
                    log.flush()
                progress.completed_rollouts += 1
            if config.runtime.checkpoint_due(progress.optimizer_steps, config.training.max_updates):
                save_checkpoint(
                    root / "checkpoints",
                    keep_last=config.runtime.checkpoint_keep,
                    model=model,
                    optimizer=optimizer,
                    progress=progress,
                    config_sha256=config.fingerprint(),
                    stateful=components,
                    generators={"sampling": generator},
                )
        _atomic_json(
            root / "status.json",
            {
                "status": "completed",
                "kind": "software_fixture_not_paper_result",
                "config_sha256": config.fingerprint(),
                **progress.state_dict(),
            },
        )
