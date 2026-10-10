"""Native image GRPO lifecycle with explicit logical budgets and evaluation EMA."""

import math
from contextlib import nullcontext

import torch

from ..algorithms import clipped_grpo_loss, flowdppo_loss
from ..config import ConfigError
from .image_contracts import prompt_advantages, sd3_schedule
from .objectives import GRPOObjective, group_advantages
from .state import PolicyShadow

ALGORITHMS = frozenset({"flow_grpo", "flow_grpo_fast", "flow_dppo", "dancegrpo"})


def _keys(value, names, label):
    if not isinstance(value, dict) or set(value) != set(names):
        raise ConfigError(f"{label} requires exactly {sorted(names)}")


def validate_grpo(c):
    a, s = c.algorithm.options, c.sampling.options
    is_flux = c.model.name == "flux"
    is_dance = c.algorithm.name == "dancegrpo"
    _keys(
        {key: value for key, value in a.items() if key != "multi_reward_advantage"},
        {
            "loss",
            "updates_per_rollout",
            "training_batches",
            "shuffle_rows",
            "zero_advantage_mode",
            "max_empty_rollouts",
            "evaluation_ema",
        } | ({"advantage_mode"} if is_dance else set()),
        "GRPO options",
    )
    _keys(
        s,
        {
            "transition_kind",
            "likelihood",
            "noise_level",
            "selection",
            "pool",
            "count",
            "logical_batch_size",
            "unselected_noise",
        } | (set() if is_flux else {"shift"}) | ({"same_latent", "endpoint_kind"} if is_dance else set()),
        "GRPO sampling",
    )
    for key in ("updates_per_rollout", "training_batches", "max_empty_rollouts"):
        if type(a[key]) is not int or a[key] < 1:
            raise ConfigError(f"{key} must be a positive integer")
    rows = c.sampling.prompts_per_rollout * c.sampling.samples_per_prompt
    if rows % a["training_batches"] or a["training_batches"] % a["updates_per_rollout"]:
        raise ConfigError("rows, logical training batches and updates must divide evenly")
    for value in (c.training.max_updates, c.runtime.checkpoint_every, c.evaluation.every_updates,
                  *c.evaluation.at_updates, *c.runtime.checkpoint_at_updates):
        if value % a["updates_per_rollout"]:
            raise ConfigError("budget/artifact intervals must align to complete GRPO rollouts")
    if type(a["shuffle_rows"]) is not bool or a["zero_advantage_mode"] not in (
        "retain",
        "filter_pad",
    ):
        raise ConfigError("invalid GRPO row selection")
    ema = a["evaluation_ema"]
    _keys(ema, {"enabled", "decay", "refresh"}, "GRPO evaluation EMA")
    if type(ema["enabled"]) is not bool or ema["refresh"] not in ("optimizer", "logical_batch"):
        raise ConfigError("invalid GRPO evaluation EMA mode")
    if (
        isinstance(ema["decay"], bool)
        or not isinstance(ema["decay"], (int, float))
        or not math.isfinite(ema["decay"])
        or not 0 <= ema["decay"] < 1
    ):
        raise ConfigError("invalid GRPO evaluation EMA decay")
    if not is_flux and c.sampling.guidance_scale != 1:
        raise ConfigError("native SD3 GRPO currently supports no-CFG recipes only")
    if s["transition_kind"] not in ("sde", "cps", "dance") or s["likelihood"] not in ("gaussian", "mse"):
        raise ConfigError("unsupported GRPO transition/likelihood")
    if is_dance:
        if (is_flux or s["transition_kind"] != "dance" or s["likelihood"] != "gaussian"
                or s["unselected_noise"] != "all" or type(s["same_latent"]) is not bool):
            raise ConfigError("image Dance requires SD3, Gaussian Dance SDE, full exploration and boolean same_latent")
        if s["endpoint_kind"] not in ("sample", "last_x0"):
            raise ConfigError("invalid Dance endpoint_kind")
        if a["advantage_mode"] not in ("prompt_global_std", "group_sample_std"):
            raise ConfigError("invalid Dance advantage_mode")
    elif s["transition_kind"] == "dance":
        raise ConfigError("Dance transition requires the DanceGRPO algorithm")
    if s["transition_kind"] == "cps" and s["likelihood"] != "mse":
        raise ConfigError("canonical CPS uses reduced negative MSE, not Gaussian likelihood")
    if is_flux and c.algorithm.name == "flow_dppo" and (
        s["transition_kind"] != "sde" or s["likelihood"] != "gaussian"
    ):
        raise ConfigError("FLUX Flow-DPPO requires SDE with Gaussian likelihood")
    if (
        isinstance(s["noise_level"], bool)
        or not isinstance(s["noise_level"], (int, float))
        or not math.isfinite(s["noise_level"])
        or not 0 < s["noise_level"] <= 1
    ):
        raise ConfigError("noise_level must be in (0,1]")
    pool = s["pool"]
    if (
        not isinstance(pool, list)
        or not pool
        or any(type(x) is not int or not 0 <= x < c.sampling.steps for x in pool)
        or pool != sorted(set(pool))
    ):
        raise ConfigError("GRPO pool must be sorted distinct valid timestep indices")
    if type(s["count"]) is not int or not 1 <= s["count"] <= len(pool):
        raise ConfigError("invalid selected timestep count")
    if s["selection"] not in ("fixed", "contiguous", "random_subset", "random_per_row"):
        raise ConfigError("invalid GRPO timestep selection")
    if s["selection"] == "random_per_row" and s["unselected_noise"] != "all":
        raise ConfigError("per-row selections require full exploration")
    if s["selection"] == "fixed" and s["count"] != len(pool):
        raise ConfigError("fixed selection must use the whole pool")
    if s["selection"] == "contiguous" and pool != list(range(pool[0], pool[-1] + 1)):
        raise ConfigError("contiguous selection requires a contiguous pool")
    if (
        type(s["logical_batch_size"]) is not int
        or s["logical_batch_size"] < 1
        or rows % s["logical_batch_size"]
    ):
        raise ConfigError("logical sampling batch must divide the rollout rows")
    if s["unselected_noise"] not in ("ode", "all"):
        raise ConfigError("unselected_noise must be ode or all")
    if s["transition_kind"] == "cps" and c.sampling.steps - 1 in pool:
        raise ConfigError("terminal zero-variance CPS cannot be trained")
    if c.algorithm.name == "flow_grpo" and (
        s["transition_kind"] != "sde" or s["unselected_noise"] != "all"
    ):
        raise ConfigError("Flow-GRPO requires full SDE rollout; windowed rollout is Fast")
    loss = a["loss"]
    _keys(
        loss,
        {"kl_mask_threshold", "advantage_clip"}
        if c.algorithm.name == "flow_dppo"
        else {"ratio_clip", "advantage_clip"},
        "GRPO loss",
    )
    clip = loss["advantage_clip"]
    if (
        isinstance(clip, bool)
        or not isinstance(clip, (int, float))
        or not math.isfinite(clip)
        or clip <= 0
    ):
        raise ConfigError("advantage_clip must be finite and positive")
    try:
        if is_flux:
            from .flux_contracts import flux_schedule

            schedule = flux_schedule(c.sampling.steps, height=c.sampling.height,
                                     width=c.sampling.width,
                                     spatial_factor=c.model.options["spatial_factor"],
                                     scheduler=c.model.options["scheduler"])
        else:
            schedule = sd3_schedule(c.sampling.steps, s["shift"])
        if s["transition_kind"] == "sde" and len(schedule) < 3:
            raise ValueError("Flow-SDE needs a nonterminal t=1 variance guard")
        dummy = torch.zeros(2)
        if c.algorithm.name == "flow_dppo":
            flowdppo_loss(
                dummy,
                dummy,
                dummy[:, None],
                dummy[:, None],
                torch.ones(2),
                dummy,
                kl_mask_threshold=loss["kl_mask_threshold"],
            )
        else:
            clipped_grpo_loss(dummy, dummy, dummy, **loss)
    except (ValueError, TypeError) as exc:
        raise ConfigError(f"invalid GRPO settings: {exc}") from exc


class GRPOState:
    def __init__(self, model, options):
        self.options = options
        self.ema = (
            PolicyShadow(model, trainable_only=True)
            if options["evaluation_ema"]["enabled"]
            else None
        )

    def stateful(self):
        return {"grpo_eval_ema": self.ema} if self.ema is not None else {}

    def evaluation(self):
        return self.ema.use() if self.ema is not None else nullcontext()

    def _refresh(self, step):
        if self.ema is not None:
            decay = min((1 + step) / (10 + step), self.options["evaluation_ema"]["decay"])
            self.ema.update(decay)

    def before_update(self, completed_updates):
        # The research loop refreshes EMA after every logical data batch, even
        # during gradient accumulation. Parameters are unchanged until the last
        # batch, so these refreshes may be performed before the combined backward.
        if self.options["evaluation_ema"]["refresh"] == "logical_batch":
            batches = self.options["training_batches"] // self.options["updates_per_rollout"]
            for _ in range(batches - 1):
                self._refresh(completed_updates)

    def after_update(self, completed_updates):
        self._refresh(completed_updates)


def dance_advantages(c, rewards, prompts):
    if c.algorithm.options["advantage_mode"] == "prompt_global_std":
        return prompt_advantages(rewards, prompts)
    k = c.sampling.samples_per_prompt
    if len(rewards) != len(prompts) or len(rewards) % k:
        raise ValueError("Dance advantages require complete prompt-major K groups")
    if any(len(set(prompts[start:start + k])) != 1 for start in range(0, len(prompts), k)):
        raise ValueError("Dance sample-std groups must contain one prompt each")
    groups = torch.arange(len(rewards), device=rewards.device) // k
    return group_advantages(rewards, groups, mode="group_sample_std", eps=1e-8)


def prepare_grpo_update(c, cache, predict, rewards, prompts, generator, device,
                        *, advantages=None, target_count=None):
    a = c.algorithm.options
    if advantages is None:
        advantages = (dance_advantages(c, rewards, prompts) if c.algorithm.name == "dancegrpo"
                      else prompt_advantages(rewards, prompts))
    if advantages.shape != rewards.shape or not torch.isfinite(advantages).all():
        raise ValueError("GRPO advantages must be finite and match reward rows")
    rows = torch.arange(len(rewards), device=device)
    nonzero = advantages.to(device) != 0
    if a["zero_advantage_mode"] == "filter_pad":
        count = int(nonzero.sum())
        if not count and not target_count:
            return None, (), {"grpo/nonzero_rows": 0, "grpo/retained_rows": 0}
        multiple = a["training_batches"]
        target = max(multiple, math.ceil(count / multiple) * multiple) if target_count is None else target_count
        if type(target) is not int or not count <= target <= len(rows) or target % multiple:
            raise ValueError("GRPO synchronized target must fit local rows and logical batches")
        zeros = rows[~nonzero]
        add = zeros[
            torch.randperm(len(zeros), generator=generator, device=device)[: target - count]
        ]
        nonzero[add] = True
        rows = rows[nonzero]
    if a["shuffle_rows"]:
        rows = rows[torch.randperm(len(rows), generator=generator, device=device)]
    rows = rows.cpu().tolist()
    loss = dict(a["loss"])
    advantages = advantages.clamp(-loss["advantage_clip"], loss["advantage_clip"])
    if c.algorithm.name == "flow_dppo":
        del loss["advantage_clip"]
    objective = GRPOObjective(cache, predict, advantages, device, c.algorithm.name, loss)
    window_size = len(rows) // a["updates_per_rollout"]
    batch_size = len(rows) // a["training_batches"]
    windows = []
    for start in range(0, len(rows), window_size):
        # Preserve logical-batch, then timestep, then row order. Physical
        # microbatches may span these items but never change the denominator.
        windows.append(
            tuple(
                (row, slot)
                for batch in range(start, start + window_size, batch_size)
                for slot in range(len(cache.selected))
                for row in rows[batch : batch + batch_size]
            )
        )
    return (
        objective,
        tuple(windows),
        {"grpo/nonzero_rows": int((advantages != 0).sum()), "grpo/retained_rows": len(rows)},
    )
