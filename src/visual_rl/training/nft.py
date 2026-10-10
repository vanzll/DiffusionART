"""Image NFT lifecycle: old-policy rollout, fresh re-noising, disjoint updates.

The old-policy EMA and optional evaluation EMA are independent. This implements
the standard mirror objective, not experimental proximal/pairwise variants.
"""

import math
from contextlib import nullcontext

import torch

from ..algorithms import nft_loss
from ..config import ConfigError
from .image_contracts import prompt_advantages as nft_advantages  # noqa: F401 -- public alias
from .image_contracts import sd3_schedule as nft_schedule
from .objectives import NFTObjective
from .rollout import endpoint_renoising
from .state import PolicyShadow


def _keys(value, names, label):
    if not isinstance(value, dict) or set(value) != set(names):
        raise ConfigError(f"{label} requires exactly {sorted(names)}")


def validate_nft(c):
    a, s = c.algorithm.options, c.sampling.options
    is_flux = c.model.name == "flux"
    required = {"loss", "old_policy", "evaluation_ema"}
    if (not isinstance(a, dict)
            or set(a) - (required | {"updates_per_rollout", "multi_reward_advantage"})
            or required - set(a)):
        raise ConfigError("NFT options require loss, old_policy and evaluation_ema")
    updates = a.get("updates_per_rollout", 1)
    if type(updates) is not int or updates not in (1, 2):
        raise ConfigError("NFT updates_per_rollout must be 1 or 2")
    if c.sampling.prompts_per_rollout * c.sampling.samples_per_prompt % updates:
        raise ConfigError("NFT disjoint windows require divisible rollout rows")
    if (c.training.max_updates % updates or c.runtime.checkpoint_every % updates
            or c.evaluation.every_updates % updates
            or any(step % updates for step in (*c.evaluation.at_updates, *c.runtime.checkpoint_at_updates))):
        raise ConfigError("NFT budget and artifacts must align to complete rollout boundaries")
    _keys(a["old_policy"], {"rate", "maximum", "delay"}, "NFT old_policy")
    _keys(a["evaluation_ema"], {"enabled", "decay"}, "NFT evaluation_ema")
    names = {"solver", "train_timestep_count", "timestep_selection"}
    _keys(s, names if is_flux else names | {"shift"}, "NFT sampling")
    if s["solver"] not in ("euler", "dpm2"):
        raise ConfigError("NFT solver must be euler or dpm2")
    if is_flux and s["solver"] != "euler":
        raise ConfigError("native FLUX NFT uses the historical Euler ODE sampler")
    if s["timestep_selection"] not in ("random_per_row", "prefix"):
        raise ConfigError("NFT timestep_selection must be random_per_row or prefix")
    if (
        type(s["train_timestep_count"]) is not int
        or not 1 <= s["train_timestep_count"] <= c.sampling.steps
    ):
        raise ConfigError("NFT train_timestep_count must be within the sampling schedule")
    if not is_flux and c.sampling.guidance_scale != 1:
        raise ConfigError("native SD3 NFT currently supports the no-CFG recipe only")
    old, ema = a["old_policy"], a["evaluation_ema"]
    if type(old["delay"]) is not int or old["delay"] < 0:
        raise ConfigError("NFT old-policy delay must be a nonnegative update count")
    for value, upper in ((old["rate"], None), (old["maximum"], 1), (ema["decay"], 1)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
            or value < 0
            or (upper is not None and value >= upper)
        ):
            raise ConfigError("invalid NFT EMA schedule")
    if type(ema["enabled"]) is not bool:
        raise ConfigError("NFT evaluation_ema.enabled must be boolean")
    loss = a["loss"]
    allowed = {"nft_beta", "adv_clip_max", "adv_mode", "eps"}
    if not isinstance(loss, dict) or set(loss) - allowed:
        raise ConfigError("unsupported NFT loss options; reference KL is not integrated")
    try:
        dummy = torch.ones(2, 1)
        nft_loss(dummy, dummy, dummy, dummy, torch.ones(2), torch.zeros(2), **loss)
        if is_flux:
            from .flux_contracts import flux_schedule

            flux_schedule(c.sampling.steps, height=c.sampling.height, width=c.sampling.width,
                          spatial_factor=c.model.options["spatial_factor"],
                          scheduler=c.model.options["scheduler"])
        else:
            nft_schedule(c.sampling.steps, s["shift"])
    except (ValueError, TypeError) as exc:
        raise ConfigError(f"invalid NFT settings: {exc}") from exc


class NFTState:
    def __init__(self, model, options):
        self.old = PolicyShadow(model, trainable_only=True)
        self.ema = (
            PolicyShadow(model, trainable_only=True)
            if options["evaluation_ema"]["enabled"]
            else None
        )
        self.options = options

    def stateful(self):
        result = {"nft_old_policy": self.old}
        if self.ema is not None:
            result["nft_eval_ema"] = self.ema
        return result

    def evaluation(self):
        return self.ema.use() if self.ema is not None else nullcontext()

    def after_update(self, successful_updates, *, refresh_old=True):
        if type(successful_updates) is not int or successful_updates < 1:
            raise ValueError("NFT refresh needs the one-based completed update count")
        if type(refresh_old) is not bool:
            raise ValueError("NFT old-policy refresh flag must be boolean")
        old = self.options["old_policy"]
        decay = min(max(successful_updates - old["delay"], 0) * old["rate"], old["maximum"])
        if refresh_old:
            self.old.update(decay)
        ema_decay = 0.0
        if self.ema is not None:
            ema_decay = min(
                (1 + successful_updates) / (10 + successful_updates),
                self.options["evaluation_ema"]["decay"],
            )
            self.ema.update(ema_decay)
        return {"nft/old_policy_decay": decay if refresh_old else 0.0,
                "nft/old_policy_refreshed": int(refresh_old),
                "nft/evaluation_ema_decay": ema_decay}


def nft_windows(c, objective):
    """Partition shuffled endpoint rows, keeping every selected time of a row together."""
    updates = c.algorithm.options.get("updates_per_rollout", 1)
    width, remainder = divmod(len(objective.states), updates)
    if not width or remainder or width % c.sampling.options["train_timestep_count"]:
        raise ValueError("NFT disjoint windows must contain complete endpoint rows")
    return tuple(tuple(range(start, start + width))
                 for start in range(0, len(objective.states), width))


@torch.no_grad()
def prepare_nft_objective(c, state, predict, endpoints, schedule, advantages, generator, device):
    """Freeze all training states and old predictions before current-policy backward.

    Draw per-item fresh noise before physical batching; changing microbatch does
    not change the logical batch. Only one old-policy parameter swap is needed.
    """
    rows, count = len(endpoints), c.sampling.options["train_timestep_count"]
    order = torch.randperm(rows, generator=generator, device=device)
    slots = []
    for _ in range(rows):
        if c.sampling.options["timestep_selection"] == "random_per_row":
            slots.append(
                torch.randperm(len(schedule) - 1, generator=generator, device=device)[:count]
            )
        else:
            slots.append(torch.arange(count, device=device))
    row_indices = order.repeat_interleave(count).cpu()
    times = schedule[torch.stack(slots).flatten().cpu()]
    if c.model.name == "flux":
        times = (times * 1000.0) / 1000.0
    x0 = endpoints[row_indices].float().cpu()
    states = torch.empty_like(x0)
    old_predictions = torch.empty_like(x0)
    for i in range(len(x0)):
        if c.model.name == "flux":
            from ..models.flux import unpack_latents

            channels, height, width = x0.shape[1:]
            packed_noise = torch.randn((1, height * width // 4, channels * 4),
                                       generator=generator, device=device, dtype=torch.float32)
            noise = unpack_latents(packed_noise, height, width)
        else:
            noise = torch.randn(
                (1, *x0.shape[1:]), generator=generator, device=device, dtype=torch.float32
            )
        states[i : i + 1] = endpoint_renoising(
            x0[i : i + 1].to(device), times[i : i + 1].to(device), noise
        ).cpu()
    with state.old.use():
        for start in range(0, len(x0), c.training.microbatch):
            stop = start + c.training.microbatch
            old_predictions[start:stop] = (
                predict(
                    states[start:stop].to(device),
                    times[start:stop].to(device),
                    row_indices[start:stop].to(device),
                )
                .float()
                .cpu()
            )
    return NFTObjective(
        x0,
        states,
        times,
        old_predictions,
        row_indices,
        advantages[row_indices],
        predict,
        device,
        options=c.algorithm.options["loss"],
        upcast_predictions=c.model.name == "flux",
    )
