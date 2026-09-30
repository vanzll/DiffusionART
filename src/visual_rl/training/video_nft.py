"""Public video NFT: old-policy Euler endpoints, fresh noising, one update.

The video contract differs from image NFT's sampler and EMA indexing. Reference:
hunyuan_public_nft.py at research revision 3ca2a36f. Evaluation uses current
parameters; there is no separate evaluation EMA or reference regularizer.
"""

import hashlib
import math
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from ..algorithms import nft_loss
from ..algorithms.nft import old_policy_decay
from ..config import ConfigError
from .objectives import NFTObjective
from .rollout import endpoint_renoising
from .state import PolicyShadow
from .video_grpo import global_std_advantages


def validate_options(c):
    options = c.algorithm.options
    if set(options) != {"loss", "old_policy", "advantage_epsilon"}:
        raise ConfigError("video NFT needs loss, old_policy and advantage_epsilon; no evaluation EMA")
    old = options["old_policy"]
    if not isinstance(old, dict) or set(old) != {"mode", "rate", "maximum"}:
        raise ConfigError("video NFT old_policy needs mode, rate and maximum")
    if old["mode"] not in ("no-ema", "slow-ema"):
        raise ConfigError("video NFT old_policy.mode must be no-ema or slow-ema")
    for value, positive in ((old["rate"], False), (old["maximum"], False),
                            (options["advantage_epsilon"], True)):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0 or (positive and value == 0)):
            raise ConfigError("invalid video NFT EMA or advantage scale")
    if old["maximum"] >= 1:
        raise ConfigError("old-policy maximum decay must be below one")
    loss = options["loss"]
    if not isinstance(loss, dict) or set(loss) - {"nft_beta", "adv_clip_max", "adv_mode", "eps"}:
        raise ConfigError("unsupported video NFT loss options")
    try:
        dummy = torch.ones(2, 1)
        nft_loss(dummy, dummy, dummy, dummy, torch.ones(2), torch.zeros(2), **loss)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"invalid video NFT loss: {error}") from error


class VideoNFTState:
    def __init__(self, model, options):
        from torch.distributed.fsdp import FullyShardedDataParallel

        if isinstance(model, FullyShardedDataParallel):
            from .fsdp_shadow import FSDPPolicyShadow

            self.old = FSDPPolicyShadow(model)
        else:
            self.old = PolicyShadow(model, trainable_only=True)
        self.options = options

    def stateful(self):
        return {"nft_old_policy": self.old}

    @contextmanager
    def use_old(self, session=None):
        context = self.old.use()
        if session is None:
            with context:
                yield
            return
        session.local("old-policy swap", context.__enter__)
        try:
            yield
        except BaseException as error:
            # A failed model collective is terminal; never enter a new vote
            # while a peer may still be in its forward. Restore locally only.
            context.__exit__(type(error), error, error.__traceback__)
            raise
        else:
            session.local("current-policy restore", lambda: context.__exit__(None, None, None))

    def after_update(self, successful_updates):
        if type(successful_updates) is not int or successful_updates < 1:
            raise ValueError("video NFT needs a positive completed-update count")
        old = self.options["old_policy"]
        decay = old_policy_decay(successful_updates - 1, old["mode"], old["rate"], old["maximum"])
        self.old.update(decay)
        return {"nft/old_policy_decay": decay, "nft/evaluation_ema_decay": 0.0}


@dataclass(frozen=True)
class VideoNFTCache:
    endpoints: torch.Tensor
    sigmas: torch.Tensor
    selected: torch.Tensor


@torch.no_grad()
def collect(predict, plan, sigmas, *, microbatch, device):
    """First model input BF16, subsequent states FP32; no trajectory cache."""
    noise, selected, _ = plan
    endpoints = torch.empty_like(noise, dtype=torch.float32, device="cpu")
    for start in range(0, len(noise), microbatch):
        end = min(start + microbatch, len(noise))
        state = noise[start:end].to(device)
        rows = torch.arange(start, end, device=device)
        for step in range(len(sigmas) - 1):
            velocity = predict(state, sigmas[step].to(device).expand(end - start), rows)
            state = state.float() + (sigmas[step + 1] - sigmas[step]).to(device) * velocity.float()
            if not torch.isfinite(state).all():
                raise ValueError("nonfinite video NFT rollout")
        endpoints[start:end].copy_(state.cpu())
    return VideoNFTCache(endpoints, sigmas.detach().float().cpu().clone(), selected.detach().cpu().clone())


def noise_seed(seed, attempt, rank, sample_index, timestep):
    values = (seed, attempt, rank, sample_index, timestep)
    digest = hashlib.sha256(":".join(str(value) for value in values).encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


@torch.no_grad()
def prepare_objective(c, cache, predict, scores, global_scores, device, *, attempt, rank=0):
    """Local CPU preparation only; old-policy forwards are a separate phase."""
    count, slots = cache.selected.shape
    rows = torch.arange(count).repeat_interleave(slots)
    indices = cache.selected.flatten()
    times = cache.sigmas[indices]
    x0 = cache.endpoints[rows].float()
    states = torch.empty_like(x0)
    for item, (row, index) in enumerate(zip(rows.tolist(), indices.tolist())):
        generator = torch.Generator(device="cpu").manual_seed(
            noise_seed(c.seed, attempt, rank, rank * count + row, index))
        noise = torch.randn((1, *x0.shape[1:]), generator=generator, dtype=torch.float32)
        states[item:item + 1] = endpoint_renoising(x0[item:item + 1], times[item:item + 1], noise)
    advantages = global_std_advantages(scores, global_scores, c.sampling.samples_per_prompt,
                                       c.algorithm.options["advantage_epsilon"])
    return NFTObjective(x0, states, times, torch.empty_like(x0), rows,
        advantages[rows], predict, device, options=c.algorithm.options["loss"])


@torch.no_grad()
def predict_old(objective, microbatch):
    """Caller activates the frozen old policy for these collective forwards."""
    for start in range(0, len(objective.states), microbatch):
        end = min(start + microbatch, len(objective.states))
        value = objective.predict(objective.states[start:end].to(objective.device),
                                  objective.sigmas[start:end].to(objective.device),
                                  objective.row_indices[start:end].to(objective.device))
        if not torch.isfinite(value).all():
            raise ValueError("nonfinite video NFT old prediction")
        objective.old_predictions[start:end].copy_(value.float().cpu())
