"""Public Hunyuan Fast-SDE/Dance update contracts, distinct from image CPS.

Reference: canonical public video backends at 3ca2a36f. Two disjoint updates,
no policy/evaluation EMA, and no reference regularizer in these recipes.
"""

import math
from dataclasses import dataclass, field

import torch

from ..algorithms import clipped_grpo_loss
from ..sampling import Transition, flow_transition
from .objectives import _indices, disjoint_windows
from .rollout import collect_rollout

ALGORITHMS = ("flow_grpo_fast", "dancegrpo")


def fast_transition(state, velocity, sigma, sigma_next, *, kind="sde", noise_level=.8,
                    sigma_max=None, innovation=None, observed=None, likelihood="gaussian"):
    """Preserve the original Gaussian Flow-SDE arithmetic, including its normalizer."""
    if kind == "ode":
        return flow_transition(state, velocity, sigma, sigma_next, kind="ode",
                               innovation=innovation, observed=observed)
    if (kind != "sde" or likelihood != "gaussian" or state.ndim < 2 or state.shape != velocity.shape
            or not math.isfinite(noise_level) or not 0 < noise_level <= 1
            or (innovation is None) == (observed is None)):
        raise ValueError("Fast video requires Gaussian Flow-SDE with one innovation or replay state")
    x, v = state.float(), velocity.float()
    def time(value):
        value = torch.as_tensor(value, dtype=torch.float32, device=x.device).reshape(-1)
        if len(value) not in (1, len(x)):
            raise ValueError("one sigma per row or one shared sigma required")
        return value.reshape((-1,) + (1,) * (x.ndim - 1))
    t, nxt = time(sigma), time(sigma_next)
    if not ((t > nxt) & (t <= 1) & (nxt >= 0)).all():
        raise ValueError("invalid reverse-time interval")
    if (t == 1).any() and (sigma_max is None or not 0 < sigma_max < 1):
        raise ValueError("sigma one requires the schedule's next sigma")
    guard = t if sigma_max is None else torch.where(t == 1, sigma_max, t)
    dt = nxt - t
    diffusion = torch.sqrt(t / (1 - guard)) * noise_level
    mean = x * (1 + diffusion**2 / (2 * t) * dt) + v * (1 + diffusion**2 * (1 - t) / (2 * t)) * dt
    std = diffusion * torch.sqrt(-dt)
    supplied = innovation if innovation is not None else observed
    if supplied.shape != x.shape or not all(torch.isfinite(value).all() for value in (x, v, supplied)):
        raise ValueError("nonfinite or incompatible transition inputs")
    sample = mean + std * innovation.to(x) if innovation is not None else observed.detach().to(x)
    logp = (-(sample.detach() - mean)**2 / (2 * std**2) - torch.log(std)
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi))).to(x.device))
    return Transition(sample, mean, std, logp.flatten(1).mean(1))


def inputs(c, shape, *, attempt, rank=0):
    options = c.sampling.options
    if c.algorithm.name == "dancegrpo":
        from .dance_video import dance_inputs

        return dance_inputs(seed=c.seed, rank=rank, attempt=attempt,
            prompts_per_rank=c.sampling.prompts_per_rollout, k=c.sampling.samples_per_prompt,
            latent_shape=shape[1:], pool=tuple(options["train_timestep_pool"]),
            count=options["train_timestep_count"], steps=c.sampling.steps)
    if c.algorithm.name != "flow_grpo_fast":
        raise ValueError("unsupported video GRPO algorithm")
    generator = torch.Generator().manual_seed((c.seed + 1_000_003 * attempt + 10_007 * rank) % (2**63 - 1))
    pool = torch.tensor(options["train_timestep_pool"])
    selected = pool[torch.randperm(len(pool), generator=generator)[:options["train_timestep_count"]]].sort().values
    noise = torch.randn(shape, generator=generator, dtype=torch.bfloat16)
    innovations = torch.randn((shape[0], len(selected), *shape[1:]), generator=generator, dtype=torch.float32)
    return noise, selected, innovations


def collect(c, predict, plan, sigmas, device):
    noise, selected, innovations = plan
    if c.algorithm.name == "dancegrpo":
        from .dance_video import collect_dance_rollout

        return collect_dance_rollout(predict, noise.to(device), sigmas, row_selected=selected,
            innovations=innovations, microbatch=c.sampling.microbatch, eta=c.sampling.options["noise_level"])
    return collect_rollout(predict, noise.float().to(device), sigmas, selected=selected.tolist(),
        stochastic=selected.tolist(), innovations={int(step): innovations[:, slot] for slot, step in enumerate(selected)},
        transition_kind="sde", likelihood="gaussian", noise_level=c.sampling.options["noise_level"],
        microbatch=c.sampling.microbatch, transition_fn=fast_transition)


@dataclass
class FastObjective:
    cache: object
    predict: object
    advantages: torch.Tensor
    device: torch.device
    options: dict = field(default_factory=dict)

    def __call__(self, items):
        rows, slots, steps = _indices(items, self.cache)
        states = self.cache.states[rows, slots].to(self.device)
        times = self.cache.sigmas[steps].to(self.device)
        velocity = self.predict(states, times, rows.to(self.device))
        replay = fast_transition(states, velocity, times, self.cache.sigmas[steps + 1],
            noise_level=self.cache.noise_level, sigma_max=float(self.cache.sigmas[1]),
            observed=self.cache.next_states[rows, slots].to(self.device))
        return clipped_grpo_loss(replay.log_prob, self.cache.old_log_probs[rows, slots].to(self.device),
                                 self.advantages[rows].to(self.device), **self.options)["per_item_loss"]


def global_std_advantages(scores, global_scores, k, eps):
    """Center complete prompt groups; normalize by global raw scalar moments."""
    local = scores.weighted.detach().float().cpu().reshape(-1, k)
    population = global_scores.weighted.detach().float().cpu().reshape(-1)
    mean = population.sum() / len(population)
    std = (population.square().sum() / len(population) - mean.square()).clamp_min(0).sqrt()
    return ((local - local.mean(1, keepdim=True)) / (std + eps)).flatten()


def prepare_objective(c, cache, predict, scores, global_scores, device):
    """Freeze complete-group advantages once, before both optimizer windows."""
    count = len(cache.endpoints)
    k = c.sampling.samples_per_prompt
    if count % k:
        raise ValueError("video advantages require complete prompt groups")
    eps = c.algorithm.options["advantage_epsilon"]
    options = dict(c.algorithm.options["loss"])
    if c.algorithm.name == "flow_grpo_fast":
        # Match the reference population moments, not unbiased std or std of
        # centered advantages. This includes all ranks before either update.
        advantages = global_std_advantages(scores, global_scores, k, eps)
        objective = FastObjective(cache, predict, advantages, device, options)
    elif c.algorithm.name == "dancegrpo":
        from .dance_video import DanceObjective

        values = scores.raw if scores.space == "raw" else scores.fixed_normalized
        weights = {name: weight for name, weight in scores.weights.items() if weight != 0}
        advantages = {}
        for name in weights:
            grouped = values[name].detach().float().cpu().reshape(-1, k)
            advantages[name] = ((grouped - grouped.mean(1, keepdim=True)) / (grouped.std(1, correction=1, keepdim=True) + eps)).flatten()
        options["component_weights"] = weights
        objective = DanceObjective(cache, predict, advantages, device, options)
    else:
        raise ValueError("unsupported video GRPO algorithm")
    return objective, disjoint_windows(count, len(cache.selected))
