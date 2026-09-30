"""Four-step CFG-distilled Wan contracts from the accepted I2V runtime.

Reference: wan_i2v_distilled_algorithms.py at research revision 3ca2a36f.
These plans intentionally differ from Hunyuan in RNG, Dance terminal sampling,
scalarized Dance advantages, and FP32 model-output arithmetic.
"""

import math
from dataclasses import dataclass, field

import torch

from ..algorithms import clipped_grpo_loss
from ..models.wan_assets import scoped_seed
from ..sampling import Transition, flow_transition
from . import video_nft
from .objectives import _indices, disjoint_windows, group_advantages
from .rollout import collect_rollout
from .video_grpo import FastObjective, fast_transition, global_std_advantages

ALGORITHMS = ("diffusionart", "diffusionnft", "flow_grpo_fast", "dancegrpo", "refl")


def inputs(*, algorithm, seed, rank, attempt, k, latent_shape, prompts_per_rank=1):
    """FP32 noise; one Fast position shared across ranks and K trajectories.

    The paper configuration has one prompt per rank. Additional local groups
    use independent named streams without changing that first group's draws.
    The caller retains entire prompt groups regardless of physical microbatch.
    """
    if algorithm not in ALGORITHMS:
        raise ValueError("unsupported distilled Wan algorithm")
    if (any(type(v) is not int or v < 0 for v in (seed, rank, attempt))
            or type(k) is not int or (k != 1 if algorithm == "refl" else k < 2) or type(prompts_per_rank) is not int
            or prompts_per_rank < 1 or not latent_shape
            or any(type(v) is not int or v < 1 for v in latent_shape)):
        raise ValueError("invalid distilled Wan rollout dimensions or seed")
    noises, innovations = [], []
    selected_count = 4 if algorithm == "diffusionnft" else 3
    if algorithm == "flow_grpo_fast":
        rng = torch.Generator().manual_seed(scoped_seed(seed, 0, attempt, "steps"))
        selected = torch.randperm(3, generator=rng)[:1]
    elif algorithm == "refl":
        selected = torch.full((prompts_per_rank, 1), 3, dtype=torch.long)
    else:
        selected = torch.arange(selected_count).repeat(prompts_per_rank * k, 1)
    for group in range(prompts_per_rank):
        rng = torch.Generator().manual_seed(scoped_seed(
            seed, rank, attempt, "noise" if group == 0 else f"noise:{group}"))
        noise = torch.randn((1 if algorithm == "dancegrpo" else k, *latent_shape), generator=rng, dtype=torch.float32)
        if algorithm == "dancegrpo":
            noise = noise.repeat(k, *([1] * len(latent_shape)))
        noises.append(noise)
        if algorithm in ("dancegrpo", "flow_grpo_fast"):
            innovations.append(torch.randn(
                (k, 3 if algorithm == "dancegrpo" else 1, *latent_shape), generator=rng, dtype=torch.float32))
    return torch.cat(noises), selected, torch.cat(innovations) if innovations else None


def dance_transition(state, velocity, sigma, sigma_next, *, kind="sde", noise_level=.25,
                     innovation=None, observed=None, **unused):
    """Native Dance likelihood at nonterminal states; terminal step is ODE.

    Wan's adapter returns FP32 fields even with BF16 transformer execution.
    Converting here makes replay retain the same rounding convention.
    """
    if kind == "ode":
        return flow_transition(state, velocity.float(), sigma, sigma_next, kind="ode")
    if (kind != "sde" or (innovation is None) == (observed is None)
            or state.ndim < 2 or velocity.shape != state.shape
            or isinstance(noise_level, bool) or not math.isfinite(noise_level) or noise_level <= 0):
        raise ValueError("Dance requires valid states and one innovation or observed transition")
    batch = len(state)
    times = torch.as_tensor(sigma, dtype=torch.float32).detach().cpu().reshape(-1).expand(batch)
    following = torch.as_tensor(sigma_next, dtype=torch.float32).detach().cpu().reshape(-1).expand(batch)
    if not ((times <= 1) & (times > following) & (following > 0)).all():
        raise ValueError("Dance likelihood is defined only at nonterminal intervals")
    target = innovation if innovation is not None else observed
    if target.shape != state.shape or not all(torch.isfinite(v).all() for v in (state, velocity, target)):
        raise ValueError("nonfinite or mismatched Dance transition")
    samples, means, probabilities, stds = [], [], [], []
    for row in range(batch):
        t, dt = times[row], following[row] - times[row]
        z, v = state[row:row + 1].float(), velocity[row:row + 1].float()
        endpoint = z - t * v
        score = -(z - endpoint * (1 - t)) / t**2
        mean = z + dt * v + (-.5 * noise_level**2 * score) * dt
        std = noise_level * math.sqrt(-dt)
        sample = (mean + innovation[row:row + 1].to(mean) * std if innovation is not None
                  else observed[row:row + 1].detach().to(mean))
        logp = (-(sample.detach() - mean)**2 / (2 * std**2) - math.log(std)
                - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi))).to(mean.device))
        samples.append(sample)
        means.append(mean)
        probabilities.append(logp.flatten(1).mean(1))
        stds.append(std)
    return Transition(torch.cat(samples), torch.cat(means),
                      state.new_tensor(stds, dtype=torch.float32), torch.cat(probabilities))


@torch.no_grad()
def collect(algorithm, predict, plan, sigmas, *, microbatch, device):
    if algorithm not in ALGORITHMS:
        raise ValueError("unsupported distilled Wan algorithm")
    expected = torch.linspace(1, 0, 5, dtype=torch.float32)
    expected = 5 * expected / (1 + 4 * expected)
    if not torch.equal(sigmas.detach().float().cpu(), expected):
        raise ValueError("distilled Wan requires four steps with shift five")
    noise, selected, innovations = plan
    if noise.dtype != torch.float32:
        raise ValueError("distilled Wan initial noise must be FP32")
    def field(state, times, rows):
        return predict(state, times, rows).float()
    if algorithm == "diffusionnft":
        return video_nft.collect(field, plan, sigmas, microbatch=microbatch, device=device)
    selection = selected.tolist() if selected.ndim == 1 else selected[0].tolist()
    stochastic = tuple(selection) if algorithm == "flow_grpo_fast" else (0, 1, 2) if algorithm == "dancegrpo" else ()
    return collect_rollout(field, noise.to(device), sigmas, selected=selection,
        row_selected=selected if selected.ndim == 2 else None,
        stochastic=stochastic,
        innovations={step: innovations[:, slot] for slot, step in enumerate(stochastic)},
        microbatch=microbatch, noise_level=.25 if algorithm == "dancegrpo" else .8,
        transition_kind="wan_dance" if algorithm == "dancegrpo" else "sde",
        transition_fn=_dance_dispatch if algorithm == "dancegrpo" else fast_transition)


def _dance_dispatch(*args, kind, **kwargs):
    return dance_transition(*args, kind="sde" if kind == "wan_dance" else kind, **kwargs)


@dataclass
class DanceObjective:
    cache: object
    predict: object
    advantages: torch.Tensor
    device: torch.device
    options: dict = field(default_factory=dict)

    def __call__(self, items):
        rows, slots, steps = _indices(items, self.cache)
        states = self.cache.states[rows, slots].to(self.device)
        times = self.cache.sigmas[steps].to(self.device)
        velocity = self.predict(states, times, rows.to(self.device)).float()
        result = dance_transition(states, velocity, times, self.cache.sigmas[steps + 1],
            noise_level=self.cache.noise_level, observed=self.cache.next_states[rows, slots].to(self.device))
        return clipped_grpo_loss(result.log_prob, self.cache.old_log_probs[rows, slots].to(self.device),
            self.advantages[rows].to(self.device), **self.options)["per_item_loss"]


def prepare_grpo(c, cache, predict, scores, global_scores, device):
    k = c.sampling.samples_per_prompt
    if c.algorithm.name == "flow_grpo_fast":
        advantages = global_std_advantages(scores, global_scores, k, c.algorithm.options["advantage_epsilon"])
        objective = FastObjective(cache, predict, advantages, device, c.algorithm.options["loss"])
    elif c.algorithm.name == "dancegrpo":
        # Wan's accepted recipe scalarizes VQ/MQ before sample-std normalization.
        # Hunyuan instead clips each component independently before weighting.
        advantages = group_advantages(scores.weighted.cpu(), torch.arange(len(cache.endpoints)) // k,
            mode="group_sample_std", eps=c.algorithm.options["advantage_epsilon"])
        objective = DanceObjective(cache, predict, advantages, device, c.algorithm.options["loss"])
    else:
        raise ValueError("prepare_grpo requires a video GRPO algorithm")
    return objective, disjoint_windows(len(cache.endpoints), len(cache.selected))
