"""Explicit flow transitions with the same sampled state used for replay.

Flow-SDE/CPS equations follow the MIT-licensed Flow-GRPO scheduler (see
THIRD_PARTY_NOTICES.md). Time is sigma in [0, 1], not an integer step index.
"""

import math
from dataclasses import dataclass

import torch


def validate_schedule(sigmas: torch.Tensor):
    if (
        sigmas.ndim != 1
        or sigmas.numel() < 2
        or not torch.isfinite(sigmas).all()
        or sigmas[0] > 1
        or sigmas[-1] < 0
        or not (sigmas[:-1] > sigmas[1:]).all()
    ):
        raise ValueError("sigmas must strictly decrease within [0, 1]")


def shifted_schedule(steps: int, shift: float, *, device=None) -> torch.Tensor:
    """Explicit static flow shift; dynamic model-specific shifts stay in adapters."""
    if type(steps) is not int or steps < 1 or not math.isfinite(shift) or shift <= 0:
        raise ValueError("steps and shift must be positive")
    sigmas = torch.linspace(1, 0, steps + 1, dtype=torch.float32, device=device)
    return shift * sigmas / (1 + (shift - 1) * sigmas)


@dataclass(frozen=True)
class Transition:
    sample: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    log_prob: torch.Tensor | None


def flow_transition(
    state,
    velocity,
    sigma,
    sigma_next,
    *,
    kind="ode",
    noise_level=0.7,
    sigma_max=None,
    innovation=None,
    observed=None,
    likelihood="gaussian",
):
    """Euler ODE, Flow-GRPO SDE, Dance SDE, or CPS.

    Stochastic sampling requires explicit innovation; replay requires the exact
    cached observed state. No hidden RNG, no stochastic-to-ODE fallback. CPS's
    original reduced negative-MSE surrogate is selected by ``likelihood=mse``.
    ``cps_zero_noise`` preserves the expanded zero-noise CPS arithmetic,
    including terminal steps, rather than replacing it with Euler arithmetic.
    """
    if state.shape != velocity.shape or state.ndim < 2 or not state.is_floating_point():
        raise ValueError("state and velocity need equal [batch, ...] shapes")
    if not torch.isfinite(state).all() or not torch.isfinite(velocity).all():
        raise ValueError("nonfinite transition inputs")
    x, v = state.float(), velocity.float()

    def time_vector(value):
        value = torch.as_tensor(value, device=x.device, dtype=torch.float32).reshape(-1)
        if value.numel() not in (1, len(x)):
            raise ValueError("sigma needs one value per sample or a scalar")
        return value.reshape((-1,) + (1,) * (x.ndim - 1))

    t, nxt = time_vector(sigma), time_vector(sigma_next)
    if (
        not torch.isfinite(t).all()
        or not torch.isfinite(nxt).all()
        or not ((t > nxt) & (t <= 1) & (nxt >= 0)).all()
    ):
        raise ValueError("invalid reverse-time interval")
    dt = nxt - t
    if kind in ("ode", "cps_zero_noise"):
        if innovation is not None or observed is not None:
            raise ValueError("ODE transitions have neither noise nor likelihood replay")
        mean = (x + dt * v if kind == "ode" else
                (x - t * v) * (1 - nxt) + (x + (1 - t) * v) * nxt)
        return Transition(mean, mean, torch.zeros_like(t), None)
    if kind not in ("sde", "dance", "cps") or likelihood not in ("gaussian", "mse"):
        raise ValueError("unsupported transition or likelihood")
    if not math.isfinite(noise_level) or not 0 < noise_level <= 1:
        raise ValueError("stochastic noise_level must lie in (0, 1]")
    if (innovation is None) == (observed is None):
        raise ValueError("supply exactly one innovation or replay observation")
    if kind == "sde":
        if (t == 1).any():
            if sigma_max is None or not math.isfinite(sigma_max) or not 0 < sigma_max < 1:
                raise ValueError("t=1 requires the actual schedule's next sigma as sigma_max")
        guard = t if sigma_max is None else torch.where(t == 1, float(sigma_max), t)
        diffusion = torch.sqrt(t / (1 - guard)) * noise_level
        mean = x * (1 + diffusion.square() * dt / (2 * t))
        mean = mean + v * (1 + diffusion.square() * (1 - t) / (2 * t)) * dt
        std = diffusion * (-dt).sqrt()
    elif kind == "dance":
        if likelihood != "gaussian":
            raise ValueError("Dance uses Gaussian transition likelihoods")
        # DanceGRPO 15cc71d, train_grpo_flux_lora.py::flux_step.
        endpoint = x - t * v
        score = -(x - endpoint * (1 - t)) / t.square()
        mean = x + dt * v + (-0.5 * noise_level**2 * score) * dt
        std = noise_level * (-dt).sqrt()
    else:
        std = nxt * math.sin(noise_level * math.pi / 2)
        x0, x1 = x - t * v, x + (1 - t) * v
        mean = x0 * (1 - nxt) + x1 * (nxt.square() - std.square()).clamp_min(0).sqrt()
    if not (std > 0).all():
        raise ValueError("zero-variance transition has no stochastic likelihood")
    supplied = innovation if innovation is not None else observed
    if supplied.shape != x.shape or not torch.isfinite(supplied).all():
        raise ValueError("noise/observed state must be finite and match state shape")
    sample = mean + std * innovation.to(x) if innovation is not None else observed.to(x)
    delta = sample.detach() - mean
    logp = -delta.square()
    if likelihood == "gaussian":
        logp = logp / (2 * std.square()) - std.log() - math.log(2 * math.pi) / 2
    return Transition(sample, mean, std, logp.flatten(1).mean(1))


def select_timesteps(pool: tuple[int, ...], count: int, generator: torch.Generator):
    if (
        not pool
        or any(type(i) is not int or i < 0 for i in pool)
        or len(set(pool)) != len(pool)
        or type(count) is not int
        or not 1 <= count <= len(pool)
    ):
        raise ValueError("invalid timestep selection")
    indices = torch.randperm(len(pool), generator=generator)[:count].tolist()
    return tuple(sorted(pool[i] for i in indices))
