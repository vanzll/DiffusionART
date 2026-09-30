"""FLUX.1's resolution-dependent FlowMatch schedule, separate from SD3 grids."""

import math

import numpy as np
import torch


def validate_asset_scheduler(config, expected):
    """The config freezes the published asset grid, not a silent override."""
    for name, wanted in expected.items():
        if config.get(name) != wanted:
            raise ValueError(f"FLUX scheduler asset {name} differs from experiment config")
    defaults = dict(num_train_timesteps=1000, invert_sigmas=False, shift_terminal=None,
                    use_karras_sigmas=False, use_exponential_sigmas=False,
                    use_beta_sigmas=False, time_shift_type="exponential", stochastic_sampling=False)
    for name, wanted in defaults.items():
        if config.get(name, wanted) != wanted:
            raise ValueError(f"FLUX scheduler asset {name} is not supported")


def flux_schedule(steps, *, height, width, spatial_factor, scheduler):
    """Mirror the research FLUX pipeline's sigma grid and calculate_shift.

    Public latents are unpacked BCHW; the image sequence consists of 2x2 patches.
    Unsupported alternate schedulers fail rather than silently changing a run.
    """
    from diffusers import FlowMatchEulerDiscreteScheduler

    if any(type(n) is not int or n < 1 for n in (steps, height, width, spatial_factor)):
        raise ValueError("FLUX steps and geometry must be positive integers")
    if height % (2 * spatial_factor) or width % (2 * spatial_factor):
        raise ValueError("FLUX image dimensions must divide by twice the VAE spatial factor")
    names = {"use_dynamic_shifting", "shift", "base_image_seq_len", "max_image_seq_len", "base_shift", "max_shift"}
    if not isinstance(scheduler, dict) or set(scheduler) != names:
        raise ValueError(f"FLUX scheduler requires exactly {sorted(names)}")
    if type(scheduler["use_dynamic_shifting"]) is not bool:
        raise ValueError("use_dynamic_shifting must be boolean")
    base, maximum = scheduler["base_image_seq_len"], scheduler["max_image_seq_len"]
    if type(base) is not int or type(maximum) is not int or not 0 < base < maximum:
        raise ValueError("FLUX requires positive base/max image sequence lengths in order")
    for key in ("shift", "base_shift", "max_shift"):
        value = scheduler[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"FLUX {key} must be finite")
    if scheduler["shift"] <= 0:
        raise ValueError("FLUX shift must be positive")
    image_seq_len = (height // spatial_factor // 2) * (width // spatial_factor // 2)
    slope = (scheduler["max_shift"] - scheduler["base_shift"]) / (maximum - base)
    mu = image_seq_len * slope + (scheduler["base_shift"] - slope * base)
    instance = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, **scheduler)
    instance.set_timesteps(sigmas=np.linspace(1.0, 1 / steps, steps), mu=mu)
    result = instance.sigmas.detach().float().cpu()
    if not torch.isfinite(result).all() or not ((result[:-1] > result[1:]).all()):
        raise ValueError("FLUX scheduler produced an invalid descending grid")
    return result
