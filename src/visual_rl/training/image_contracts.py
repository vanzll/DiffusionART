"""Shared SD3 scheduler and image-benchmark reward-statistics conventions."""

import math
from contextlib import contextmanager

import torch


@contextmanager
def tf32_policy(enabled):
    if enabled is None:
        yield
        return
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = enabled
        torch.backends.cudnn.allow_tf32 = enabled
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn


def sd3_schedule(steps, shift):
    """Use the native SD3 FlowMatch grid for both training and evaluation."""
    from diffusers import FlowMatchEulerDiscreteScheduler

    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    if (
        isinstance(shift, bool)
        or not isinstance(shift, (float, int))
        or not math.isfinite(shift)
        or shift <= 0
    ):
        raise ValueError("shift must be finite and positive")
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=shift)
    scheduler.set_timesteps(steps)
    return scheduler.sigmas.detach().float().cpu()


@torch.no_grad()
def prompt_advantages(rewards, prompts):
    """Prompt-text centering, global population std +1e-4, FP64 statistics.

    Matches current-batch PerPromptStatTracker(global_std=True), followed by
    its per-rollout clear. Does not filter zero-signal rows or choose clipping.
    """
    if (
        rewards.ndim != 1
        or len(rewards) != len(prompts)
        or not len(prompts)
        or not torch.isfinite(rewards).all()
    ):
        raise ValueError("invalid reward population")
    values = rewards.detach().double().cpu()
    denominator = values.std(unbiased=False) + 1e-4
    result = torch.empty_like(values)
    for prompt in set(prompts):
        mask = torch.tensor([p == prompt for p in prompts])
        result[mask] = (values[mask] - values[mask].mean()) / denominator
    return result.float()
