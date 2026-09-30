"""Final-step ReFL with a stopped exact endpoint cotangent, K=1.

VAE/RM input gradients are computed before reconstructing the policy graph.
This is the chain rule for the continuous endpoint objective, not an MP4 STE
and not intermediate-x0 extrapolation or full-trajectory backpropagation.
"""

import math
from dataclasses import dataclass

import torch

from ..config import ConfigError


def validate_options(c):
    if set(c.algorithm.options) != {"reward_loss_scale"}:
        raise ConfigError("ReFL requires only explicit reward_loss_scale")
    scale = c.algorithm.options["reward_loss_scale"]
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ConfigError("ReFL reward_loss_scale must be positive and finite")
    if c.reward.normalization != "fixed" or not math.isclose(sum(c.reward.weights.values()), 1.):
        raise ConfigError("video ReFL needs a convex mixture of fixed-normalized rewards")
    if c.sampling.samples_per_prompt != 1:
        raise ConfigError("ReFL requires K=1")
    if (c.sampling.options.get("train_timestep_pool") != [c.sampling.steps - 1]
            or c.sampling.options.get("train_timestep_count") != 1):
        raise ConfigError("this ReFL recipe differentiates only the final denoising step")


@dataclass
class ReFLObjective:
    cache: object
    predict: object
    latent_gradients: torch.Tensor
    losses: torch.Tensor
    device: torch.device

    def __call__(self, items):
        rows = torch.tensor(items, dtype=torch.int64)
        states = self.cache.states[rows, 0].detach().to(self.device)
        sigma = self.cache.sigmas[self.cache.selected[0]].to(self.device).expand(len(rows))
        field = self.predict(states, sigma, rows.to(self.device))
        shape = (-1,) + (1,) * (states.ndim - 1)
        endpoint = states.float() - sigma.reshape(shape) * field.float()
        if not torch.allclose(endpoint.detach(), self.cache.endpoints[rows].to(self.device), atol=2e-3, rtol=2e-3):
            raise RuntimeError("ReFL endpoint changed before its policy VJP")
        cotangent = self.latent_gradients[rows].detach().to(self.device)
        product = (endpoint * cotangent).flatten(1).sum(1)
        # Preserve original loss values while applying the unnormalized VJP.
        return self.losses[rows].to(self.device) + (product - product.detach())


def prepare_objective(c, parts, cache, predict, prompts, device):
    from .refl_vjp import staged_latent_vjp

    parts.adapter.transformer.eval()
    wan = c.model.name == "wan_i2v_distilled"
    decode_fn = getattr(parts, "refl_decode", None)
    if wan and decode_fn is None:
        raise ValueError("Wan ReFL requires its differentiable causal-cache decoder")
    gradient, losses, metrics = staged_latent_vjp(cache.endpoints, parts.adapter, parts.scorer, prompts,
        device=device, fps=c.sampling.fps, precision=c.training.precision,
        reward_loss_scale=c.algorithm.options["reward_loss_scale"], decode_fn=decode_fn,
        vae_precision="fp32" if wan else c.training.precision)
    objective = ReFLObjective(cache, predict, gradient, losses, device)
    metrics.update({"train/refl/differentiable_steps": 1., "train/refl/guidance_normalized": 0.,
                    "train/refl/update_index": float(c.sampling.steps - 1),
                    "train/refl/update_sigma": float(cache.sigmas[-2])})
    return objective, (tuple(range(len(cache.endpoints))),), metrics
