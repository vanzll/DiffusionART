"""Native LTX-Video 2B 0.9.6 distilled stochastic transitions.

Equations follow Lightricks/LTX-Video commit
50f0aef1297574a4b29acbdb874e3a363a6d5f2e, schedulers/rf.py.
This is re-noising of the predicted clean state, not an Euler ODE step.
The caller supplies every innovation, including the zero-weight last draw.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

LTX_VIDEO_CHECKPOINT = "ltxv-2b-0.9.6-distilled-04-25.safetensors"
LTX_VIDEO_WEIGHT_REVISION = "8984fa25007f376c1a299016d0957a37a2f797bb"
LTX_VIDEO_CODE_REVISION = "50f0aef1297574a4b29acbdb874e3a363a6d5f2e"
LTX_VIDEO_ALLOWED_INFERENCE_STEPS = (1.0, 0.9937, 0.9875, 0.9812, 0.975, 0.9094, 0.725, 0.4219)
# Actual LinearQuadratic(8) schedule; metadata only lists rounded validation
# values. Never use that rounded allowlist as the physical solver schedule.
LTX_VIDEO_NATIVE_SIGMAS = (1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0)


def ltx_video_schedule(*, device=None) -> Tensor:
    """Official LinearQuadratic(8) physical timesteps, then terminal zero."""
    return torch.tensor(LTX_VIDEO_NATIVE_SIGMAS, dtype=torch.float32, device=device)


def pack_ltx_latents(latents: Tensor) -> Tensor:
    if latents.ndim != 5 or not latents.is_floating_point() or min(latents.shape) < 1:
        raise ValueError("LTX latents must be nonempty floating [B,C,F,H,W]")
    return latents.permute(0, 2, 3, 4, 1).reshape(latents.shape[0], -1, latents.shape[1])


def unpack_ltx_latents(tokens: Tensor, shape) -> Tensor:
    if len(shape) != 5 or min(shape) < 1:
        raise ValueError("LTX latent shape must be nonempty [B,C,F,H,W]")
    b, c, f, h, w = shape
    if tokens.shape != (b, f * h * w, c):
        raise ValueError("packed tokens do not match the latent shape")
    return tokens.reshape(b, f, h, w, c).permute(0, 4, 1, 2, 3)


def draw_ltx_transition_innovation(latents: Tensor, *, generator=None) -> Tensor:
    """Draw in the official scheduler's [B,F*H*W,C] order, then unpack.

    Initial noise is drawn packed with its dedicated generator, then unpacked;
    decode noise is drawn unpacked. Transition draws must preserve the current
    packed state's strides as well as shape to align the global RNG stream.
    """
    packed = pack_ltx_latents(latents)
    # randn_like preserves dense noncontiguous strides. A plain randn(shape)
    # is not equivalent when the packed current state is noncontiguous.
    noise = (
        torch.randn_like(packed)
        if generator is None
        else torch.empty_like(packed).normal_(generator=generator)
    )
    return unpack_ltx_latents(noise, latents.shape)


def _sigma(value, state: Tensor, name: str) -> Tensor:
    value = torch.as_tensor(value, device=state.device, dtype=torch.float32)
    if value.ndim == 0 or value.shape == (1,):
        value = value.reshape(1).expand(len(state))
    if value.shape != (len(state),) or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be a finite scalar or [batch]")
    return value


def _expand(value: Tensor, state: Tensor) -> Tensor:
    return value.reshape((-1,) + (1,) * (state.ndim - 1))


@dataclass(frozen=True)
class LTXVideoTransition:
    sample: Tensor
    mean: Tensor
    predicted_clean: Tensor
    velocity_coefficient: Tensor
    state_coefficient: Tensor
    std: Tensor
    innovation: Tensor


def ltx_video_stochastic_transition(
    state: Tensor,
    velocity: Tensor,
    sigma,
    next_sigma,
    *,
    innovation: Tensor,
) -> LTXVideoTransition:
    """Exact native algebra with fixed draws and native FP32 promotion.

    Float32 scalar coefficients are broadcast exactly as in the upstream packed
    per-token path. Frozen cache records, not inferred Euler coefficients, must
    be used by replay. The returned mean is a *stochastic transition mean*.
    """
    if (
        state.ndim < 2
        or not state.is_floating_point()
        or velocity.shape != state.shape
        or innovation.shape != state.shape
        or velocity.device != state.device
        or innovation.device != state.device
        or not velocity.is_floating_point()
        or not innovation.is_floating_point()
        or innovation.dtype != state.dtype
    ):
        raise ValueError(
            "state, velocity and innovation need equal floating shape/device; noise dtype matches state"
        )
    if not all(torch.isfinite(x).all() for x in (state, velocity, innovation)):
        raise ValueError("nonfinite native LTX transition input")
    s, n = _sigma(sigma, state, "sigma"), _sigma(next_sigma, state, "next_sigma")
    if ((s <= 0) | (s > 1) | (n < 0) | (n >= s)).any():
        raise ValueError("native LTX transition requires 0 <= next_sigma < sigma <= 1")
    # Upstream multiplies float32 timestep[B,1,1] with BF16 sample[B,N,C],
    # promoting the first transition to FP32. Do not cast the result back.
    clean = state - _expand(s, state) * velocity
    mean = _expand(1 - n, state) * clean
    sample = mean + _expand(n, state) * innovation
    return LTXVideoTransition(
        sample, mean, clean, -s * (1 - n), 1 - n, n, innovation.detach().clone()
    )


@dataclass(frozen=True)
class LTXVideoReplayStep:
    """All policy-independent data required to replay one native transition."""

    state: Tensor
    sigma: Tensor
    next_sigma: Tensor
    old_velocity: Tensor
    old_mean: Tensor
    next_state: Tensor
    innovation: Tensor
    velocity_coefficient: Tensor
    state_coefficient: Tensor
    std: Tensor

    @classmethod
    def capture(cls, state, velocity, sigma, next_sigma, innovation):
        transition = ltx_video_stochastic_transition(
            state, velocity, sigma, next_sigma, innovation=innovation
        )

        # A detached CPU snapshot is immutable with respect to live tensors.
        def freeze(x):
            return x.detach().cpu().clone()

        return cls(
            *(
                freeze(x)
                for x in (
                    state,
                    _sigma(sigma, state, "sigma"),
                    _sigma(next_sigma, state, "next_sigma"),
                    velocity,
                    transition.mean,
                    transition.sample,
                    innovation,
                    transition.velocity_coefficient,
                    transition.state_coefficient,
                    transition.std,
                )
            )
        )

    def replay(self, velocity: Tensor) -> LTXVideoTransition:
        return ltx_video_stochastic_transition(
            self.state.to(device=velocity.device),
            velocity,
            self.sigma,
            self.next_sigma,
            innovation=self.innovation.to(device=velocity.device),
        )


def ltx_video_normalized_mean_mse(
    current_mean: Tensor, target_mean: Tensor, velocity_coefficient: Tensor
) -> Tensor:
    """Per-item MSE/a², without selecting ART weights, targets or masks.

    Algebraically equals velocity MSE for a frozen affine transition with a!=0.
    This utility alone does not establish ART restoration/mask equivalence.
    """
    if current_mean.shape != target_mean.shape or current_mean.ndim < 2:
        raise ValueError("mean regression needs equal [batch,...] tensors")
    a = _sigma(velocity_coefficient, current_mean, "velocity_coefficient")
    if (a == 0).any():
        raise ValueError("mean regression requires nonzero velocity coefficients")
    residual = (current_mean - target_mean.detach().to(current_mean)) / _expand(a, current_mean)
    return residual.square().flatten(1).mean(1)


def require_ltx_video_art_replay_support(*, native_coefficients: bool, explicit_mask: bool):
    """Fail closed until caller explicitly handles native means and mask semantics."""
    if not native_coefficients or not explicit_mask:
        raise ValueError(
            "Native stochastic LTX needs a=-sigma*(1-next_sigma), frozen native means, "
            "normalized mean MSE/a^2 and an explicitly chosen equivalent ART mask; "
            "the generic Euler/Q2 replay path is not compatible"
        )
