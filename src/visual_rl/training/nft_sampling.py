"""Deterministic NFT endpoints, without trajectory/replay prediction caches.

Recurrence source: flow_grpo_alg/diffusers_patch/nft_solver.py at
3ca2a36f6b20c53a06813013234666d9c033a660: run_sampling, dpm_step,
convert_model_output, ddim_update and multistep_dpm_solver_second_order_update.
That source attributes its solver to MixGRPO; see THIRD_PARTY_NOTICES.md.
Here ``dpm2`` means multistep DPM-Solver++ with the midpoint correction,
not a two-model-evaluation explicit midpoint/RK2 integrator.
"""

from collections.abc import Callable

import torch

from ..sampling import validate_schedule


@torch.no_grad()
def collect_nft_endpoints(
    predict: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    noise: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    solver: str = "dpm2",
    microbatch: int = 8,
    quantize_model_time: bool = True,
) -> torch.Tensor:
    """Return detached CPU fp32 endpoints for explicit descending sigmas.

    ``predict(state, sigma_batch, row_indices)`` owns conditioning/CFG and must
    return a finite floating velocity on the state's device with its shape.
    Rows index the original noise batch; each physical call has at most
    ``microbatch`` rows. Model calls and each completed step use ``noise.dtype``;
    velocities/states are upcast to fp32 for updates and DPM scalar coefficients
    use fp64, matching the reference's casts. No RNG is consumed here.

    Sigmas are validated in [0, 1], converted to fp32 as in the NFT pipeline,
    and validated again to reject collapsed intervals. No resampling, shifting,
    or terminal padding is performed. For clean endpoints supply terminal zero;
    a positive terminal sigma deliberately returns a partially denoised state.
    ``shifted_schedule`` is not necessarily the reference scheduler's grid.

    With ``quantize_model_time=True``, only model inputs are floor(sigma*1000)
    /1000, reproducing the SD3 pipeline's integer model-time convention through
    this normalized-float API. Integration always uses the unquantized sigmas.
    Training re-noising/model times are outside this API and are not quantized.

    DPM2 uses one prediction per interval, one rolling previous x0 estimate,
    and first/final DDIM(eta=0) updates (algebraically Euler), preserving their
    reference arithmetic. ``euler`` uses direct fp32 velocity Euler at every
    interval. Neither mode returns old-policy predictions or likelihoods.
    This covers the recurrence, not full pipeline or paper-result parity.
    """
    if not callable(predict):
        raise ValueError("predict must be callable")
    if solver not in ("dpm2", "euler"):
        raise ValueError("solver must be 'dpm2' or 'euler'")
    if type(microbatch) is not int or microbatch < 1:
        raise ValueError("microbatch must be a positive integer")
    if type(quantize_model_time) is not bool:
        raise ValueError("quantize_model_time must be bool")
    if (
        not isinstance(noise, torch.Tensor)
        or noise.ndim < 2
        or noise.numel() == 0
        or not noise.is_floating_point()
        or not torch.isfinite(noise).all()
    ):
        raise ValueError("noise must be a nonempty finite floating [batch, ...] tensor")
    if not isinstance(sigmas, torch.Tensor) or not sigmas.is_floating_point():
        raise ValueError("sigmas must be a floating tensor")
    validate_schedule(sigmas)
    schedule = sigmas.detach().to(device="cpu", dtype=torch.float32).clone()
    validate_schedule(schedule)
    # CPU scalar tensors retain fp64 coefficient arithmetic without promoting
    # fp32 latent tensors, as in the reference's scalar/tensor operations.
    schedule64 = schedule.double()
    endpoints = torch.empty_like(noise, device="cpu", dtype=torch.float32)
    count = len(schedule) - 1
    for start in range(0, len(noise), microbatch):
        end = min(start + microbatch, len(noise))
        rows = torch.arange(start, end, device=noise.device)
        state = noise[start:end].detach().clone()
        previous_x0 = None
        for index in range(count):
            sigma = schedule[index]
            model_sigma = (sigma * 1000).floor() / 1000 if quantize_model_time else sigma
            times = model_sigma.to(state.device).expand(end - start)
            velocity = predict(state, times, rows)
            if (
                not isinstance(velocity, torch.Tensor)
                or velocity.shape != state.shape
                or velocity.device != state.device
                or not velocity.is_floating_point()
                or not torch.isfinite(velocity).all()
            ):
                raise ValueError("predict must return a finite floating velocity matching state")
            sample, velocity = state.float(), velocity.float()
            if solver == "euler":
                updated = sample + (schedule[index + 1] - sigma) * velocity
            else:
                x0 = sample - sigma * velocity
                s, t = schedule64[index], schedule64[index + 1]
                if index == 0 or index == count - 1:
                    # Deterministic ddim_update, including its operation order;
                    # omit only the unused random draw and zero noise term.
                    epsilon = (sample - (1 - s) * x0) / s
                    updated = (1 - t) * x0 + torch.sqrt(t**2) * epsilon
                else:
                    prev = schedule64[index - 1]
                    lambda_t = torch.log(1 - t) - torch.log(t)
                    lambda_s = torch.log(1 - s) - torch.log(s)
                    lambda_prev = torch.log(1 - prev) - torch.log(prev)
                    h, h_prev = lambda_t - lambda_s, lambda_s - lambda_prev
                    r = h_prev / h
                    delta = (1.0 / r) * (x0 - previous_x0)
                    coefficient = (1 - t) * (torch.exp(-h) - 1.0)
                    updated = (t / s) * sample - coefficient * x0 - 0.5 * coefficient * delta
                previous_x0 = x0
            state = updated.to(noise.dtype)
            if not torch.isfinite(state).all():
                raise ValueError(f"nonfinite NFT state after step {index}")
        endpoints[start:end].copy_(state.float().cpu())
    return endpoints
