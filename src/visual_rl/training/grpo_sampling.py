"""Logical-batch FlowGRPO/Fast sampling with row-specific replay caches.

Selection reference: flow_grpo_alg/diffusers_patch/sd3_pipeline_with_logprob_fast.py
at 3ca2a36f6b20c53a06813013234666d9c033a660. Transition equations from that
revision's sd3_sde_with_logprob.py are reused through the portable rollout and
sampling modules, not reimplemented here. See THIRD_PARTY_NOTICES.md.
"""

import math
from collections.abc import Callable
from numbers import Real

import torch

from ..sampling import validate_schedule
from .rollout import Rollout, collect_rollout


@torch.no_grad()
def collect_grpo_rollout(
    predict: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    noise: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    pool: tuple[int, ...],
    count: int,
    selection: str,
    logical_batch_size: int,
    microbatch: int,
    transition_kind: str,
    likelihood: str,
    noise_level: float,
    unselected_noise: str,
    generator: torch.Generator,
    flux_precision: bool = False,
    endpoint_kind: str = "sample",
) -> Rollout:
    """Select once per logical batch, then partition only model execution.

    ``predict(state, sigma_batch, global_rows)`` must be deterministic and
    row-separable for physical-microbatch invariance. The first model call keeps
    the initial latent dtype; later states, innovations and transitions are FP32
    by default. ``flux_precision`` instead rounds each resulting latent back to
    the initial dtype, draws innovations in packed coordinates, and preserves
    zero-noise CPS arithmetic outside a CPS window. Old likelihoods remain those
    of pre-rounding samples, as in the reference FLUX pipeline.
    Model precision/conditioning belong to
    ``predict``; model times are the supplied sigmas, without quantization.

    Each equal-sized logical batch draws its selection, then one full-batch
    innovation per stochastic step in chronological order on ``generator``'s
    device, before any prediction. Innovations and returned caches are detached
    CPU copies independent of input storage. Changing logical batch size changes
    this random stream; changing physical microbatch does not. This stream is
    not bitwise historical Python-random/Diffusers RNG parity.

    ``fixed`` preserves pool order and requires count == len(pool).
    ``contiguous`` requires an increasing consecutive pool and samples uniformly
    over all valid windows; ``random_subset`` sorts a uniform subset. Every
    eligible pool step must support a stochastic transition, even if not drawn.
    CPS transitions ending at zero cannot appear in the pool and always use ODE.
    SDE terminal transitions may be selected or stochastic in ``all`` mode.
    ``random_per_row`` independently shuffles each trajectory's eligible steps,
    and requires full exploration. Dance's ``last_x0`` scores the last
    pre-transition endpoint while retaining the sampled transition for replay.

    The merged ``selected`` is only the first logical batch's representative;
    use ``indices(rows, slots)`` for replay. ``stochastic`` is the union across
    batches, not a per-row stochasticity mask. Only selected steps are cached.
    """
    if not callable(predict):
        raise ValueError("predict must be callable")
    for name, value in (("logical_batch_size", logical_batch_size), ("microbatch", microbatch)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if (
        not isinstance(noise, torch.Tensor)
        or noise.ndim < 2
        or noise.numel() == 0
        or not noise.is_floating_point()
        or not torch.isfinite(noise).all()
        or not torch.isfinite(noise.float()).all()
    ):
        raise ValueError("noise must be a nonempty finite FP32-representable [batch, ...] tensor")
    if len(noise) % logical_batch_size:
        raise ValueError("rows must be divisible by logical_batch_size")
    if not isinstance(sigmas, torch.Tensor) or not sigmas.is_floating_point():
        raise ValueError("sigmas must be a floating tensor")
    validate_schedule(sigmas)
    schedule = sigmas.detach().to(device="cpu", dtype=torch.float32).clone()
    validate_schedule(schedule)
    if selection not in ("fixed", "contiguous", "random_subset", "random_per_row"):
        raise ValueError("unsupported timestep selection")
    if transition_kind not in ("sde", "dance", "cps"):
        raise ValueError("transition_kind must be sde, dance or cps")
    if transition_kind == "dance" and (flux_precision or likelihood != "gaussian"):
        raise ValueError("image Dance requires SD3 and Gaussian likelihoods")
    if endpoint_kind not in ("sample", "last_x0") or (
        endpoint_kind == "last_x0" and transition_kind != "dance"
    ):
        raise ValueError("last_x0 endpoint selection is Dance-only")
    if likelihood not in ("gaussian", "mse"):
        raise ValueError("likelihood must be gaussian or mse")
    if unselected_noise not in ("ode", "all"):
        raise ValueError("unselected_noise must be ode or all")
    if (
        isinstance(noise_level, bool)
        or not isinstance(noise_level, Real)
        or not math.isfinite(noise_level)
        or not 0 < noise_level <= 1
    ):
        raise ValueError("noise_level must lie in (0, 1]")
    if not isinstance(generator, torch.Generator):
        raise ValueError("generator must be an explicit torch.Generator")
    steps = len(schedule) - 1
    if (
        not isinstance(pool, tuple)
        or not pool
        or any(type(i) is not int or not 0 <= i < steps for i in pool)
        or len(set(pool)) != len(pool)
    ):
        raise ValueError("pool must be a nonempty tuple of unique valid integer steps")
    if type(count) is not int or not 1 <= count <= len(pool):
        raise ValueError("count must be a positive integer no larger than pool")
    if selection == "fixed" and count != len(pool):
        raise ValueError("fixed selection requires count == len(pool)")
    if selection == "contiguous" and pool != tuple(range(pool[0], pool[0] + len(pool))):
        raise ValueError("contiguous selection requires an increasing consecutive pool")
    terminal_cps = transition_kind == "cps" and schedule[-1] == 0
    if terminal_cps and steps - 1 in pool:
        raise ValueError("pool cannot select a terminal zero-variance CPS transition")
    if transition_kind == "sde" and schedule[0] == 1 and schedule[1] == 0:
        raise ValueError("SDE at sigma=1 requires an interior next sigma for sigma_max")

    caches = []
    if type(flux_precision) is not bool:
        raise ValueError("flux_precision must be boolean")
    if flux_precision and (noise.ndim != 4 or any(n % 2 for n in noise.shape[-2:])):
        raise ValueError("FLUX rollout requires BCHW latents with even spatial dimensions")
    indices = []
    stochastic_union = set()
    for start in range(0, len(noise), logical_batch_size):
        row_selected = None
        if selection == "fixed":
            selected = pool
        elif selection == "contiguous":
            offset = int(
                torch.randint(
                    len(pool) - count + 1, (), generator=generator, device=generator.device
                ).item()
            )
            selected = pool[offset : offset + count]
        elif selection == "random_subset":
            order = torch.randperm(len(pool), generator=generator, device=generator.device)
            selected = tuple(sorted(pool[i] for i in order[:count].cpu().tolist()))
        else:
            eligible = torch.tensor(pool, dtype=torch.int64, device=generator.device)
            row_selected = torch.stack([
                eligible[torch.randperm(len(pool), generator=generator, device=generator.device)[:count]]
                for _ in range(logical_batch_size)
            ]).cpu()
            selected = tuple(row_selected[0].tolist())
            if unselected_noise != "all":
                raise ValueError("per-row selection requires a fully stochastic rollout")
        stochastic = (
            tuple(range(steps - int(terminal_cps)))
            if unselected_noise == "all"
            else tuple(sorted(selected))
        )
        logical_noise = noise[start : start + logical_batch_size].detach()
        noise_shape = logical_noise.shape
        if flux_precision:
            batch, channels, height, width = noise_shape
            noise_shape = (batch, height * width // 4, channels * 4)
        innovations = {
            step: torch.randn(
                noise_shape,
                generator=generator,
                device=generator.device,
                dtype=torch.float32,
            ).cpu()
            for step in stochastic
        }
        if flux_precision:
            from ..models.flux import unpack_latents

            innovations = {step: unpack_latents(value, height, width)
                           for step, value in innovations.items()}

        def predict_global(state, times, rows):
            velocity = predict(state, times, rows + start)
            if (
                not isinstance(velocity, torch.Tensor)
                or velocity.shape != state.shape
                or velocity.device != state.device
                or not velocity.is_floating_point()
                or not torch.isfinite(velocity).all()
            ):
                raise ValueError("predict must return a finite floating velocity matching state")
            return velocity

        cache = collect_rollout(
            predict_global,
            logical_noise,
            schedule,
            selected=selected,
            stochastic=stochastic,
            innovations=innovations,
            transition_kind=transition_kind,
            likelihood=likelihood,
            noise_level=float(noise_level),
            microbatch=microbatch,
            preserve_state_dtype=flux_precision,
            unselected_kind="cps_zero_noise"
            if flux_precision and transition_kind == "cps" else "ode",
            row_selected=row_selected,
            endpoint_kind=endpoint_kind,
        )
        caches.append(cache)
        indices.append(row_selected if row_selected is not None else
                       torch.tensor(selected, dtype=torch.long).repeat(logical_batch_size, 1))
        stochastic_union.update(stochastic)

    fields = (
        "endpoints",
        "states",
        "old_velocities",
        "next_states",
        "old_log_probs",
        "old_means",
        "transition_stds",
    )
    merged = {name: torch.cat([getattr(cache, name) for cache in caches]) for name in fields}
    if any(not torch.isfinite(value).all() for value in merged.values()):
        raise ValueError("nonfinite GRPO rollout cache")
    return Rollout(
        **merged,
        selected=caches[0].selected,
        sigmas=schedule,
        stochastic=tuple(sorted(stochastic_union)),
        transition_kind=transition_kind,
        likelihood=likelihood,
        noise_level=float(noise_level),
        step_indices=torch.cat(indices),
    )
