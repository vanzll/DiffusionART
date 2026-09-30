"""Frozen ART reward/endpoint statistics. See algorithm-provenance.md."""

import torch

from ._validation import batch, scalar


@torch.no_grad()
def art_scalar_covariance(
    endpoints: torch.Tensor,
    rewards: torch.Tensor,
    row_group_indices: torch.Tensor,
    group_count: int,
    *,
    reduction: str = "unbiased",
    direction_mode: str = "historical",
    scale_mode: str = "unit_normalized_covariance",
    guidance_gain: float = 1.0,
    zero_signal_mode: str = "historical",
    zero_signal_atol: float = 0.0,
    eps: float = 1e-5,
) -> dict[str, torch.Tensor]:
    """Complete-group covariance, never a microbatch estimator.

    Historical direction reduction divides covariance by reward MAD before
    RMS normalization with an epsilon floor. ``normalized_aggregate`` instead
    normalizes the aggregate covariance to exact unit RMS when active.
    ``threshold`` zero handling explicitly gates raw covariance and reward range;
    it is not silently applied to the historical fp32 computation.
    """
    batch(endpoints, "endpoints")
    eps = scalar(eps, "eps", positive=True)
    gain = scalar(guidance_gain, "guidance_gain")
    atol = scalar(zero_signal_atol, "zero_signal_atol")
    if reduction not in ("unbiased", "population"):
        raise ValueError("reduction must be unbiased or population")
    if direction_mode not in ("historical", "normalized_aggregate"):
        raise ValueError("unknown direction_mode")
    if zero_signal_mode not in ("historical", "threshold"):
        raise ValueError("unknown zero_signal_mode")
    if (zero_signal_mode == "historical" and atol != 0) or (
        zero_signal_mode == "threshold" and atol <= 0
    ):
        raise ValueError("threshold mode requires positive atol; historical requires zero")
    if scale_mode not in (
        "reward_mad",
        "raw_covariance",
        "reward_std_normalized_covariance",
        "unit_normalized_covariance",
    ):
        raise ValueError("unknown scale_mode")
    if isinstance(group_count, bool) or not isinstance(group_count, int) or group_count < 1:
        raise ValueError("group_count must be a positive integer")
    if row_group_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("row_group_indices must be integer indices")
    x = endpoints.detach().float().flatten(1)
    r = rewards.detach().to(device=x.device, dtype=torch.float32).reshape(-1)
    groups = row_group_indices.to(device=x.device, dtype=torch.long).reshape(-1)
    if r.numel() != len(x) or groups.numel() != len(x):
        raise ValueError("rewards and groups must match the endpoint count")
    if (groups < 0).any() or (groups >= group_count).any():
        raise ValueError("group index out of range")
    if not torch.isfinite(x).all() or not torch.isfinite(r).all():
        raise ValueError("endpoints and rewards must be finite")
    counts = torch.bincount(groups, minlength=group_count).float()
    sums = x.new_zeros(group_count).index_add_(0, groups, r)
    square_sums = x.new_zeros(group_count).index_add_(0, groups, r.square())
    means = sums / counts.clamp_min(1)
    stds = (square_sums / counts.clamp_min(1) - means.square()).clamp_min(0).sqrt()
    centered = r - means[groups]
    mad = x.new_zeros(group_count).index_add_(0, groups, centered.abs()) / counts.clamp_min(1)
    numerator = x.new_zeros(group_count, x.shape[1]).index_add_(0, groups, centered[:, None] * x)
    denominator = (counts - int(reduction == "unbiased")).clamp_min(1)
    covariance = numerator / denominator[:, None]
    covariance = torch.where((counts > 1)[:, None], covariance, 0.0)
    rms = covariance.square().mean(1).sqrt()
    magnitudes = {
        "reward_mad": mad,
        "raw_covariance": rms,
        "reward_std_normalized_covariance": rms / (stds + eps),
        "unit_normalized_covariance": torch.ones_like(rms),
    }[scale_mode] * gain
    signal = covariance / torch.where(mad > 0, mad, 1.0)[:, None]
    signal_rms = signal.square().mean(1).sqrt()
    active = (counts > 1) & (mad > 0) & (signal_rms > 0) & (magnitudes > 0)
    if zero_signal_mode == "threshold":
        maxima = r.new_full((group_count,), -torch.inf).scatter_reduce_(0, groups, r, reduce="amax")
        minima = r.new_full((group_count,), torch.inf).scatter_reduce_(0, groups, r, reduce="amin")
        active &= (maxima - minima > atol) & (rms > atol)
        magnitudes = torch.where(active, magnitudes, 0.0)
    if direction_mode == "historical":
        directions = signal / signal_rms.clamp_min(eps)[:, None]
    else:
        # Float64 norm avoids underflow for tiny but nonzero aggregates.
        norm = covariance.double().square().mean(1).sqrt()
        directions = (covariance.double() / torch.where(norm > 0, norm, 1.0)[:, None]).float()
    directions = torch.where(active[:, None], directions, 0.0)
    shape = (group_count, *endpoints.shape[1:])
    return {
        "group_covariance": covariance.reshape(shape),
        "group_directions": directions.reshape(shape),
        "group_guidance_magnitudes": magnitudes,
        "group_active": active,
        "group_counts": counts,
        "group_reward_means": means,
        "group_reward_stds": stds,
        "group_reward_abs_means": mad,
        "group_covariance_rms": rms,
    }


@torch.no_grad()
def art_pairwise_covariance(
    endpoints: torch.Tensor,
    scores: torch.Tensor,
    group_size: int = 8,
    *,
    seed: int = 0,
    round_index: int = 0,
    group_offset: int = 0,
    any_preference_active: bool | None = None,
    reduction: str = "unbiased",
    direction_mode: str = "historical",
    zero_signal_mode: str = "historical",
    zero_signal_atol: float = 0.0,
    eps: float = 1e-5,
) -> dict[str, torch.Tensor]:
    """Seeded disjoint comparisons; ties receive 0.5, never arbitrary winners.

    Supply globally reduced ``any_preference_active`` for multi-rank tie anchors.
    None deliberately means local activity only. RNG state is private to this call.
    """
    batch(endpoints, "endpoints")
    if group_size not in (2, 4, 8) or len(endpoints) % group_size:
        raise ValueError("pairwise covariance requires complete K=2/4/8 groups")
    scores = scores.detach().to(device=endpoints.device, dtype=torch.float32).reshape(-1)
    if scores.numel() != len(endpoints) or not torch.isfinite(scores).all():
        raise ValueError("scores must be finite with one value per endpoint")
    labels = torch.full_like(scores, 0.5)
    pairs = []
    for group in range(len(endpoints) // group_size):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            (int(seed) + 1_000_003 * (int(round_index) + int(group_offset) + group)) % (2**63 - 1)
        )
        pair = (
            (torch.randperm(group_size, generator=generator) + group * group_size)
            .reshape(-1, 2)
            .to(scores.device)
        )
        left, right = pair.unbind(1)
        label = torch.where(
            scores[left] > scores[right], 1.0, torch.where(scores[left] < scores[right], 0.0, 0.5)
        )
        labels[left], labels[right] = label, 1.0 - label
        pairs.append(pair)
    groups = torch.arange(len(endpoints), device=endpoints.device) // group_size
    result = art_scalar_covariance(
        endpoints,
        labels,
        groups,
        len(endpoints) // group_size,
        reduction=reduction,
        direction_mode=direction_mode,
        zero_signal_mode=zero_signal_mode,
        zero_signal_atol=zero_signal_atol,
        eps=eps,
    )
    preference = labels != 0.5
    any_active = (
        bool(preference.any()) if any_preference_active is None else bool(any_preference_active)
    )
    active = result["group_active"][groups] & preference
    anchor = ~preference & any_active
    result.update(
        {
            "binary_rewards": labels,
            "pair_rows": torch.cat(pairs),
            "row_preference_active": preference,
            "row_active": active,
            "row_guidance_magnitudes": result["group_guidance_magnitudes"][groups]
            * preference.float(),
            "row_restore_anchor": anchor,
            "row_trainable": active | anchor,
        }
    )
    return result
