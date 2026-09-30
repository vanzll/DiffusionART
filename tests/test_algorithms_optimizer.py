"""Optimizer-state parity against standalone equations and uneven microbatches."""

import pytest
import torch

from visual_rl.algorithms import (
    art_loss,
    clipped_grpo_loss,
    flowdppo_loss,
    nft_loss,
    old_policy_decay,
    refl_loss,
)


def reference_terms(kind, prediction, old, data):
    """Direct reference equations, deliberately independent of kernel helpers."""
    advantage, sigma, xt, x0, direction = data
    if kind == "art":
        target = old - direction
        return sigma * (prediction - target).square().mean(1)
    if kind == "nft":
        plus = xt - sigma[:, None] * (0.6 * prediction + 0.4 * old) - x0
        minus = xt - sigma[:, None] * (1.6 * old - 0.6 * prediction) - x0
        positive_scale = plus.detach().double().abs().mean(1).clamp_min(1e-5)
        negative_scale = minus.detach().double().abs().mean(1).clamp_min(1e-5)
        probability = advantage.clamp(-2, 2) / 4 + 0.5
        positive_loss = (plus.square() / positive_scale[:, None]).mean(1)
        negative_loss = (minus.square() / negative_scale[:, None]).mean(1)
        return 2 * (probability * positive_loss / 0.6 + (1 - probability) * negative_loss / 0.6)
    if kind == "refl":
        return -(0.3 * prediction[:, 0] + 0.7 * prediction[:, 1])
    ratio = (prediction.mean(1) - old.mean(1)).exp()
    if kind == "grpo":
        return torch.maximum(-advantage * ratio, -advantage * ratio.clamp(0.97, 1.03))
    kl = (prediction - old).square().mean(1) / (2 * 0.4**2)
    keep = ~((kl >= 0.001) & (((ratio > 1) & (advantage > 0)) | ((ratio < 1) & (advantage < 0))))
    return torch.where(keep.detach(), -advantage * ratio, 0.0)


def kernel_terms(kind, prediction, old, data):
    advantage, sigma, xt, x0, direction = data
    if kind == "art":
        return art_loss(
            prediction,
            old,
            sigma,
            direction,
            torch.ones(len(prediction)),
            direction_normalization="none",
            velocity_mask_metric="fixed_mse",
            corrective_mask_enabled=False,
        )["per_item_loss"]
    if kind == "nft":
        return nft_loss(prediction, old, x0, xt, sigma, advantage, nft_beta=0.6, adv_clip_max=2)[
            "per_item_loss"
        ]
    if kind == "refl":
        return refl_loss({"a": prediction[:, 0], "b": prediction[:, 1]}, {"a": 0.3, "b": 0.7})[
            "per_item_loss"
        ]
    if kind == "grpo":
        return clipped_grpo_loss(prediction.mean(1), old.mean(1), advantage, ratio_clip=0.03)[
            "per_item_loss"
        ]
    return flowdppo_loss(
        prediction.mean(1),
        old.mean(1),
        prediction,
        old,
        torch.tensor(0.4),
        advantage,
        kl_mask_threshold=0.001,
    )["per_item_loss"]


@pytest.mark.parametrize("kind", ["art", "nft", "grpo", "flowdppo", "refl"])
@pytest.mark.parametrize("chunks", [(7,), (2, 4, 1), (1, 1, 1, 1, 1, 1, 1)])
def test_weighted_microbatch_values_gradients_and_adamw_transitions(kind, chunks):
    generator = torch.Generator().manual_seed(73)
    features = torch.randn(7, 3, generator=generator)
    initial = torch.randn(3, 2, generator=generator) * 0.1
    full = torch.nn.Parameter(initial.clone())
    micro = torch.nn.Parameter(initial.clone())
    old_full = features @ initial
    old_micro = old_full.clone()
    optimizers = [torch.optim.AdamW([p], lr=0.02, weight_decay=0.01) for p in (full, micro)]
    advantage = torch.tensor([2.0, -2.0, 0.4, -0.3, 1.3, -0.8, 0.0])
    sigma = torch.linspace(0.1, 0.9, 7)
    xt = torch.randn(7, 2, generator=generator)
    x0 = torch.randn(7, 2, generator=generator)
    direction = torch.randn(7, 2, generator=generator)
    data = (advantage, sigma, xt, x0, direction)
    weights = torch.tensor([0.3, 2.0, 0.0, 1.1, 0.6, 1.8, 0.7])
    # The denominator belongs to the coordinator and need not equal local N.
    denominator = 13.0
    for step in range(4):
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        loss_full = (
            reference_terms(kind, features @ full, old_full, data) * weights
        ).sum() / denominator
        loss_full.backward()
        start, total = 0, 0.0
        for size in chunks:
            rows = slice(start, start + size)
            terms = kernel_terms(
                kind, features[rows] @ micro, old_micro[rows], tuple(t[rows] for t in data)
            )
            loss = (terms * weights[rows]).sum() / denominator
            total += float(loss.detach())
            loss.backward()
            start += size
        assert total == pytest.approx(float(loss_full.detach()), abs=2e-6, rel=2e-6)
        torch.testing.assert_close(micro.grad, full.grad, atol=3e-6, rtol=3e-6)
        for optimizer in optimizers:
            optimizer.step()
        torch.testing.assert_close(micro, full, atol=3e-6, rtol=3e-6)
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                optimizers[1].state[micro][key],
                optimizers[0].state[full][key],
                atol=3e-6,
                rtol=3e-6,
            )
        if kind == "nft":
            # Optimizer first, then old-policy refresh at zero-based index.
            expected_decay = min(step * 0.2, 0.6)
            old_full = expected_decay * old_full + (1 - expected_decay) * (features @ full.detach())
            decay = old_policy_decay(step, "slow-ema", rate=0.2)
            old_micro = decay * old_micro + (1 - decay) * (features @ micro.detach())
            torch.testing.assert_close(old_micro, old_full, atol=3e-6, rtol=3e-6)
        # Other algorithms deliberately reuse the original rollout across updates.
