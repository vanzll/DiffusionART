"""Row-bounded ReFL VJPs; model placement only, never CPU activation offload."""

from functools import partial
from types import MethodType

import torch
from torch.utils.checkpoint import checkpoint

from visual_rl.models.base import module_dtype
from visual_rl.rewards.inputs import ContinuousRGBInput
from visual_rl.rewards.scalar import ScoreResult, finite_number

_COMPONENTS = ("VQ", "MQ", "TA")


def _freeze(module):
    module.requires_grad_(False).eval()
    for parameter in module.parameters():
        parameter.grad = None


def _hunyuan_output(decoder, hidden):
    return decoder.conv_out(decoder.conv_act(decoder.conv_norm_out(hidden)))


def _hunyuan_forward(decoder, hidden_states):
    if not torch.is_grad_enabled():
        return decoder._refl_native_forward(hidden_states)
    hidden_states = checkpoint(decoder.conv_in, hidden_states, use_reentrant=False)
    hidden_states = checkpoint(decoder.mid_block, hidden_states, use_reentrant=False)
    for block in decoder.up_blocks:
        hidden_states = checkpoint(block, hidden_states, use_reentrant=False)
    return checkpoint(_hunyuan_output, decoder, hidden_states, use_reentrant=False)


def configure_hunyuan_vae(vae):
    """Freeze and checkpoint native decoder regions; return their count.

    Adapted from research refl_vae_memory.py at 3ca2a36f. Idempotent; native
    no-grad forward, decode, state-dict keys and all tiling settings are retained.
    Configure only the ReFL native loader, not arbitrary injected test VAEs.
    """
    decoder = getattr(vae, "decoder", None)
    required = ("conv_in", "mid_block", "up_blocks", "conv_norm_out", "conv_act", "conv_out")
    if (type(decoder).__name__ != "HunyuanVideoDecoder3D"
            or any(not isinstance(getattr(decoder, key, None), torch.nn.Module) for key in required)
            or type(decoder.mid_block).__name__ != "HunyuanVideoMidBlock3D"
            or not len(decoder.up_blocks)
            or any(type(block).__name__ != "HunyuanVideoUpBlock3D" for block in decoder.up_blocks)):
        raise RuntimeError("ReFL requires complete native Hunyuan decoder blocks")
    if any(hasattr(module, "_checkpoint_wrapped_module") for module in decoder.modules()):
        raise RuntimeError("configure a fresh Hunyuan VAE without fine-grained wrappers")
    _freeze(vae)
    if hasattr(vae, "disable_gradient_checkpointing"):
        vae.disable_gradient_checkpointing()
    if not hasattr(decoder, "_refl_native_forward"):
        decoder._refl_native_forward = decoder.forward
        decoder.forward = MethodType(_hunyuan_forward, decoder)
    return len(decoder.up_blocks) + 3


def configure_video_reward(scorer):
    """Freeze native VideoReward and install non-reentrant Qwen2-VL checkpoints."""
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )

    model = scorer.model
    names = {"Qwen2VLDecoderLayer", "Qwen2VLVisionBlock"}
    count = sum(type(module).__name__ in names for module in model.modules())
    if not count:
        raise RuntimeError("native VideoReward Qwen2-VL blocks missing")
    _freeze(scorer)
    if getattr(scorer, "_refl_checkpoint_regions", None) is None:
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=partial(
                checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            ),
            check_fn=lambda module: type(module).__name__ in names,
        )
        scorer._refl_checkpoint_regions = count
    return scorer._refl_checkpoint_regions


def _assert_frozen(module, name):
    if any(part.training for part in module.modules()):
        raise ValueError(f"ReFL {name} must be in eval mode")
    if any(p.requires_grad or p.grad is not None for p in module.parameters()):
        raise ValueError(f"ReFL {name} must have frozen weights and no weight gradients")


def _decode(adapter, latent, device, precision, decode_fn=None):
    # Exactly one model-to-VAE affine per native decode, including rematerialization.
    raw = adapter.to_vae_latents(latent)
    raw = raw.to(dtype=module_dtype(adapter.vae, raw.dtype))
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
        decoded = adapter.vae.decode(raw, return_dict=False)[0] if decode_fn is None else decode_fn(raw)
    if (not isinstance(decoded, torch.Tensor) or decoded.ndim != 5
            or decoded.shape[:2] != (1, 3) or not torch.isfinite(decoded).all()):
        raise ValueError("ReFL VAE must return finite BCTHW RGB for one row")
    # Reward's (frames + 1) * .5 must run in FP32, not BF16.
    return decoded.clamp(-1, 1).float().permute(0, 2, 1, 3, 4)


def _row_vjp(endpoint, adapter, scorer, prompt, device, fps, precision, scale, decode_fn, vae_precision):
    metrics = {}
    vae = adapter.vae
    vae.to(device)
    with torch.no_grad():
        preview = _decode(adapter, endpoint.to(device), device, vae_precision, decode_fn)
    vae.to("cpu")
    scorer.to(device)
    with torch.enable_grad():
        pixels = preview.detach().requires_grad_(True)
        inputs = ContinuousRGBInput(pixels, [prompt], fps, "minus_one_one")
        inputs.validate()
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
            scores = scorer(inputs)
        if (not isinstance(scores, ScoreResult) or scores.fixed_normalized is None
                or any(set(values) != set(_COMPONENTS)
                       for values in (scores.raw, scores.fixed_normalized, scores.weights))):
            raise ValueError("ReFL requires VQ/MQ/TA fixed-normalized scores and weights")
        for prefix, values in (("raw", scores.raw), ("normalized", scores.fixed_normalized)):
            for key, value in values.items():
                if (not isinstance(value, torch.Tensor) or value.shape != (1,)
                        or not torch.isfinite(value).all()):
                    raise ValueError("ReFL scores must be finite one-row tensors")
                metrics[f"{prefix}_{key}"] = value.detach().item()
        loss = -scale * sum(finite_number(scores.weights[key], key) * scores.fixed_normalized[key]
                            for key in _COMPONENTS)
        if not loss.requires_grad or not torch.isfinite(loss).all():
            raise ValueError("ReFL loss must be finite and input-differentiable")
        pixel_grad, = torch.autograd.grad(loss.sum(), pixels)
        if not torch.isfinite(pixel_grad).all():
            raise ValueError("nonfinite ReFL pixel gradient")
        loss_cpu = loss.detach().cpu()
    del scores, loss, inputs, pixels
    scorer.to("cpu")
    # Detached stage boundaries need not occupy GPU memory during VAE replay.
    # This is not saved-activation offload: checkpoint activations stay on GPU.
    preview = preview.detach().cpu()
    pixel_grad = pixel_grad.detach().cpu()
    vae.to(device)
    with torch.enable_grad():
        latent = endpoint.to(device).detach().requires_grad_(True)
        rematerialized = _decode(adapter, latent, device, vae_precision, decode_fn)
        if (rematerialized.shape != preview.shape
                or not torch.allclose(rematerialized.detach().cpu(), preview, atol=2e-3, rtol=0)):
            raise RuntimeError("ReFL VAE rematerialization differs from preview (atol=2e-3)")
        del preview
        latent_grad, = torch.autograd.grad(rematerialized, latent, grad_outputs=pixel_grad.to(device))
        if not torch.isfinite(latent_grad).all():
            raise ValueError("nonfinite ReFL latent gradient")
        gradient_cpu = latent_grad.detach().cpu()
    del rematerialized, latent, latent_grad, pixel_grad
    vae.to("cpu")
    return gradient_cpu, loss_cpu, metrics


def staged_latent_vjp(endpoints, adapter, scorer, prompts, *, device, fps, precision,
                      reward_loss_scale=1.0, decode_fn=None, vae_precision=None):
    """Return (CPU latent cotangents, CPU [B] losses, scalar metrics).

    Each row is d(-sum(weights * fixed_normalized) * scale)/dx0, WITHOUT
    population averaging or gradient normalization. The caller averages once.
    Only one row's activations is live; VAE and RM weights alternate residency.
    Both modules must already be frozen/eval and are left on CPU even on failure.
    CPU execution is supported for tests; native configuration is caller-owned.
    An optional decode_fn accepts already-denormalized raw latents and returns
    BCTHW RGB. vae_precision can differ from reward precision (Wan VAE is FP32).
    """
    device = torch.device(device)
    try:
        _assert_frozen(adapter.vae, "VAE")
        _assert_frozen(scorer, "scorer")
        if (not isinstance(endpoints, torch.Tensor) or endpoints.ndim != 5
                or not endpoints.is_floating_point() or any(n == 0 for n in endpoints.shape)
                or not torch.isfinite(endpoints).all()):
            raise ValueError("endpoints must be finite nonempty floating BCTHW latents")
        if (isinstance(prompts, str) or len(prompts) != endpoints.shape[0]
                or any(not isinstance(p, str) or not p.strip() for p in prompts)):
            raise ValueError("one nonempty prompt per endpoint required")
        if precision not in ("bf16", "fp32"):
            raise ValueError("ReFL precision must be bf16 or fp32")
        vae_precision = precision if vae_precision is None else vae_precision
        if vae_precision not in ("bf16", "fp32") or (decode_fn is not None and not callable(decode_fn)):
            raise ValueError("invalid ReFL VAE precision or decoder")
        if finite_number(fps, "fps") <= 0:
            raise ValueError("fps must be positive")
        scale = finite_number(reward_loss_scale, "reward_loss_scale")
        gradients = torch.empty_like(endpoints, device="cpu")
        losses, totals = [], {}
        # Establish nonoverlapping residency even if the caller left both on device.
        scorer.to("cpu")
        adapter.vae.to("cpu")
        for row, prompt in enumerate(prompts):
            gradient, loss, metrics = _row_vjp(
                endpoints[row:row + 1].detach(), adapter, scorer, prompt,
                device, fps, precision, scale, decode_fn, vae_precision,
            )
            gradients[row:row + 1].copy_(gradient)
            losses.append(loss)
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.) + value
        losses = torch.cat(losses)
        metrics = {f"train/refl/{key}": value / len(prompts)
                   for key, value in totals.items()}
        metrics["train/refl/loss"] = losses.double().mean().item()
        _assert_frozen(adapter.vae, "VAE")
        _assert_frozen(scorer, "scorer")
    finally:
        # Attempt both offloads even when the first one fails; never hide that failure.
        try:
            scorer.to("cpu")
        finally:
            adapter.vae.to("cpu")
    return gradients, losses, metrics
