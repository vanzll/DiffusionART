"""Exact untiled Wan2.1 ReFL decode with GPU-resident temporal checkpoints.

Adapted from wan_i2v_refl.py at research reference 3ca2a36f. Unlike that
reference's decode_fn, this callback accepts ALREADY denormalized latents.
It never changes the ordinary VAE decode entrypoint or its tiling settings.
"""

import inspect
from functools import partial

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint


def _validate_vae(vae):
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d, WanDecoder3d

    if not isinstance(getattr(vae, "decoder", None), WanDecoder3d) or not isinstance(
        getattr(vae, "post_quant_conv", None), WanCausalConv3d
    ):
        raise ValueError("ReFL requires the native Wan causal decoder and post_quant_conv")
    if getattr(vae.config, "patch_size", None) is not None or getattr(
        vae.config, "is_residual", False
    ):
        raise ValueError("ReFL supports only the unpatched Wan2.1 VAE")
    count = sum(isinstance(module, WanCausalConv3d) for module in vae.decoder.modules())
    if not count:
        raise ValueError("native Wan causal decoder blocks are required")
    if any(p.is_floating_point() and p.dtype != torch.float32 for p in vae.parameters()):
        raise ValueError("Wan ReFL requires FP32 VAE parameters")
    if any(b.is_floating_point() and b.dtype != torch.float32 for b in vae.buffers()):
        raise ValueError("Wan ReFL requires FP32 VAE buffers")
    return count


def configure_wan_vae(vae) -> int:
    """Validate and freeze an FP32 native Wan2.1 VAE, idempotently.

    Return the native causal-convolution count/cache capacity, matching the
    reference configure_wan_vae. No wrappers are installed: the explicit decode
    callback below checkpoints one complete decoder invocation per latent frame.
    Reporting may retain native tiling; this callback is always untiled.
    """
    count = _validate_vae(vae)
    vae.requires_grad_(False).eval()
    for parameter in vae.parameters():
        parameter.grad = None
    vae.disable_gradient_checkpointing()
    return count


def _decoder_chunk(vae, first_chunk, frame, *cache):
    # Replay owns a fresh container; tensors retain their cross-frame graph.
    working = list(cache)
    kwargs = {"feat_cache": working, "feat_idx": [0]}
    if "first_chunk" in inspect.signature(vae.decoder.forward).parameters:
        kwargs["first_chunk"] = first_chunk
    with torch.autocast(device_type=frame.device.type, enabled=False):
        output = vae.decoder(frame, **kwargs)
    return output, *working


def decode_wan_checkpointed(vae, raw_latents: Tensor) -> Tensor:
    """Decode raw BCTHW latents to FP32 BCTHW pixels clamped to [-1, 1].

    Call configure_wan_vae first. No latent affine, spatial tiles, cache detach,
    activation offload, or OOM fallback is applied. All tensors stay on the input
    device. Every invocation owns its temporal cache, including during backward.
    No-grad previews use the same native primitives without checkpoint overhead.
    """
    count = _validate_vae(vae)
    if any(m.training for m in vae.modules()) or any(p.requires_grad for p in vae.parameters()):
        raise ValueError("call configure_wan_vae first: Wan ReFL requires frozen eval weights")
    if (
        not isinstance(raw_latents, Tensor)
        or raw_latents.ndim != 5
        or not raw_latents.is_floating_point()
        or any(n == 0 for n in raw_latents.shape)
        or raw_latents.shape[1] != vae.config.z_dim
    ):
        raise ValueError("raw latents must be nonempty floating BCTHW with z_dim channels")
    if any(p.device != raw_latents.device for p in vae.parameters()):
        raise ValueError("Wan VAE and raw latents must be on the same device")
    with torch.autocast(device_type=raw_latents.device.type, enabled=False):
        x = vae.post_quant_conv(raw_latents.float())
        cache = (None,) * count
        frames = []
        for index in range(x.shape[2]):
            decode = partial(_decoder_chunk, vae, index == 0)
            inputs = (x[:, :, index : index + 1], *cache)
            result = (
                checkpoint(decode, *inputs, use_reentrant=False)
                if torch.is_grad_enabled()
                else decode(*inputs)
            )
            frames.append(result[0])
            cache = result[1:]
        decoded = torch.cat(frames, dim=2)
        if not torch.isfinite(decoded).all():
            raise ValueError("Wan VAE decoder returned nonfinite pixels")
        return decoded.clamp(-1.0, 1.0)
