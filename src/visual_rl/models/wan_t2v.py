"""Wan2.1 T2V components, distinct from the distilled first-frame I2V adapter.

No optional libraries or assets are loaded at import time. The native DiT
zero-pads text *before* projection; those pad tokens participate in attention.
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .base import (
    VelocityAdapter,
    batch_tensor,
    check_component_config,
    checked_velocity,
    guidance_scale,
    positive,
    sigma_batch,
    validate_latents,
)
from .wan import decode_wan_differentiable

WAN_T2V_SOURCE_REVISION = "9737cba9c1c3c4d04b33fcad41c111989865d315"
WAN_T2V_NATIVE_REVISION = "37ec512624d61f7aa208f7ea8140a131f93afc9a"
WAN_T2V_DIFFUSERS_REVISION = "0fad780a534b6463e45facd96134c9f345acfa5b"
WAN_T2V_LATENTS_MEAN = (
    -0.7571,
    -0.7089,
    -0.9113,
    0.1075,
    -0.1745,
    0.9653,
    -0.1517,
    1.5508,
    0.4134,
    -0.0715,
    0.5517,
    -0.3632,
    -0.1922,
    -0.9497,
    0.2503,
    -0.2921,
)
WAN_T2V_LATENTS_STD = (
    2.8184,
    1.4541,
    2.3275,
    2.6558,
    1.2196,
    1.7708,
    2.6052,
    2.0743,
    3.2687,
    2.1526,
    2.8652,
    1.5579,
    1.6382,
    1.1253,
    2.8251,
    1.9160,
)
WAN_T2V_DIFFUSERS_LORA_TARGETS = ("to_q", "to_k", "to_v", "to_out.0")
WAN_T2V_NATIVE_LORA_TARGETS = ("q", "k", "v", "o")


@dataclass(frozen=True)
class WanT2VConfig:
    latents_mean: tuple[float, ...] = WAN_T2V_LATENTS_MEAN
    latents_std: tuple[float, ...] = WAN_T2V_LATENTS_STD
    cfg_scale: float | None = 6.0
    timestep_scale: float = 1000.0
    text_len: int = 512
    vae_stride: tuple[int, int, int] = (4, 8, 8)
    patch_size: tuple[int, int, int] = (1, 2, 2)
    transformer_api: str = "native"
    vae_api: str = "native"

    def __post_init__(self):
        for name in ("latents_mean", "latents_std", "vae_stride", "patch_size"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not self.latents_mean or len(self.latents_mean) != len(self.latents_std):
            raise ValueError("Wan requires one mean and std per latent channel")
        if any(not math.isfinite(v) for v in self.latents_mean):
            raise ValueError("latents_mean must be finite")
        for value in self.latents_std:
            positive(value, "latents_std")
        for name in ("vae_stride", "patch_size"):
            values = getattr(self, name)
            if len(values) != 3 or any(type(v) is not int or v < 1 for v in values):
                raise ValueError(f"{name} requires three positive integers")
        if type(self.text_len) is not int or self.text_len < 1:
            raise ValueError("text_len must be a positive integer")
        if self.transformer_api not in ("native", "diffusers"):
            raise ValueError("unknown Wan transformer API")
        if self.vae_api not in ("native", "diffusers"):
            raise ValueError("unknown Wan VAE API")
        positive(self.timestep_scale, "timestep_scale")
        guidance_scale(self.cfg_scale)

    @property
    def latent_channels(self):
        return len(self.latents_mean)

    def latent_shape(self, *, height: int, width: int, frames: int):
        if any(type(x) is not int or x < 1 for x in (height, width, frames)):
            raise ValueError("height, width and frames must be positive integers")
        st, sh, sw = self.vae_stride
        if (frames - 1) % st or height % sh or width % sw:
            raise ValueError("video dimensions must align with Wan VAE strides")
        shape = (self.latent_channels, (frames - 1) // st + 1, height // sh, width // sw)
        if any(a % b for a, b in zip(shape[1:], self.patch_size)):
            raise ValueError("latent dimensions must align with transformer patches")
        return shape


@dataclass(frozen=True)
class WanT2VConditioning:
    prompt_embeds: Tensor
    prompt_attention_mask: Tensor
    negative_prompt_embeds: Tensor | None = None
    negative_prompt_attention_mask: Tensor | None = None
    # Use the native scheduler's actual INT64 time conditions for exact parity.
    # Physical sigma remains independent: int truncation loses information.
    timesteps: Tensor | None = None


def masked_wan_context(embeds: Tensor, mask: Tensor, *, text_len: int) -> Tensor:
    """Trim right padding and restore native pre-projection zero padding.

    A DiT attention mask would remove projected pad tokens, changing the native
    model. T5 must already have used this mask while producing its embeddings.
    """
    if (
        not isinstance(embeds, Tensor)
        or embeds.ndim != 3
        or not embeds.is_floating_point()
        or min(embeds.shape) < 1
    ):
        raise ValueError("prompt embeddings must be nonempty floating [B,L,D]")
    if (
        not isinstance(mask, Tensor)
        or mask.shape != embeds.shape[:2]
        or mask.device != embeds.device
        or ((mask != 0) & (mask != 1)).any()
    ):
        raise ValueError("prompt attention mask must be binary [B,L] on the embedding device")
    if type(text_len) is not int or text_len < embeds.shape[1]:
        raise ValueError("prompt embeddings exceed the native text length")
    valid = mask.bool()
    if (~valid[:, :-1] & valid[:, 1:]).any() or not valid.any(dim=1).all():
        raise ValueError("prompt attention mask requires a nonempty right-padded prefix")
    if not torch.isfinite(embeds[valid]).all():
        raise ValueError("valid prompt embeddings must be finite")
    # where also removes NaN/Inf from masked padding without 0*NaN propagation.
    cleaned = torch.where(valid.unsqueeze(-1), embeds, 0)
    return torch.nn.functional.pad(cleaned, (0, 0, 0, text_len - embeds.shape[1]))


class WanT2VAdapter(VelocityAdapter):
    """One physical transformer call for the entire batch (including both CFG branches)."""

    def __init__(self, transformer, vae, config: WanT2VConfig = WanT2VConfig()):
        if config.transformer_api == "diffusers":
            check_component_config(
                transformer, in_channels=config.latent_channels, out_channels=config.latent_channels
            )
            model_config = getattr(transformer, "config", None)
            if model_config is not None and getattr(model_config, "image_dim", None) is not None:
                raise ValueError("T2V requires a transformer without image conditioning")
        else:
            for name, wanted in (
                ("model_type", "t2v"),
                ("in_dim", config.latent_channels),
                ("out_dim", config.latent_channels),
            ):
                if hasattr(transformer, name) and getattr(transformer, name) != wanted:
                    raise ValueError(f"native Wan {name} disagrees with T2V config")
            if hasattr(transformer, "text_len") and transformer.text_len != config.text_len:
                raise ValueError("native transformer text_len disagrees with T2V config")
        model_config = getattr(transformer, "config", None)
        actual_patch = getattr(transformer, "patch_size", None)
        if actual_patch is None and model_config is not None:
            actual_patch = getattr(model_config, "patch_size", None)
        if actual_patch is not None and tuple(actual_patch) != config.patch_size:
            raise ValueError("transformer patch_size disagrees with T2V config")
        if hasattr(vae, "z_dim") and vae.z_dim != config.latent_channels:
            raise ValueError("native VAE latent channels disagree with T2V config")
        check_component_config(
            vae,
            z_dim=config.latent_channels,
            latents_mean=config.latents_mean,
            latents_std=config.latents_std,
        )
        super().__init__(transformer, vae)
        self.config = config

    @classmethod
    def from_pretrained(
        cls,
        path,
        *,
        config=WanT2VConfig(),
        revision=None,
        torch_dtype=None,
        local_files_only=True,
        native_root=None,
        asset_preflight=None,
    ):
        if config.transformer_api == "native" and config.vae_api == "native":
            from .wan_t2v_provenance import (
                import_native_components,
                verify_native_assets,
                verify_native_source,
            )

            if not local_files_only:
                raise ValueError("native factory permits offline local loading only")
            if torch_dtype not in (None, torch.float32):
                raise ValueError(
                    "native Wan parameters must remain FP32; use BF16 forward autocast"
                )
            source = verify_native_source(native_root, expected_revision=WAN_T2V_SOURCE_REVISION)
            revision = WAN_T2V_NATIVE_REVISION if revision is None else revision
            path = verify_native_assets(
                path,
                revision=revision,
                expected_revision=WAN_T2V_NATIVE_REVISION,
                asset_preflight=asset_preflight,
            )
            from .wan_t2v_dependencies import check_native_wan_dependencies

            dependencies = check_native_wan_dependencies()
            WanModel, WanVAE = import_native_components(source)
            # Official native weights are FP32. In particular the native time
            # embedding/projection explicitly run FP32 under BF16 forward AMP.
            transformer = WanModel.from_pretrained(
                str(path), local_files_only=True, torch_dtype=torch.float32
            )
            vae = WanVAE(
                vae_pth=str(path / "Wan2.1_VAE.pth"), dtype=torch.float32, device="cpu"
            ).model
            adapter = cls(transformer, vae, config)
            adapter.native_dependency_report = dependencies
            return adapter
        if config.transformer_api != "diffusers" or config.vae_api != "diffusers":
            raise ValueError("inject components explicitly when mixing native and Diffusers APIs")
        from diffusers import AutoencoderKLWan, WanTransformer3DModel

        revision = WAN_T2V_DIFFUSERS_REVISION if revision is None else revision
        torch_dtype = torch.bfloat16 if torch_dtype is None else torch_dtype
        kwargs = dict(revision=revision, local_files_only=local_files_only, torch_dtype=torch_dtype)
        transformer = WanTransformer3DModel.from_pretrained(path, subfolder="transformer", **kwargs)
        # Native VAE computations use float32; forward precision is independent.
        vae = AutoencoderKLWan.from_pretrained(
            path,
            subfolder="vae",
            **{
                **kwargs,
                "torch_dtype": torch.float32,
            },
        )
        return cls(transformer, vae, config)

    def predict_velocity(self, latents, sigma, conditioning: WanT2VConditioning):
        c, config = conditioning, self.config
        validate_latents(latents, 5, config.latent_channels)
        if any(a % b for a, b in zip(latents.shape[2:], config.patch_size)):
            raise ValueError("latent dimensions must align with transformer patches")
        batch_tensor(c.prompt_embeds, latents, "prompt_embeds", 3)
        prompt = masked_wan_context(
            c.prompt_embeds, c.prompt_attention_mask, text_len=config.text_len
        )
        sigma = sigma_batch(sigma, latents)
        batch_tensor(c.timesteps, latents, "timesteps", 1)
        if (
            c.timesteps.dtype != torch.int64
            or (c.timesteps < 0).any()
            or (c.timesteps > config.timestep_scale).any()
        ):
            raise ValueError("native timesteps must be INT64 within the training time range")
        t = c.timesteps
        hidden = latents
        if config.cfg_scale is not None:
            batch_tensor(c.negative_prompt_embeds, latents, "negative_prompt_embeds", 3)
            negative = masked_wan_context(
                c.negative_prompt_embeds, c.negative_prompt_attention_mask, text_len=config.text_len
            )
            if prompt.shape[2] != negative.shape[2]:
                raise ValueError("positive and negative text feature dimensions differ")
            hidden, t = torch.cat([hidden, hidden]), torch.cat([t, t])
            prompt = torch.cat([prompt, negative])
        if config.transformer_api == "native":
            seq_len = math.prod(a // b for a, b in zip(latents.shape[2:], config.patch_size))
            output = torch.stack(
                self.transformer(
                    list(hidden.unbind()), t=t, context=list(prompt.unbind()), seq_len=seq_len
                )
            )
        else:
            output = self.transformer(
                hidden_states=hidden, timestep=t, encoder_hidden_states=prompt, return_dict=False
            )[0]
        expected_batch = len(latents) * (2 if config.cfg_scale is not None else 1)
        if output.shape != (expected_batch, *latents.shape[1:]):
            raise ValueError("Wan T2V velocity must match the full latent/CFG batch")
        if config.cfg_scale is not None:
            conditional, unconditional = output.float().chunk(2)
            output = unconditional + config.cfg_scale * (conditional - unconditional)
        # Native flow output is dz/dsigma, with no sign inversion.
        return checked_velocity(output, latents)

    def to_vae_latents(self, latents):
        validate_latents(latents, 5, self.config.latent_channels)
        dtype = torch.float64 if latents.dtype == torch.float64 else torch.float32
        mean = torch.tensor(self.config.latents_mean, device=latents.device, dtype=dtype)
        std = torch.tensor(self.config.latents_std, device=latents.device, dtype=dtype)
        return latents.to(dtype) * std.view(1, -1, 1, 1, 1) + mean.view(1, -1, 1, 1, 1)

    def _decode_raw(self, latents):
        if self.config.vae_api == "native":
            # Inject WanVAE.model, not the wrapper. Affine normalization already ran.
            decoded = self.vae.decode(latents, scale=(0.0, 1.0))
            if not torch.isfinite(decoded).all():
                raise ValueError("Wan VAE decoder returned nonfinite pixels")
            return decoded.clamp(-1.0, 1.0)
        return decode_wan_differentiable(self.vae, latents)
