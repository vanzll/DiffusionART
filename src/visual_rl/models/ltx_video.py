"""Optional native LTX-Video 2B 0.9.6 distilled component adapter.

No LTX/Diffusers/Transformers imports occur until explicit asset loading.
This adapter supplies native velocity and decode; rollout/replay selection is
separate. LTX-2, 2.5, 13B and later checkpoints are outside this ABI.
"""

import importlib
import json
import struct
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import MethodType

import torch
from torch import Tensor, nn

from .base import (
    VelocityAdapter,
    batch_tensor,
    channel_count,
    check_component_config,
    checked_velocity,
    module_dtype,
    positive,
    sigma_batch,
    validate_latents,
)
from .ltx_video_sampling import (
    LTX_VIDEO_ALLOWED_INFERENCE_STEPS,
    LTX_VIDEO_CHECKPOINT,
    LTX_VIDEO_CODE_REVISION,
    pack_ltx_latents,
    unpack_ltx_latents,
)
from .source_pin import import_pinned_source

LTX_VIDEO_LORA_TARGETS = ("to_q", "to_k", "to_v", "to_out.0")
LTX_VIDEO_TEXT_ENCODER_ID = "PixArt-alpha/PixArt-XL-2-1024-MS"
LTX_VIDEO_TEXT_ENCODER_REVISION = "b89adadeccd9ead2adcb9fa2825d3fabec48d404"


def _ltx_bf16_lora_linear_forward(module, value):
    # Keep FP32 trainable masters and PEFT's module/state-dict identity. This
    # permanent method also executes during activation-checkpoint recomputation.
    with torch.autocast(device_type=value.device.type, dtype=torch.bfloat16):
        return torch.nn.functional.linear(value, module.weight, module.bias)


def install_ltx_video_lora_precision(transformer):
    """Confine BF16 autocast to existing PEFT A/B GEMMs, never native norms."""
    if getattr(transformer, "_ltx_video_lora_precision_installed", False):
        return
    branches = []
    for module in transformer.modules():
        if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            continue
        if (
            not isinstance(module.lora_A, nn.ModuleDict)
            or not isinstance(module.lora_B, nn.ModuleDict)
            or set(module.lora_A) != set(module.lora_B)
        ):
            raise ValueError("native LTX requires matching linear PEFT A/B branches")
        for name in module.lora_A:
            for linear in (module.lora_A[name], module.lora_B[name]):
                if not isinstance(linear, nn.Linear):
                    raise ValueError("native LTX LoRA BF16 GEMMs require existing Linear branches")
                if linear.weight.dtype != torch.float32:
                    raise ValueError("native LTX LoRA trainable masters must remain FP32")
                if "forward" in vars(linear) and not getattr(
                    linear, "_ltx_video_bf16_lora_gemm", False
                ):
                    raise ValueError("refuse to replace an unknown LoRA Linear forward override")
                branches.append(linear)
    # Validate the entire set before mutating; failed validation installs none.
    for linear in branches:
        linear.forward = MethodType(_ltx_bf16_lora_linear_forward, linear)
        linear._ltx_video_bf16_lora_gemm = True
    if branches:
        transformer._ltx_video_lora_precision_installed = True


@dataclass(frozen=True)
class LTXVideoConfig:
    latent_channels: int = 128
    caption_channels: int = 4096
    spatial_compression: int = 32
    temporal_compression: int = 8
    scaling_factor: float = 1.0
    per_channel_normalize: bool = True
    causal_temporal_positioning: bool = True
    decode_timestep: float = 0.05
    decode_noise_scale: float = 0.025

    def __post_init__(self):
        channel_count(self.latent_channels)
        channel_count(self.caption_channels)
        channel_count(self.spatial_compression)
        channel_count(self.temporal_compression)
        positive(self.scaling_factor, "scaling_factor")
        if not 0 <= self.decode_timestep <= 1 or not 0 <= self.decode_noise_scale <= 1:
            raise ValueError("decode timestep and noise scale must be in [0,1]")
        if not all(
            isinstance(x, bool)
            for x in (self.per_channel_normalize, self.causal_temporal_positioning)
        ):
            raise ValueError("normalization and causal positioning must be boolean")


@dataclass(frozen=True)
class LTXVideoConditioning:
    prompt_embeds: Tensor
    attention_mask: Tensor
    frame_rate: float = 30.0

    def select(self, rows):
        return type(self)(self.prompt_embeds[rows], self.attention_mask[rows], self.frame_rate)


def ltx_video_latent_shape(
    batch: int, height: int = 704, width: int = 1216, frames: int = 153, *, config=LTXVideoConfig()
):
    if any(type(x) is not int or x < 1 for x in (batch, height, width, frames)):
        raise ValueError("batch, resolution and frames must be positive integers")
    if (
        height % config.spatial_compression
        or width % config.spatial_compression
        or (frames - 1) % config.temporal_compression
    ):
        raise ValueError("native LTX grid requires H/W divisible by32 and frames=8n+1")
    return (
        batch,
        config.latent_channels,
        (frames - 1) // config.temporal_compression + 1,
        height // config.spatial_compression,
        width // config.spatial_compression,
    )


def ltx_video_coordinates(shape, *, frame_rate=30.0, device=None, config=LTXVideoConfig()):
    """Native patch-size1 top-left pixel coordinates; temporal axis is seconds."""
    positive(frame_rate, "frame_rate")
    if len(shape) != 5 or min(shape) < 1:
        raise ValueError("coordinates need a nonempty [B,C,F,H,W] shape")
    batch, _, frames, height, width = shape
    grid = torch.stack(
        torch.meshgrid(
            torch.arange(frames, device=device),
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        ),
        dim=0,
    ).reshape(3, -1)
    grid = (
        grid
        * torch.tensor(
            (config.temporal_compression, config.spatial_compression, config.spatial_compression),
            device=device,
        )[:, None]
    )
    if config.causal_temporal_positioning:
        grid[0] = (grid[0] + 1 - config.temporal_compression).clamp_min(0)
    coords = grid.to(torch.float32).unsqueeze(0).repeat(batch, 1, 1)
    coords[:, 0] = coords[:, 0] * (1.0 / frame_rate)
    return coords


class LTXVideoAdapter(VelocityAdapter):
    def __init__(self, transformer, vae, config=LTXVideoConfig(), *, native_forward_dtype=None):
        if native_forward_dtype not in (None, torch.bfloat16):
            raise ValueError("native checkpoint forward precision must be explicit BF16")
        check_component_config(
            transformer,
            in_channels=config.latent_channels,
            out_channels=config.latent_channels,
            caption_channels=config.caption_channels,
            timestep_scale_multiplier=1000,
            causal_temporal_positioning=config.causal_temporal_positioning,
        )
        check_component_config(
            vae, scaling_factor=config.scaling_factor, latent_channels=config.latent_channels
        )
        super().__init__(transformer, vae)
        self.config = config
        # Explicit factory/FSDP contract; FP32 master parameters do not reveal
        # the compute dtype supplied by a sharding wrapper during its forward.
        self.native_forward_dtype = native_forward_dtype
        if config.per_channel_normalize:
            for name in ("mean_of_means", "std_of_means"):
                stats = getattr(vae, name, None)
                if (
                    not isinstance(stats, Tensor)
                    or stats.numel() != config.latent_channels
                    or not torch.isfinite(stats).all()
                    or (name == "std_of_means" and (stats <= 0).any())
                ):
                    raise ValueError(f"native per-channel normalization requires valid VAE {name}")

    def predict_velocity(self, latents, sigma, conditioning: LTXVideoConditioning):
        validate_latents(latents, 5, self.config.latent_channels)
        c = conditioning
        batch_tensor(c.prompt_embeds, latents, "prompt_embeds", 3)
        batch_tensor(c.attention_mask, latents, "attention_mask", 2)
        if c.prompt_embeds.shape[1:] != (c.attention_mask.shape[1], self.config.caption_channels):
            raise ValueError("text embedding shape and attention mask do not match")
        mask = c.attention_mask
        if (
            not ((mask == 0) | (mask == 1)).all()
            or not mask.bool().any(1).all()
            or not torch.isfinite(c.prompt_embeds).all()
        ):
            raise ValueError("text needs finite embeddings and a nonempty binary mask per row")
        t = sigma_batch(sigma, latents).unsqueeze(-1)
        coords = ltx_video_coordinates(
            latents.shape, frame_rate=c.frame_rate, device=latents.device, config=self.config
        )
        dtype = self.native_forward_dtype or module_dtype(self.transformer, latents.dtype)
        install_ltx_video_lora_precision(self.transformer)
        # Official mixed_precision=False uses BF16 weights/inputs without a
        # global autocast context. CUDA autocast can upcast normalization and
        # change subsequent modulation rounding even when every input matches.
        # Preserve that native policy for a BF16 backbone, including PEFT with
        # FP32 adapter parameters. PEFT handles its own branch input casting.
        precision = (
            torch.autocast(device_type=latents.device.type, enabled=False)
            if dtype == torch.bfloat16
            else nullcontext()
        )
        # One real physical batch call. Do not loop over prompts here.
        with precision:
            tokens = self.transformer(
                pack_ltx_latents(latents).to(dtype),
                indices_grid=coords,
                encoder_hidden_states=c.prompt_embeds.to(dtype),
                encoder_attention_mask=mask,
                timestep=t,
                skip_layer_mask=None,
                return_dict=False,
            )[0]
        output = unpack_ltx_latents(tokens, latents.shape)
        return checked_velocity(output, latents)

    def to_vae_latents(self, latents):
        validate_latents(latents, 5, self.config.latent_channels)
        # Upstream casts to VAE dtype before applying external channel statistics.
        z = latents.to(dtype=module_dtype(self.vae, latents.dtype))
        if self.config.per_channel_normalize:
            mean = self.vae.mean_of_means.to(z).reshape(1, -1, 1, 1, 1)
            std = self.vae.std_of_means.to(z).reshape(1, -1, 1, 1, 1)
            return z * std + mean
        return z / self.config.scaling_factor

    def decode(self, latents, *, innovation: Tensor | None = None, generator=None):
        """Native conditioned decode; callers cache/supply the actual noise draw.

        Sampling without an explicit innovation uses the caller's generator.
        Replay/evaluation should supply cached noise to pair base/post results.
        Output is native pixels in [-1,1], before any presentation clipping.
        """
        validate_latents(latents, 5, self.config.latent_channels)
        if getattr(self.vae, "use_hw_tiling", False) or getattr(self.vae, "use_z_tiling", False):
            raise ValueError("pinned native LTX VAE tiled paths drop decode timestep conditioning")
        if innovation is not None and generator is not None:
            raise ValueError("supply either decode innovation or generator")
        if innovation is None:
            innovation = (
                torch.randn_like(latents)
                if generator is None
                else torch.empty_like(latents).normal_(generator=generator)
            )
        if (
            innovation.shape != latents.shape
            or innovation.device != latents.device
            or innovation.dtype != latents.dtype
            or not torch.isfinite(innovation).all()
            or not torch.isfinite(latents).all()
        ):
            raise ValueError("decode noise must match finite latent shape/device/dtype")
        # Official uses float32 broadcast scales; retain this promotion/order.
        scale = torch.full(
            (len(latents), 1, 1, 1, 1), self.config.decode_noise_scale, device=latents.device
        )
        z = latents * (1 - scale) + innovation * scale
        raw = self.to_vae_latents(z)
        timestep = torch.full((len(latents),), self.config.decode_timestep, device=raw.device)
        _, _, f, h, w = raw.shape
        target = (
            len(raw),
            3,
            f * self.config.temporal_compression,
            h * self.config.spatial_compression,
            w * self.config.spatial_compression,
        )

        # Fail before downstream image processors can hide nonfinite pixels.
        def check_output(module, inputs, output):
            if isinstance(output, Tensor) and not torch.isfinite(output).all():
                raise ValueError("LTX VAE decoder returned nonfinite pixels")

        decoder = getattr(self.vae, "decoder", None)
        hook = (
            decoder.register_forward_hook(check_output) if isinstance(decoder, nn.Module) else None
        )
        try:
            with torch.autocast(device_type=raw.device.type, enabled=False):
                output = self.vae.decode(
                    raw, return_dict=False, timestep=timestep, target_shape=target
                )[0]
        finally:
            if hook is not None:
                hook.remove()
        if (
            not isinstance(output, Tensor)
            or output.ndim != 5
            or output.shape[:2] != (len(latents), 3)
            or not torch.isfinite(output).all()
        ):
            raise ValueError("LTX VAE must return finite [B,3,T,H,W] pixels")
        if torch.is_grad_enabled() and latents.requires_grad and not output.requires_grad:
            raise RuntimeError("LTX VAE detached the latent graph")
        return output


@torch.no_grad()
def encode_ltx_video_prompts(
    prompts, tokenizer, text_encoder, *, device=None, max_length=256, frame_rate=30.0
):
    if (
        not isinstance(prompts, (list, tuple))
        or not prompts
        or any(not isinstance(p, str) or not p.strip() for p in prompts)
    ):
        raise ValueError("prompts must be a nonempty sequence of nonempty strings")
    if type(max_length) is not int or max_length != 256:
        raise ValueError("the native 0.9.6 text profile uses max_length=256")
    parameter = next(text_encoder.parameters(), None)
    encoder_device = parameter.device if parameter is not None else torch.device("cpu")
    device = encoder_device if device is None else torch.device(device)
    # Official preprocessing strips whitespace only; no prompt enhancer/LLM.
    text = tokenizer(
        [p.strip() for p in prompts],
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    mask = text.attention_mask.to(encoder_device)
    embeddings = text_encoder(text.input_ids.to(encoder_device), attention_mask=mask)[0]
    return LTXVideoConditioning(embeddings.to(device), mask.to(device), frame_rate)


def read_ltx_video_checkpoint_config(checkpoint):
    """Read only the bounded safetensors JSON header, never model tensors."""
    checkpoint = Path(checkpoint)
    if checkpoint.name != LTX_VIDEO_CHECKPOINT:
        raise ValueError(f"expected the exact {LTX_VIDEO_CHECKPOINT} checkpoint")
    with checkpoint.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("truncated safetensors header")
        length = struct.unpack("<Q", prefix)[0]
        if not 0 < length <= 16 * 1024 * 1024:
            raise ValueError("invalid or excessive safetensors metadata header")
        header = stream.read(length)
        if len(header) != length:
            raise ValueError("truncated safetensors metadata")
    configs = json.loads(json.loads(header)["__metadata__"]["config"])
    if (
        configs.get("allowed_inference_steps") != list(LTX_VIDEO_ALLOWED_INFERENCE_STEPS)
        or configs["transformer"].get("in_channels") != 128
        or configs["transformer"].get("out_channels") != 128
        or configs["vae"].get("latent_channels") != 128
        or configs["scheduler"].get("sampler") != "LinearQuadratic"
        or configs["scheduler"].get("shifting") is not None
    ):
        raise ValueError("checkpoint metadata does not match native 2B 0.9.6 distilled")
    return configs


@dataclass(frozen=True)
class LTXVideoComponents:
    adapter: LTXVideoAdapter
    text_encoder: nn.Module
    tokenizer: object


def load_ltx_video_components(
    checkpoint, text_encoder_path, *, native_root, text_encoder_revision=None, device="cpu"
):
    """Explicit optional load, without prompt enhancers or scheduler substitution.

    Supply a clean native source checkout at the exact recorded code revision.
    The caller places/fine-tunes the transformer and owns optimizer, sharding
    and checkpoint identity. Native forward explicitly requires BF16 compute
    parameters/inputs without outer global autocast. Remote T5 assets require their exact
    pinned repository/revision; local assets require the caller's asset receipt.
    """
    read_ltx_video_checkpoint_config(checkpoint)
    local_text = Path(text_encoder_path).is_dir()
    if not local_text and (
        str(text_encoder_path) != LTX_VIDEO_TEXT_ENCODER_ID
        or text_encoder_revision != LTX_VIDEO_TEXT_ENCODER_REVISION
    ):
        raise ValueError(
            "remote T5 encoder/tokenizer require the exact pinned PixArt repository/revision"
        )
    package = import_pinned_source(
        native_root, package="ltx_video", revision=LTX_VIDEO_CODE_REVISION
    )
    root = Path(package.__file__).resolve().parent.parent

    def native_component(module_name, class_name):
        module = importlib.import_module(module_name)
        origin = getattr(module, "__file__", None)
        expected = root.joinpath(*module_name.split(".")).with_suffix(".py")
        if origin is None or Path(origin).resolve() != expected:
            raise ValueError("native LTX component origin differs from the pinned source")
        return getattr(module, class_name)

    CausalVideoAutoencoder = native_component(
        "ltx_video.models.autoencoders.causal_video_autoencoder", "CausalVideoAutoencoder"
    )
    Transformer3DModel = native_component(
        "ltx_video.models.transformers.transformer3d", "Transformer3DModel"
    )
    from transformers import T5EncoderModel, T5Tokenizer

    transformer = Transformer3DModel.from_pretrained(str(checkpoint)).to(device, torch.bfloat16)
    vae = CausalVideoAutoencoder.from_pretrained(str(checkpoint)).to(device, torch.bfloat16)
    options = {"revision": text_encoder_revision} if not local_text else {}
    encoder = T5EncoderModel.from_pretrained(
        text_encoder_path, subfolder="text_encoder", **options
    ).to(device, torch.bfloat16)
    encoder.requires_grad_(False).eval()
    tokenizer = T5Tokenizer.from_pretrained(text_encoder_path, subfolder="tokenizer", **options)
    return LTXVideoComponents(
        LTXVideoAdapter(transformer, vae, native_forward_dtype=torch.bfloat16), encoder, tokenizer
    )
