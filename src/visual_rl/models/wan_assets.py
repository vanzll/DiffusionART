"""Bounded native Wan2.1 assets, online/fixed references and CPU conditioning.

Pretrained loading requires the historical diffusers 0.33.1 runtime. The
conditioning-only ABI also supports 0.40.0 for small native CPU oracle tests;
this is not permission to load converted 0.33.1 weights on another version.
No runners, downloads, implicit image caches, model offload hooks or Wan2.2.
The caller owns policy placement, batching, autocast and training preparation.
"""

import hashlib
import random
from contextlib import contextmanager
from io import BytesIO
from types import MethodType

import numpy as np
import torch

from .wan import WanI2VAdapter, WanI2VConditioning, WanI2VConfig

NATIVE_DIFFUSERS_VERSION = "0.33.1"
CONDITIONING_DIFFUSERS_VERSIONS = ("0.33.1", "0.40.0")


def reference_mode(options):
    mode = options.get("reference_mode", "online_flux")
    if mode not in ("online_flux", "fixed_images"):
        raise ValueError("Wan reference_mode must be online_flux or fixed_images")
    return mode


def load_fixed_reference(path, expected_sha256):
    """Decode the checked file itself, without generation or a JPEG re-encode."""
    from PIL import Image

    if path is None or (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
                       or any(c not in "0123456789abcdef" for c in expected_sha256)):
        raise ValueError("Wan fixed training requires an image and image_sha256")
    with open(path, "rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        if digest.hexdigest() != expected_sha256:
            raise ValueError(f"reference image hash differs: {path}")
        stream.seek(0)
        with Image.open(stream) as source:
            return source.convert("RGB")


def _seed(seed):
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    return seed


def scoped_seed(seed, rank, attempt, stream):
    """Canonical SHA256('{seed}:{rank}:{attempt}:{stream}') LE64, mod 2**63-1."""
    _seed(seed)
    if any(type(v) is not int or v < 0 for v in (rank, attempt)):
        raise ValueError("rank and attempt must be nonnegative integers")
    if not isinstance(stream, str) or not stream:
        raise ValueError("stream must be a nonempty string")
    digest = hashlib.sha256(f"{seed}:{rank}:{attempt}:{stream}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def _device(device):
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("native Wan staging supports CPU or CUDA only")
    return device


@contextmanager
def isolated_rng(seed, device):
    """Restore Python, NumPy, CPU and selected CUDA RNG, including on failure.

    This scopes process-global state and must not run concurrently in threads.
    Other CUDA devices are neither seeded nor initialized by this helper.
    """
    _seed(seed)
    device = _device(device)
    devices = []
    if device.type == "cuda":
        devices = [device.index if device.index is not None else torch.cuda.current_device()]
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed % 2**32)
            torch.random.default_generator.manual_seed(seed)
            if devices:
                with torch.cuda.device(devices[0]):
                    torch.cuda.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def require_native_version(version=NATIVE_DIFFUSERS_VERSION):
    import diffusers

    if version != NATIVE_DIFFUSERS_VERSION or diffusers.__version__ != version:
        raise RuntimeError(
            f"Wan assets require diffusers=={NATIVE_DIFFUSERS_VERSION}; "
            f"requested {version}, installed {diffusers.__version__}. "
            "Converted native weights must use the pinned runtime, not a silent ABI upgrade."
        )


def _conditioning_version():
    import diffusers

    if diffusers.__version__ not in CONDITIONING_DIFFUSERS_VERSIONS:
        raise RuntimeError(f"unsupported Wan conditioning ABI: diffusers {diffusers.__version__}")


def _modules(component):
    if isinstance(component, torch.nn.Module):
        modules = [component]
    else:
        modules = list(dict.fromkeys(
            m for m in component.components.values() if isinstance(m, torch.nn.Module)
        ))
    if not modules:
        raise ValueError("staging requires torch modules or a native pipeline")
    if getattr(component, "hf_device_map", None) or any(
        getattr(m, "hf_device_map", None) or hasattr(child, "_hf_hook")
        or hasattr(child, "_diffusers_hook")
        for m in modules for child in m.modules()
    ):
        raise ValueError("remove device maps/automatic offload hooks before manual staging")
    return modules


def _freeze(module):
    module.requires_grad_(False).eval()
    for parameter in module.parameters():
        parameter.grad = None


def _offload(modules):
    error = None
    for module in modules:
        try:
            module.to("cpu")
        except Exception as exc:
            error = error or exc
    if error is not None:
        raise RuntimeError("frozen CPU offload failed; residency is not guaranteed") from error


@contextmanager
def staged(component, device):
    """Freeze/eval and stage weights; attempt ALL CPU cleanups even if .to fails.

    Freezing is permanent. Never pass the trainable Wan policy to this helper.
    A FLUX pipeline is one frozen stage; its own modules are resident together.
    """
    device = _device(device)
    modules = _modules(component)
    try:
        for module in modules:
            _freeze(module)
        for module in modules:
            module.to(device)
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            yield component
    finally:
        _offload(modules)


def _wan_config(pipe):
    config = getattr(pipe, "config", None)
    if (getattr(config, "expand_timesteps", False)
            or getattr(config, "boundary_ratio", None) is not None
            or getattr(pipe, "transformer_2", None) is not None):
        raise ValueError("only single-transformer Wan2.1 first-frame conditioning is supported")
    vae = pipe.vae
    if getattr(vae.config, "patch_size", None) is not None or getattr(vae, "use_tiling", False):
        raise ValueError("only unpatched, untiled Wan2.1 VAE conditioning is supported")
    for name, dtype in (("text_encoder", torch.bfloat16), ("image_encoder", torch.float32),
                        ("vae", torch.float32)):
        module = getattr(pipe, name, None)
        if not isinstance(module, torch.nn.Module):
            raise ValueError(f"Wan requires a native {name}")
        _modules(module)
        if any(p.is_floating_point() and p.dtype != dtype for p in module.parameters()):
            raise ValueError(f"Wan {name} weights must be {dtype}")
    if getattr(pipe, "hf_device_map", None):
        raise ValueError("manual Wan staging does not support device maps")
    temporal, spatial = pipe.vae_scale_factor_temporal, pipe.vae_scale_factor_spatial
    if any(type(v) is not int or v < 1 for v in (temporal, spatial)):
        raise ValueError("invalid Wan VAE compression")
    return WanI2VConfig(tuple(vae.config.latents_mean), tuple(vae.config.latents_std),
                        temporal_compression=temporal, cfg_scale=None)


def latent_shape(pipe, *, height, width, frames):
    """Return unbatched CTHW; reject geometry the native pipeline would round."""
    config = _wan_config(pipe)
    spatial, temporal = pipe.vae_scale_factor_spatial, config.temporal_compression
    if (any(type(v) is not int or v < 1 for v in (height, width, frames))
            or height % spatial or width % spatial or (frames - 1) % temporal):
        raise ValueError("video dimensions must align with Wan VAE compression")
    if pipe.vae.config.z_dim != config.latent_channels:
        raise ValueError("Wan VAE normalization channel count differs")
    shape = (config.latent_channels, (frames - 1) // temporal + 1,
             height // spatial, width // spatial)
    transformer = getattr(pipe, "transformer", None)
    # Components-only pipelines have no policy. Wan2.1 uses (1, 2, 2) patches.
    patch = (1, 2, 2) if transformer is None else tuple(transformer.config.patch_size)
    if (len(patch) != 3 or any(type(p) is not int or p < 1 for p in patch)
            or any(n % p for n, p in zip(shape[1:], patch))):
        raise ValueError("latent dimensions must align with Wan transformer patches")
    if transformer is not None:
        tc = transformer.config
        if (tc.in_channels != 2 * config.latent_channels + temporal
                or tc.out_channels != config.latent_channels or tc.image_dim is None):
            raise ValueError("requires a CLIP-conditioned Wan2.1 I2V transformer")
    return shape


def _prompt_geometry(prompt, height, width):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("one nonempty prompt string is required")
    if any(type(v) is not int or v < 1 for v in (height, width)):
        raise ValueError("height and width must be positive integers")


def _cpu_tensor(value, name, ndim, dtype):
    if (not isinstance(value, torch.Tensor) or value.ndim != ndim or value.shape[0] != 1
            or not value.is_floating_point() or any(n == 0 for n in value.shape)):
        raise ValueError(f"native {name} must be a nonempty floating single-sample tensor")
    value = value.detach().to(device="cpu", dtype=dtype)
    if not torch.isfinite(value).all():
        raise ValueError(f"native {name} contains nonfinite values")
    return value


@torch.no_grad()
def condition_image(pipe, prompt, image, *, height, width, frames, device, seed, cfg=False):
    """Return existing WanI2VConditioning on CPU, with no policy/noise RNG use.

    Text/CLIP embeddings are BF16; CLIP weights and VAE conditioning are FP32.
    Native prepare_latents encodes [preprocessed image, zero PIXEL frames],
    takes the VAE mode and normalizes ALL frames. Never pad first-frame latents.
    cfg=False returns negative_prompt_embeds=None, not an empty CFG branch.
    """
    _conditioning_version()
    _prompt_geometry(prompt, height, width)
    if type(cfg) is not bool:
        raise ValueError("cfg must be boolean")
    from PIL import Image

    if not isinstance(image, Image.Image) or image.mode != "RGB":
        raise ValueError("image must be one decoded RGB PIL image")
    shape = latent_shape(pipe, height=height, width=width, frames=frames)
    device = _device(device)
    modules = [pipe.text_encoder, pipe.image_encoder, pipe.vae]
    with isolated_rng(seed, device):
        for module in modules:
            _freeze(module)
        _offload(modules)
        with staged(pipe.text_encoder, device):
            positive, negative = pipe.encode_prompt(
                prompt=prompt, negative_prompt="", do_classifier_free_guidance=cfg,
                num_videos_per_prompt=1, max_sequence_length=512, device=device,
                dtype=torch.bfloat16,
            )
            positive = _cpu_tensor(positive, "prompt_embeds", 3, torch.bfloat16)
            if cfg:
                negative = _cpu_tensor(negative, "negative_prompt_embeds", 3, torch.bfloat16)
                if negative.shape != positive.shape:
                    raise ValueError("native positive/negative prompt shapes differ")
            elif negative is not None:
                raise ValueError("native conditional-only encoding returned a CFG branch")
        with staged(pipe.image_encoder, device):
            embeds = _cpu_tensor(pipe.encode_image(image, device=device),
                                 "image_embeds", 3, torch.bfloat16)
        pixels = pipe.video_processor.preprocess(image, height=height, width=width).float()
        if tuple(pixels.shape) != (1, 3, height, width) or not torch.isfinite(pixels).all():
            raise ValueError("native image preprocessing returned invalid pixels/geometry")
        with staged(pipe.vae, device):
            result = pipe.prepare_latents(
                image=pixels.to(device), batch_size=1, num_channels_latents=shape[0],
                height=height, width=width, num_frames=frames, dtype=torch.float32,
                device=device, latents=torch.zeros((1, *shape), device=device, dtype=torch.float32),
            )
            if not isinstance(result, tuple) or len(result) != 2:
                raise ValueError("unsupported native Wan prepare_latents return ABI")
            condition = _cpu_tensor(result[1], "image_condition", 5, torch.float32)
        temporal = pipe.vae_scale_factor_temporal
        if tuple(condition.shape) != (1, shape[0] + temporal, *shape[1:]):
            raise ValueError("native Wan image conditioning geometry mismatch")
        mask = condition[:, :temporal]
        if not torch.all(mask[:, :, 0] == 1) or not torch.all(mask[:, :, 1:] == 0):
            raise ValueError("native Wan must return the first-frame-only mask")
    return WanI2VConditioning(positive, embeds, condition, negative)


@torch.no_grad()
def generate_reference(flux, prompt, *, height, width, device, seed):
    """Online native FLUX (30, 3.5, 512), then default JPEG save/reopen in RAM.

    Always calls FLUX. Errors propagate; there is no cached-image fallback.
    The returned RGB PIL image owns its pixels independently of the byte buffer.
    """
    _conditioning_version()
    _prompt_geometry(prompt, height, width)
    if height % 16 or width % 16:
        raise ValueError("FLUX reference dimensions must be multiples of 16")
    from PIL import Image

    with isolated_rng(seed, device), staged(flux, device):
        image = flux(
            prompt, height=height, width=width, num_inference_steps=30,
            guidance_scale=3.5, max_sequence_length=512,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        ).images[0]
    if not isinstance(image, Image.Image) or image.size != (width, height):
        raise ValueError("native FLUX returned a non-image or changed reference geometry")
    with BytesIO() as buffer:
        image.save(buffer, format="JPEG")
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB")


def load_components(c, *, vae=None):
    """Load a CPU components-only pipeline from a local native Diffusers snapshot.

    Pass adapter.vae to share its FP32 VAE; no transformer is loaded or moved.
    Uses model.options base_checkpoint/base_revision, never policy weights.
    """
    require_native_version()
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    from transformers import CLIPVisionModel

    checkpoint = c.model.options["base_checkpoint"]
    common = dict(revision=c.model.options.get("base_revision"), local_files_only=True)
    if vae is None:
        vae = AutoencoderKLWan.from_pretrained(
            checkpoint, subfolder="vae", torch_dtype=torch.float32, **common)
    image_encoder = CLIPVisionModel.from_pretrained(
        checkpoint, subfolder="image_encoder", torch_dtype=torch.float32, **common)
    pipe = WanImageToVideoPipeline.from_pretrained(
        checkpoint, transformer=None, vae=vae, image_encoder=image_encoder,
        torch_dtype=torch.bfloat16, **common)
    _wan_config(pipe)
    for module in (pipe.text_encoder, pipe.image_encoder, pipe.vae):
        _freeze(module)
        module.to("cpu")
    return pipe


def load_adapter(c, device, *, vae=None):
    """Load FP32 master policy/VAE and reuse WanI2VAdapter; leave VAE on CPU.

    c.model.checkpoint is the converted transformer-only directory. Frozen VAE
    comes from base_checkpoint. FSDP policies remain on CPU for caller wrapping.
    """
    require_native_version()
    device = _device(device)
    from diffusers import AutoencoderKLWan, WanTransformer3DModel

    from .finetune import FineTuneConfig, prepare_policy

    transformer = WanTransformer3DModel.from_pretrained(
        c.model.checkpoint, revision=c.model.revision,
        local_files_only=True, torch_dtype=torch.float32)
    if c.runtime.distributed == "fsdp":
        stabilize_wan_time_embedder_dtype_lookup(transformer)
    elif c.model.options.get("gradient_checkpointing", False):
        transformer.enable_gradient_checkpointing()
    transformer = prepare_policy(transformer, FineTuneConfig(**c.model.options["finetune"]))
    if vae is None:
        vae = AutoencoderKLWan.from_pretrained(
            c.model.options["base_checkpoint"], subfolder="vae",
            revision=c.model.options.get("base_revision"),
            local_files_only=True, torch_dtype=torch.float32)
    if getattr(vae.config, "patch_size", None) is not None or getattr(vae, "use_tiling", False):
        raise ValueError("only unpatched, untiled Wan2.1 VAE is supported")
    if any(p.dtype != torch.float32 for m in (transformer, vae) for p in m.parameters()):
        raise ValueError("Wan master policy and VAE must be FP32")
    config = WanI2VConfig(tuple(vae.config.latents_mean), tuple(vae.config.latents_std),
                         temporal_compression=2 ** sum(vae.config.temperal_downsample),
                         cfg_scale=None if c.sampling.guidance_scale == 1
                         else c.sampling.guidance_scale)
    adapter = WanI2VAdapter(transformer, vae, config).eval()
    adapter.vae.to("cpu")
    if c.runtime.distributed != "fsdp":
        adapter.transformer.to(device)
    return adapter


def _stable_wan_condition_forward(self, timestep, encoder_hidden_states,
                                 encoder_hidden_states_image=None):
    # 0.33.1 algebra, with a direct weight lookup valid during FSDP unsharding.
    timestep = self.timesteps_proj(timestep)
    dtype = self.time_proj.weight.dtype
    if timestep.dtype != dtype and dtype != torch.int8:
        timestep = timestep.to(dtype)
    temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
    timestep_proj = self.time_proj(self.act_fn(temb))
    encoder_hidden_states = self.text_embedder(encoder_hidden_states)
    if encoder_hidden_states_image is not None:
        encoder_hidden_states_image = self.image_embedder(encoder_hidden_states_image)
    return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image


def stabilize_wan_time_embedder_dtype_lookup(model):
    """Bind the 0.33.1 FSDP fix to THIS embedder only; never mutate its class."""
    require_native_version()
    condition = model.condition_embedder
    cls = type(condition)
    if (cls.__module__ != "diffusers.models.transformers.transformer_wan"
            or cls.__name__ != "WanTimeTextImageEmbedding"):
        raise TypeError("unexpected pinned Wan condition embedding class")
    if getattr(condition, "_wan_fsdp_dtype_lookup_stabilized", False):
        return False
    condition.forward = MethodType(_stable_wan_condition_forward, condition)
    condition._wan_fsdp_dtype_lookup_stabilized = True
    return True


def load_models(c, device):
    """Return (adapter, Wan components pipe, online FLUX pipe or None).

    model.checkpoint/revision: converted distilled transformer.
    model.options base_checkpoint/base_revision: Wan VAE/text/CLIP snapshot.
    model.options reference_checkpoint/reference_revision: online FLUX snapshot,
    not loaded or required in explicit reference_mode='fixed_images'.
    All loads are local-only. VAE is shared, frozen modules remain CPU, and
    only non-FSDP policy is moved to device. No VideoReward or global TF32 setup.
    """
    require_native_version()
    fixed = reference_mode(c.model.options) == "fixed_images"
    if not fixed:
        from diffusers import FluxPipeline

    for key in (("base_checkpoint",) if fixed else ("base_checkpoint", "reference_checkpoint")):
        if not isinstance(c.model.options.get(key), str) or not c.model.options[key]:
            raise ValueError(f"Wan native loading requires model.options.{key}")
    adapter = load_adapter(c, device)
    pipe = load_components(c, vae=adapter.vae)
    if fixed:
        pipe.set_progress_bar_config(disable=True)
        return adapter, pipe, None
    flux = FluxPipeline.from_pretrained(
        c.model.options["reference_checkpoint"],
        revision=c.model.options.get("reference_revision"),
        torch_dtype=torch.bfloat16, local_files_only=True)
    modules = _modules(flux)
    for module in modules:
        _freeze(module)
    _offload(modules)
    pipe.set_progress_bar_config(disable=True)
    flux.set_progress_bar_config(disable=True)
    return adapter, pipe, flux
