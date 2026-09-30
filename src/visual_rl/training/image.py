"""Image-policy replay and native scalar scoring, shared by image algorithms."""

from dataclasses import dataclass, fields, replace

import torch

from ..models import FluxConditioning, SD3Conditioning
from ..rewards.image_dispatch import score_image_batch, validate_image_metadata
from ..rewards.scalar import ScoreResult, weighted_score


class ConditionedImagePolicy:
    """Bind immutable prompt rows for rollout and repeated timestep replay.

    The adapter owns CFG/VAE coordinates; this object only indexes conditioning
    and selects forward autocast. It does not sample noise or choose an update
    objective. Gradients flow to policy parameters, never text-encoder tensors.
    """

    def __init__(self, adapter, conditioning, *, autocast_dtype=None, replay_input_dtype=None):
        if not isinstance(conditioning, (SD3Conditioning, FluxConditioning)):
            raise ValueError("image replay requires SD3 or FLUX conditioning")
        if autocast_dtype not in (None, torch.bfloat16, torch.float16):
            raise ValueError("image replay supports fp32, bf16 or scaled fp16")
        if replay_input_dtype not in (None, torch.float32, torch.bfloat16):
            raise ValueError("replay input dtype must be fp32 or bf16")
        if replay_input_dtype is not None and not isinstance(conditioning, FluxConditioning):
            raise ValueError("explicit replay input precision is currently FLUX-only")
        self.replay_input_dtype = replay_input_dtype
        values = {}
        for field in fields(conditioning):
            value = getattr(conditioning, field.name)
            values[field.name] = None if value is None else value.detach().clone()
        self.conditioning = replace(conditioning, **values)
        self.adapter, self.autocast_dtype = adapter, autocast_dtype
        self.device = next(adapter.trainable_module.parameters()).device
        self.count = len(conditioning.prompt_embeds)
        if self.count < 1:
            raise ValueError("conditioning batch must be nonempty")

    def __call__(self, state, sigma, rows):
        return self._predict(state, sigma, rows, rollout=False)

    def rollout(self, state, sigma, rows):
        return self._predict(state, sigma, rows, rollout=True)

    def _predict(self, state, sigma, rows, *, rollout):
        if (
            rows.ndim != 1 or rows.dtype not in (torch.int32, torch.int64)
            or len(rows) != len(state) or (rows < 0).any() or (rows >= self.count).any()
        ):
            raise ValueError("one valid integer conditioning index per state is required")
        values = {}
        for field in fields(self.conditioning):
            value = getattr(self.conditioning, field.name)
            # FLUX text IDs describe token positions, not batch rows.
            if value is not None:
                if field.name != "text_ids":
                    value = value[rows.to(value.device)]
                value = value.to(self.device)
            values[field.name] = value
        conditioning = replace(self.conditioning, **values)
        options = {}
        if isinstance(conditioning, SD3Conditioning) and self.autocast_dtype == torch.float16:
            # Match Accelerate's FP16 forward wrapper, including FP32 CFG mixing.
            options["upcast_output"] = True
        if isinstance(conditioning, FluxConditioning):
            latent_dtype = self.autocast_dtype or torch.float32
            state = state.to(latent_dtype if rollout else (self.replay_input_dtype or latent_dtype))
            options["timestep_dtype"] = latent_dtype if rollout else torch.float32
        with torch.autocast(
            device_type=self.device.type, dtype=self.autocast_dtype,
            enabled=self.autocast_dtype is not None,
        ):
            return self.adapter(state.to(self.device), sigma.to(self.device), conditioning, **options)


@dataclass(frozen=True)
class ScoredImages:
    pixels: torch.Tensor
    scores: ScoreResult


@torch.no_grad()
def decode_and_score_images(adapter, endpoints, prompts, scorer, *, decode_microbatch,
                            score_microbatch, metadata=None, evaluation=False):
    """Decode once, map native [-1,1] RGB to [0,1], then use the native scorer.

    CPU pixels can be saved to PNG by the artifact layer. Scorer preprocessing
    (including PickScore quantization) is not replaced by this helper. This is
    ordinary scalar rollout/evaluation, explicitly not the ReFL gradient path.
    """
    if (
        endpoints.ndim != 4 or len(endpoints) < 1 or len(prompts) != len(endpoints)
        or not endpoints.is_floating_point() or not torch.isfinite(endpoints).all()
    ):
        raise ValueError("finite BCHW endpoints and paired prompts are required")
    if any(type(x) is not int or x < 1 for x in (decode_microbatch, score_microbatch)):
        raise ValueError("decode/score microbatches must be positive integers")
    validate_image_metadata(scorer, prompts, metadata)
    device = next(adapter.vae.parameters()).device
    pixel_batches, results = [], []
    for start in range(0, len(endpoints), decode_microbatch):
        stop = min(start + decode_microbatch, len(endpoints))
        native = adapter.decode(endpoints[start:stop].to(device))
        if native.ndim != 4 or native.shape[1] != 3 or not torch.isfinite(native).all():
            raise ValueError("image VAE must return finite BCHW RGB")
        pixels = (native.float() / 2 + 0.5).clamp(0, 1)
        results.append(score_image_batch(
            scorer, pixels, prompts[start:stop], microbatch=score_microbatch,
            metadata=None if metadata is None else metadata[start:stop],
            evaluation=evaluation,
        ))
        pixel_batches.append(pixels.cpu())
    first = results[0]
    for result in results:
        if (result.space != first.space or result.weights != first.weights
                or result.normalization != first.normalization or result.raw.keys() != first.raw.keys()):
            raise ValueError("reward contract changed within one image batch")
        if set(result.diagnostics or {}) != set(first.diagnostics or {}):
            raise ValueError("diagnostic contract changed within one image batch")
    raw = {key: torch.cat([result.raw[key].detach().cpu() for result in results])
           for key in first.raw}
    diagnostics = None if first.diagnostics is None else {
        key: torch.cat([result.diagnostics[key].detach().cpu() for result in results])
        for key in first.diagnostics
    }
    scores = weighted_score(raw, first.weights, space=first.space,
                            normalization=first.normalization, diagnostics=diagnostics)
    if scores.weighted.shape != (len(endpoints),):
        raise ValueError("scorer must return one reward per image")
    return ScoredImages(torch.cat(pixel_batches), scores)
