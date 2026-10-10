# Model Adapter Contract

Status: component-level foundation on release base `5183083`, now including
optional SD3 text loading and full-weight/LoRA setup. Production task runners,
distributed integration and pretrained/GPU reproduction acceptance remain
unfinished. Model code never references the research checkout.

## Text And Trainable Parameters

`visual_rl.models.sd3_text.SD3TextEncoder` accepts an injected SD3 pipeline or
loads its text components with an explicit model revision and offline default.
The loader passes `transformer=None, vae=None`, so it does not load a second
policy or decoder. Both CLIP encoders and T5 stay frozen/eval. Empty negatives
are encoded, not replaced with zero tensors.

`encode(prompt, max_sequence_length=..., cfg_scale=...,
num_images_per_prompt=...)` returns `SD3Conditioning` with prompt-major repeats
for both token and pooled vectors. `cfg_scale=None` disables negatives; 0 or 1
are still explicit CFG settings. Sequence length is not guessed from a model
family: the inspected historical SD3 training helper used 128, while other
recipes may use different lengths. Use each runtime's actual setting.
The helper uses public Diffusers rather than copying the historical encoder;
its real tiny CLIP/T5 and offline-load tests are not a pretrained parity claim.

`visual_rl.models.finetune.prepare_policy(model, FineTuneConfig(...))` runs
before optimizer/shadow/DDP creation. Full-weight mode selects all weights in
FP32. LoRA uses explicit rank, alpha and linear target-module suffixes, Gaussian
initialization, zero dropout, no trainable bias, no DoRA and no rank-stabilized
rescaling. Its trainable parameters use FP32; frozen base precision is not
changed. Forward autocast and activation checkpointing are separate choices.
It refuses an already adapted/wrapped or quantized policy rather than stacking
another adapter or silently switching a training recipe.

The inspected SD3 recipe used rank 32, alpha 64 and targets
`attn.add_k_proj`, `attn.add_q_proj`, `attn.add_v_proj`, `attn.to_add_out`,
`attn.to_k`, `attn.to_out.0`, `attn.to_q`, `attn.to_v`. These are provenance,
not universal defaults for other models. `parameter_inventory()` records
trainable/frozen parameter counts by dtype, counting tied parameters once.

`visual_rl.training.image.ConditionedImagePolicy` connects frozen conditioning
to cached rollout/replay rows. `decode_and_score_images` maps VAE `[-1,1]` RGB
to clamped `[0,1]`, invokes the native scalar scorer, and returns detached CPU
pixels and explicit scores. It does not choose an algorithm, mutate update
windows, or serve as the differentiable ReFL path. A real tiny SD3/LoRA/VAE/CLIP
test exercises this chain through two disjoint ART optimizer updates, including
frozen old predictions and unchanged VAE/reward parameters.

**Precision boundary:** the affine formulas and CPU differentiability are
verified, but decode is **not claimed to be numerically identical to historical
mixed-precision execution**. This slice explicitly disables decoder autocast and
uses the injected VAE parameter dtype. In particular, the research Hunyuan ReFL
caller can run a float32 VAE under CUDA BF16 autocast; that is not this policy.
Historical GPU pixel/VJP parity must be checked with each recipe's actual VAE
dtype and autocast policy before a runner claims reproduction. There is no GPU
or BF16 full-model decode validation in this handoff.

## Public API

All adapters are `torch.nn.Module` instances with:

```python
adapter.predict_velocity(latents, sigma, conditioning)  # Tensor, same shape as latents
adapter(latents, sigma, conditioning)                   # same operation
adapter.decode(latents)                                # native VAE image/video tensor
adapter.trainable_module                               # injected transformer, not VAE
adapter.to_vae_latents(latents)                         # explicit single affine transform
```

Construct with `Adapter(transformer, vae, config)`. Components are loaded
elsewhere; construction does not load assets, move devices, install adapters,
or change transformer parameter flags. It freezes the injected VAE **in place**,
clears its stale gradients, and keeps it in eval mode even under `adapter.train()`.
Optimizers should use trainable parameters of `adapter.trainable_module`.

`sigma` is the **already schedule-shifted** flow noise level in `[0, 1]`, as a
scalar, `[1]`, or `[B]`. It is not a step index or a raw unshifted schedule value.
The scheduler owns resolution-dependent shifts, step selection, and integration.
The returned field is `dz/dsigma`; for example `x0 = z - sigma * velocity`.
Adapters never rescale velocity using the VAE scale or negate the field.

Image state is unpacked `[B,C,H,W]`; video state is `[B,C,T,H,W]`. Every field
must return the same shape, excluding I2V conditioning channels. Conditioning
batches must already match `B`; there is no implicit repeat or text encoding.
Caller owns transformer dtype/device and autocast. Inputs and conditioning must
be compatible with those components, including floating dtypes and attention
dimensions. No automatic model `.to()` or hidden CUDA autocast is performed.

`decode` takes **model-coordinate** latents, applies one affine transform in
float32 (float64 inputs retain float64), casts to the VAE parameter dtype, and
calls the decoder with autocast disabled. Do not feed `to_vae_latents(z)` back
into `decode`: that would scale twice. It does not detach, enter `no_grad`,
convert to PIL/uint8, select frames, resize, or apply `(pixels + 1) / 2`.
The result is the native VAE tensor, conventionally `[-1,1]`; Wan's native clamp
is preserved, while image/Hunyuan outputs are not additionally clamped.
Frozen VAE weights still permit `d pixels / d latents`, as required by ReFL.
Caller `no_grad` is honored. A visibly detached inference decoder is rejected
when gradients were requested; this guard does not prove an arbitrary custom
decoder's complete gradient correctness.

## Family Semantics

| Adapter | Model time | Guidance | Decode affine |
| --- | --- | --- | --- |
| `SD3Adapter` | float32 `sigma * timestep_scale` (default 1000) | Optional classic CFG, negative then positive | `z / scaling_factor + shift_factor` |
| `FluxAdapter` | `(sigma * scheduler_timestep_scale).to(z.dtype) / 1000` | Embedded guidance only; no classic CFG | unpacked `z / scaling_factor + shift_factor` |
| `HunyuanVideoAdapter` | `(sigma * timestep_scale).long()` | Native public FastVideo raw embedded guidance | `z / scaling_factor`, no shift |
| `WanI2VAdapter` | float32 `sigma * timestep_scale` | Optional classic CFG, positive then negative | per-channel `z * latents_std + latents_mean` |

Configs are frozen dataclasses. Image scale and shift and Wan channel statistics
are mandatory, not inferred from private checkpoints. Available component
metadata is checked for matching scales, channels, and guidance settings.
Opaque wrappers without `.config` remain the injector's responsibility.
`cfg_scale=None` disables CFG; any finite nonnegative value, including 0 or 1,
explicitly evaluates both branches. Both branches retain autograd. Guidance is
always `unconditional + scale * (conditional - unconditional)`.

### SD3.5

`SD3Config(scaling_factor, shift_factor, latent_channels=16,
timestep_scale=1000, cfg_scale=None)`.
`SD3Conditioning(prompt_embeds, pooled_prompt_embeds,
negative_prompt_embeds=None, negative_pooled_prompt_embeds=None)`.

Uses the Diffusers SD3 joint-attention transformer ABI. CFG concatenates all
negative rows followed by all positive rows, duplicating the entire timestep
vector in that same order (not `repeat_interleave`). Guidance rescale,
skip-layer guidance, IP adapters, and per-call attention kwargs are not included.

For a separately loaded SD3 pipeline, use named fields: the pipeline return
order differs from the conditioning dataclass positional order.

```python
pos, neg, pooled_pos, pooled_neg = pipeline.encode_prompt(...)
conditioning = SD3Conditioning(
    prompt_embeds=pos,
    pooled_prompt_embeds=pooled_pos,
    negative_prompt_embeds=neg,
    negative_pooled_prompt_embeds=pooled_neg,
)
config = SD3Config(
    scaling_factor=pipeline.vae.config.scaling_factor,
    shift_factor=pipeline.vae.config.shift_factor,
    latent_channels=pipeline.transformer.config.in_channels,
    cfg_scale=7.0,
)
adapter = SD3Adapter(pipeline.transformer, pipeline.vae, config)
```

Request negative embeddings from `encode_prompt` whenever `cfg_scale` is not
None. Device residency and inference/offload hooks remain caller-owned; such
hooks have not been validated for differentiable adapter execution.

### FLUX.1

`FluxConfig(scaling_factor, shift_factor, latent_channels=16,
embedded_guidance_scale=3.5, scheduler_timestep_scale=1000)`.
`FluxConditioning(prompt_embeds, pooled_prompt_embeds, text_ids)`.

Use `embedded_guidance_scale=None` for a transformer without guidance embeddings
(such as the corresponding schnell architecture). A mismatch with a declared
`transformer.config.guidance_embeds` raises an error. This is **not** true CFG,
FLUX Kontext, FLUX.2, or an image-editing adapter.

Public latents are deliberately unpacked rather than the historical pipeline's
packed scheduler state. `pack_latents`/`unpack_latents` expose the reversible
conversion: 2x2 spatial patches in row-major patch order with channel-major
`[channel, patch_row, patch_column]` features. H and W must both be even.
Transformer output is unpacked back into the caller's coordinates. Decoder
inputs are already unpacked, so no geometry guessing or second unpack occurs.
Text IDs are unbatched `[text_sequence,3]`. Image IDs are generated as
`[0, patch_row, patch_column]`, without batch repetition. Caller-supplied text
IDs are preserved. The cast before `/1000` intentionally retains historical
BF16 timestep rounding; replacing it with `sigma.to(dtype)` is not equivalent.

### Public HunyuanVideo 13B

`HunyuanVideoConfig(scaling_factor=0.476986, latent_channels=16,
timestep_scale=1000, embedded_guidance_raw=6018,
guidance_dtype=torch.bfloat16)`.
`HunyuanVideoConditioning(prompt_embeds, attention_mask)`.

This wraps the injected **FastVideo `hunyuan_hf` model**, not a generic modern
Diffusers Hunyuan transformer or a HunyuanVideo-1.5 SAGE model. The native forward
receives `hidden_states`, `encoder_hidden_states`, `timestep`, `guidance`,
`encoder_attention_mask`, and `return_dict=False`. Its embeddings include a
leading pooled-CLIP token followed by text tokens; the mask covers only the
text tokens. There is no separate `pooled_projections` argument.

The historical raw guidance is a singleton `[6018]` in BF16, which rounds to
6016. It is not user-space 6.018, and is not multiplied by another 1000 inside
this adapter. Positive sigma timesteps are truncated to integer values. There
is no classic negative-prompt CFG. No loader is provided: native FastVideo
dependencies, sequence-parallel initialization, and checkpoint compatibility
must be supplied and verified externally. CPU tests check the exact call ABI,
not execution of the full FastVideo transformer.

### Wan2.1 First-Frame I2V

`WanI2VConfig(latents_mean, latents_std, temporal_compression=4,
timestep_scale=1000, cfg_scale=5)`.
`WanI2VConditioning(prompt_embeds, image_embeds, image_condition,
negative_prompt_embeds=None)`.

Supported architecture: **one** Wan2.1 transformer with CLIP image conditioning.
No Wan2.2 high/low-noise model selection, last-frame conditioning, expanded
per-token timesteps, or first-frame state clamping is implemented.

`image_condition` is `[mask_channels, normalized_condition_latents]` along C.
The transformer sees `[noise_state, mask_channels, condition_latents]`.
There are `temporal_compression` mask channels: all ones at latent frame 0,
all zeros thereafter. The noise state's first frame is **not overwritten**.
Each CFG branch receives identical CLIP embeddings and identical VAE condition;
only the text changes. The concatenated CFG batch is positive first, negative
second; float32 guidance combination matches the research predictor.

`first_frame_condition(normalized_video_latents, temporal_compression=4)` only
prepends this mask. The caller must first preprocess the reference image,
construct a pixel video `[image, zero pixel frames]`, encode it with the frozen
VAE posterior **mode**, and normalize **all** latent frames with
`(raw - latents_mean) / latents_std`. Encoding one image and appending zero
latents is not equivalent. The [Wan runner](wan-runner.md) implements this
image/text preparation through the native pipeline; the adapter also accepts
an explicitly prepared condition tensor.

Wan decode uses `post_quant_conv` and the native causal decoder one latent frame
at a time, with a fresh per-call cache retaining cross-frame autograd. It accepts
the native decoder ABI with or without `first_chunk`. It preserves all frames
and the native `[-1,1]` clamp. Patched VAEs and enabled tiling are rejected.
The ReFL runner uses a separate FP32 decoder callback with per-frame GPU
checkpointing and staged reward VJPs, preserving the same temporal gradients.
It does not alter ordinary reporting decode or automatically introduce spatial
tiling/CPU activation offload. Pretrained asset loading is described in the
Wan runner guide; full-resolution GPU memory acceptance remains outstanding.

## Optional Image Factories

Only SD3 and FLUX expose lazy component factories:

```python
from visual_rl.models import SD3Adapter, SD3Config

config = SD3Config(scaling_factor=1.5305, shift_factor=0.0609, cfg_scale=7.0)
adapter = SD3Adapter.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium",
    config=config,
    revision="<explicit-checkpoint-revision>",
    local_files_only=True,
)
```

The config values must match the selected VAE. Factories load only the
`transformer` and `vae` subfolders, never text encoders, schedulers, or pipelines.
Diffusers is imported only when the factory is called. `local_files_only=True`
is the default; downloads require an explicit `False`. Authentication and cache
placement use standard Hugging Face environment/credential storage (`HF_TOKEN`,
`HF_HOME`, etc.). No token, private model path, or private cache path is embedded.

## Source Attribution And Repair Audit

Read-only research revision:
`3ca2a36f6b20c53a06813013234666d9c033a660`, repository
`flow_grpo_video_baseline_longrun_20260928`. Exact functions inspected:

- `flow_grpo_alg/diffusers_patch/sd3_pipeline_with_logprob.py::pipeline_with_logprob`:
  negative-first CFG and `/scaling_factor + shift_factor`; its latest file
  revision is `a95ff7244b93de8bbed12c6800cc2cf2c0befd34`.
- `flow_grpo_alg/diffusers_patch/flux_pipeline_with_logprob.py::pipeline_with_logprob`:
  packed field invocation, embedded guidance, timestep cast/division, unpack and
  VAE affine. Also inspected its historical implementation at
  `b44be52784e98fcecf1d771aa243af3c3b193bef`.
- `flow_grpo_alg/video/backends/hunyuan_public_flowart.py::predict_velocity`,
  `collect_ode`, `raw_endpoint_to_vae`: native call ABI and integer time mapping.
  The reporting helper's `detach` is intentionally not copied into decode.
- `flow_grpo_alg/video/refl_core.py::freeze`, `validate_hunyuan_vae`,
  `configure_hunyuan_vae`, `staged_latent_vjp`: single-scale differentiable
  decode, also inspected at successful revision
  `d2effaed5563de422d52de7c37984ae598f05af6`.
- `flow_grpo_alg/video/backends/wan_i2v_flowart.py::predict_velocity`,
  `prepare_conditioning`, `denormalize_latents`: conditional-first CFG and
  prepared I2V conditioning. Reporting-only detach is not copied.
- `flow_grpo_alg/video/backends/wan_i2v_refl.py::denormalize_latents`,
  `_decoder_chunk`, `decode_fn`: differentiable channel affine and causal cache
  semantics. These files match successful revision
  `14379aefd6c205e33d18c0504a69a55e8e29413d`; the intervening public Hunyuan
  changes were inspected and do not change its predictor/scale contract.

Pinned public upstream inspected locally at
`DanceGRPO@15cc71d53cc2e6e18a68ee607d5fb6ba9a99e344`:

- `fastvideo/models/hunyuan_hf/modeling_hunyuan.py::HunyuanVideoTransformer3DModel.forward`
  (pooled token extraction, singleton native guidance, no separate pooled input).
- `fastvideo/train_grpo_hunyuan.py::run_sample_step`, `grpo_one_step`, and decode call sites
  (integer `sigma*1000`, raw BF16 guidance, single `/0.476986`).

Diffusers API references inspected from installed version **0.39.0**:
`FluxPipeline._pack_latents`, `_unpack_latents`,
`WanImageToVideoPipeline.prepare_latents`, `AutoencoderKLWan._decode`,
`WanResidualBlock.forward`, and `WanResample.forward`. Wan mask construction
agrees with the pinned research-prepared condition contract. These local API
checks do **not** establish full compatibility with the research environment's
Diffusers 0.33.1. Full FastVideo, old-version, pretrained, CUDA, distributed, and
full-resolution memory validation remain external integration work.

The research incident index (generation 40) and relevant successful-commit diffs
were read once. ReFL records distinguish verified differentiable-decode repairs
from full-resolution memory repairs; this slice preserves the former contract
without claiming the latter. No experiment, remote launch, or research edit
was performed. This document is the scoped integration handoff, not a new
experiment ledger.

A read-only AST-extracted comparison against the exact public Hunyuan and Wan
`predict_velocity` functions at `14379aefd6c205e33d18c0504a69a55e8e29413d`
matched field values and input gradients using CPU doubles with distinct
per-row sigmas. This checks predictor adaptation, not mixed-precision decode.

## CPU Verification

Run from the release checkout with Torch and pytest; Diffusers tests skip as a
module when the optional dependency is absent:

```sh
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 \
python -B -m pytest -p no:cacheprovider tests/test_models*.py -q
```

Tests include exact CFG ordering with distinct per-row sigmas, both branch
gradients (including scales 0 and 1), frozen VAE policy-to-pixel gradients,
single-scale/shift numerical oracles, BF16 time/guidance behavior, rectangular
noncontiguous FLUX packing, explicit mask semantics, malformed-input rejection,
lazy/offline factory arguments, and import without optional backends.
Tiny real SD3, FLUX, and Wan transformers are compared with independent direct
forwards and input VJPs; real image, Hunyuan, and Wan VAEs verify decode behavior.
Wan additionally checks cross-frame derivatives and cache isolation.
The first-frame helper is compared with real
`WanImageToVideoPipeline.prepare_latents`, including full zero-padded pixel-video
encoding and unchanged noise state.

Verification on 2026-09-29: Torch 2.13.0 / Diffusers 0.39.0 ran 44 tests together,
then the added pipeline-preparation parity test passed separately: 45 passing
tests total, no model downloads. The coordinator's new virtualenv also passed
35 Torch-only tests with a process-local `visual_rl.__path__` adjustment for its
installed main package shadowing this isolated namespace. Its Diffusers module
test collection skipped because Diffusers was absent, and the subprocess
import-isolation test was deselected there (already passing in the original
environment). No parent package or dependency configuration was changed.
