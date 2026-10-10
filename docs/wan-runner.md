# Wan I2V Integration

The `wan_i2v_distilled` backend connects DiffusionART, DiffusionNFT, Flow-GRPO-Fast
DanceGRPO and final-step ReFL to the same single-device/FSDP lifecycle used by Hunyuan: MP4
scoring, fixed-reference evaluation, retained local videos, optimizer-step/cost
metrics, and full-state checkpoint/resume.

Full-budget paper profiles inherit
[automatic paper assets](video-installation.md#automatic-paper-assets): pinned
base/FLUX/reward/backbone, the nonquantized LightX2V source and CPU conversion,
and exact public ConsistID training texts. The archived fixed eval JPEGs have
no verified public download. Supply one ready `WAN_EVAL_MANIFEST` path or one
`WAN_EVAL_RAW_MANIFEST` JSONL path with existing images and image hashes;
preparation infers the manifest SHA and copies fixed image bytes unchanged.
Missing raw eval input fails before model downloads. Online FLUX **training**
references remain online; no fabricated fixed eval bundle is substituted.

## Native Assets

The policy is the full, nonquantized LightX2V CFG/step-distilled Wan2.1 I2V 14B
transformer, converted to Diffusers format. The runner does not interpret the
original single-file checkpoint as an already converted model directory.
Required config environment variables:

| Variable | Asset |
| --- | --- |
| `WAN_DISTILLED_TRANSFORMER`, `WAN_DISTILLED_REVISION` | Converted transformer-only directory and source identity |
| `WAN_BASE_MODEL`, `WAN_BASE_REVISION` | Wan2.1 I2V Diffusers snapshot, including VAE, text/CLIP encoders and processors |
| `FLUX_REFERENCE_MODEL`, `FLUX_REFERENCE_REVISION` | FLUX.1-dev snapshot for online training reference images |
| `DANCEGRPO_ROOT` | Clean native source checkout at `15cc71d53cc2e6e18a68ee607d5fb6ba9a99e344` |
| `VIDEO_REWARD_MODEL`, `VIDEO_REWARD_REVISION` | VideoReward checkpoint directory and revision |
| `VIDEO_REWARD_BACKBONE` | Local Qwen2-VL-2B-Instruct snapshot |

Native model loading deliberately requires Diffusers **0.33.1**, matching the
converted weights and accepted runtime. Historical environment evidence also
uses Torch 2.6.0, Transformers 4.46.1 and Hugging Face Hub 0.30.2. A clean,
public Linux/CUDA environment recipe and checkpoint conversion command are in
[native video setup](video-installation.md). The generic image-development `uv` environment is not
a substitute for this native environment. No pretrained GPU acceptance has
been performed for the open-source runner yet.

The model loader preserves FP32 master policy, FP32 VAE/CLIP weights and BF16
text/FLUX weights. Frozen models are staged separately from the sharded policy.
The legacy FSDP time-embedding dtype fix applies only to this loaded instance,
not to the upstream class or other algorithms.

## Data and Reference Images

Set `WAN_TRAIN_MANIFEST`/`WAN_TRAIN_SHA256` to a hashed **text-only** JSONL
manifest. Each training prompt generates a new reference image with FLUX,
30 steps, guidance 3.5 and 512 maximum text tokens, followed by the native JPEG
save/reopen conversion. This work is inside training GPU-hours. It is not
silently replaced by cached references.

Set `WAN_EVAL_MANIFEST`/`WAN_EVAL_SHA256` to a hashed I2V JSONL manifest. Each
row has `sample_id`, `prompt`, `image` and `image_sha256`; image paths resolve
relative to the manifest. Evaluation always uses these fixed images, never
fresh FLUX outputs. See [prompt data](prompt-data.md) for the schema.

### Precomputed References

`model.options.reference_mode: fixed_images` uses hashed I2V JSONL training
images instead of online FLUX. It applies to all five algorithms; group samples
share the selected prompt's reference. No FLUX is loaded or called by training
in this mode. Missing mode retains the original online recipe.

The new dataset plan keeps all 27,815 ConsisID training rows and extends the
VidProM test set to 1,000, preserving the original100 prompt/image prefix.
Exact normalized and token-set Jaccard>=0.8 train/test overlap is excluded.
Precomputation changes the reference-image distribution and cost, not the
algorithm's objective, K, sampler or update settings. It is a separate dataset
recipe, not a claim that historical online runs used fixed training images.
One-time FLUX generation cost is separate from training GPU-hours.

See [dataset preparation](video-installation.md#precomputed-i2v-train-and-test)
for image generation, publication and a recipe bound to the actual HF commit.

Native `prepare_latents` encodes the reference image followed by zero **pixel**
frames, takes the VAE mode, normalizes all latent frames and prepends the
first-frame mask. Padding an encoded first frame is not equivalent.

## Configs and Execution

The five short integration examples all use the same CLI:

```bash
# In the native environment, after preparing the assets/manifests above:
python -m visual_rl.train --config configs/examples/wan_art.yaml --validate
torchrun --standalone --nproc-per-node=8 -m visual_rl.train \
  --config configs/examples/wan_art.yaml --output-dir outputs/wan-art-check
```

Use `wan_nft.yaml`, `wan_fast.yaml`, `wan_dance.yaml` or `wan_refl.yaml` to change algorithm.
These are four-update integration examples, not frozen paper configs. For
multiple nodes use the standard torchrun rendezvous arguments; global prompt
count must divide world size. `runtime.distributed=none` selects a single
device, subject to full-model memory requirements.

All examples use P32, four sampling steps, shift five, no CFG, 53 frames at
400x640/8 FPS, RM 4 FPS and VQ:MQ=1:1. The nodes are
`[1, 0.9375, 0.8333333333, 0.625, 0]`. Physical rollout/update microbatches are
8/4 for group-based methods (K8) and 1/1 for ReFL (K1), and can be reduced
without changing K or the original loss denominator.

| Algorithm | Sampling and update contract |
| --- | --- |
| DiffusionART | ODE, all first three positions, two disjoint row windows, original covariance/MSE/mask |
| DiffusionNFT | Old-policy ODE endpoints, fresh noising at all four positions, one mirror update, slow-EMA shadow |
| Flow-GRPO-Fast | One position uniformly selected from the first three, shared across all ranks/K; SDE there, ODE elsewhere |
| DanceGRPO | Shared initial noise per prompt, independent SDE innovations at the first three positions, clean terminal ODE |
| ReFL | Current-policy ODE, K1, one final-step update through continuous VideoReward; no guidance normalization |

Wan Dance scalarizes rewards **before** group sample-standard-deviation
normalization and PPO clipping, matching its accepted runtime. Hunyuan Dance's
componentwise objective remains separate. All Wan model-output arithmetic is
FP32 after BF16 forward computation; neither its initial noise nor Euler
recurrence is rounded to BF16. Hunyuan evaluation uses its own BF16 recurrence.

ReFL applies the Wan latent affine exactly once, then decodes in FP32 with
non-reentrant GPU checkpointing per latent frame. Temporal caches remain
differentiable across frames. The frozen reward model uses BF16 and its own
checkpointed blocks. Frozen VAE/reward weights are staged separately; retained
activations are not automatically offloaded to CPU. Ordinary MP4 reporting and
evaluation keep their native decode path. Continuous optimization scores are
logged separately under `train/refl/*`; comparison curves use MP4 scores.
Full-model memory must still be measured. There is no automatic tiling or
OOM fallback that silently changes the differentiable objective.

The shared full-state checkpoint includes optimizer, prompt cursor, RNG and
NFT old-policy shadow. FSDP resume requires the original topology and supported
Torch version/layout. Model-only research checkpoints are not silently treated
as exact training-resume checkpoints. JSONL metrics currently expose
`train/optimizer_step`, raw/fixed-normalized component scores, and both elapsed
and training GPU-hours. Optional [W&B export](logging.md) uses those same records.

## Evidence and Remaining Work

CPU tests cover all five algorithms through the actual CLI, FFmpeg, evaluation
manifest/gallery, four updates, interrupted/full-state resume and real two-rank
FSDP. Native RNG and Dance transition/log-likelihood/gradient arithmetic are
compared against the accepted research source. Small native Wan decoder and
pipeline conditioning tests cover full zero-pixel-video construction.

These checks do not prove 14B CUDA memory/performance or paper reproduction.
Remaining acceptance includes the native environment/assets, frozen prompt
ordering and paper configuration provenance, and pretrained
distributed runs. Generic prompt ordering is explicitly distinct from the
historical DistributedSampler order.
