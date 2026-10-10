# Hunyuan Video Integration

The `hunyuan_public` backend connects single-device and full-shard DiffusionART,
DiffusionNFT, Flow-GRPO-Fast-SDE, DanceGRPO and final-step ReFL with VideoReward MP4 scoring,
evaluation galleries and full checkpoint/resume. Native loading targets the DanceGRPO FastVideo source
at `15cc71d53cc2e6e18a68ee607d5fb6ba9a99e344`; it does not substitute the
Diffusers transformer implementation.

Full-budget `configs/paper/video/hunyuan_*.yaml` and cover profiles inherit
[automatic paper assets](video-installation.md#automatic-paper-assets): pinned
HF model/reward/backbone, the exact public training texts and frozen VidProM
validation100, native text view and cached conditioning. Preparation precedes
torchrun and still requires the installed native environment on an authorized
training node. Bind ready conditioned manifests to skip encoding, or
`HUNYUAN_TRAIN_RAW_MANIFEST`/`HUNYUAN_EVAL_RAW_MANIFEST` for custom text JSONL.
Missing user SHA bindings are inferred; public pins stay strict. This does not
claim pretrained acceptance, showcase150 identity or historical paper parity.

This is an integration backend, not yet the released HY13B paper recipe.
Pretrained Linux/CUDA loading and full-model GPU memory/performance still need
acceptance. In particular, the shared image environment is not a verified
replacement for the native FastVideo/VideoAlign dependency stack. A small CPU
fixture exercises orchestration and real FFmpeg, not native model execution.
Use [native video setup](video-installation.md) for the pinned environment and
asset preparation commands.

## Scientific Contract

- FP32 policy parameters with BF16 autocast; FP32-only execution is rejected
  because native conditioning and attention require the mixed-precision path.
- Embedded guidance is 6018, without external classifier-free guidance.
- DiffusionART: shifted ODE sampling, per-video timestep selection and frozen full-group
  covariance before the two disjoint updates. Old velocity is reconstructed
  from the cached ODE transition; current velocity is the model output.
- The VAE receives raw latents divided by 0.476986. Denormalization happens in
  the decoded dtype, then FP32 conversion, multiplication by 255 and uint8
  truncation, matching the reporting reference.
- MP4s use H.264 CRF25/yuv420p. Complete raw-file transport replaces stdin;
  publication requires a successful encoder exit, full frame-count decode and
  matching dimensions/FPS. Decode verification supports old and new FFmpeg
  passthrough option names. It does not regenerate samples on encoding failure.
- Train and evaluation video FPS must agree. VideoReward reports raw and
  fixed-normalized scores separately, applying configured weights once.
- Evaluation may override resolution, sampling steps and seed. Every scored
  video is retained with an identity manifest and an offline HTML gallery.

`model.options` supplies `native_root`, `finetune`, `gradient_checkpointing`,
`ffmpeg`, `ffprobe` and `allow_tf32`. `reward.options.backbone` is the local
Qwen2-VL backbone; model and reward checkpoint identities must be explicit.
The existing config validator rejects unsupported algorithms and distributed
modes rather than silently changing their behavior.

## Cached Conditioning

The preparer creates a new prompt manifest and per-prompt safetensors files:

```bash
python -m visual_rl.models.hunyuan_assets \
  --manifest "$PROMPT_MANIFEST" --sha256 "$PROMPT_SHA256" \
  --model-root "$HUNYUAN_TEXT_MODEL" --native-root "$DANCEGRPO_ROOT" \
  --output "$CONDITIONING_OUTPUT" --device cuda
```

Run this only in the compatible native GPU environment. The source manifest
is unchanged. Prepare `HUNYUAN_TEXT_MODEL` with the asset-layout command in
[native video setup](video-installation.md#hunyuan). Bind the returned manifest
path and checksum in the experiment config. Loading checks prompt identity,
native revision, tensor file checksum
and the pooled-token-plus-text-mask shape. Complete asset-bundle provenance
is separate from the available [environment installer](video-installation.md).

## Current Verification

The CPU fixture runs the normal CLI with injected small policy/VAE/scorer
components, real MP4 encoding/scoring, evaluation HTML, and an interruption
followed by exact model/optimizer/prompt-stream resume. Separate checks compare
RGB quantization with `VideoProcessor` and row-specific caches with a full
trajectory cache. This does not establish pretrained GPU memory/performance or
historical paper-run ordering parity.

The runner retains every training video and requires responsive output storage.
JSONL and optional [isolated W&B export](logging.md) share the same metric names.
Full pretrained GPU acceptance remains outstanding.

## Algorithm Selection

All recipes use the same CLI, model, reward, evaluation and checkpoint
interfaces. Change the algorithm configuration, not the runner source.

| Config | Rollout | Advantage and update |
| --- | --- | --- |
| `configs/examples/hunyuan_art.yaml` | ODE, 9 row-specific training positions | Frozen group covariance, two disjoint MSE updates |
| `configs/examples/hunyuan_fast.yaml` | Gaussian SDE at 2 of the first 6 positions, ODE elsewhere | Group-centered weighted reward / global population std; clipped likelihood replay |
| `configs/examples/hunyuan_dance.yaml` | Native Dance SDE at all positions, 9 row-specific training positions | Per-component group sample std; clip each component objective before weighted mixing |
| `configs/examples/hunyuan_nft.yaml` | Old-policy Euler ODE endpoints | Fresh per-item re-noising, frozen old predictions and one full mirror update; slow old-policy EMA |
| `configs/examples/hunyuan_nft_onpolicy.yaml` | Same NFT path | Copy current to old after every update, without EMA |
| `configs/examples/hunyuan_refl.yaml` | Current-policy Euler ODE, K=1 | One final-step update through frozen VAE and continuous VideoReward |
| `configs/examples/hunyuan_pairwise_k{2,4,8}.yaml` | ART ODE with K=2/4/8 | Disjoint binary/tie comparisons, prompt-global covariance, two disjoint MSE updates |

Fast uses independent initial latents; Dance shares the initial latent within
each prompt group but uses independent transition innovations. Dance scores
the last predicted clean endpoint, not the noisy terminal state. Neither video
baseline uses policy/evaluation EMA or a reference regularizer. Fast uses
Gaussian transition likelihoods, not the image Fast-CPS negative-MSE surrogate.
Both baselines freeze advantages before the two disjoint optimizer updates.

NFT instead performs one optimizer update over all K trajectories and selected
timesteps. Its training positions include the final positive-sigma position;
they identify fresh-noised endpoint states, not cached rollout states. Private
CPU seeds bind noise to the attempt, rank, sample and timestep, independently
of physical batching. Advantage normalization is group centering followed by
global scalar reward population std, as in the video Fast-SDE recipe.

NFT's first post-update old-policy decay is zero; subsequent decay is
`min((completed_updates - 1) * rate, maximum)`. No-EMA always copies current
parameters. Evaluation uses current parameters, not old-policy or a separate
EMA. The CPU old-policy shadow is included in full-resume checkpoints. FSDP
swaps only local shards and invalidates retained gathered caches on entry/exit;
it never creates a second full GPU model.

### Pairwise Feedback

Use `hunyuan_pairwise_k2.yaml`, `hunyuan_pairwise_k4.yaml` or
`hunyuan_pairwise_k8.yaml` with the same installation and asset variables as
scalar ART. For example, on an allocated multi-GPU host:

```bash
torchrun --standalone --nproc-per-node=8 \
  -m visual_rl.train --config configs/examples/hunyuan_pairwise_k2.yaml
```

These short integration recipes fix the prompt count and physical microbatches;
only K and experiment/output names differ. K is videos per prompt, not pairs.
For multi-node training, use the launch arguments below rather than `--standalone`.

The opt-in `algorithm.options.pairwise: {seed: 42}` compares fixed-normalized
VQ within seeded disjoint pairs, producing winner 1, loser 0, exact tie 0.5.
Only those labels enter covariance; scalar reward magnitudes remain reporting
and evaluation data. Raw pair directions aggregate over the complete prompt
group before normalization and stay frozen through both update windows.

Tie rows carry restoration only when at least one rank has a non-tied pair.
All rows, including inactive rows, retain their place in the original loss
denominator. A globally untrainable rollout resamples without advancing the
optimizer step or completed-rollout count. Its actual compute remains in
elapsed/training GPU-hours, and its attempt and activity are logged. Persistent
all-tie feedback can therefore keep resampling without making update progress;
the runner does not fabricate a winner or silently switch to scalar training.

The single-device and FSDP paths use the same rule. Checkpoint/resume includes
the prompt cursor and attempt number, so the pair seeds continue consistently.
Evaluation and its HTML always retain the ordinary scalar component scores.

### ReFL Gradients And Reporting

ReFL optimizes the actual final endpoint's continuous RGB reward, not an
intermediate clean-sample extrapolation. The ODE prefix is detached; only the
last denoising step is differentiated. Each prompt generates one video. The
negative weighted fixed-normalized reward gradient is propagated through
frozen VideoReward and VAE modules without gradient normalization, group
centering, or a straight-through MP4 codec estimator.

The implementation stages the exact chain-rule VJPs: frozen VAE preview,
reward-to-pixel gradient, rematerialized VAE-to-latent gradient, then the last
policy step. Only one of VAE and reward model is resident on the GPU at a time.
Their weights move between CPU and GPU; activations are not automatically
offloaded to CPU. Coarse native VAE regions and reward-model blocks use
non-reentrant GPU checkpointing. Native tiling stays unchanged; smaller tiles
are not assumed mathematically equivalent. OOM is a reported failure, never a
silent change of reward, frame sampling, or gradient policy.

`train/refl/loss`, `train/refl/raw_*` and `train/refl/normalized_*` describe the
differentiable training objective. The ordinary `train/reward/*` metrics and
evaluation manifests remain MP4-based, exactly as for other algorithms. These
two scoring paths are deliberately distinguished, not claimed identical.
`train/optimizer_step` and both elapsed/training-only GPU-hour clocks retain
their shared definitions. Rematerialized pixels and policy endpoints are checked
against those used for the cotangent before an update. A zero global policy
gradient fails before AdamW can apply weight decay.

The example files run four updates for integration, not the full paper budget.
They require explicit environment variables for local assets and prepared
conditioning manifests:

- `HUNYUAN_MODEL`, `HUNYUAN_REVISION`: local model bundle and its source revision.
- `DANCEGRPO_ROOT`: clean checkout at the pinned native commit above.
- `VIDEO_REWARD_MODEL`, `VIDEO_REWARD_REVISION`, `VIDEO_REWARD_BACKBONE`:
  local VideoReward checkpoint, source revision, and Qwen2-VL backbone.
- `HUNYUAN_TRAIN_MANIFEST`, `HUNYUAN_TRAIN_SHA256`,
  `HUNYUAN_EVAL_MANIFEST`, `HUNYUAN_EVAL_SHA256`: paths and checksums returned
  by the conditioning preparer for the corresponding datasets.

For a single node with eight GPUs, in the compatible native environment:

```bash
torchrun --standalone --nproc-per-node=8 -m visual_rl.train \
  --config configs/examples/hunyuan_fast.yaml
```

Choose another config for another algorithm. The global prompt count is 32,
K is 8, and physical microbatches default to 1. Adjust physical microbatches in
the config to fit hardware; changing them does not change K or the loss
denominator. Full-model memory acceptance is still pending. Multi-node launches
use ordinary torchrun rendezvous settings; do not launch independent standalone
jobs on each node.

## Full-Shard Execution

Set `runtime.distributed: fsdp` with `model.options.finetune.mode: full`.
Global prompts per rollout must divide by the number of ranks; every rank
keeps complete K-sample groups (and an even trajectory count for two-update
algorithms). Launch the same
entrypoint with ordinary torchrun allocation settings. Model/reward assets,
hashed manifests and outputs must be visible at the configured paths to all
workers. No cluster-specific script or agent plugin is required.

The wrapper uses FULL_SHARD, FP32 master parameters/reductions/buffers,
BF16 forward autocast, native block-level wrapping, and optionally non-reentrant
activation checkpointing for every transformer block. It does not CPU-offload
policy parameters. Every update microbatch reduces its gradient shards;
`no_sync` is deliberately not used. Clipping uses FSDP's global gradient norm,
not a separate norm on each local shard.

Unlike replicated image DDP, FSDP inference still participates in model
collectives. Rollout forward counts match across ranks. Evaluation assigns
global indices round-robin, padding only the final forward round when necessary.
Padding trajectories are not scored, saved or counted. Rank zero publishes the
combined identity-checked manifest and gallery. A model forward/backward failure
is terminal for the process group; the launcher must stop the peer workers.

Full-state checkpoints store local model/AdamW shards and per-rank RNG/data
state without assembling the full model. Resume requires the same topology,
wrapping/layout, config and PyTorch version. Pass the explicit checkpoint
manifest from `checkpoints/latest.json` to `--resume`. See the
[distributed notes](distributed-execution.md) for publication/error boundaries.

Two-process CPU tests exercise actual FSDP (not a fake collective wrapper),
including a three-video evaluation on two ranks and interrupted training
followed by exact resume. Native pretrained GPU and frozen historical sampler
ordering acceptance remain separate gates.
