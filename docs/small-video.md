# Small native text-to-video recipes

Development status: native adapters, frozen sampling/reward/data contracts and
CPU replay/resume tests are implemented. Pretrained official GPU parity,
eight-GPU training, actual memory/throughput and raw-reward sanity evidence are
pending. The checked-in training gate deliberately remains
`pending_scientific_decision` until the native ART replay/mask choice is resolved.
Do not interpret configuration validation or CPU fixtures as a mature recipe.

LoRA is the default entry. This scope is Wan2.1 T2V1.3B and the exact
LTX-Video2B0.9.6 distilled checkpoint, separate from existing Wan I2V14B/Hunyuan/H3.
See [assets/install/preflight](small-video-assets.md), [Wan native details](wan-t2v.md),
[LTX native details/license](ltx-video.md), and [checkpoint internals](small-video-resume.md).
Existing image and video environments/configurations are retained.

## Frozen native inference and replay

| Model | Native video | Native generation | Trainable attention LoRA suffixes |
| --- | --- | --- | --- |
| Wan2.1 T2V1.3B | 832×480,81frames,16fps | 50steps,CFG6,shift8,order2 UniPC bh2 | q,k,v,o |
| LTX-Video2B0.9.6 distilled | 1216×704,153frames,30fps | 8steps,CFG1,STG0,checkpoint LinearQuadratic,stochastic re-noising | to_q,to_k,to_v,to_out.0 |

Wan physical sigmas and INT64 time conditions are stored separately; full native
corrector/predictor history determines each transition. LTX physical sigmas are
`[1,.99375,.9875,.98125,.975,.909375,.725,.421875,0]` in FP32. The rounded
checkpoint allowlist is a validation table, not the physical schedule. LTX initial
noise is drawn in packed `[B,F*H*W,C]` layout; transition and decode innovation
layout/RNG draws follow the native pipeline. Decode retains time.05/noise.025 and
external per-channel VAE normalization. Prompt enhancement is disabled.

Both retain the complete native generation trajectory. All native input positions
with nonzero physical sigma and affine velocity coefficient are eligible, including
the last input before sigma0; sigma0 itself is never a training input. Original
trains all50/8 positions. Flash chooses2 without replacement from first4 eligible
positions, once per rollout and shared across ranks/windows, using a private
checkpointed RNG. `resolved-protocol.json` records the entire physical sigma
and selection table, cache branch, shapes, optimizer, parallel mesh and hashes.

The canonical Euler `x+delta_sigma*v` replay cannot be substituted for either
native solver. Under frozen native history/state the conditional mean is `a*v+b`.
The proposed named contract uses native mean-MSE/a², evaluated as cancellation-free
velocity residuals, with exactly canonical sigma weights, targets, restoration and
Q2 **reference** masking. Q2 is not the native stochastic-kernel KL. This scientific
choice remains pending; no alternative solver or mask is silently selected.

The transformer executes real physical batches. Wan CFG combines2B branches in
one transformer invocation; its native patch/unpatch helpers still operate per row.
Physical rollout/update microbatches are independently8 by default, adjustable to
4/2/1 with accumulation while preserving the complete global loss denominator.
Actual microbatch, forward shapes, stage time and every rank's allocated/reserved
peaks must be reported; no 80GB claim follows from a larger-memory machine or cap.

## Reward, data and optimizer

A rollout contains P16 prompt groups × K8 videos =128 videos. On the complete K8,
independently for VQ and MQ:

`z_j = (r_j - mean_K(r_j)) / (std_K(r_j, correction=0) + 1e-4)`

`R = (z_VQ + z_MQ) / 2`

There is no second normalization after averaging. The frozen complete-group
endpoint/reward covariance supplies both promptwise K4/K4 windows, each64 global
videos. One epoch and exactly two actual global AdamW steps occur per rollout;
timestep traversal, DP ranks and gradient accumulation add no optimizer steps.
The loss divides by actual global selected sample×timestep count, including masked
items. TA is mandatory finite telemetry, with zero training weight.

VideoReward uses the pinned canonical DanceGRPO implementation, raw scores,
4fps uniform preprocessing and unchanged assets. A separate canonical reward
Python process runs after decode on the same rank GPU and exits before backward;
there is no ninth scorer GPU or external service. Model and scorer environments
remain separate. Set `SMALL_VIDEO_REWARD_PYTHON` to that new environment's Python.
The stage cost and peak memory include scorer execution.

Use raw VQ, raw MQ and their raw equal mean for the sanity learning curve. The
mean group-standardized target is naturally near0 and is not a reward-growth
curve. Also inspect raw TA, group population std/near-zero fraction, loss,
finite/zero gradients, mask/anchor fractions, actual optimizer count, native
selected sigmas, phase duration and per-rank memory.

DanceGRPO50k train and VidProM validation come from the pinned
`vanzl/DiffusionART-Prompts` revision. Full source hashes and normalized text
leakage checks are in the asset manifest. Acceptance uses exactly8 fixed held-out
rows selected with Random42; identical prompts, initial seeds, stochastic draws
and decode draws are paired before/after. General held-out evaluation remains
available without spending optimizer steps.

BF16 forward, FP32 optimizer parameters, frozen text/VAE/scorer, LoRA32/alpha32/
dropout0, gradient checkpointing, CP1, noEMA and KL0 are fixed. LoRA LR1e-4;
full LR1e-5; clip1; canonical AdamW betas(.9,.999),epsilon1e-8,weight_decay1e-4.
LoRA defaults to DDP, full to FSDP. Each recipe uses a single node of8 physical
GPUs including its reward phases. Different recipes may run independently on
owned idle nodes; existing scientific jobs/services/leases must remain protected.

## Eight configurations and bounded launch

After completing the installation/assets/official-parity gates, select one row.
All commands run from the public checkout on an already-owned complete8GPU node.
Schedulers/lease acquisition are site responsibilities; no site hostnames or
private GPU-control implementation is baked into this repository.

| Combination | `RECIPE` |
| --- | --- |
| Wan LoRA Original (default Wan entry) | wan_t2v_1_3b_lora |
| Wan LoRA Flash | wan_t2v_1_3b_lora_flash |
| Wan full Original | wan_t2v_1_3b_full |
| Wan full Flash | wan_t2v_1_3b_full_flash |
| LTX LoRA Original (default LTX entry) | ltx_video_2b_distilled_lora |
| LTX LoRA Flash | ltx_video_2b_distilled_lora_flash |
| LTX full Original | ltx_video_2b_distilled_full |
| LTX full Flash | ltx_video_2b_distilled_full_flash |

```bash
export RECIPE=wan_t2v_1_3b_lora
export SMALL_VIDEO_ASSETS="$SV_ROOT/assets"
export SMALL_VIDEO_OUTPUT="$SV_ROOT/runs/$RECIPE-seed42-unique-attempt"
export SMALL_VIDEO_REWARD_PYTHON="$SV_ROOT/env-reward/bin/python"
export PYTHONDONTWRITEBYTECODE=1
# Validate profile/assets first. Current science-pending gate prevents training.
"$SV_MODEL_ENV/bin/python" -m visual_rl.train --config "configs/small_video/$RECIPE.yaml" --validate
# Checkpoint4 then stop this owned attempt at a complete two-window boundary.
SMALL_VIDEO_PAUSE_AFTER_UPDATE=4 timeout -k 30s 43200s   "$SV_MODEL_ENV/bin/torchrun" --standalone --nproc_per_node=8   -m visual_rl.train --config "configs/small_video/$RECIPE.yaml"
# Supply the exact complete checkpoint receipt printed by that attempt.
export RESUME_RECEIPT=/absolute/path/to/complete/checkpoint-receipt.json
unset SMALL_VIDEO_PAUSE_AFTER_UPDATE
timeout -k 30s 43200s "$SV_MODEL_ENV/bin/torchrun" --standalone --nproc_per_node=8   -m visual_rl.train --config "configs/small_video/$RECIPE.yaml" --resume "$RESUME_RECEIPT"
```

The cumulative hard limit is10 actual optimizer calls, including the first4 and
remaining6. The configuration stays at10 throughout; restart never resets budget.
An admitted but uncommitted step or a checkpoint behind the durable step journal
fails closed for training. Only a matching complete rollout boundary may resume.
At10 the owned test completes paired post-eval, final checkpoint and named export,
then exits so its launcher can release its generation-fenced lease. No long-run
50/200-rollout learning stage is part of this acceptance budget.

Keep two complete resumable checkpoints and `final/COMPLETE.json`. Final LoRA
export is a PEFT adapter; full export is the native transformer `policy.pt` plus
immutable base/source/protocol identity. Adam/RNG continuation uses the complete
checkpoint, not an inference-only export. A fresh process must verify load/eval/
resume before any combination is accepted.

For a larger independent hashed held-out manifest, evaluation alone supports:

```bash
timeout -k 30s 43200s "$SV_MODEL_ENV/bin/torchrun" --standalone --nproc_per_node=8   -m visual_rl.small_video_eval --config "configs/small_video/$RECIPE.yaml"   --checkpoint "$RESUME_RECEIPT" --manifest /absolute/heldout.jsonl   --sha256 ACTUAL_HELDOUT_SHA256 --label independent-heldout
```

This is a general capability, not a requirement to expand this round's8-row eval.
Short-run raw-score changes, including flat/negative changes, are reported with
paired uncertainty and all eight samples. Ten updates do not demonstrate long-run
stability, generalization or absence of reward exploitation.

## Acceptance evidence

All8 GPU cells are currently pending. CPU fixtures establish software/algebra
properties only. The observed allocation has inconsistent name/PCI metadata and
about140GiB memory per device, so it cannot establish8×H80080GB support. No process
memory cap resolves that hardware evidence gap. Publish actual hardware identity,
raw paired scores/uncertainty, complete checkpoint/fresh-load results and measured
memory/time before promoting any recipe's support status.
