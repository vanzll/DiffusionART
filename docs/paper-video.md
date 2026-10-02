# Paper Video Profiles

`configs/paper/video/*.yaml` are runnable full-budget native training profiles,
not the four-update integration examples. They inherit the complete public
interfaces in `configs/examples/`, pin available scientific settings, and use
**corrected common evaluation**, not verbatim historical evaluation. Pretrained
GPU execution, exact historical ordering and paper-number parity are not claimed.
No new sampler, evaluator, objective, asset downloader or launcher is introduced.

## Coverage And Provenance

The inventory is [reproduction-inventory.json](reproduction-inventory.json).
Each resolved YAML records its exact run ID, recorded runtime SHA (including
honest nulls), separately inspected reference SHA/path, evaluation distinction,
budget basis and artifact policy. Reference paths are research Git paths, not
public files promised to exist in this release. Read-only research reference:
`flow_grpo_video_baseline_longrun_20260928` at
`3ca2a36f6b20c53a06813013234666d9c033a660`; no remote run status was refreshed.

| YAML Stem | Historical Run | Runtime SHA | Budget / Reference |
| --- | --- | --- | --- |
| `hunyuan_art_721` | `gyvtocue` | `163d0696` (recorded, object unavailable) | 400 / `flowart_main721_val150.json` at `3ca2a36f` |
| `hunyuan_fast_11` | `ontwcush` | `a2a3dad8` | 400 / `fastsde_vqmq11.json` |
| `hunyuan_dance_11` | `1a9gbm4p` | `a2a3dad8` | 400 / `dancegrpo_vqmq11.json` |
| `hunyuan_nft_ema_721` | `bjb9ed8b` | `75778e76` | 400 / `nft_slowema_eval8fps.json` |
| `hunyuan_nft_onpolicy_721` | `y5vn56yr` | `75778e76` | 400 / `nft_noema_eval8fps.json` |
| `hunyuan_scalar_vq_k8` | `1iwf5sj9` | `75778e76` | 400 / `flowart_scalar_vq_k8_eval8fps.json` |
| `hunyuan_pairwise_vq_k2` | `x9mzbm4c` | `ccab6ad8` | 400 / `flowart_pairwise_vq_k2_eval8fps.json` |
| `hunyuan_pairwise_vq_k4` | `sdi90cn1` | `ccab6ad8` | 400 / `flowart_pairwise_vq_k4_eval8fps.json` |
| `hunyuan_pairwise_vq_k8` | `p6jdpon4` | `ccab6ad8` | 400 / `flowart_pairwise_vq_k8_eval8fps.json` |
| `cover_hunyuan_art_11` | `4dnkgeto` | `8b2f8a8d` | 400 / `flowart_vqmq11_curves.json` |
| `cover_hunyuan_nft_ema_11` | `njpjfulm` | `8b2f8a8d` | 400 / `nft_slowema_vqmq11_curves.json` |
| `cover_hunyuan_fast_11_long` | `76zs2v12` | `45c9c44a` | Bounded 400 reference, long-run horizon unresolved |
| `wan_art_11` | `08ca4146a532752c473f1f37` | unresolved | 400 / `flowart.json` at `3ca2a36f` |
| `wan_fast_11` | `31223c0be629580537f55ae2` | `c56245ec` | 400 / `fastsde.json` |
| `wan_dance_11` | `ef06ee7200e554ef1d5f4e5f` | `51269742` | 400 / `dancegrpo-u400.json` |
| `wan_nft_ema_11` | `a6ce1ae3fe255baf889d5c03` | unresolved | 400 / `nft-slowema-recovery-u400.json` at `3ca2a36f` |
| `hunyuan_refl_11` | `a0ba6dc83810b89403515067` | `14379aef` | 400 / `vqmq11-final-gpu.json` |
| `wan_refl_11` | `16cc3f7450393e017fd6e1de` | unresolved | 400 / `refl-gpu.json` at `3ca2a36f` |

Hunyuan reference JSONs live in `runners/t2v_formal_batch_v1/`, ReFL in
`runners/t2v_refl32_v1/`, Wan in `runners/wan_i2v_distilled32_v1/`.
Targeted `git show` checks covered the named runtime backends, ART loss terms,
Fast `MAIN_PLAN` (not its legacy 4-of-9 default), Dance componentwise loss,
NFT `old_policy_decay` and normalized-x0 objective, Wan algorithm dispatch,
ReFL final-step selector, and the unbounded lifecycle-only opt-in.
The incident index was read once. In particular, do not import promoted
reference-MSE experiments into these KL0 profiles merely because an operational
repair shares their lineage; a remote report is not scientific equivalence proof.

## Scientific Settings

All runs use full FP32 master policies, BF16 forwards, TF32 enabled, AdamW
`lr=1e-5`, betas `.9/.999`, epsilon `1e-8`, weight decay `1e-4`, grad clip `1`.
P32 is **global prompt groups**, not per rank. K8 means 256 videos per rollout;
pairwise K2/4/8 means 64/128/256 without changing P. ReFL K1 means 32.
ART/Fast/Dance make two disjoint updates per rollout (200 rollouts for U400);
NFT/ReFL make one (400 rollouts). No replay is added to equalize video counts.

Hunyuan: original public 13B guidance-distilled model, not HY1.5; 53 frames,
480x480, video8 FPS, 16 linear-grid shifted steps, shift5, embedded guidance6018,
external CFG disabled. ART/Dance select nine row-specific positions from the
first15; Fast selects two of first6 per rank/prompt group, Gaussian Flow-SDE
noise `.8`, ODE elsewhere. Dance has shared initial prompt noise, independent
innovations, eta `.25`, per-component sample-std advantages and componentwise
PPO clipping. Fast instead centers the full group and divides scalar rewards
by global population std plus `1e-4`; ratio clips are `1e-5`/`1e-4` respectively.
Neither uses policy or evaluation EMA, reference KL, or image Fast-CPS.

ART uses unbiased, unit-normalized endpoint covariance, velocity-linear-sigma
loss, RMS direction, corrective velocity mask
with scheduler-reference metric and Q2 noise `.8`/threshold `1e-5`. Pairwise
compares only fixed-normalized VQ in seeded disjoint pairs: winner1/loser0/tie.5.
Only binary labels enter covariance; ties anchor only with a global preference.
Scalar ART uses exact constant-reward protection and no zero-signal restoration;
inactive rows retain the original loss denominator. Globally all-zero scalar
rollouts skip updates but retain their actual cost. Pairwise keeps historical
covariance and tie-anchor semantics. Scalar VQ remains scalar; evaluation reports
scalar scores in both cases.

NFT uses old-policy ODE endpoints, fresh endpoint re-noising, normalized-x0
mirror loss (`beta=1`, clip5, stabilization floor `1e-5`) and one full update.
Hunyuan selects nine of all16 positions. Slow old-policy decay after update N
is `min((N-1)*.00125,.6)`; on-policy always copies with decay0. Evaluation uses
current weights, never the old shadow or a separate EMA.

Wan: full nonquantized LightX2V Wan2.1 I2V14B CFG/step-distilled policy,
400x640/53 frames/video8, four shift5 nodes `[1,.9375,5/6,.625,0]`, FP32
model-output/Euler arithmetic, one conditional branch. ART/Dance use all first3
positions; Fast selects one of first3 shared across all ranks; NFT uses all4.
Wan Dance scalarizes **before** group sample-std normalization, unlike Hunyuan.
Training FLUX references are generated online (30 steps, guidance3.5, max512
tokens, native JPEG reopen); evaluation images are fixed and hashed.

ReFL is final-step reward-only adaptation (indices15/3), not random-late or
first-raw ReFL. It differentiates continuous fixed-normalized RGB reward through
frozen RM/VAE and only the last policy step, without gradient normalization,
pretraining loss, reference KL or EMA. MP4 comparison scores remain separate.

VideoReward is pinned to `4f26600130683e6f1de9f5d463887f28e8ef995c`.
The native scorer uses RM4 FPS, uniform even frame count (26 of 53 video8
frames), `num_frames=None`, max200704 pixels and batch1. These are implementation
contracts, not extra unsupported YAML options. Reward weights are fixed
normalized TA/VQ/MQ=.7/.2/.1 for 721, VQ/MQ=.5/.5 for 11, VQ1 for scalar/pairwise.

## Assets And Execution

Use the pinned Linux/CUDA environment and preparation commands in
[video-installation.md](video-installation.md), [hunyuan-runner.md](hunyuan-runner.md)
and [wan-runner.md](wan-runner.md). There is no download at config validation.
Set local public-asset paths, not private cluster mounts or credentials:

```bash
export DANCEGRPO_ROOT="$HOME/assets/DanceGRPO"
export VIDEO_REWARD_MODEL="$HOME/assets/VideoReward"
export VIDEO_REWARD_BACKBONE="$HOME/assets/Qwen2-VL-2B-Instruct"
export PAPER_VIDEO_OUTPUT_ROOT="$HOME/outputs/paper-video"
export HUNYUAN_MODEL="$HOME/assets/HunyuanVideo"
export HUNYUAN_TRAIN_MANIFEST="$HOME/data/hunyuan-train/prompts.jsonl"
export HUNYUAN_EVAL_MANIFEST="$HOME/data/hunyuan-eval/prompts.jsonl"
export HUNYUAN_TRAIN_SHA256="$(shasum -a 256 "$HUNYUAN_TRAIN_MANIFEST" | awk '{print $1}')"
export HUNYUAN_EVAL_SHA256="$(shasum -a 256 "$HUNYUAN_EVAL_MANIFEST" | awk '{print $1}')"
python -m visual_rl.train --config configs/paper/video/hunyuan_art_721.yaml --validate
```

Hunyuan JSONLs must be outputs of the conditioning preparer, with hashed
safetensors and pooled CLIP token. Pin the Hunyuan local snapshot to
`e8c2aaa66fe3742a32c11a6766aecbf07c56e773` and native source checkout to
`15cc71d53cc2e6e18a68ee607d5fb6ba9a99e344`. Qwen2-VL backbone source is
`895c3a49bc3fa70a340399125c650a463535e71c`; its identity is an asset prerequisite,
not a runtime YAML revision field. The reward checkpoint file SHA256 is
`48375908e6112de9f0248402db156a23b480709a6960b091c598c6f4c88d21b9`.

For Wan, additionally bind the converted transformer and manifests:

```bash
export WAN_DISTILLED_TRANSFORMER="$HOME/assets/wan-distilled-converted"
export WAN_DISTILLED_REVISION=fef288b326f4fed6d2983b9800c35363da31fcfe
export WAN_BASE_MODEL="$HOME/assets/Wan2.1-I2V-14B-480P-Diffusers"
export FLUX_REFERENCE_MODEL="$HOME/assets/FLUX.1-dev"
export WAN_TRAIN_MANIFEST="$HOME/data/wan-train.jsonl"
export WAN_EVAL_MANIFEST="$HOME/data/wan-eval.jsonl"
export WAN_TRAIN_SHA256="$(shasum -a 256 "$WAN_TRAIN_MANIFEST" | awk '{print $1}')"
export WAN_EVAL_SHA256="$(shasum -a 256 "$WAN_EVAL_MANIFEST" | awk '{print $1}')"
python -m visual_rl.train --config configs/paper/video/wan_art_11.yaml --validate
```

Wan train JSONL is text-only; eval JSONL has `image`/`image_sha256`. Base/FLUX
revisions are pinned in YAML. The LightX2V source checkpoint revision and checksum
are provenance; they do not assert byte identity of converted tensors. Use the
existing converter and preserve its report. No private asset path is required.

On an authorized allocation, the exact historical logical distribution is
four nodes x eight GPUs (32 complete rank-local prompt groups):

```bash
torchrun --nnodes=4 --nproc-per-node=8 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  -m visual_rl.train --config configs/paper/video/hunyuan_fast_11.yaml
```

Run the same command once per node with standard allocation rendezvous values;
choose any table YAML. An eight-GPU single-node launch is also supported using
`--standalone --nproc-per-node=8`, preserving global P32 but **not** historical
rank-local ordering. All assets/output paths must be visible across ranks.
Hunyuan microbatches1/1 are allowed memory adaptations; Wan uses8/4 except
ReFL1/1. Neither changes K, timestep count, loss denominator or update count.
No launch was performed for this change.

## Evaluation And Remaining Gaps

- Common evaluation is current-policy deterministic ODE, seed312+index,
  video8/RM4, every10 updates including0/final, retaining all scored samples.
  Hunyuan uses30 steps/640x640/shift7, separately from training shift5.
  Wan uses four steps/400x640/shift5. Bind a held-out hashed100
  prompt manifest; supplying arbitrary data does not establish paper identity.
- Historical Hunyuan evaluation used a separate shift7 sampler, some runs used
  video18 FPS, sparse steps0/10/20/40/60/100/200/300/400, and retained32 of100.
  Final MQ rescoring on matched32 retained clips is a separate historical
  lineage, not implemented or reproduced by these YAMLs. Even eval8 historical
  profiles remain labeled corrected-eval because cadence and retention differ.
- The ART721 reference at `3ca2a36f` is a later150-prompt contract preserving
  training science, not the `gyvtocue` runtime snapshot. Common evaluation cannot
  label validation100/showcase50 subsets; do not mix their headline statistics.
- The schema requires positive integer `max_updates`. `76zs2v12` used the
  lifecycle-only unbounded opt-in at `45c9c44a`. Neither supplied inventory nor
  inspected source gives its paper-plotted endpoint. The long YAML is explicitly
  a runnable **400-update bounded reference**, not full long-run reproduction.
  Set a verified plotted horizon before using it as the cover long-run profile;
  no guessed2000/1000 horizon or invented `max_updates: null` is supplied.
- `gyvtocue` runtime SHA is recorded but unavailable in the inspected local
  object databases. Wan ART/NFT/ReFL runtime SHAs are null in the inventory;
  reference code SHAs are not substituted as actual run SHAs. Recover these
  before claiming exact-runtime provenance. Complete dataset/converted-asset
  manifests and clean pretrained GPU acceptance remain outstanding.
- Generic prompt-stream seed42 differs from the historical sampler seed1237,
  and native retry/seed ownership differs. These are training-semantic recipes,
  not bitwise trajectory reproductions. No unsupported sampler flag is invented.
- Research checkpoints were model-only every100 rollouts: 200 optimizer steps
  for ART/Fast/Dance,100 for NFT/ReFL. YAML uses those evidenced step intervals,
  but native checkpoints are full-resume and retained, not historical keep1.
  Native media retention saves all scored evaluation videos; training MP4s are
  temporary reward inputs and are removed after scoring. Historical train
  retention was interval10 updates and ranks0/16. No unsupported artifact interval
  is added. Checkpoints need substantial capacity and responsive I/O;
  select a writable shared `PAPER_VIDEO_OUTPUT_ROOT` before launch. The native
  runner has no bounded primary/fallback failover; do not claim automatic recovery.

## Local Verification

`tests/test_paper_video_configs.py` validates all18 resolved profiles through
`get_backend`, checks independent frozen scientific constants and inventory IDs,
P/K ratios, binary-only pairwise behavior, no stale inherited algorithm/sampler
fields, native reward preprocessing, zero-based NFT EMA and truthful horizon gaps.
It uses synthetic environment paths and performs no asset loading or GPU work.

Verification on 2026-09-30 against the dirty release worktree: **43 passed**
in 7.17 seconds; Ruff reported **All checks passed**. This is CPU config/contract
evidence, not pretrained execution or exact historical-runtime acceptance.

```bash
.venv/bin/python -m pytest -q tests/test_paper_video_configs.py
.venv/bin/ruff check tests/test_paper_video_configs.py
```
