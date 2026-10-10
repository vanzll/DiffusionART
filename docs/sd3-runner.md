# Native SD3 Runner

This is the first integrated real-model runner, not a paper reproduction
certificate. It supports single-device SD3/SD3.5, DiffusionART and native
PickScore or Aesthetic with FP32, BF16 or scaled FP16 forward precision. It does not silently substitute
for unsupported algorithms or distributed strategies. Optional [W&B logging](logging.md)
uses the same metrics without changing the training objective.
DiffusionNFT is available through the same entrypoint; its distinct sampling,
update and EMA contract is documented in [the NFT guide](sd3-nft.md).
Flow-GRPO, Fast and Flow-DPPO configurations are in [the GRPO guide](sd3-grpo.md).
ART/NFT and the GRPO family also support DDP; see [distributed execution](distributed-execution.md)
for global batch ownership, torchrun commands and distributed checkpoint resume.
The [Aesthetic guide](sd3-aesthetic.md) provides asset and reward-switch commands.

## Full Experiment (Recommended)

For the complete eight-GPU PickScore experiment, accept the SD3.5 Hub license,
configure Hugging Face and W&B access, and start training directly:

```bash
uv sync --locked --python 3.10 --extra diffusers
uv run --locked --extra diffusers visual-rl \
  --config configs/paper/image/sd3_pickscore_art.yaml --nproc-per-node 8
```

The launcher prepares the pinned SD3.5, PickScore and processor snapshots from
Hugging Face and downloads/converts the configured prompt lists before starting
GPU workers. Existing downloads are reused; missing files are acquired through
the Hub download/cache APIs. No separate `hf download` command or local model
path is required for this recipe. `VISUAL_RL_ASSET_ROOT` selects a writable asset
directory; explicit local asset bindings remain optional overrides.

Models are loaded locally after preparation, so backend `local_files_only`
does not disable the launcher's download stage. `--prepare-only` separates
preparation from training when desired, but is not a required first step.
See [Quick Start](quickstart.md) for credentials, storage and recovery.

## Install And Prepare

The remaining sections describe the optional four-update offline integration
example, not the recommended complete experiment above. Its YAML has no automatic
asset declaration and expects pre-cached weights; the manual commands below apply only
to that integration path.

Run commands from the repository root. Install the optional dependencies:

```bash
uv sync --locked --python 3.10 --extra diffusers
uv run --locked --extra diffusers visual-rl --config configs/examples/sd3_pickscore.yaml --validate
```

Validation checks configuration only: it does not claim assets, CUDA capacity
or storage have been verified. The example uses four supplied illustration
prompts, K=2 and four successful optimizer updates. These small budgets check
integration; they are not the paper dataset, K or training budget.

For pretrained execution, use an authorized CUDA GPU supporting the configured precision and
sufficient VRAM. A measured full-model VRAM requirement has not yet been
established. Do not infer it from the CPU tiny-model tests. The SD3 model is
gated: accept its terms on the
[model page](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)
and authenticate using an existing Hugging Face credential store or `HF_TOKEN`.
Never put token values in YAML or commands saved to logs.

Choose writable, persistent `HF_HOME` and output locations with sufficient
capacity before downloading. The base-model repository contains large weights
and alternate copies; inspect `hf download --dry-run` before fetching it.
The versions below were resolved from the publishers' metadata; they are not
yet asserted to be the exact asset revisions of every historical paper run.

```bash
uv run --locked --extra diffusers hf download stabilityai/stable-diffusion-3.5-medium --revision b940f670f0eda2d07fbb75229e779da1ad11eb80
uv run --locked --extra diffusers hf download yuvalkirstain/PickScore_v1 --revision a4e4367c6dfa7288a00c550414478f865b875800
uv run --locked --extra diffusers hf download laion/CLIP-ViT-H-14-laion2B-s32B-b79K --revision 1c2b8495b28150b8a4922ee1c8edee224c284c0c --include '*.json' --include '*.txt'
```

The integration example loads its manually cached pinned revision offline.
Unlike the complete paper recipe, it does not declare automatic asset preparation.

## Train And Inspect

```bash
uv run --locked --extra diffusers visual-rl --config configs/examples/sd3_pickscore.yaml
```

The run writes a resolved config and parameter inventory, one JSONL record per
successful optimizer update, all evaluation PNGs and identity/score manifests,
and full-state checkpoints. With the example, evaluations occur at updates
0, 2 and 4, while checkpoints occur at 2 and 4. `status.json` records completion.
An exception or incomplete evaluation is not reported as success.

Important JSONL fields:

| Field | Meaning |
| --- | --- |
| `train/optimizer_step` | Successful optimizer updates, not microbatches or rollouts |
| `train/reward` | Selected training space; the example uses native PickScore / 26 |
| `train/reward/raw/pickscore` | Native paired CLIP logit, without the /26 scale |
| `train/grad_norm/before_clip` | Aggregated parameter gradient before clipping |
| `cost/training_gpu_hours` | Timed text preparation, rollout, scoring and updates times allocated GPU count |
| `cost/elapsed_gpu_hours` | Training plus evaluation/checkpoint/artifact time since the run cost clock starts |

The cost clock currently starts after model and optimizer setup; it is not
W&B `_runtime` or allocation-wide billing time. CPU executions report zero GPU
hours. No retry cost is discarded. Evaluation uses separate per-sample noise
generators; it does not consume the training sampling generator.

Checkpoint receipts include checksums. To resume at a completed rollout
boundary, use the original configuration and output directory:

```bash
uv run --locked --extra diffusers visual-rl --config configs/examples/sd3_pickscore.yaml --resume outputs/sd35-pickscore-integration/checkpoints/latest.json
```

The checkpoint includes policy weights, optimizer, prompt cursor, CPU/CUDA RNG,
the sampling generator and, for FP16, the dynamic loss scaler. Resume verifies both the checkpoint fingerprint and
the existing directory's run/config identity before loading assets. A separate
`cost.json` records completed artifact-publication time, which cannot be stored
inside the checkpoint before that checkpoint finishes writing. Final status
costs are preserved for a no-op completed resume. A hard process/host failure
before cost publication can still leave an unmeasured tail; that is not a
zero-cost recovery or allocation-wide accounting guarantee.

Newer training logs or cost/status records than the selected checkpoint
cause an explicit refusal, not silent history splicing or an inexact claim of
continuation. ART's two-update windows are indivisible for resume;
budgets and checkpoint/evaluation intervals must be multiples of two.

## Scientific And Runtime Boundaries

Set `training.precision: fp16` for SD3 FP16 execution (single device or DDP).
Policy parameters and optimizer state remain FP32. Text encoders and transformer
autocast use FP16; transformer outputs are promoted before CFG/loss arithmetic,
as in Accelerate's mixed-precision wrapper. VAE and scalar scorers remain FP32.
Gradients are unscaled before clipping. On gradient overflow all ranks reduce
the loss scale together and replay the same frozen update window, with at most
16 retries, without advancing optimizer, EMA or successful-step counters.
Retries consume real training/GPU time. Persistent overflow fails explicitly;
nonfinite forward losses are not hidden by scaling. This same-window retry is
an explicit runtime recovery rule, not historical Accelerate's skip-on-overflow
behavior. FP16 support alone does not certify full historical-run parity.
`train/overflow_retries`, `train/loss_scale` and `train/next_loss_scale` record it.
FP32/BF16 runs do not instantiate a scaler. FLUX FP16 is not enabled.

- Prompt slots are generated by the named `PromptStream` order. They define
  independent K-sample groups; this is not yet verified against each historical
  distributed sampler, especially around dataset epoch boundaries.
- The trajectory cache is frozen once per rollout. Rows are shuffled before
  two disjoint updates; each selected timestep remains attached to its row.
  Masked and zero-signal states stay in the original loss denominator. Scalar
  ART requires exact constant-reward protection and `restore_zero_signal=false`;
  globally all-zero rollouts skip updates while retaining their actual cost.
- LoRA, its target layers, covariance settings, mask and timestep weighting are
  configuration-owned. VAE and reward weights stay frozen. This runner uses
  FP32 VAE decode and FP32 PickScore; transformer autocast is configured separately.
- The local writer/lock is synchronous; W&B export runs in an isolated process.
  Full artifact-publication and pretrained distributed acceptance are separate.
  Do not deploy this development runner on an unreliable shared mount
  and claim it has the research cluster's recovery guarantees.
- No production GPU validation or paper reward reproduction has been performed.
  Real tiny SD3/CLIP/T5/VAE local assets exercise the actual loader/CLI path
  offline. Interrupted-boundary tests establish exact CPU continuation.

```bash
uv run --locked --extra dev --extra diffusers pytest -q tests/test_sd3_runner.py
```
