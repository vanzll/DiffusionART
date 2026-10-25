# Paper Image Training Candidates

These are **training_candidate_not_fullhistorical_reproduction** profiles, not
certified paper reproduction recipes. They use the existing native image
backends and pinned release-example assets. No training was launched.

## Coverage

| Model / benchmark | Methods |
| --- | --- |
| SD3.5-M / geneval | ART, GRPO, Fast, DPPO, NFT EMA, NFT on-policy, Dance |
| SD3.5-M / ocr | ART, GRPO, Fast, DPPO, NFT EMA, NFT on-policy, Dance |
| SD3.5-M / pickscore | ART, GRPO, Fast, DPPO, NFT EMA, NFT on-policy, Dance |
| SD3.5-M / four rewards | ART, GRPO, Fast, DPPO, NFT EMA, NFT on-policy, Dance |
| FLUX.1-dev / GenEval | ART, Fast, DPPO, NFT EMA |
| SD3.5-M / OCR restoration | ART one-update ablation |

FLUX NFT EMA freezes P48/K24, 144 rows per
rank, one full-rollout update, five of six fresh-noising positions, BF16,
rollout/update microbatch six and independent old/evaluation shadows.

All four FLUX profiles use the same 12 LoRA target suffixes, rank 32 and
alpha 64, including context-attention and feed-forward projections.
Regression tests check the exact targets and identical trainable parameter
inventories on a small native FLUX model. Per-method update counts, EMA,
learning rates and mask settings are unchanged; this is not a new GPU
performance reproduction result. Adapters trained with the old four-target
config require that original config to resume.

## Assets And Validation

The four-reward profiles use a separate
[setup and score contract](multireward.md), with HPSv2.1 rather than aesthetic
prediction. Their candidate training budget is 1000 successful optimizer updates.
Historical runtime and result parity are not certified.

### Automatic PickScore Entry Point

The seven SD3 PickScore paper recipes inherit `configs/paper/pickscore-assets.yaml`.
After accepting the base-model license and configuring Hub/W&B credentials,
run the selected recipe directly:

```bash
uv run --locked --extra diffusers visual-rl \
  --config configs/paper/image/sd3_pickscore_art.yaml --nproc-per-node 8
```

The parent prepares the pinned HF base model, reward model and processor, plus
the configured train/eval prompt lists, before launching workers. Existing
downloads are reused and missing files are downloaded automatically. Neither
manual `hf download` nor model/manifest environment variables are required for
this route. Set `VISUAL_RL_ASSET_ROOT` for a writable cache and `PAPER_OUTPUT_ROOT`
for outputs; the latter defaults to `outputs`. `--prepare-only` is optional,
while `--validate` checks the planned config without downloading assets.

### Other Benchmarks And Local Overrides

Each YAML inherits its integration recipe and overrides scientific values from
the selected training contract. GenEval/OCR prompts are downloaded from the pinned
[HF dataset](https://huggingface.co/datasets/vanzl/DiffusionART-Prompts).
Their public Flow-GRPO splits preserve annotations and repeated evaluation rows;
they are not certified byte-identical copies of private historical data.
GenEval/OCR still need their task-specific worker setup; see
[GenEval](geneval.md), [OCR](sd3-ocr.md) and
[four-reward setup](multireward.md). The four-reward recipe also prepares its
declared HF model/scorer snapshots automatically, but not the separate GenEval
worker bundle. Recipes without automatic asset declarations use the local-asset
setup in their backend guide.

The following bindings apply to those task-specific setups or optional local
overrides of automatically prepared assets, not mandatory manual downloads for
the PickScore Quick Start:

- `SD3_MODEL` or `FLUX_MODEL`: local snapshots of the model revision pinned in YAML.
- `PICKSCORE_MODEL`, `PICKSCORE_PROCESSOR`: local pinned model/processor snapshots.
- `GENEVAL_MANIFEST`, `GENEVAL_MANIFEST_SHA256`, `GENEVAL_PYTHON`: detector worker assets.
- `OCR_MANIFEST`, `OCR_MANIFEST_SHA256`: OCR worker assets.
- For each selected benchmark, `GENEVAL`, `OCR`, or `PICKSCORE` prefix:
  `*_TRAIN_MANIFEST`, `*_TRAIN_SHA256`, `*_EVAL_MANIFEST`, `*_EVAL_SHA256`.
  These are optional overrides of the automatically prepared public splits.
  Use full benchmark manifests rather than the small example fixtures.
- `PAPER_OUTPUT_ROOT`: caller-selected writable output root.

Only variables required by the chosen inheritance chain need to be set. Model
revisions are pinned from release examples; archived revisions were `main`.
These pins do not establish historical artifact identity.

For a task-specific recipe, complete its required worker/data setup before
validation. Validation itself never downloads models or starts training:

```bash
visual-rl --config configs/paper/image/sd3_geneval_art.yaml --validate
```

Profiles select DDP. Historical comparisons used eight ranks. This document is
not a remote launcher or GPU-allocation authorization.

## Budgets And Cadence

For a training-only short check, set `evaluation.every_updates: 0` and
`evaluation.at_updates: []`. The inherited evaluation manifest, hash and sampling
protocol remain recorded but inactive. Also clear explicit checkpoint points
when shortening the update budget. Restoring the original cadence reactivates
the original evaluation protocol; a short check does not validate that lifecycle.

All candidates stop at **1000 successful optimizer updates**.
Historical W&B indexing, overflow skips and overshoot parity are not certified.

All public paper profiles evaluate every 50 successful optimizer updates,
including initial and final evaluation. Extra `at_updates` are empty.
Image paper recipes also save a complete training checkpoint every 200 successful
optimizer updates for recovery. Archived explicit save points are retained, so
some recipes save more frequently. Checkpoint cadence is independent of evaluation.

Global prompt count follows
`P = train_batch_size * num_batches_per_epoch * 8 / K`, not copied from a
neighboring method. Most GenEval/GRPO profiles use P48/K24; OCR ART uses P64/K16,
and PickScore ART/Fast/NFT use P64/K18. Training AdamW learning rate, betas,
epsilon, weight decay, clipping, precision and TF32 are explicit in each YAML.

GRPO-family candidates use two disjoint updates and archived logical batch
counts (16 SD3, 48 FLUX). Typical historical SD3 accumulation was eight batches.
`sd3_ocr_art_one_update.yaml` inherits the OCR ART profile and explicitly sets
`algorithm.options.updates_per_rollout=1`: one whole rollout window per successful update,
with the same restoration loss, P64/K16 and artifact cadence. The 1000-update
budget therefore consumes twice as many rollouts as the two-update profile.
Runtime source and exact ordering parity remain unresolved for this ablation.
NFT defaults to one full-rollout update (normally equivalent to sixteen source
batches), with old-policy decay separate from evaluation EMA. **OCR NFT EMA
differs:** beta 0.1, decay type 2, accumulation eight and two disjoint
updates. Its config explicitly enables two windows; the old shadow remains
frozen through both, then refreshes once at the completed rollout boundary.
Evaluation EMA refreshes after each successful update. Both one- and two-update
methods use the common 50-update evaluation cadence.

## Semantic Gaps

- Historical `DistributedKRepeatSampler` shuffled mixed rows across ranks.
  Current candidates use prompt-major local complete groups. Selection,
  filtering/padding, gradient accumulation order and RNG parity remain pending;
  valid backend configuration is not reproduction evidence.
- NFT old-policy rate/maximum/delay and evaluation EMA are configured separately.
  Exact historical EMA timing and successful-update versus epoch indexing remain unverified.
- SD3 DPPO candidates start at base weights; predecessor checkpoints are unavailable.
- Dance A2 declares `prompt_global_std`, ratio clip 1e-5, eta 0.8, K24,
  shared per-prompt initialization and nine-of-ten per-row selections.
  `last_x0` and the kernel are native DanceGRPO 15cc71d adaptations.
  Endpoint, terminal eligibility and exact sharing/order are not historically certified.
- Fast CPS uses the source branch's reduced negative-MSE likelihood even when
  archived `logprob_type=gaussian`: the research CPS branch ignores that label.
  SD3 window [0,5) and FLUX window [0,3) are preserved.
- Asset pins, text length, LoRA module choices and physical decode/worker
  batching inherit release examples. Historical asset hashes, data hashes and
  full environment identity remain unresolved.
- Full-budget CPU/GPU execution, clean-environment acceptance and paper-curve
  equivalence have not been established.

## Configuration Tests

`tests/test_paper_image_configs.py` validates the 26 backend configurations
against independently fixed scientific constants, without private archives or
historical run mappings. Checks retain optimizer settings, batch sizes, sampling,
LoRA targets and per-method loss settings. The seven four-reward candidates are
separately checked by `tests/test_paper_multireward_configs.py`, including
sum-reporting and algorithm-specific normalization branches.

Removing historical provenance changes the full configuration fingerprint.
Resume existing checkpoints with their original configuration; the cleaned
profiles do not bypass checkpoint identity validation.
