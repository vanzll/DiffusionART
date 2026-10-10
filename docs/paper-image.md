# Paper Image Training Candidates

These are **training_candidate_not_fullhistorical_reproduction** profiles, not
certified paper reproduction recipes. They use the existing native image
backends and pinned release-example assets. No training was launched.

## Coverage

| Model / benchmark | Methods | Source runs |
| --- | --- | --- |
| SD3.5-M / geneval | ART, GRPO, Fast, DPPO, NFT EMA, NFT on-policy, Dance | 11pq0iy3, wxissl19, kj09p7ee, ienj38uu, 7zirilwj, u64dtm5f, 2fryhvvf |
| SD3.5-M / ocr | ART, GRPO, Fast, DPPO, NFT EMA, NFT on-policy, Dance | 9c0zrzlw, x4cywydr, g3949kwe, x76d50t3, uxe86tf6, s408j3eh, 92ahoxm2 |
| SD3.5-M / pickscore | ART, GRPO, Fast, DPPO, NFT EMA, NFT on-policy, Dance | mofumvbw, dosfn01k, 0qvzsp36, gzqhxeg0, pm9ouohi, 6wvdti47, bt84xzy0 |
| SD3.5-M / four rewards | ART, GRPO, Fast, DPPO, NFT EMA, NFT on-policy, Dance | 3t35bwps, nv2hv2l1, ye14mp5t, dvmnd177, ujnglsgl, ldpbaea3, gpxub5g5 |
| FLUX.1-dev / GenEval | ART, Fast, DPPO, NFT EMA | z9gkdemj, dmrybr9d, yghomcbn, asge8y6v |
| SD3.5-M / OCR restoration | ART one-update ablation | v4nt8roi |

FLUX NFT EMA `asge8y6v` is recovered from a separate archived W&B response and
its actual runtime `98d76252de5f194df39a6441f33ddf7e2843e06d`. The compact archive
alone does not contain it. Its dedicated runtime freezes P48/K24, 144 rows per
rank, one full-rollout update, five of six fresh-noising positions, BF16,
rollout/update microbatch six and independent old/evaluation shadows.

## Assets And Validation

The four-reward profiles use a separate
[setup and score contract](multireward.md), with HPSv2.1 rather than aesthetic
prediction. Their seven recipes have their own allowlisted W&B evidence in
`paper-multireward-evidence.json`. Main-paper plotting stops at 500 steps;
the candidate training budget remains 1000. Historical baseline runtime source
is not recovered; reference mappings are explicitly marked, not certified.

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
its own run. GenEval/OCR still need their task-specific worker setup and benchmark
manifests; see [GenEval](geneval.md), [OCR](sd3-ocr.md) and
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
  Supply the full benchmark manifests and their actual SHA256 values, not the
  small example manifests. Historical data hashes are not recoverable here.
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

All candidates stop at **1000 successful optimizer updates**, the plotted
horizon. Explicit archived `paper_max_wandb_step=1000` is preserved; legacy
`num_epochs=100000` is not treated as the release budget. Historical W&B
zero-based indices, overflow skips and 1002-step overshoot are not certified.

ART and the three recovered FLUX runs use archived evaluation updates
500/800/1000 and an explicit checkpoint at 1000. Evaluation at zero remains enabled by the
nonempty update list, and enabled evaluation includes the final update.
Legacy baselines convert `eval_freq=30` epochs into candidate update cadence:
60 updates for two-update GRPO-family methods, 30 for one-update NFT.
Image paper recipes also save a complete training checkpoint every 200 successful
optimizer updates for recovery. Archived explicit save points are retained, so
some recipes save more frequently. Evaluation cadence remains unchanged.

Global prompt count is derived per run as
`P = train_batch_size * num_batches_per_epoch * 8 / K`, not copied from a
neighboring method. Most GenEval/GRPO profiles use P48/K24; OCR ART uses P64/K16,
and PickScore ART/Fast/NFT use P64/K18. Training AdamW learning rate, betas,
epsilon, weight decay, clipping, precision and TF32 follow each archived run.

GRPO-family candidates use two disjoint updates and archived logical batch
counts (16 SD3, 48 FLUX). Typical historical SD3 accumulation was eight batches.
`sd3_ocr_art_one_update.yaml` inherits the OCR ART profile and explicitly sets
`algorithm.options.updates_per_rollout=1`. Its archived `v4nt8roi` accumulation
is sixteen rather than eight: one whole rollout window per successful update,
with the same restoration loss, P64/K16 and artifact cadence. The 1000-update
budget therefore consumes twice as many rollouts as the two-update profile.
Runtime source and exact ordering parity remain unresolved for this ablation.
NFT defaults to one full-rollout update (normally equivalent to sixteen source
batches), with old-policy decay separate from evaluation EMA. **OCR NFT EMA
`uxe86tf6` differs:** beta 0.1, decay type 2, accumulation eight and two disjoint
updates. Its config explicitly enables two windows; the old shadow remains
frozen through both, then refreshes once at the completed rollout boundary.
Evaluation EMA refreshes after each successful update. Its evaluation cadence
is 60 updates, not the one-update methods' 30.

## Semantic Gaps

- Historical `DistributedKRepeatSampler` shuffled mixed rows across ranks.
  Current candidates use prompt-major local complete groups. Selection,
  filtering/padding, gradient accumulation order and RNG parity remain pending;
  valid backend configuration is not reproduction evidence.
- NFT decay types 0/1/2 map to existing old-policy rate/maximum/delay using
  `scripts/train_nft_sd3.py::return_decay` at research
  `3ca2a36f6b20c53a06813013234666d9c033a660`. EMA defaults come from that
  source, not invented extra archived fields. Exact historical EMA timing and
  successful-update versus epoch indexing remain unverified.
- SD3 DPPO sources resumed at 700/500/800 updates. Candidates start at base
  weights, not those unavailable predecessor checkpoints.
- Dance A2 declares `prompt_global_std`, ratio clip 1e-5, eta 0.8, K24,
  shared per-prompt initialization and nine-of-ten per-row selections.
  `last_x0` and the kernel are native DanceGRPO 15cc71d adaptations.
  Source commits were recovered for GenEval/OCR (`aa3cb43c...`) and PickScore
  (`db5aecf8...`), but their SD3 code bodies remain unavailable. Endpoint,
  terminal eligibility and exact sharing/order are not historically certified.
- Fast CPS uses the source branch's reduced negative-MSE likelihood even when
  archived `logprob_type=gaussian`: the research CPS branch ignores that label.
  SD3 window [0,5) and FLUX window [0,3) are preserved.
- Asset pins, text length, LoRA module choices and physical decode/worker
  batching inherit release examples. Historical asset hashes, data hashes and
  full environment identity remain unresolved.
- Full-budget CPU/GPU execution, clean-environment acceptance and paper-curve
  equivalence have not been established.

## Evidence Tests

`tests/test_paper_image_configs.py` validates the original 26 backend configurations
against [the bundled scientific evidence](paper-image-evidence.json), extracted
from archived W&B configs independently of the release YAML. Only explicit
scientific fields are included; full source snapshots are identified by SHA256.
Normal checks require no private archive. Optional `PAPER_IMAGE_ARCHIVE` and
`PAPER_FLUX_NFT_ARCHIVE` point to original snapshots for further cross-checking;
only that optional original-file check skips when absent. No credentials or
private host paths are stored in the bundled evidence or profiles. The seven
four-reward candidates are separately checked by
`tests/test_paper_multireward_configs.py`, including sum-reporting and the
algorithm-specific normalization branches.
