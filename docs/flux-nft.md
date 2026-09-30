# FLUX DiffusionNFT

Select `model.name: flux` and `algorithm.name: diffusionnft` through the same
image CLI. The [example](../configs/examples/flux_nft_pickscore.yaml) uses
FLUX.1-dev, PickScore, LoRA and four updates. It is not a paper profile. The
[full-budget GenEval config](../configs/paper/image/flux_geneval_nft_ema.yaml)
uses P48/K24 and eight ranks, mapped to the archived `asge8y6v` runtime; see the
[paper profile guide](paper-image.md) for provisioning and remaining parity gaps.

## Commands

Prepare the pinned model/scorer assets as described in [the FLUX guide](flux-runner.md).
Its license, offline asset checks, writable storage and pending full-model GPU
capacity validation also apply here. Run from the repository root:

```bash
uv run --locked --extra diffusers visual-rl --config configs/examples/flux_nft_pickscore.yaml --validate
uv run --locked --extra diffusers visual-rl --config configs/examples/flux_nft_pickscore.yaml
uv run --locked --extra diffusers visual-rl --config configs/examples/flux_nft_pickscore.yaml --resume outputs/flux-nft-pickscore-integration/checkpoints/latest.json
```

Training metrics, evaluation files, GPU-hour accounting and strict resume
identity are shared with [the image runner](sd3-runner.md). GPU hours include
old-policy predictions and update preparation. This does not reproduce the
historical W&B runtime clock; do not splice the two cost series.

## Algorithm Contract

- The old-policy shadow generates deterministic Euler endpoints using FLUX's
  resolution-dependent schedule and embedded guidance. No two-branch CFG,
  integer SD3 timestep truncation, DPM solver or likelihood ratio is inserted.
- Each selected training timestep receives fresh standard-normal noise in
  FLUX's packed coordinate order. Endpoints and re-noised training states are
  FP32, including under BF16 transformer autocast. Rollout latents remain BF16
  between steps when BF16 precision is selected.
- Current and old predictions are cast to FP32 before forming the positive and
  negative mirror branches, matching `standard_flux_loss`. Each branch's
  squared endpoint residual is divided by its detached FP64 mean absolute
  residual (floor 1e-5). Advantage clipping and its scale factor are retained.
- Rewards are centered per prompt and divided by the rollout population's
  standard deviation plus 1e-4. Zero-advantage rows are retained. One optimizer
  update consumes the entire rollout and all selected training positions;
  physical microbatch does not change this denominator or draw new samples.
- The example selects five of six schedule positions independently per row.
  It uses old-policy decay `min(0.001 * successful_updates, 0.5)`, refreshed only
  after a completed update. The optional evaluation EMA has its own state and
  warmup, capped at 0.9. Neither mechanism is an ART EMA.
- Full local checkpoints retain optimizer, both enabled shadows, prompt cursor,
  random states and cost records. They are not weight-only restarts.

The shared framework freezes logical training inputs before physical batching.
Historical distributed samplers and microbatch-dependent random draw ordering
are not claimed bit-identical; their paper-profile acceptance remains separate.
SD3's solver, input precision and loss arithmetic are not changed by this switch.

## Tests

```bash
uv run --locked --extra dev --extra diffusers --extra image-rewards --extra ocr pytest -q tests/test_flux_nft_contracts.py tests/test_flux_nft_runner.py tests/test_native_nft.py
```

These CPU checks use locally saved tiny real models with network access disabled.
Real pretrained GPU curves, Linux deployment and multi-rank replication are
still required before this development backend is marked release-ready.
