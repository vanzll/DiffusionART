# Native FLUX Runner

FLUX uses the same CLI, reward adapters, artifact format and checkpoint lifecycle
as SD3. The native FLUX backend supports single-device DiffusionART,
[DiffusionNFT](flux-nft.md) and [Flow-GRPO/Fast/DPPO](flux-grpo.md). Other algorithms fail validation rather than
silently selecting ART. This is a
development integration, not yet an accepted paper reproduction.
ART/NFT and Flow-GRPO/Fast/DPPO also have [native DDP execution](distributed-execution.md),
including parallel evaluation and full-state resume.

## Run The Integration Example

This section uses the optional offline `configs/examples/` integration route.
Manual downloads here are not requirements for recipes that declare automatic
asset preparation; see [paper recipe setup](paper-image.md#assets-and-validation).

Run from the repository root on an authorized BF16-capable CUDA GPU. Full-model
VRAM and Linux/GPU acceptance remain unmeasured; the tiny CPU tests are not a
capacity estimate. Choose writable persistent cache/output locations first.

Accept the publisher's [FLUX.1-dev license and access conditions](https://huggingface.co/black-forest-labs/FLUX.1-dev).
Use the existing Hugging Face credential store or `HF_TOKEN`, never credentials
in config files. The revision below was resolved from publisher metadata; it is
not asserted to be the exact asset revision of historical paper runs. The loader
checks the configured VAE affine transform and scheduler against downloaded assets.

```bash
uv sync --locked --python 3.10 --extra diffusers
uv run --locked --extra diffusers visual-rl --config configs/examples/flux_pickscore.yaml --validate
uv run --locked --extra diffusers hf download black-forest-labs/FLUX.1-dev --revision 3de623fc3c33e44ffbe2bad470d0f45bccf2eb21 --include 'model_index.json' 'scheduler/*' 'transformer/*' 'vae/*' 'text_encoder/*' 'text_encoder_2/*' 'tokenizer/*' 'tokenizer_2/*' --dry-run
```

Inspect the download size, then repeat the last command without `--dry-run`.
Prepare the pinned PickScore and processor assets using the download commands
in [the SD3 guide](sd3-runner.md#install-and-prepare), excluding its SD3 download.
The training command uses offline asset access:

```bash
uv run --locked --extra diffusers visual-rl --config configs/examples/flux_pickscore.yaml
uv run --locked --extra diffusers visual-rl --config configs/examples/flux_pickscore.yaml --resume outputs/flux-pickscore-integration/checkpoints/latest.json
```

The example deliberately uses P=2, K=2 and four updates for integration. It uses
six sampling steps, all six replay positions and 28-step evaluation. These are
not a substitute for the paper's data, rollout budget or exact LoRA targets.
All evaluation images are saved at updates 0/2/4, checkpoints at 2/4.
JSONL metrics, raw versus fixed-normalized rewards and GPU-hour definitions are
the same as [SD3](sd3-runner.md#train-and-inspect). [W&B export](logging.md)
and [DDP execution](distributed-execution.md) use the common runtime.

## Numerical Contract

- FLUX's CLIP pooled output and T5 tokens are frozen, detached and repeated in
  prompt-major order. Guidance 3.5 is an embedding, not two-branch SD3 CFG.
- The sigma grid uses packed image-token count and the asset's resolution shift.
  Unsupported alternate grids are rejected. SD3's rational shift is not reused.
- In BF16 rollout, each Euler transition is computed in FP32, then the next
  latent is rounded back to BF16. FP32 transition means remain separately cached.
- Historical rollout casts scheduler timesteps to the latent dtype before
  division by 1000. Update replay uses FP32 timesteps and BF16 latent inputs.
  These distinct paths are intentional; changing them can change initial drift.
- ART reconstructs old/current velocity from the FP32 Euler means, preserving
  the historical subtraction/division rather than replacing it with a nominally
  equivalent direct prediction. Two disjoint updates retain the original
  denominator and configured mask/restoration semantics.
- BF16 training rounds covariance directions to the rollout latent dtype before
  loss normalization, as in the historical FLUX implementation. This rounding
  can affect corrective decisions near the mask boundary.
- The inverse VAE affine uses rollout precision before casting to FP32 for
  decoding, matching the historical image path. VAE and neural scorers remain
  frozen FP32; trainable policy weights remain
  FP32 with BF16 forward autocast. No CPU offload or quantization is implicit.

## Offline Verification

```bash
uv run --locked --extra dev --extra diffusers --extra image-rewards --extra ocr pytest -q tests/test_flux_contracts.py tests/test_flux_text.py tests/test_flux_runner.py
```

Tests create small real Diffusers/Transformers assets locally and prohibit
network access in CLI tests. They exercise text encoding, policy gradients,
VAE/PickScore scoring, update/evaluation logs and checkpoint continuation.
Pretrained asset acceptance, historical distributed sampling parity and actual
paper-curve reproduction remain release requirements, not claims of these tests.
