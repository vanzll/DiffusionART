# Native SD3 DiffusionNFT

The same `sd3` backend now supports `algorithm.name: diffusionnft`. This is a
single-device integration of the standard mirror objective, not a certified
paper recipe. Asset installation, local output requirements and common metrics
are described in [the SD3 guide](sd3-runner.md).

```sh
uv run --locked --extra diffusers visual-rl --config configs/examples/sd3_nft_pickscore.yaml --validate
uv run --locked --extra diffusers visual-rl --config configs/examples/sd3_nft_pickscore.yaml
uv run --locked --extra diffusers visual-rl --config configs/examples/sd3_nft_pickscore.yaml --resume outputs/sd35-nft-pickscore-integration/checkpoints/latest.json
```

The four-update example reuses the pinned SD3.5/PickScore assets from the ART
example. It uses K=2 and a four-prompt integration dataset, not the paper's data
budget. Model download/access and GPU capacity remain prerequisites. Validation
checks configuration without downloading or starting a model. Native GPU and
historical-result acceptance are still pending.

## Training Contract

1. Generate K endpoints per prompt with the **old policy**, using deterministic
   DPM-Solver++ second-order multistep sampling. This is one model call per
   interval, not an RK2 solver with two calls. `euler` is a separate explicit
   choice. The schedule comes from SD3's FlowMatch scheduler with an explicit
   shift and 1000 training timesteps, not an interchangeable uniform grid.
2. Decode and score those endpoints with native PickScore. Center scores within
   each prompt text, divide by the global population standard deviation plus
   1e-4, and clip advantages inside the NFT objective. Statistics use float64;
   the resulting advantages use float32. Equal-reward rows are retained.
3. Randomly select the configured number of distinct timesteps **for each
   trajectory**, or select a prefix if explicitly configured. Draw fresh noise
   and form forward-noised endpoint states. These are not the rollout states.
4. Freeze old-policy predictions on those states. Accumulate the mirror,
   branchwise-normalized endpoint loss over the complete logical batch, then
   perform **one** optimizer update by default. The explicit
   `algorithm.options.updates_per_rollout: 2` partitions the shuffled endpoint
   rows into two disjoint windows, keeping every selected time of each row
   together. Each window retains its original mean denominator. Physical
   microbatching does not change the logical population or update count.
5. Refresh old-policy parameters only after the complete rollout's successful
   updates. Independently refresh the optional evaluation EMA after each
   successful update. In the two-window recipe, the same frozen old policy is
   used for both windows; its decay is indexed by the final completed update.
   Failed updates refresh neither state.

The default old-policy decay is `min(completed_updates * 0.001, 0.5)`, starting
at 0.001 after update 1. Set its rate and maximum to zero for a current-policy
copy. `delay` expresses an explicit initial zero-decay period. Evaluation EMA
uses `min((1 + completed_updates)/(10 + completed_updates), configured_decay)`.
These schedules must not be conflated with video NFT's different defaults.

Rollout model times follow the reference SD3 NFT integer-time convention;
re-noising/training times remain floating scheduler times. The BF16/FP16 paths use
initial rollout noise in the configured precision and round rollout states back to that dtype at each
interval. Fresh training noise is always FP32, independently of forward
precision. Integration arithmetic is FP32 with FP64 DPM coefficients.

Evaluation uses the evaluation EMA when enabled, otherwise the current policy,
and the ordinary deterministic Euler evaluation path. Evaluation never
overwrites the trainable policy or the old-policy shadow. Checkpoints save the
current model, optimizer, both applicable shadows, data cursor, sampling RNG and
progress. Resume and artifact intervals must land on a completed rollout
boundary: every update by default, every two updates for the optional two-window
recipe. The OCR NFT paper candidate uses that second recipe; ART's defaults
remain unchanged.

## Evidence and Boundaries

The migration reference is research commit
`3ca2a36f6b20c53a06813013234666d9c033a660`:
`scripts/train_nft_sd3.py`, `flow_grpo_alg/ema.py`,
`flow_grpo_alg/stat_tracking.py`, and
`flow_grpo_alg/diffusers_patch/{nft_pipeline_with_logprob,nft_solver}.py`.
These are provenance references, not runtime dependencies on the private repo.

Tests cover the solver recurrence, state selection, separate EMA schedules,
full-batch gradients, retained zero-advantage rows, actual small SD3/VAE/RM
training, and exact interrupted/resumed model and shadow state. Trainable-only
shadows avoid copying frozen LoRA base parameters or buffers.

This is not a claim of identical historical random streams: prompt ordering
and noise generation are now explicit, and unused deterministic-solver random
draws are not retained. The independent fixture establishes recurrence behavior,
not pretrained GPU result parity. Historical data/sampler ordering, asset and
environment hashes and pretrained GPU result acceptance remain pending. DDP,
W&B and the image reward adapters are connected through the common runner.

```sh
uv run --locked --extra dev --extra diffusers pytest -q tests/test_native_nft.py tests/test_nft_sampling.py tests/test_sd3_runner.py
```
