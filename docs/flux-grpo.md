# FLUX Flow-GRPO, Fast And DPPO

The native image runner supports `flow_grpo`, `flow_grpo_fast` and `flow_dppo` with FLUX.1-dev.
The [full SDE example](../configs/examples/flux_grpo_pickscore.yaml) and
[Fast CPS example](../configs/examples/flux_fast_pickscore.yaml) share model,
reward, assets and runtime configuration. Their different samplers and update
settings are explicit config, not a substitution of ART or NFT losses.
The [Flow-DPPO example](../configs/examples/flux_dppo_pickscore.yaml) uses the
research KL-ADV transfer with first-three-position Gaussian SDE sampling.

These are four-update P2/K4 PickScore integration examples, not the paper's
multi-rank GenEval profiles. Prepare the pinned model/scorer assets and access
permissions using [the FLUX guide](flux-runner.md). Pretrained GPU capacity and
Linux acceptance remain unverified.

```bash
uv run --locked --extra diffusers visual-rl --config configs/examples/flux_grpo_pickscore.yaml --validate
uv run --locked --extra diffusers visual-rl --config configs/examples/flux_grpo_pickscore.yaml
uv run --locked --extra diffusers visual-rl --config configs/examples/flux_fast_pickscore.yaml --validate
uv run --locked --extra diffusers visual-rl --config configs/examples/flux_fast_pickscore.yaml
uv run --locked --extra diffusers visual-rl --config configs/examples/flux_fast_pickscore.yaml --resume outputs/flux-fast-pickscore-integration/checkpoints/latest.json
uv run --locked --extra diffusers visual-rl --config configs/examples/flux_dppo_pickscore.yaml --validate
uv run --locked --extra diffusers visual-rl --config configs/examples/flux_dppo_pickscore.yaml
```

## Sampling And Replay

- Full Flow-GRPO samples SDE transitions throughout the trajectory and trains
  the configured positions. The example excludes the final position from loss,
  but not from stochastic sampling.
- Fast selects a window per logical sampling batch; physical microbatches do
  not independently choose windows. The example trains the first three of six
  positions using CPS and reduced negative-MSE log-prob, without a Gaussian
  variance denominator. Other windows must be requested explicitly.
- FLUX model inputs return to the rollout latent dtype after every transition.
  Innovations are FP32 in packed FLUX coordinate order. Reference old log-probs
  are computed on the FP32 transition sample before latent rounding; replay
  uses the rounded cached next state. This intentionally preserves the historical
  implementation, not an exact probability mass for quantized states. Initial
  ratios therefore need not be exactly one in BF16.
- Rollout casts timestep before division by 1000; replay uses FP32 timesteps.
  Fast CPS outside its stochastic window uses the expanded zero-noise CPS mean,
  including deterministic evaluation. It is not silently replaced with a
  different finite-precision Euler expression.
- VAE inverse scaling uses rollout precision before the FP32 decoder cast,
  matching the reference FLUX image path. Neural reward models remain FP32.
- Flow-DPPO samples Gaussian SDE at the first three positions and Euler ODE
  elsewhere, including evaluation. It does not reuse Fast's CPS likelihood.
  Its mask uses the feature-mean Gaussian KL between current/old transition
  means divided by twice the actual transition variance, including the Euler
  time interval. High-KL updates are removed only when the ratio moves in the
  advantage-favored direction; there is no PPO ratio clipping. Advantages are
  clipped to +/-5 before the objective, matching the reference trainer.

## Update And Artifacts

Clipped-ratio loss, prompt-centered/global-population-standardized advantages,
zero-advantage filtering with padding, logical training batches and their
original denominators are shared with [the SD3 GRPO integration](sd3-grpo.md).
All three examples perform two disjoint logical update windows. GRPO/Fast have
an independent evaluation EMA refreshed with the configured logical-batch
cadence. Flow-DPPO disables evaluation EMA, as in the research FLUX transfer.
No NFT old-policy EMA is introduced. The DPPO algorithm flag changes the loss;
its SDE sampling and disabled evaluation EMA are explicit recipe settings.

The existing checkpoint path stores optimizer, enabled evaluation shadow,
prompt stream and RNG. Evaluation saves every image and score manifest. Cost
clocks include sampling/scoring/replay updates; see the [image runner](sd3-runner.md)
for exact training versus elapsed GPU-hour definitions.

```bash
uv run --locked --extra dev --extra diffusers --extra image-rewards --extra ocr pytest -q tests/test_flux_grpo_sampling.py tests/test_flux_grpo_runner.py tests/test_native_grpo.py
```

Tests use offline tiny real models, not pretrained reward-curve reproductions.
Logical RNG streams are stable under physical batching, but are not asserted
bit-identical to historical Python-random and distributed loader streams.
