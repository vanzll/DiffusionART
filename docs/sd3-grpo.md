# Native SD3 GRPO Family

The `sd3` backend supports Flow-GRPO, Flow-GRPO-Fast, Flow-DPPO and DanceGRPO through the
same entrypoint as ART/NFT. These are single-device integration examples, not
paper recipes. Install and prepare assets using [the SD3 guide](sd3-runner.md).

```sh
uv run --locked --extra diffusers visual-rl --config configs/examples/sd3_fast_pickscore.yaml --validate
uv run --locked --extra diffusers visual-rl --config configs/examples/sd3_fast_pickscore.yaml
uv run --locked --extra diffusers visual-rl --config configs/examples/sd3_fast_pickscore.yaml --resume outputs/sd35-fast-pickscore-integration/checkpoints/latest.json
```

Use `sd3_grpo_pickscore.yaml` for full Flow-SDE sampling or
`sd3_dppo_pickscore.yaml` for the KL-advantage mask objective. Each config has a
separate output directory. They inherit the same assets; algorithm-specific
options remain explicit and unsupported settings are rejected.

## Sampling And Updates

- **Flow-GRPO:** stochastic Flow-SDE throughout the rollout, Gaussian transition
  likelihood, clipped importance ratio. The example trains the first nine of
  ten transitions; the final transition is sampled but not trained.
- **Fast:** select a contiguous window or random subset per logical sampling
  batch. Only selected transitions are stochastic with `unselected_noise: ode`.
  The example uses CPS with reduced negative-MSE likelihood, not Gaussian
  normalization. SDE with an explicit likelihood is a separate configuration.
- **Flow-DPPO:** the equal-variance transition KL and advantage/ratio sign gate
  replace PPO ratio clipping. Advantages are clipped before this objective.
  The example uses the same CPS rollout as Fast.
- **DanceGRPO:** `sd3_dance_pickscore.yaml` uses the native Dance flow-SDE
  equation at upstream revision `15cc71d`, Gaussian likelihood, full stochastic
  exploration (including the last transition), independent per-row timestep
  shuffles and `endpoint_kind: last_x0`. Reward sees the final pre-transition
  extrapolated endpoint, while likelihood replay retains the sampled next state.
  `group_sample_std` uses complete K groups, unbiased sample std and `1e-8`;
  `prompt_global_std` is the separately named historical image adaptation.
  The short example uses the former and disables evaluation EMA. Shared initial
  noise is per prompt, not shared across unrelated prompts. This is an explicit
  SD3 adaptation of the upstream kernel, not an asserted copy of the original
  upstream training recipe. FLUX Dance is not registered.

Replay uses each row's exact sampled next state and selected timestep. Selected
timesteps may differ across logical sampling batches. A terminal CPS transition
with zero variance cannot be selected for training; its final deterministic
transition is explicit. Sampling settings are stored with the run.

`sampling.options.logical_batch_size` controls timestep-selection sharing.
`sampling.microbatch` controls physical model execution only. Innovations are
drawn before physical partitioning. Changing the latter does not select new
timesteps or draw different noise. Floating GPU kernels can still differ with
batch shape; this is not a bitwise GPU invariance promise.
In the CPU BF16 regression, a singleton tail changed velocity by up to 0.00390625
and CPS log-probability by 2.20e-7; direct model calls reproduced the difference,
while equal-sized replay and the tested batch sizes 2/4 matched exactly. Do not
generalize those tolerances to pretrained GPU models. Physical-batch changes
need numerical acceptance before being used for a paper reproduction.

`training_batches` fixes the number of logical data batches, while
`updates_per_rollout` sets optimizer boundaries. The logical batch count is
retained after zero-advantage filtering and random zero-row padding. Loss is
divided by each resulting logical update window's item count, not the number
of physical microbatches. Subsequent DPPO masking does not shrink this
denominator. `retain` is an explicit alternative to `filter_pad`.

All-zero batches in `filter_pad` mode do not advance optimizer or EMA. They
consume real training/GPU time and log attempt counters at the unchanged
optimizer step. `max_empty_rollouts` bounds consecutive empty rollouts and
fails explicitly instead of spinning indefinitely. It is not a limit on valid
updates or a claim that all infrastructure failures are recoverable.

## Evaluation EMA

Rollouts use current policy weights; there is no NFT old-policy EMA. Evaluation
EMA is optional and separate. The reference image loop refreshes it after every
logical data batch, including during gradient accumulation. `refresh:
logical_batch` preserves that behavior: pre-boundary refreshes use unchanged
parameters and the current update count, then the last refresh uses updated
weights and the incremented count. Decay is
`min((1+completed_updates)/(10+completed_updates), decay)`.

`refresh: optimizer` is a distinct explicit setting. Neither mode keys EMA to
physical microbatch size. Full checkpoints include EMA state. Evaluations use
deterministic Euler sampling and independent seeds.

## Verification Scope

Migration references at research commit
`3ca2a36f6b20c53a06813013234666d9c033a660`: `scripts/train_sd3_fast.py`,
`scripts/train_sd3.py`, `flow_grpo_alg/counterfactual_transitions.py`,
`flow_grpo_alg/ema.py` and `flow_grpo_alg/diffusers_patch/sd3_*logprob*.py`.
The private repository is not an installation dependency.

Tests cover transition equations, old-likelihood replay, row-specific selection,
microbatch-independent RNG, filtering/padding, EMA refresh, real small SD3
training and exact interrupted resume. Historical Python/Torch random streams
and distributed sampler ordering are not asserted identical. GPU and full paper
result reproduction, reference KL penalties, remaining reward models and
distributed runtime still require integration and acceptance.

```sh
uv run --locked --extra dev --extra diffusers pytest -q tests/test_grpo_sampling.py tests/test_native_grpo.py
```
