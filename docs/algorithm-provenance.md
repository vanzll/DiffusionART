# Algorithm Kernel Provenance

## Scope and Attribution

This document describes the pure-PyTorch kernels and their source lineage.
The kernels load no models and require no private research-repository import.
Native trainers and task-specific acceptance are documented in the runner
guides; kernel tests alone do not establish paper-result reproduction.

The implementation adapts equations and behavior from the author's research
repository, retaining attribution to its DiffusionNFT, DanceGRPO, Flow-GRPO,
Flow-DPPO, DiffusionART, and ReFL implementations. It does not claim
authorship of those algorithms or assign them a new license. Flow-GRPO's MIT
notice and DiffusionNFT's Apache-2.0 license are preserved in
`THIRD_PARTY_NOTICES.md` and `licenses/`. DanceGRPO and UniRL also declare
Apache-2.0 upstream; their native modules are not bundled here. Complete the
remaining source/asset license review before a public release.

Read-only reference: the author's research source snapshot at verified commit
`3ca2a36f6b20c53a06813013234666d9c033a660`. The independent tests do not
require access to that private repository or the author's filesystem.

## Source Manifest

Paths below are relative to the reference checkout. Hashes are full **Git blob
SHA-1 object IDs**, not abbreviated commits or filesystem SHA-256 hashes.

| Source file | Blob object ID | Functions/behavior used |
| --- | --- | --- |
| `flow_grpo_alg/covariance_guidance.py` | `ec94a152098fc88f24a71ebc86b0d4b378668fb5` | `covariance_guidance_from_grouped_rows`, `disjoint_pair_binary_rewards` |
| `flow_grpo_alg/transition_mean_guidance.py` | `0e8322dc8684744f4328025ff34c5f108c7dbd9a` | `rms_normalize_per_sample`, MSE branches and corrective masks of `x0_covariance_guidance_loss_terms` |
| `flow_grpo_alg/nft_objective.py` | `f293b6a2b8e9734d5f834e1b2999248104973ff0` | `clip_nft_advantages`, mirror/standard/normalized_x0 branch of `diffusion_nft_loss_terms` |
| `flow_grpo_alg/flowdppo_loss.py` | `c6960349c313921e1a850af9960ad3b7a20163f4` | `transition_kl_per_sample`, `flowdppo_policy_loss` |
| `flow_grpo_alg/video/backends/hunyuan_public_flowart.py` | `d5566c5f9ebd3c3de9edee5a8429a58039754643` | `frozen_covariance`, `loss_terms`, inactive-row restoring anchors |
| `flow_grpo_alg/video/backends/hunyuan_sage.py` | `909c6ee80f25e21bb4b642549fe648c0ebba6d7b` | `flow_art_endpoint_covariance_contract`, aggregate scalarization before normalization |
| `flow_grpo_alg/video/pairwise_feedback.py` | `4bec97a724936427afdbb4189dec8ebfd57e2824` | `frozen_pairwise_covariance`, tie anchors and group seed offset |
| `flow_grpo_alg/video/backends/hunyuan_public_nft.py` | `724573e630c9e432fa2ea4fb4d3f6fb69a532b6b` | `old_policy_decay`, `loss_terms`, reference-MSE dtype and detach contract |
| `flow_grpo_alg/video/backends/hunyuan_public_dancegrpo.py` | `e4b1f44f4b18ec84505d87539c9eea327261ad4e` | `loss_terms`, componentwise clipping, ratio clip 1e-4 |
| `flow_grpo_alg/video/backends/hunyuan_public_fast_sde.py` | `d53c92c01b01b35a3ea2ad6b450cb83b8517cd19` | `loss_terms`, ratio clip 1e-5 |
| `flow_grpo_alg/video/backends/hunyuan_public_refl.py` | `d01766cb2319d952dd05d969f41110391c8861b6` | `update_policy`, unnormalized negative-reward gradient contract |
| `flow_grpo_alg/video/refl_core.py` | `d9c355db753fd3e23fe9154564dfc48863a01f2f` | `reward_loss`, fixed-normalized component rewards |

The DiffusionART reference source additionally pins DanceGRPO revision
`15cc71d53cc2e6e18a68ee607d5fb6ba9a99e344`. That is source context, not a
claim that this slice validates the full upstream runtime.

The SD3 Dance adapter independently follows `fastvideo/train_grpo_flux_lora.py`
at that upstream revision: native flow-SDE mean/variance, detached observed
likelihood, and final pre-transition x0 scoring. Its upstream-style config uses
group sample std plus `1e-8`. Per-prompt shared initialization and SD3 timestep
selection are explicit adaptations. Archived SD3 A2 configs instead declare
global reward std and evaluation EMA. W&B run commit and code metadata agree on
`aa3cb43cc7c362ba24aa5296c14e39ef36582d7c` (GenEval/OCR) and
`db5aecf8fbb6e002527ef8aff09e75a53ece4312` (PickScore), but the full runtime
sources have not been recovered. These identities are not proof of A2 parity.

The local incident/progress notes and recent applicable Git diffs were inspected.
In particular, `60ba9322eb6a671c687c0be20dcd0a1d83b05c4d` added opt-in raw
zero-signal filtering. That change is present in the supplied canonical commit
and is exposed explicitly here. The covariance telemetry repair
`8bed470a627184dfc65537dd1cc0c3cbaffbb040` affects global diagnostic reduction,
which is intentionally outside these kernels. Reading incident notes is not
evidence that a remote runtime repair has been revalidated here.

## Public API Contract

All names are exported from `visual_rl.algorithms`. Losses return a dictionary
whose `per_item_loss` is a differentiable `[N]` tensor. They never divide by N,
active count, microbatch count, timestep count, or process count. Feature MSE
and KL use a **mean over all non-batch axes**, not a feature sum. The coordinator
must compute `sum(weights * per_item_loss) / chosen_global_denominator`, including
any correction required for DDP's gradient averaging. Masked zeros remain in the
chosen historical item denominator. Do not average microbatch means.

Frozen statistics must be built on complete groups before item slicing. All
rollout/reference predictions, rewards used as advantages, sigmas, guidance,
and masks are detached. Only current predictions/log probabilities carry loss
gradients, except ReFL, whose reward graph must remain differentiable.

### ART Covariance

`art_scalar_covariance(endpoints, rewards, row_group_indices, group_count, *,
reduction="unbiased", direction_mode="historical", scale_mode="unit_normalized_covariance",
guidance_gain=1., zero_signal_mode="historical", zero_signal_atol=0., eps=1e-5)`:

- Inputs are `[N, ...]`, `[N]`, integer `[N]`, and a positive group count.
  Empty/singleton groups return zero covariance and are inactive.
- Rewards are the already scalarized, fixed-normalized reward. For multiple
  components, first form `sum(weight[c] * reward[c])`; never normalize individual
  covariance directions and then add them as a substitute for this aggregate.
- Historical fp32 computation is `C = sum((r - mean(r)) * x) / max(K-1, 1)`.
  `reduction="population"` explicitly selects K instead. Reward standard
  deviation always uses the population moment formula, even with unbiased C.
- Historical direction is `S = C / MAD` (use 1 when MAD is zero), followed by
  `S / max(RMS(S), eps)`. `direction_mode="normalized_aggregate"` is a new,
  explicit convention: normalize C directly to unit RMS with a float64 norm
  for active groups. These are not bit-equivalent below the historical floor;
  the historical activity gate still applies in both modes.
- Scale modes are `unit_normalized_covariance`, `raw_covariance`,
  `reward_std_normalized_covariance`, and `reward_mad`, matching the source.
  Historical inactive groups can retain nonzero magnitude but have zero direction.
- Scalar profiles use `zero_signal_mode="exact"`: exactly constant rewards
  produce zero covariance, direction and magnitude, even when FP32 summation
  rounds their mean. Real nonconstant weak signals retain the existing reduction.
- `zero_signal_mode="threshold"` requires positive `zero_signal_atol` and
  gates both raw covariance RMS and reward range, then zeros inactive magnitudes.
  This optional kernel cutoff is rejected by native scalar runners; historical mode
  remains available for pairwise compatibility. No random replacement direction
  is ever generated.
- Outputs include `group_covariance`, `group_directions` shaped `[G, ...]`,
  `group_guidance_magnitudes`, `group_active`, counts, and reward statistics.

`art_pairwise_covariance(endpoints, scores, group_size=8, *, seed=0,
round_index=0, group_offset=0, any_preference_active=None, ...)` uses complete
contiguous K=2/4/8 groups and the same covariance options. Each group gets a
private CPU generator seeded by `(seed + 1000003 * (round_index + group_offset
+ group_index)) % (2**63-1)`. Disjoint pairs receive winner1/loser0, ties0.5/0.5.
The global torch RNG is unchanged. Outputs additionally include `pair_rows`,
`binary_rewards`, and `row_preference_active`, `row_active`,
`row_guidance_magnitudes`, `row_restore_anchor`, `row_trainable`.
The coordinator must supply the globally OR-reduced `any_preference_active`
for distributed tie anchors; None means local activity only. The coordinator
also owns all-inactive rollout skipping and optimizer-budget accounting.

`training/video_pairwise.py` connects these rules to Hunyuan ART. The FSDP
coordinator MAX-reduces preference/guidance/tie activity outside local-error
callbacks, then finalizes anchors and the shared skip decision. Both windows
retain the original row/item denominator. Pairwise is an explicit opt-in;
scalar ART and the other video algorithms do not enter this branch.

### ART Target and Mask

`art_loss(current_velocity, old_velocity, sigma, guidance_direction,
guidance_magnitude, *, ...)` consumes per-item `[N, ...]` predictions/directions
and `[N]` scalars. Defaults match DiffusionART: `velocity_linear_sigma`, RMS
direction normalization, velocity-space `scheduler_reference_metric`, enabled
corrective mask, reference noise0.8 and KL threshold1e-5. The default metric
requires a negative `ode_velocity_coefficient[N]`.

For velocity MSE, `target = stopgrad(old_velocity - magnitude * direction)`;
the whole feature-mean squared residual is weighted by sigma or, with
`loss_space="velocity_uniform"`, one. `loss_space="x0"` requires `state` and
uses `target = stopgrad(state - sigma * old_velocity + displacement)`.
Only x0 supports `target_projection_radius`; direction normalization may be
`rms` or `none`. Optional `reference_prediction` and `reference_mse_beta` add
unmasked reference MSE, matching the backend's regularizer.

The mask removes active rows only when movement energy is >= threshold AND
`mean((current-old) * desired_corrective_step) > 0`. Overshoot corrections are
therefore retained. Explicit `restore_anchor_mask` retains requested inactive
rows with positive sigma. Image scalar runners require `restore_zero_signal=false`:
zero-signal rows contribute neither guidance nor restoration, but retain their
original loss denominator. There is no refill. Globally all-zero scalar rollouts
skip optimizer updates without advancing training steps; their actual cost is
counted. A locally inactive rank still executes both full update windows when
another rank has signal. Video paper recipes instead set `restore_zero_signal=true`
to preserve the historical old-policy anchors; anchor-only windows are not skipped.
Both choices use the original denominator and exact constant-reward protection.
Pairwise callers retain `row_restore_anchor` and the
historical covariance mode. Disabling the mask does not
implicitly activate zero-guidance rows or change the denominator.

Metrics are `fixed_mse`, `noise_normalized_v`, `q2_sde_kl`, or the equivalent
`scheduler_reference_metric`. Q2 uses initial sigma approximately1: std=noise,
coefficient=dt; otherwise std=sqrt(noise^2*sigma/(1-sigma)*(-dt)) and
coefficient=dt*(1+noise^2/2). Energy is coefficient^2*MSE/(2*std^2), with source
epsilon floors. `mask_space="ode_mean"` requires `fixed_mse`, both
`current_ode_mean` and `old_ode_mean`, and dt. Extra outputs include `target`,
`keep_mask`, `masked`, unmasked loss, policy loss, and reference MSE.

### NFT

`nft_loss(prediction, old_prediction, x0, xt, sigma, advantages, *, nft_beta=1.,
adv_clip_max=5., adv_mode="all", reference_prediction=None, reference_mse_beta=0.,
eps=1e-5)` implements only mirror/standard/normalized_x0. Each branch uses
`positive = beta*current + (1-beta)*old` and
`negative = (1+beta)*old - beta*current`. Endpoint residuals are divided by
their **own detached float64 mean absolute residual**, floored at eps. With
`p = clamp(clip(advantage)/adv_clip_max/2 + .5, 0, 1)`, the per-item loss is
`adv_clip_max * (p*positive_loss/beta + (1-p)*negative_loss/beta)`.
The source permits finite nonzero beta, including negative beta. Optional
advantage modes are `positive_only`, `negative_only`, `one_only`, and `binary`;
these transform advantages, not the loss into a one-branch-only variant.
Reference MSE preserves fp32 reference dtype even with bf16 predictions.

`old_policy_decay(step, mode, rate=.00125, maximum=.6)` uses a zero-based
completed-update index: first refresh is0, then `min(step*rate, maximum)` for
`slow-ema`, or always0 for `no-ema`. The coordinator performs the refresh
**after** optimizer.step: `old = decay*old + (1-decay)*current`. The kernel
does not mutate a shadow model or parameter tree. Image NFT has a separately
configured, one-based rate/maximum/delay schedule. The optional two-window
image recipe freezes the same old policy through both windows and refreshes
once at the complete-rollout boundary; its evaluation EMA updates per step.

### GRPO, FlowDPPO, ReFL

`clipped_grpo_loss(log_prob, old_log_prob, advantages, *, ratio_clip,
advantage_clip=5., component_weights=None)` requires `[N]` log probabilities
already reduced according to the sampler. `ratio_clip` is deliberately required:
Video DanceGRPO uses1e-4, Fast-SDE uses1e-5; SD3 Dance paper candidates retain
their archived1e-5 setting. The equation is
`max(-A*exp(new-old), -A*clamp(exp(new-old), 1-clip, 1+clip))`.
Mapping advantages require matching explicit weights, and each component is
clipped **before** weighted summation. Tensor advantages mean scalarized input.

`transition_kl_per_sample(current_mean, old_mean, transition_std, eps=1e-8)`
returns feature-mean `(current-old)^2/(2*max(std,eps)^2)` in fp32. Supply the
actual Gaussian transition std (including sqrt(-dt) where applicable), not the
scheduler's unscaled diffusion coefficient. Scalar, per-item and featurewise
broadcast stds are supported. Old means/stds are frozen.

`flowdppo_loss(log_prob, old_log_prob, current_mean, old_mean, transition_std,
advantages, *, kl_mask_threshold=1e-5)` returns `-A*ratio`, except zero when
KL>=threshold AND either `(ratio>1 AND A>0)` or `(ratio<1 AND A<0)`.
It does not clip ratios or advantages. Equality at ratio1 is retained.

The FLUX adapter follows `geneval_flux_flowdppo_two_updates_8gpu` and the
`flowdppo_enabled` branch of `scripts/train_flux.py` at research revision
`3ca2a36f6b20c53a06813013234666d9c033a660`: first-three-position SDE, Gaussian
likelihood, noise0.7, ODE elsewhere, KL threshold1e-5, two disjoint updates,
and no evaluation EMA. The trainer clips advantages to +/-5 before calling
the loss. This is the paper's controlled FLUX transfer, not an assertion that
all upstream Flow-DPPO profiles use these hyperparameters. The short integration
example changes benchmark/budget; the full GenEval paper profile is separate.

`refl_loss(rewards, component_weights, *, reward_loss_scale=1.)` accepts matching
nonempty component mappings of differentiable `[N]` rewards and finite,
nonnegative weights summing to1. It returns negative weighted reward; no group
centering, detachment, normalization or reward-model execution is inserted.
Generic component names are allowed rather than hardcoding VQ/MQ/TA.

## Deliberately Not Implemented

Alternative ART estimators (finite tilt, all-pairs normalized directions,
pair-local guidance, state-specific/posterior covariance), linear-guidance and
lateral-restoration objectives, ART PPO clipping, NFT alternative objectives,
NFT corrective masking, endpoint/reference KL variants, and one-branch-only
NFT loss modes are not implemented. Unsupported loss/mask mode names fail,
rather than silently falling back. `normalized_aggregate` and population
covariance are selectable extensions, not claims of canonical video defaults.

Group/global advantage standardization, reward calibration, transition
sampling/log densities, noise schedules, cache replay, selected timesteps,
distributed sufficient statistics, model/EMA storage, checkpointing, reporting,
and inactive-rollout skip policies belong to integration. The ReFL kernel supplies
the differentiable reward objective; the connected video runners implement the
staged VAE/reward-model VJP pipeline. No full benchmark, convergence, paper
result, or GPU parity is claimed by these kernel tests.

## Validation

Initial kernel acceptance: 50 CPU tests passed with Torch 2.13.0 / pytest 9.1.1
and Torch 2.14.0 / pytest 9.1.1. These kernel checks use no pretrained model
or GPU. Runner guides separately describe native and distributed integration
evidence; pretrained acceptance remains pending.

`tests/test_algorithms.py` uses independent numeric values and analytic gradients,
not research imports. It covers reduction conventions, four scale modes, detached
covariance, zero signal, seeded ties, global anchors, target/mask behavior,
NFT branch normalization and mixed precision, clipping, KL and reward gradients.
`tests/test_algorithms_optimizer.py` compares all five losses against standalone
equations using weighted uneven microbatches, an external denominator, four
AdamW transitions (parameters and moments), and post-step NFT EMA refreshes.

Additional read-only validation executed selected function ASTs directly from
the pinned Git objects in memory, verifying source blobs against checkout files.
Historical covariance matched bit-for-bit in all four scale modes. ART matched
values/gradients for three loss spaces times four velocity mask metrics; NFT
matched fp32/bf16 values/gradients with fp32 reference penalties; FlowDPPO matched
values/gradients. The independent tests remain runnable without those sources.

Test command from the release repository:

```sh
uv run --locked --extra dev pytest -q tests/test_algorithms.py tests/test_algorithms_optimizer.py
```
