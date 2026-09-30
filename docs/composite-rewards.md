# Composing Image Rewards

The native SD3 runner can score the same decoded images with several frozen
scorers. Select `reward.name: composite`; list model assets under `components`
and literal coefficients under `weights`. ART, NFT and the GRPO family all
consume the resulting scalar without changing their update schedule.

Components include PickScore, Aesthetic, CLIPScore and an OCR adapter whose
native Paddle/Linux acceptance is pending. This interface is
not the paper's four-reward recipe: that experiment uses GenEval, CLIPScore,
PickScore and HPSv2, with per-component group-normalized advantages combined
before covariance guidance. ART's optional training-side construction is
described below; the remaining native scorers and frozen paper profile are
still required. A weighted sum of raw scores must not be used as a silent
replacement.

## Run The Example

Install the [SD3 dependencies and assets](sd3-runner.md), then prepare the
[Aesthetic assets](sd3-aesthetic.md), including `AESTHETIC_HEAD_PATH`.
From the repository root:

```bash
uv run --locked --extra diffusers visual-rl --config configs/examples/sd3_composite.yaml --validate
uv run --locked --extra diffusers visual-rl --config configs/examples/sd3_composite.yaml
```

The [config](../configs/examples/sd3_composite.yaml) inherits the four-update
SD3 integration example and replaces only the reward block and output name.
Its illustrative total is `PickScore_raw / 26 + 0.1 * Aesthetic_raw`. This is
neither a tuned paper ratio nor an empirical calibration of Aesthetic scores.
The scoring weights are not divided by their sum.

Each component pins its own model and processor revisions. There is no shared
top-level checkpoint or implicit asset substitution. Missing assets fail
offline by default. All selected scorers currently remain resident on the same device;
the composition does not promise the memory footprint of a single scorer.
Per-component `microbatch` limits each scoring call, additionally capped by
the top-level scoring microbatch. Neither value changes K or update batch size.

## Score Semantics

- `normalization: raw`: combine each scorer's native `raw` values directly.
- `normalization: fixed`: every component requires explicit `mean`, positive
  `std`, and `source` under `fixed_statistics`. The transform is
  `(raw - mean) / std`, applied once before weighting.
- Child totals and child normalization are not composed. In particular,
  PickScore cannot accidentally be divided by 26 twice.
- A zero coefficient retains the component in scoring and reporting. Its
  failures and nonfinite scores are still errors; zero is not a disable switch.
  To omit a scorer entirely, remove it from both `components` and `weights`
  (and from `fixed_statistics`, when supplied).
- Components and weight keys must match exactly. Nested composites, unknown
  options and per-child scale overrides are rejected during configuration
  validation. A component returns one score per image without broadcasting.

Training and evaluation log the total plus `reward/raw/<component>` and, when
statistics are configured, `reward/fixed_normalized/<component>`. Evaluation
sample manifests retain all component values, coefficients and normalization
provenance. Thus totals can be recomputed without guessing their units.

Config inheritance merges mappings; when removing a component, use a standalone
config without `extends` rather than assuming an omitted nested entry is
deleted. Validation detects leftover components whose weights no longer match.
The generic scalar API orders keys deterministically; this is not a claim of
bitwise equality with every historical Python-list accumulation order.

## ART Joint Advantages

`algorithm.options.multi_reward_advantage` explicitly selects a different
input to ART covariance, while leaving scalar reward reporting unchanged:

```yaml
algorithm:
  options:
    multi_reward_advantage:
      weights: {pickscore: 1.0, aesthetic: 1.0}
      score_space: fixed_normalized
      eps: 0.0001
      renormalize_combined: false
```

For each selected component, center scores by their prompt-text group mean and
divide by that group's population standard deviation plus epsilon. Combine
these advantages with the specified **positive weights divided by their sum**.
Those are algorithm-side weights, not the literal scalar-reporting coefficients
under `reward.weights`. Tied groups have mathematically zero advantage;
the implementation retains historical FP64 centering arithmetic rather than
adding an implicit tie/numerical threshold. Conflicting component preferences
may cancel in the combined guidance score.

Statistics use detached FP64 CPU values. Prompt text defines the advantage
groups over the complete rollout, not a scoring or update microbatch. The
result becomes the terminal scalar input to the existing covariance estimator;
it is not clipped, converted to pairwise labels or normalized a second time by
default. Existing covariance normalization, masks, restoration choices, row
shuffle and two disjoint update windows remain unchanged. The native runner's
covariance groups retain their prompt-slot contract; historical distributed
prompt selection/group assembly still needs separate paper-profile acceptance.

`score_space` must be explicitly `raw` or `fixed_normalized`. Fixed statistics
must exist for the latter. This distinction matters even after group
normalization because epsilon has units: `(raw/26 - mean(raw/26)) / (std(raw/26)
+ eps)` is not exactly `(raw - mean(raw)) / (std(raw) + eps)`. The original
PickScore wrapper used `/26`; do not silently replace that input with raw logits
when binding the four-reward paper recipe.

`renormalize_combined: true` explicitly subtracts the global combined mean and
divides by global population std plus epsilon after combining. The inspected
paper four-reward config sets this to **false**. No such renormalization is
implied when the option is absent. Algorithm diagnostics use
`train/multi_reward/*`; `train/reward` and eval scores retain their ordinary
scalar meaning. Other algorithms reject this ART-specific option.

After preparing the same assets as the composite example:

```bash
uv run --locked --extra diffusers visual-rl --config configs/examples/sd3_joint_guidance.yaml --validate
uv run --locked --extra diffusers visual-rl --config configs/examples/sd3_joint_guidance.yaml
```

This two-component example tests the training path; it does not substitute
Aesthetic for GenEval/CLIPScore/HPSv2 in a claimed paper reproduction. Omitting
`multi_reward_advantage` preserves the original weighted-scalar ART behavior.

The reference is research `scripts/train_sd3.py` at
`3ca2a36f6b20c53a06813013234666d9c033a660`, blob
`c14c702a91cd7851e38adab3e3d126dc3abd722e`: functions
`group_mean_group_std_advantages`, `multi_reward_group_normalized_advantages`
and `covariance_guidance_scores`. The portable implementation sorts component
keys for deterministic accumulation; this may differ in the final FP64 bits
from historical dictionary insertion order. It does not insert a numerical
zero threshold into these formulas. Nonfinite calculated statistics fail
explicitly rather than being clamped or silently used for training.

## Verification Boundary

Tests run the scorers through actual offline tiny-model loaders and the CLI
for ART, NFT, Flow-GRPO, Fast and Flow-DPPO. They check train/eval totals against
explicit formulas and preserve component scores in manifests. These tests
establish integration behavior, not pretrained GPU reward or paper-curve
reproduction. No differentiable ReFL composition is implied by this no-grad
image scorer.
