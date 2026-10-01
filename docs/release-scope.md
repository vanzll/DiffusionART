# Visual Diffusion RL: Release Scope

Development snapshot. The standalone training package and configuration-driven
entrypoint are implemented; pretrained GPU acceptance and exact historical
paper-result reproduction remain pending. This document tracks the intended
coverage and release gates. Repository names and published URLs remain unchanged.

## Boundaries

- Modify this code-release repository only. Read the research repository and
  paper sources without changing them or running experiments.
- Build reusable T2I, T2V and I2V infrastructure, with DiffusionART as one method.
- Reproduce every main-paper experiment, including motivation probes,
  restoration ablations, cover comparisons and video pairwise K experiments.
- Include video ReFL explicitly, irrespective of its final paper placement.
- Do not change repository visibility, names, published links or W&B permissions
  as part of this planning phase. Keep website development separate.

## Required Coverage

| Surface | Model / benchmark | Required methods or experiments |
| --- | --- | --- |
| T2I | SD3.5-M; GenEval, OCR, PickScore | DiffusionART, Flow-GRPO, Fast/CPS, DanceGRPO, Flow-DPPO, NFT EMA and on-policy |
| T2I | FLUX.1-dev; GenEval | DiffusionART, Fast, Flow-DPPO; NFT EMA implementation and pending result verification |
| Mechanisms | Main-paper motivation and restoration figures | Branch, gradient and update-structure probes; final panel membership must be frozen against the live manuscript |
| T2V | HunyuanVideo-13B | DiffusionART, Fast-SDE, DanceGRPO, NFT EMA and on-policy; matched-reward cover comparison |
| T2V pairwise | HunyuanVideo-13B, VQ-only | Scalar K8 and pairwise K2/K4/K8; binary/tie feedback semantics |
| I2V | Paper's distilled Wan-14B profile | DiffusionART, Fast-SDE, DanceGRPO, NFT EMA |
| ReFL | T2V and I2V | Actual final-step differentiable reward training profiles |

`reproduction-inventory.json` records candidate run identities and known runtime
commits. It does not assert that all runs are publicly accessible or that the
listed code has passed a new reproduction test.

The available manuscript snapshot predates later ledger amendments. Freeze the
live Arxiv manuscript and its actual figure/table inputs before declaring the
inventory complete. Figure numbering alone is not a stable experiment identity.

## Two Reproduction Contracts

**Paper profiles** preserve the settings actually used for each reported result:
source revision, model/reward assets, prompts and seeds, sampling, optimizer,
update partition, precision, reward preprocessing, evaluation and plotting.

**Upstream-reference profiles** document the reference repository revision,
equations and supported adaptations. A paper-specific adaptation must not be
silently relabeled as the exact original recipe. Fixes that change scientific
behavior get a separate profile rather than rewriting historical results.

Important distinctions to preserve:

- T2I Fast/CPS and video Fast-SDE are different sampler profiles.
- NFT old-policy predictions, EMA schedule and update timing are algorithmic
  state, not interchangeable logging or optimizer utilities.
- Main DiffusionART profiles retain their historical mask. No-mask, numerical
  covariance guards, zero-guidance restoration and refill are explicit variants.
- Pairwise grouping, aggregation, normalization and tie values must match the
  actual experiment. Do not substitute scalar magnitudes for binary feedback.
- Video ReFL uses a differentiable reward path and a separate common MP4 scoring
  path. Label the UniRL-style final-step adaptation; do not claim identity with
  every term of the original ImageReward/ReFL objective.
- Historical video evaluation FPS and retained-video MQ rescoring differ across
  runs. Preserve those facts and inputs in paper-reproduction metadata; provide
  the corrected common evaluator separately for new experiments.
- Preserve raw versus normalized metrics, plotted horizons, smoothing and
  threshold calculations. Elapsed GPU-hours and training-only GPU-hours are
  separate measures; physical cost includes failed attempts when applicable.

## Framework Shape

Keep shared facilities small and algorithm loops explicit:

- `algorithms/`: pure losses, guidance, advantages, masks and policy-state rules.
- `models/`: model, scheduler, conditioning and VAE adapters.
- `rewards/`: scalar/pairwise scorers and differentiable video adapters.
- `training/`: rollout/update orchestration, distributed execution and precision.
- `evaluation/`: frozen datasets, media identity, scoring and HTML galleries.
- `configs/`: general examples plus immutable paper profiles.
- `tests/`: mathematical, adapter, integration and replay tests.
- `docs/`: method contracts, extension guide and reproduction index.

Do not force ReFL's differentiable path into an MP4-only reward interface, or
force all algorithms to share one rollout/update schedule. Start from the
working implementations; introduce abstractions only where they remove real
duplication without hiding algorithm semantics.

Use optional dependencies and tested environment locks per supported backend.
A single package does not require an untested universal Conda environment.

## Generic Reliability, Not Private Cluster Operations

Retain portable validation, atomic publication, configurable output paths,
bounded background media work, explicit artifact failures, distributed error
coordination and well-defined checkpoint/resume semantics. A model-only
checkpoint is an evaluation artifact, not an exact training-resume checkpoint.

Exclude private mount names, machine addresses, credentials, allocations,
GPU-hold programs, private queue/agent machinery and production logs. Distributed
examples should accept user-provided launch settings. Storage policy should be
configurable rather than encoding a particular cluster's disk failover policy.

## Validation and Release Gates

1. Freeze manuscript membership and recover run provenance. Several early T2I
   baseline config snapshots lack an exact runtime SHA; check W&B code metadata
   and original source snapshots before calling them exactly reproducible.
2. Audit licenses and redistribution terms for upstream code, models, rewards,
   benchmark data and visual assets. Preserve notices; do not select a license
   or redistribute restricted assets without resolving compatibility.
3. Compare migrated kernels with pinned reference/runtime code: outputs, losses,
   gradients and optimizer/EMA transitions. Test zero signal, ties, masking,
   loss denominators, timestep selection and accumulation invariance.
4. Validate each model/reward adapter, then single-device and distributed runs.
   Match preprocessing and metric meanings. Record tolerance and hardware;
   cross-device bitwise equality is not a general promise.
5. Execute documented paper recipes on a clean environment, retaining evidence
   of settings, progress, evaluation and source identity. Mark CPU-only tests,
   GPU smoke tests and full reproductions distinctly.
6. Publish a concise visual README: teaser, results, supported methods, setup,
   runnable recipes, extension examples, reproduction table and citations.
   Link only verified accessible W&B evidence after privacy review.

Import source as a clean snapshot, not the research Git history. Subsequent
public development can use normal commits; source attribution remains required.

## Appendix Triage

Recommended after the required scope: four-reward joint optimization,
mask/no-mask robustness, reward-weight sensitivity and pairwise assignment
variants. Keep implemented core variants easy to expose, but do not port every
historical failed attempt or exploratory script. Cancelled image pairwise
cross-algorithm panels are not mandatory main-paper reproductions.

Open decisions: final framework name, compatible license, appendix priorities,
and the minimum advertised hardware for each validated profile.
