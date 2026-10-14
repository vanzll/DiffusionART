# Release Review

Reviewed on 2026-10-10 against code snapshot
`7254461903b736942c6c8c99cec74e310beb4650` and the released Arxiv manuscript.
This review covers core objectives, connected runners, candidate configurations,
evaluation, checkpoint/resume, assets and packaging. An independent read-only
reviewer checked algorithms against primary upstream sources. The initial
review did not change scientific code, training configurations, weights or
results. The author subsequently authorized the SD3 scheduler fix and
four-reward coverage repair below.

## Findings

### 1. Main-paper four-reward training coverage: implemented

The paper jointly optimizes GenEval, PickScore, CLIPScore and HPSv2.1 for 500
plotted steps. The original snapshot had no HPS scorer or four-reward recipe.
The author approved repairing this gap, without addressing the other findings.
The release now includes the official HPSv2.1 adapter, pinned assets, seven
four-reward configuration candidates and independently retrieved scientific
evidence. See [the four-reward guide](multireward.md).

Reporting uses GenEval + PickScore logit/26 + CLIPScore logit/30 + HPS cosine.
ART uses the equal-weight mean of individually group-normalized component
scores for covariance. GRPO-family candidates explicitly select this joint
advantage branch. NFT candidates retain the inspected reference's scalar-total
normalization; an archived configuration flag alone does not establish which
branch executed. Source commits for those historical baselines remain
unavailable, and exact historical runtime parity is not certified.

The ART source budget is 1000 updates, not the first 500 steps shown in the
paper. The approved public zero-signal semantics remain unchanged. No new GPU
training or pretrained-model result acceptance was performed for this repair.

### 2. SD3 ART scheduler migration discrepancy: fixed

The reviewed snapshot's `backends/sd3.py::_schedule` chose `shifted_schedule`
for ART and `sd3_schedule` for baselines. Training and evaluation both reused
this algorithm-dependent choice.
At 40 steps and shift 3, a CPU comparison found ART's last positive sigma
`0.07142857`, versus `0.00892857` for the baseline scheduler, with maximum
corresponding-node difference `0.0625`. This is not floating-point rounding.

The baseline path follows the
[Diffusers scheduler](https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py).
The effect on reward and which method benefits have not been measured.

Follow-up: the historical SD3 ART runs `11pq0iy3`, `9c0zrzlw` and `mofumvbw`
report runtime commit `b52a36dd62f7b733c3ee97c91cb0500bda69e632`. That commit's
`scripts/train_sd3.py` uses the model's Diffusers scheduler through
`sd3_pipeline_with_logprob`/`retrieve_timesteps`, without a custom sigma override,
for both ten-step training and forty-step evaluation. The custom ART grid was a
migration discrepancy, not established evidence of a historical paper confound.
Exact historical source authentication for every baseline remains incomplete.

The author authorized aligning SD3 ART with its actual historical implementation.
`_schedule` now uses `sd3_schedule` for every SD3 method, for training and eval.
Video grids, FLUX scheduling, transition kinds and losses are unchanged.
Regression tests cover all six SD3 methods against the native scheduler and
the real small-model ART training/evaluation paths with distinct step counts.
Do not treat continuing a pre-fix ART checkpoint under the corrected grid as
an unchanged recipe; this fix changes that snapshot's sampling behavior.

### 3. Important adaptations are not verbatim historical/upstream recipes

- Image scalar ART uses the author's later-approved exact zero-signal suppression,
  no restoration on those rows, and unchanged original loss denominators.
  Pairwise semantics are unchanged. Refill is absent. This is intentional,
  but it is not identity with all historical paper runs.
- Wan Dance scalarizes VQ/MQ before group standardization and PPO clipping
  (`training/wan_video.py::prepare_grpo`). Hunyuan Dance instead standardizes
  and clips components separately before weighted summation. The Wan recipe
  is an explicit adaptation, not verbatim upstream multi-component DanceGRPO.
- Video Fast uses independent initial noise and potentially noncontiguous
  selected positions. Upstream Fast advertises shared-prefix sampling, but
  [its trainer makes shared initial noise optional](https://github.com/yifan123/flow_grpo/blob/879042cf5707f8b90daa98d147d7deac2317c5da/scripts/train_sd3_fast.py).
  Historical video-runtime parity and any performance disadvantage remain
  unverified.
- Video ReFL is the paper's final-step, differentiable-reward protocol. It is
  not every setting or objective term of the original ImageReward paper or
  UniRL's experimental implementation.

Keep these distinctions explicit in recipe provenance. Do not silently change
an adapted historical baseline or treat an approved ART change as an accidental
bug. No confirmed baseline reward handicap was established by this review;
that is not evidence that every recipe has equal performance.

### 4. Recovery is implemented, but real-model CUDA acceptance is pending

Complete checkpoints retain policy, optimizer, RNG, data cursor, progress,
algorithm/shadow state and logging/cost boundaries. Checksum-verified rank
files precede manifest/latest publication; retention runs after publication.
CPU tests cover save/load and continued updates. Same topology/layout and
compatible runtime are required; FSDP resharding is not implemented.

These tests do not establish real pretrained-model CUDA/FSDP save/restore
correctness. Before advertising seamless recovery, validate a full save,
restart, next optimizer update and NFT/EMA restoration on the intended GPU
topology. W&B recovery uses authoritative local history and a fresh grouped
run; it does not require cloud rewind. Compare global optimizer steps and
recorded cumulative GPU-hours, not the new run's cloud `_runtime`.

`training/fsdp_checkpoint.py::save_fsdp_checkpoint` is synchronous and does
not provide filesystem deadlines or process-crash recovery. A hung shared
filesystem or SIGBUS is not made safe by exception handling or free-space
checks. Retention of one checkpoint also does not mean a one-checkpoint peak:
the new complete publication must coexist with the previous recovery point.

### 5. Video artifact recovery and export remain limited

Video encoding has input checks, bounded subprocesses and full output decode
validation. However, `backends/hunyuan_fsdp.py` treats rollout decoding/scoring
errors as coordinated terminal errors, not a recoverable skipped rollout.
Do not claim continuous recovery from repeated media/storage failures.

The [export utility](model-export.md) supports image local/DDP checkpoints,
not video FSDP/Hunyuan/Wan. A portable trained-video inference export and its
round-trip loading test are still needed for the complete release workflow.

### 6. Licensing and GPU-cost parity are release gates

At the time of this review, there was no project-wide license. This item was
resolved on 2026-10-13 by adopting Apache-2.0 for original project code.
Preserved upstream notices and separate model/data redistribution terms still
apply; see `LICENSE` and `THIRD_PARTY_NOTICES.md`.

CPU-backed policy/EMA shadows add host transfers. This does not by itself
change a loss, but hardware-matched timing is untested and may differ from the
original training implementations. Measure rollout, update, scoring and
evaluation costs separately before attributing timing differences to algorithms.
For very small PPO clipping thresholds, also verify on-policy likelihood
replay under the actual precision and physical batches; the
[Flow-GRPO FAQ](https://github.com/yifan123/flow_grpo#faq) warns about this issue.

## Verified Core Semantics

- ART: covariance aggregation and RMS normalization, sigma weighting, frozen
  rollout predictions across two disjoint update windows, original-window
  denominators and corrective displacement masking are connected. The paper
  and code use opposite velocity conventions; `target = old - G` has the
  correct sign in code. Mask weights and rollout targets are detached.
- NFT: clipped advantages map to branch probabilities; positive/negative
  mirror predictions use separate detached mean-absolute endpoint residual
  denominators. Beta and advantage-scale factors are retained. Direct CPU
  comparison to the
  [official loss](https://github.com/NVlabs/DiffusionNFT/blob/cbb14f84b8312b620390dfcbe2fab69c1383b104/scripts/train_nft_sd3.py)
  at beta 1 and 0.1 found loss differences below `9e-7` and maximum gradient
  difference below `1.2e-7`. Old-policy EMA and evaluation EMA are separate;
  old policy remains frozen across both windows in the two-update recipe.
- Flow-GRPO/Fast: detached observations, feature-mean Gaussian likelihoods,
  transition standard deviation including `sqrt(-dt)`, and contiguous image
  Fast windows follow the inspected
  [reference transitions](https://github.com/yifan123/flow_grpo/blob/879042cf5707f8b90daa98d147d7deac2317c5da/flow_grpo/diffusers_patch/sd3_sde_with_logprob.py).
  CPS retains the reference negative-MSE surrogate instead of substituting a
  Gaussian likelihood.
- Hunyuan Dance: per-component sample-standardization, advantage clipping and
  separate PPO clipping before weighting follow
  [DanceGRPO 15cc71d](https://github.com/XueZeyue/DanceGRPO/blob/15cc71d53cc2e6e18a68ee607d5fb6ba9a99e344/fastvideo/train_grpo_hunyuan.py).
- ReFL: the unnormalized reward/VAE cotangent reaches the final-step endpoint
  through staged VJPs; earlier policy steps are detached. Continuous tensor
  preprocessing is differentiable, without MP4/PIL straight-through gradients.
  See the [UniRL reference](https://github.com/Tencent-Hunyuan/UniRL/blob/5b651cf4f6264d511c4595c4baf700729065c1c2/experimental/refl/models/wan21.py)
  and the explicit final-step adaptation above.
- Shared update plumbing: logical denominators, DDP world correction, padding
  exclusion and successful-update counters are covered by tests. Asset
  preparation, immutable identities, atomic publication and scalar-only W&B
  isolation do not depend on private queue/storage/GPU-holder machinery.

## Validation

### Paper Recipe Corrections (2026-10-14)

Author-approved scope: restore Hunyuan paper rollout/update microbatches to 8/8
(ReFL stays 1/1; Wan stays 8/4), restore scalar video zero-guidance old-policy
anchors, and add matched TA/VQ/MQ=7:2:1 Fast/Dance configurations. Image scalar
zero-guidance rows remain zero-loss with the original window denominator;
exact constant-reward protection stays enabled without a weak-signal threshold.
Pairwise handling, learning rates, masks, seeds, sampler and update boundaries
are unchanged. All paper profiles now evaluate every 50 optimizer updates;
video remains 8 FPS / RM 4 FPS. This cadence is not historical evaluation parity.

Evidence: `8b2f8a8d` Hunyuan ART runtime uses microbatches8/8 and
`restore_anchor_mask=~active`; Fast721 `a419ds0n` at `64cadf05` and Dance721
`5ttd9ord` at `8aa9801d` profiles/W&B config agree on microbatches8/8 and
weights .7/.2/.1. Their observed endpoints348/368 are not relabeled400.
The new configs specify the 400-update bounded recipe. Historical image
microbatches remain unchanged, including FLUX NFT's actual runtime6/6.

Video scalar activity now includes restoration anchors, so both unsharded and
distributed runners execute anchor-only windows instead of skipping them.
Regression coverage includes exact-zero versus real weak covariance, original
denominators, corrective restoration after drift, local/global inactive ranks,
unchanged pairwise ties, and the new profiles. No pretrained GPU run is implied.
Historical sampler/order gaps and unavailable runtime commits remain unresolved.

Validation: algorithm/scalar-runner/paper-profile/pairwise tests passed 231 with
one archived-evidence skip; accumulation/scaled-update/FSDP/no-signal/profile
tests passed 210 with one archived-evidence skip (these sets overlap).
Asset-preparation/config tests passed 53 after updating the obsolete sparse-eval
expectation. Targeted Ruff and diff checks passed. The independent reviewer found
no blocking issue in the final code/config diff; real pretrained CUDA throughput
and reward reproduction remain untested in this batch.

### Execution Provenance and Early ART Audit (2026-10-14)

Live W&B exports now capture the loaded checkout's commit and dirty status in
`config.execution_source`, separate from historical `config.provenance`.
Each exporter has its own local source record, including after resume. Git
inspection is bounded and unavailable identity is explicit. Standalone replay
only attributes source when supplied its original `--source-record`; it never
infers old training code from the replay checkout. Source files, diffs and Git
remote credentials are not uploaded. A dirty or unpublished commit still
requires a separate source/patch handoff for reproduction.
Validation: 22 W&B exporter tests passed, including real offline SDK exports,
timeout handling and replay attribution; targeted Ruff checks passed. No GPU
work or mutation of the active acceptance run was performed.

The remote acceptance discussion and local bundle identify its running source
as `1a02e397efe90299574b954aed758c9de524a0a9`. Its extra changes over `78364cb`
are the Wan hook fix already integrated here, checkpoint-space preflight and
optional resume verification. The latter two have CPU evidence but not full
GPU save/resume acceptance; this change does not import them.

At this audit, run `869917a2b848438194c761f89ab19e2e` had six updates (three
rollouts), not six independent reward batches. Fixed-normalized weighted
rewards were -0.35827, -0.05455, -0.16806; historical `gyvtocue` had -0.22724,
-0.09061, -0.26065 at the corresponding first three rollouts. Neither sequence
increased monotonically. Public prompt shuffling uses seed 42 versus historical
sampler seed 1237; rollout/update microbatch is 1 versus 8. LR, optimizer, reward
weights, training FPS/RM FPS and mask settings agree. Eval video FPS differs
(corrected public 8 versus historical 18), so historical evaluation is not an
identical protocol. This establishes no reward-regression cause or performance
parity; continue the fixed-setting run and compare later training/eval points.

### Wan Empty Hook Registry Fix (2026-10-14)

Integrated the focused fix from remote acceptance commit `db58aa3`: native
Diffusers cleanup can leave an empty `HookRegistry`, which manual staging
previously rejected on the next rollout. Only a known empty registry is now
allowed; active hooks, unknown hook objects and device maps remain rejected.
The regression reproduces the original failure, then checks repeated staging
and active-hook rejection. Training math and checkpoint behavior are unchanged.
Focused Wan asset, runner, online/deferred evaluation, video and ReFL tests:
134 passed, 1 skipped on CPU; targeted Ruff checks passed. No new GPU run.

### Earlier Validation

Initial snapshot validation on macOS, Python 3.10, without pretrained-model
downloads or GPU tasks:

- `ruff check src tests`: passed.
- `python -m pytest -q`: 2131 passed, 4 skipped, 34 subtests passed;
  elapsed 702.77 seconds. Warnings include upstream dtype deprecations.
- `python -m visual_rl.train --config configs/examples/cpu_smoke.yaml --validate`:
  passed. This is configuration validation, not a paper experiment.
- `uv build`: source distribution and wheel built successfully.

After the SD3 scheduler fix: `ruff check src tests` passed; focused pytest
coverage passed 269 tests in 252.20 seconds. This includes native scheduler
parity, ART train/eval, GRPO/NFT, two-rank image training/resume, paper image
configurations and unchanged FLUX/video grid contracts. No GPU tasks were run.

No scientific implementation defect was confirmed in the inspected core loss
kernels. The follow-up scheduler migration defect above was subsequently
confirmed and fixed. This does not certify bug-free execution, historical curve
identity, GPU mixed-precision parity or unbiased end-to-end timing. Highest-value next
checks are four-reward coverage, real-model CUDA recovery,
video export and licensing. Historical SD3 Dance source authentication remains
open. README presentation and the project website were not changed.
