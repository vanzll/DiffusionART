# Small native T2V development progress

Canonical base: `b51e6ecc9ec1c993144a0e21ec579b6dd062bd01`.
Branch: `codex/small-video-8gpu-20261104`; draft PR: https://github.com/vanzll/DiffusionART/pull/11. Local coordinator reviews; no automatic merge.

## Binding protocol and budget

Eight recipes cover native Wan2.1 T2V1.3B and LTX-Video2B0.9.6 distilled, each with LoRA/full and Original/Flash. Every cell is capped at ten cumulative actual global optimizer calls (five rollouts), including checkpoint4, fresh load, and the remaining six calls. Base/post evaluation uses the same eight held-out prompts and seeds. Historical 50/200-rollout budgets are superseded. Existing scientific image runs, reward services, supervisors and leases are protected; new work uses only verified available whole eight-GPU nodes through their original generation-fenced controller.

P16 prompt groups × K8 videos are scored on complete groups. Raw VQ/MQ are separately group-standardized and averaged once; TA is monitoring only. Frozen covariance supplies disjoint K4/K4 windows, exactly two optimizer calls per rollout. A near-zero standardized reward mean is not evidence of improvement: report raw VQ, raw MQ and their equal raw mean, paired base/post changes and uncertainty. Ten calls provide short sanity evidence only.

## Implemented and CPU checked

Separate native backends, pinned assets and preflight, complete native rollout caches, pending scientific replay gate, variable-length batching, LoRA/full policies, bounded eight-GPU runner, separate canonical VideoReward environment, full policy/Adam/RNG/data/timestep-selection checkpoints, exports and generic held-out evaluation are implemented. New dependencies load optionally; old model defaults and checkpoint interfaces are preserved.

261 new CPU tests and 261 affected existing tests passed (522 public cases covered, no skips). These include native schedule/cache checks, denominator and partition-gradient equivalence, real two-rank CPU FSDP BF16/FP32-master/Adam/resume checks, and a nonzero PEFT save/load/export/disable roundtrip. One independent code review completed; device-placement and asset-receipt binding findings were fixed. CPU algebra and fixtures are not GPU or training acceptance.

Both model and canonical reward environments were installed into fresh isolated Python3.10 venvs and passed dependency/import checks. Native Wan additionally pins FlashAttention2.7.4.post1 with Torch2.6/CUDA12.4 ABI=False and rejects a competing FA3 default. All 51 required asset files (52.44GB) passed fixed revision/checksum preflight. A real canonical reward smoke on two existing full-profile native LTX videos now passed; this is independent of training acceptance.

## Actual native GPU evidence

Passing native LTX source: `c015d8b23fd874e4937b974d6d2d14a0246e1f9f`.

| Check | Evidence | Limit |
| --- | --- | --- |
| LTX LoRA pretrained native batch2 | 115/115 comparisons, maximum absolute error0; matched full-profile videos | Single-device inference, no backward |
| LTX full pretrained native batch2 | 99/99 comparisons, maximum absolute error0; matched videos | Eval-only BF16 views retain FP32 masters; no distributed/backward proof |
| Wan LoRA source47ed68a7d04e01614d6d5123ec03d7a7cd1ccab4 | 100/154 teacher comparisons failed; strict native frozen-history affine replay audit failed | No acceptance; diagnostic investigation remains open |
| Eight-GPU end-to-end training | All eight cells remain0/10 actual optimizer calls | Reward/backward/checkpoint/resume/effectiveness evidence pending |

LTX uses native1216×704,153frames/30fps/eight stochastic transitions,CFG1/STG0 with actual random-draw and decode conditioning alignment. Wan uses832×480,81frames/16fps/50native UniPC transitions,CFG6/shift8. Actual inference transformer batch exceeds1. Failed attempts remain retained. Native samplers, dimensions and numerical thresholds were not changed to obtain passing LTX results.

Wan's proposed frozen-history affine replay passed the tested CPU arithmetic but failed the real native CUDA AMP audit. A source-pinned solver-only CUDA diagnostic now reproduces exact same-velocity snapshot replay in all50 positions while native AMP einsum returnsBF16 and violates the unchanged3e-5 affine gate at positions46/47/48 (maximum relative error6.7012e-5). An explicitly diagnostic AMP-off control returnsFP32 and passes (maximum2.3821e-7). No weights are loaded in this small-graph diagnostic; it is not model/base/training acceptance. This establishes a finite-precision incompatibility of the current affine simplification with native AMP, independently of the full model's unresolved batch-parity mismatches. Native sampling precision and tolerances remain unchanged; scientific approval is required before selecting a different actual-mean training definition.

## Scientific and hardware limits

The proposed native conditional normalized-mean regression with canonical Q2 reference masking requires an explicit scientific decision. This mask is not claimed as a native stochastic-kernel KL. Checked-in recipes fail closed before training. LTX's independent decision is pending; Wan's additional real-GPU replay failure must be resolved independently.

The allocation reports H800 names but143771MiB and PCI device2335 (H200 in NVIDIA's driver table). This is not eight H80080GB validation. Successful native parity uses one device with physical batch2, not an eight-GPU recipe acceptance. No training reward trend or long-run stability is established. See `docs/small-video.md` for public commands, protocol and evidence requirements.

## Reward diagnostic and operational fixes

Two fixed native LTX videos passed canonical raw VideoReward scoring on source `1e0a31d37bb4659892d82956045838378295241a`: VQ/MQ means1.248046875 and1.8984375, with all VQ/MQ/TA finite. Reward loading18.31s, scoring2.86s, parent phase22.32s; scorer peak allocated5.262GB, reserved6.325GB. One physical visible GPU was shared by the diagnostic parent and its owned one-shot scorer child; no existing reward service or extra device was used. This establishes a scoring path, not learning or eight-GPU acceptance.

Failures exposed three implementation issues: receipt source IDs/list schema, Torch2.6 bare CUuuid metadata, and native fastvideo PEP420 namespace provenance. All were fixed with targeted tests; source-pin regular-package defaults stay unchanged. A real cold reward import checked fixed native inference and prompt modules with CUDA uninitialized. The already-observed native checkpoint11352 is now passed explicitly to avoid mutable latest-directory selection; native signature/worker forwarding/cwd cleanup were checked on CPU.

Latest targeted checks: Wan diagnostic25passed; reward UUID/process/parity71passed; namespace source/worker57passed. These overlap earlier groups and must not be summed as a new unique total. Sanitized numerical/source/hash evidence is in [the diagnostic record](docs/evidence/small-video-pretraining-diagnostics.json). Private artifacts retain full failed-attempt, lifecycle and video evidence; no hostnames, credentials, weights or videos are included in this record. All eight cells remain0/10 actual global optimizer calls, with no training W&B run.
