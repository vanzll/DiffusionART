# Small native T2V development progress

Canonical base: `b51e6ecc9ec1c993144a0e21ec579b6dd062bd01`.
Branch: `codex/small-video-8gpu-20261104`; draft PR: https://github.com/vanzll/DiffusionART/pull/11. Local coordinator reviews; no automatic merge.

## Binding protocol and budget

Eight recipes cover native Wan2.1 T2V1.3B and LTX-Video2B0.9.6 distilled, each with LoRA/full and Original/Flash. Every cell is capped at ten cumulative actual global optimizer calls (five rollouts), including checkpoint4, fresh load, and the remaining six calls. Base/post evaluation uses the same eight held-out prompts and seeds. Historical 50/200-rollout budgets are superseded. Existing scientific image runs, reward services, supervisors and leases are protected; new work uses only verified available whole eight-GPU nodes through their original generation-fenced controller.

P16 prompt groups × K8 videos are scored on complete groups. Raw VQ/MQ are separately group-standardized and averaged once; TA is monitoring only. Frozen covariance supplies disjoint K4/K4 windows, exactly two optimizer calls per rollout. A near-zero standardized reward mean is not evidence of improvement: report raw VQ, raw MQ and their equal raw mean, paired base/post changes and uncertainty. Ten calls provide short sanity evidence only.

## Implemented and CPU checked

Separate native backends, pinned assets and preflight, complete native rollout caches, pending scientific replay gate, variable-length batching, LoRA/full policies, bounded eight-GPU runner, separate canonical VideoReward environment, full policy/Adam/RNG/data/timestep-selection checkpoints, exports and generic held-out evaluation are implemented. New dependencies load optionally; old model defaults and checkpoint interfaces are preserved.

261 new CPU tests and 261 affected existing tests passed (522 public cases covered, no skips). These include native schedule/cache checks, denominator and partition-gradient equivalence, real two-rank CPU FSDP BF16/FP32-master/Adam/resume checks, and a nonzero PEFT save/load/export/disable roundtrip. One independent code review completed; device-placement and asset-receipt binding findings were fixed. CPU algebra and fixtures are not GPU or training acceptance.

Both model and canonical reward environments were installed into fresh isolated Python3.10 venvs and passed dependency/import checks. Native Wan additionally pins FlashAttention2.7.4.post1 with Torch2.6/CUDA12.4 ABI=False and rejects a competing FA3 default. All 51 required asset files (52.44GB) passed fixed revision/checksum preflight. Real GPU reward evaluation remains pending.

## Actual native GPU evidence

Passing native LTX source: `c015d8b23fd874e4937b974d6d2d14a0246e1f9f`.

| Check | Evidence | Limit |
| --- | --- | --- |
| LTX LoRA pretrained native batch2 | 115/115 comparisons, maximum absolute error0; matched full-profile videos | Single-device inference, no backward |
| LTX full pretrained native batch2 | 99/99 comparisons, maximum absolute error0; matched videos | Eval-only BF16 views retain FP32 masters; no distributed/backward proof |
| Wan LoRA source47ed68a7d04e01614d6d5123ec03d7a7cd1ccab4 | 100/154 teacher comparisons failed; strict native frozen-history affine replay audit failed | No acceptance; diagnostic investigation remains open |
| Eight-GPU end-to-end training | All eight cells remain0/10 actual optimizer calls | Reward/backward/checkpoint/resume/effectiveness evidence pending |

LTX uses native1216×704,153frames/30fps/eight stochastic transitions,CFG1/STG0 with actual random-draw and decode conditioning alignment. Wan uses832×480,81frames/16fps/50native UniPC transitions,CFG6/shift8. Actual inference transformer batch exceeds1. Failed attempts remain retained. Native samplers, dimensions and numerical thresholds were not changed to obtain passing LTX results.

Wan's proposed frozen-history affine replay is proved only under the tested CPU arithmetic. The real CUDA AMP native solver failed its strict audit; current-velocity-dependent einsum rounding is a hypothesis under explicit diagnostic investigation, not an established root cause. The native multi-step trajectory must not be silently replaced by Euler or a different precision policy. Full base-parity mismatches also require separate localization.

## Scientific and hardware limits

The proposed native conditional normalized-mean regression with canonical Q2 reference masking requires an explicit scientific decision. This mask is not claimed as a native stochastic-kernel KL. Checked-in recipes fail closed before training. LTX's independent decision is pending; Wan's additional real-GPU replay failure must be resolved independently.

The allocation reports H800 names but143771MiB and PCI device2335 (H200 in NVIDIA's driver table). This is not eight H80080GB validation. Successful native parity uses one device with physical batch2, not an eight-GPU recipe acceptance. No training reward trend or long-run stability is established. See `docs/small-video.md` for public commands, protocol and evidence requirements.
