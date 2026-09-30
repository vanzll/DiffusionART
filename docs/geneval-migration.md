# GenEval Migration Contract

Status: source audit, annotation transport and isolated native scorer implemented.
Configuration and setup are described in [geneval.md](geneval.md). Pretrained
GPU parity and clean installation acceptance remain outstanding.

## Reference Identity

The research client is `flow_grpo_alg/rewards.py::geneval_score` at research
revision `3ca2a36f6b20c53a06813013234666d9c033a660`. It delegates inference to
[reward-server revision 8245af22](https://github.com/yifan123/reward-server/blob/8245af22b75fe640950a880ad31fe702a5d9a014/reward_server/gen_eval.py).
The current deployment was inspected read-only on 2026-09-29. Its Git HEAD is
that revision, but the evaluator is locally modified:

- The OpenAI ViT-L-14 asset loads from an explicit local checkpoint through
  `open_clip.openai.load_openai_model`, with `precision="fp16"`. The image
  transform uses size 224 and the model's visual mean/std.
- Detector inference is chunked using `GENEVAL_DETECTOR_BATCH_SIZE`, default 1.
  CUDA cache is emptied after each detector chunk.
- Asset paths are local deployment settings and will not be copied into this
  framework. Scoring formulas are unchanged in the inspected diff.

Current deployment checksums:

| File | SHA256 |
| --- | --- |
| `reward_server/gen_eval.py` | `e18b1577f7bc32ccbd3483edaafa9784a130b0e09412e8c9265e76647780a394` |
| `app_geneval.py` | `b3b1686feaf61b7af1355d9ef748f20f832c78b366dadf7f8ab2e9637608ed0f` |
| `reward_server/object_names.txt` | `608f6a0e5c8ca1c7a92b818430141fe268c85b7930b50f4160f1d65893b5bacd` |

This establishes current source identity, not which dirty tree every historical
paper run used. Historical run-level provenance and model hashes remain required.

## Semantics To Preserve

The client rounds floating RGB to uint8, JPEG-encodes with Pillow defaults, then
the server decodes that JPEG. Native in-process inference must not bypass this
roundtrip or silently change its quality. Preserve RGB input to the detector.

Training consumes **dense `scores`**, not binary `strict_rewards`. Strict
correctness remains a separate diagnostic. `only_strict=True` skips ordinary
correctness calculation, not dense reward calculation. The dense formula can
produce negative values and contains a historical failed-position contribution;
changing these behaviors requires a separate recipe, not a hidden migration fix.

Include-clause order matters: a position target references an earlier clause.
Count, color, position and exclude rules must be tested independently against
the pinned evaluator. Class names must retain their exact vocabulary and order.
The detector's confidence threshold differs for the counting tag.

Dataset rows may contain repeated prompt text intentionally. The inspected
research test manifest contains 2,212 rows, representing 553 prompts repeated
four times. Do not add a second fourfold expansion. The open-source manifest
format is described in [prompt-data.md](prompt-data.md); annotations belong in
its `metadata` object and must retain their prompt association.

## Remaining Acceptance

- Pin detector checkpoint/config, class list, OpenCLIP weights, tokenizer,
  transforms and compatible MMDetection/MMCV environment.
- Preserve the upstream license and resolve original evaluator/data provenance
  before vendoring or redistributing assets.
- Keep incompatible detector dependencies isolated from the training environment;
  do not reuse the old unauthenticated pickle transport as the public interface.
- Verify dense, ordinary and strict outputs against the deployed implementation,
  including failed-position behavior, excess objects, exclusions and colors.
- Execute the integrated native loader, training/evaluation diagnostics and
  composite reward paths on pretrained assets in a clean Linux environment.

Annotation transport tests and CPU model fixtures do not satisfy these gates.
