# LTX native pretrained parity

`python -m visual_rl.small_video_parity` compares the new LTX adapter/rollout
against the **actual official LTXVideoPipeline and RectifiedFlowScheduler** at
[`50f0aef`](https://github.com/Lightricks/LTX-Video/tree/50f0aef1297574a4b29acbdb874e3a363a6d5f2e).
The tool is LTX-only. It does not establish official Wan parity, native ART
training equivalence, reward compatibility or eight-GPU acceptance.

No pretrained inference occurs with`--plan`. Real execution requires explicit
`--device cuda`, a completed LTX asset receipt and a new output directory. The
tool does not download assets, load a reward model, create an optimizer or
control GPU reservations/leases. The caller must admit it onto an authorized
idle GPU through their existing resource controls.

## Inputs and independent reference

Use the [asset preparation guide](small-video-assets.md) and its native model
environment. The full profile is704×1216,153frames,30fps; the tool fixes physical
batch2 and uses the first two frozen held-out eval8 prompts. Their actual T5
attention-mask lengths must differ or the tool fails. It executes CFG1/STG0,
eight native stochastic steps, checkpoint scheduler configuration, no prompt
enhancement, decode timestep0.05 and decode noise scale0.025. No small substitute
grid or hidden per-row transformer loop is used.

The official pipeline is constructed from receipt-verified native transformer,
VAE, T5 and tokenizer components, with all prompt-enhancement components set to
None. Its scheduler is the official`RectifiedFlowScheduler.from_config` called
on the exact checkpoint scheduler metadata, the same final class/config path
as official`from_pretrained`. This avoids reading every weight tensor merely
to recover scheduler metadata. The reference then executes the original
official pipeline call and native sampler unchanged.

`--finetune base|lora|full` selects the candidate only; the default is`base`.
The official unwrapped base completes its entire reference run **before** the
candidate is changed. LoRA then uses the shared`prepare_policy` with rank32,
alpha32, dropout0, Gaussian A and verified zero B. Full mode makes every
original transformer parameter trainable FP32. Native BF16 forwards disable
outer global autocast, matching the official`mixed_precision=False` reference.
Only existing LoRA A/B Linear GEMMs enable BF16 autocast; their trainable
master weights remain FP32. The permanent branch wrappers preserve PEFT
module types, parameter names and checkpoint identity and run again during
activation-checkpoint recomputation. No optimizer is constructed or called.
T5 and VAE must remain frozen and in evaluation mode throughout.

The report records actual trainable/frozen dtype/count inventories and all
attention projection names/shapes. Each of28 native layers has two attentions,
so each suffix`to_q`,`to_k`,`to_v`,`to_out.0` must match56 projections,224 total.
LoRA additionally checks eight batch2 disabled-adapter forwards against the
captured official base and compares zero-enabled output to disabled output
exactly. Full mode checks that every original parameter is trainable FP32.
Single-device full parity uses explicitly reported BF16 parameter/buffer views
of the FP32 masters through`torch.func.functional_call` on the actual native
transformer. This helper rejects gradients, train mode, checkpointing and
DDP/FSDP wrappers. It restores original master identities even after an
exception. It is an inference-only simulation of compute precision, never a
production training fallback or full-backward/distributed acceptance claim.
The production full path requires its separately validated BF16-compute FSDP
contract. Actual packed inputs are explicitly BF16; coordinate/time kwargs
remain FP32. Tolerances below are unchanged across all three candidate modes.

Observation wrappers capture the actual initial latents and post-initial global
RNG, transformer inputs/kwargs/velocity, native scheduler state/x0/noise/mean/
next state, decode noise and pre-postprocessing VAE pixels. They forward all
original arguments and return original results. The scheduler's mean is observed
from its actual`add_noise` inputs as`(1-next_sigma)*actual_x0`; it is not computed
using this repository's transition helper. A pipeline-local Torch namespace
proxy records its actual`randn_like` decode call without replacing global Torch
or changing the sampler. The final zero-weight stochastic draw is retained.

The adapter independently re-encodes text and first predicts from frozen
official states, checking actual model input, mask, time, coordinates and
velocity. It then runs`collect_native_video` as a separate full recurrence from
the captured initial latent and exact post-initial global RNG state. Every
state, velocity, conditional mean, next state and actual innovation is compared.
Both paths must make eight physical transformer calls of batch2. Packed state
strides and the global RNG state after all transitions must also match.

Default adapter decode is tested by restoring the official pre-decode RNG and
calling`adapter.decode(endpoint)` without supplying an innovation. The actual
default innovation and storage mapping must equal the official draw. This
prevents a correct cached-noise injection from concealing a wrong default draw.
Native VAE pixels are checked for finiteness **before** presentation clipping
and compared before MP4 encoding.

Two primary-code pitfalls are checked explicitly:

- The actual physical sigma table is LinearQuadratic(8):
  `[1,.99375,.9875,.98125,.975,.909375,.725,.421875,0]`, represented in FP32.
  Checkpoint`allowed_inference_steps` is a rounded validation allowlist, not
  the physical solver schedule.
- Official initial noise is generated packed`[B,F*H*W,C]` with a dedicated
  generator, then unpacked. Transitions use global`randn_like(packed_state)`;
  decode uses global`randn_like(unpacked_endpoint)`. Strides determine the
  assignment of random values, so equal shapes/seeds alone are insufficient.

The first real LoRA run exposed an additional implementation discrepancy:
candidate global autocast was enabled while the unwrapped official reference
used`mixed_precision=False`. Disabled LoRA and zero-enabled LoRA matched each
other exactly but already differed from the official first velocity, before
trajectory divergence. This identifies a precision-contract mismatch; it does
not establish which individual internal operator first diverged. The corrected
tool records actual input dtype/stride/autocast and metadata-only hooks at
native normalization/projection and representative A/B boundaries for the next
real comparison. The original failed report remains evidence; no GPU pass is
claimed for the correction before a new report exists.

## Commands

From the repository root, inspect the plan without inference:

```bash
python -m visual_rl.small_video_parity \
  --asset-root "$SMALL_VIDEO_ASSETS" --plan
```

On an explicitly admitted GPU, using the model environment from asset setup:

```bash
# Three transformer passes (official, frozen-input adapter, full recurrence),
# plus a disabled-adapter pass for LoRA,
# two full native video decodes and MP4 verification can take several minutes.
# Stderr reports phases and every native rollout step. No lease changes occur.
timeout -k 15s 1800s "$SV_MODEL_ENV/bin/python" -m visual_rl.small_video_parity \
  --asset-root "$SMALL_VIDEO_ASSETS" \
  --receipt "$SMALL_VIDEO_ASSETS/preflight-ltx.json" \
  --manifest configs/assets/small_video.json \
  --output /absolute/new/ltx-native-parity-base --device cuda --seed 42 --finetune base
```

Repeat with distinct new output paths and`--finetune lora` or`--finetune full`
for their actual candidate parameter/forward contracts. These inference checks
do not perform updates or establish optimizer/backward/checkpoint acceptance.
Detailed trajectories can occupy several GB; add`--save-tensors` only when they
are needed. It is disabled by default.

The caller selects GPU visibility before launch; the tool uses one visible
current device. It records its actual model name and physical memory rather
than claiming eight80GB devices. TF32 is disabled for the comparison. No
training or reward steps are consumed; this is base inference acceptance.

## Tolerances and artifacts

| Comparison | Absolute tolerance | Relative tolerance |
| --- | --- | --- |
|schedule, masks, coordinates, model inputs, RNG and innovations|0|0|
|FP32 trajectory states/means/next states|1e-5|1e-5|
|BF16 velocity and raw decoded pixels|1/128|1/128|

Every element must meet its threshold; a small global MSE does not hide a bad
element. The report includes max absolute error, RMS error, threshold fraction,
shapes and actual dtypes for every comparison. The BF16 tolerance is explicit;
it is not a claim of bitwise model equality. All outputs must be finite.

`report.json` records the result, asset/source identity, seeds, profile,
actual physical batch counts, layout, memory, concrete parameter/target
inventories and per-step errors. The`forward_precision` and`precision_boundaries`
sections record actual call
dtypes/strides/autocast plus boundary outputs and stored parameter dtypes,
including FP32 LoRA masters with BF16 branch results. Default output contains
the report and four
matched MP4s. Optional`--save-tensors` writes`trace.pt` with the actual observed
native trajectory, all innovations, RNG states, teacher-forced and disabled
predictions and candidate cache. Four matched MP4 files contain
both official and adapter outputs for both prompts, using the existing encoder
and full frame-count/FPS decode verification. Video encoding clips only after
raw finite-pixel parity checks. An error report with partial step/batch history
is retained on failure; partial tensor traces require`--save-tensors`.
Exit0 means all checks passed; mismatches exit1 and
runtime/asset errors fail rather than substituting zeros or changing the solver.

CPU tests validate observation preservation, cleanup, strict comparison, RNG
mapping, signature checks and the no-GPU default gate with small fixtures.
They also exercise real PEFT zero/disabled parity, exact target coverage and
full FP32 parameter/frozen-service contracts on small CPU modules when PEFT is
available.
Targeted tests also cover outer-autocast isolation, actual BF16 A/B results,
FP32 master gradients, checkpoint recomputation, strict save/load and copied
method identity, and the restricted full-view helper with master restoration.
They are explicitly **not** pretrained parity evidence. No real GPU execution
is claimed by this document until a corresponding report and artifacts exist.
