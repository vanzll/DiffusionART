# Wan2.1 native inference parity

`python -m visual_rl.small_video_wan_parity --plan` is the default, model-free
path. It prints the immutable revisions, full profile, fixed tolerances and
required execution arguments without importing Torch or native model packages.

Install the [separate pinned Wan FA2 add-on](small-video-assets.md) first.
Explicit CUDA parity performs the lazy dependency/binary preflight before
constructing official WanT2V or loading weights. Missing/wrong FA2, a mismatched
Torch/CUDA/ABI, or an FA3 default-branch override produces an explicit dependency
failure report instead of reaching the first official attention assertion.
This preflight neither changes native attention nor executes a GPU kernel.

Real execution requires an explicitly authorized CUDA device and a verified
`wan` asset receipt:

```bash
python -m visual_rl.small_video_wan_parity \
  --device cuda --asset-root "$ASSET_ROOT" --receipt "$WAN_ASSET_RECEIPT" \
  --owned-root "$TASK_ROOT" --output "$TASK_ROOT/parity-wan-lora" \
  --finetune lora
```

Use `--finetune full` for the full-weight initialization path and a new output
directory. This command does not manage GPU leases or occupancy; the launcher
must have admitted its device before execution. It does not run a reward model,
optimizer or ART update, and it leaves the training science gate unchanged.

Reference code is official `Wan-Video/Wan2.1` commit
`9737cba9c1c3c4d04b33fcad41c111989865d315`, native model revision
`37ec512624d61f7aa208f7ea8140a131f93afc9a`. The fixed profile is832×480,
81frames,16fps,50 native UniPC steps,CFG6,shift8. The receipt binds source and
model hashes. The tool uses two verified held-out prompts selected by shortest
and longest text, and requires their actual captured T5 token lengths to differ.

Official `WanT2V.generate` supports a serial physical batch1. The reference
therefore executes two original calls with distinct prompts and seeds. Hooks
observe native text prefixes, conditional/unconditional transformer outputs,
actual initial noise, physical sigmas, INT64 time conditions, native states and
history-dependent means. A raw VAE decoder hook checks finite pixels before
native clipping can hide overflow. These wrappers preserve the original calls
and their computations. The report explicitly identifies this serialB1
reference; it is not described as an official batched reference.

The candidate reuses those FP32 transformer and raw VAE modules without loading
a second model copy. It prepares either fullFP32 parameters or native attention
LoRA32/alpha32/dropout0, checks all240 self/cross attention projection targets,
and records trainable/frozen counts. Zero-initialized LoRA is also compared
with the adapter disabled. Every candidate transformer block sees a real
physical batch4: two video inputs and their two CFG branches.

For every one of the50 frozen official states, the candidate compares individual
CFG branches and guided velocity. It separately executes a complete B2 native
trajectory from the captured actual noise, retaining all50 transitions and
running the existing native conditional-affine history audit. It compares each
state, velocity and mean. Native sigmas and INT64 conditions must match exactly;
Euler substitution or a shortened/OOD profile is unsupported.

Tolerances are frozen before execution: BF16-influenced comparisons use
`atol=rtol=1/128`; FP32 same-input VAE decoding uses `atol=rtol=1e-5`. The
same-input VAE comparison isolates affine/decode correctness; the separate
full-trajectory decode comparison includes upstream BF16 trajectory error.
Failures remain failures in the report, and the tool never relaxes tolerance
or changes the sampler to pass.

Default artifacts are `metrics.json` and real reference/candidateMP4 files.
Detailed tensors stay in CPU memory; `--save-tensor-trace` explicitly opts into
a potentially multi-GB tensor artifact. The normal path avoids writing those
large arrays. An interrupted/failed attempt retains its diagnostics; use a
new owned output directory rather than overwriting evidence.

The CPU tests exercise observer signatures, cleanup, raw-overflow detection,
variable-length padding, comparator thresholds and model-free planning. Their
synthetic scheduler is an instrumentation fixture, not a native UniPC proof.
Neither these tests nor fixed-input pretrained inference parity alone validate
ART replay,8GPU training, checkpoint correctness, reward improvements or80GB
hardware capacity. Those require their own evidence.

An opt-in localizes failures without changing the default parity path.
Add `--solver-audit-only` to the same explicit CUDA/receipt/output
arguments to import only the pinned native solver file: no T5, transformer,
VAE or FlashAttention is loaded. It runs a deterministic small B2 tensor through
all50 native transitions with AMP disabled and BF16 AMP enabled. It records
native `einsum` input/output dtypes, unchanged-velocity snapshot replay identity,
and the original `3e-5` fixed-perturbation affine gate at every position.
Native sigmas, INT64 times, scheduler equations and production precision stay
unchanged. The synthetic graph is a solver diagnostic, not model inference.

The observer also has CPU-covered first-transition capture/cleanup support.
The composed B1/pre-PEFT B4/post-PEFT B4 diagnostic remains unexposed pending
its own integration coverage; it is not an available CLI execution mode yet.

All opt-in reports say `diagnostic_notaccepted`, even if all measured checks
pass. They execute no optimizer or reward calls and cannot approve training.
CPU tests additionally offer a clearly labeled, test-only emulation of CUDA's
BF16 `einsum` policy. CPU autocast policies differ from CUDA; neither CPU
algebra nor that emulation proves the native CUDA solver affine.
