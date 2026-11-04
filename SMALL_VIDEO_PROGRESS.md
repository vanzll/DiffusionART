# Small native T2V development progress

Base: b51e6ecc9ec1c993144a0e21ec579b6dd062bd01.
Branch: codex/small-video-8gpu-20261104; PR targets main, never auto-merge.

Binding acceptance is ten cumulative actual global optimizer calls per
combination (five rollouts), including checkpoint4/fresh-load/remaining6.
Base/post uses the same8 held-out prompts/seeds. Older50/200-rollout budgets
are superseded. Protect existing unrelated scientific jobs/services/owners;
only owned idle whole8GPU nodes are eligible for new GPU work.

Native Wan1.3B T2V and LTX2B0.9.6 adapters, full native trajectory/replay math,
pinned assets/preflight, complete checkpoint counters/RNG, Original/Flash
configurations and the bounded runner are implemented. CPU integrated batch:
231passed,1FFmpeg/ffprobe skip; later parity/config/checkpoint batch113passed.
Split consolidated checks after phased scorer integration:232 new tests and261 affected legacy tests passed (493 total, no skips). The initial combined run exposed fixture-registration and test-spy errors; both corrected, original failures retained privately. Four added single-node/eight-GPU topology guards also passed (498 public tests covered); the private fenced adapter passed17 fake CPU tests and shell syntax checks, with no operational action. Independent review completed once; CUDA DDP placement and custom receipt-binding findings were fixed. GPU validation remains pending.

Primary-code checks corrected LTX's rounded checkpoint allowlist into the
actual LinearQuadratic physical sigma table, and its initial/decode RNG layout.
Native UniPC stores true integer time conditions separately from physical
sigma and preserves conditional frozen-history affine replay.

The named native normalized-mean/canonical-Q2-reference training contract
requires a scientific decision. Checked-in profiles fail closed until that
choice. This is distinct from the independent official base-parity gate.

All8 pretrained GPU/training/checkpoint/effectiveness cells remain pending.
The observed allocation reports roughly140GiB per GPU and inconsistent
name/PCI metadata; no8×H80080GB acceptance is claimed. CPU fixtures alone
establish software/algebra properties, not a mature training recipe.

See docs/small-video.md for protocol, commands, metrics and evidence limits.

A real nonzero PEFT adapter roundtrip now verifies named export, strict fresh loading into the identical immutable base, output equality and disabled-adapter base parity. This adds one CPU test; it does not replace pretrained GPU acceptance.

Cold Python3.10 model-environment installation, editable package import without PYTHONPATH, pip check, native LTX component imports and both model-free parity plans passed. The model lock excludes unused Decord (the separate reward reader remains pinned) and uses OpenCV4.10.0.84 headless to avoid a desktop libGL dependency. Official Wan imports query CUDA and await admitted GPU validation. The canonical reward environment is still a separate environment and has no new cold-install/GPU proof.
