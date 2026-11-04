# Small-video checkpoint, resume and final export

`training.small_video_checkpoint` adds a frozen identity and cumulative budget
envelope to the existing canonical local checkpoint writer. It saves the full
policy state, including frozen base and LoRA parameters, complete Adam state,
Python/NumPy/torch/CUDA RNG states already owned by the process, explicitly
named extra torch generators, the independent rollout-selection RNG, data
cursor and actual optimizer-window membership.

`SmallVideoIdentity` freezes the model and fine-tuning/selection modes, immutable
native code/model/scorer/data revisions, data/protocol/resolved-config hashes,
and the same8 paired evaluation prompt/seed fingerprint. The cumulative limit
is10 actual global optimizer steps, two windows per rollout, across all resumes.
The protocol hash must cover the resolved native solver/schedule/replay/mask,
reward/batch settings and paired evaluation seeds. A different identity fails
restoration.

Construct `SmallVideoResumeState` with the policy, optimizer, `PromptStream`,
`RolloutSelection`, identity and an owned persistent checkpoint root. Use
`state.budget` for window admission; after restoration it is the restored budget
object. The local pre/post optimizer hooks persist an admission before each
actual optimizer step and its committed counter after the step. A checkpoint
older than the journal, an unresolved admission, or an odd/mid-rollout boundary
cannot resume. This prevents restarting an older checkpoint from silently
expanding the10-step budget. Mid-rollout recovery would require the frozen
trajectory/reward/cache and is deliberately unsupported. Keep the original
journal with the checkpoint; do not delete it or create a replacement run root
to bypass consumed steps.

After both disjoint windows complete, update `Progress.optimizer_steps` and
`completed_rollouts`, clear gradients with `zero_grad(set_to_none=True)`, then
call `save_small_video_checkpoint`. The selection RNG rollout count must agree
with completed rollouts. The canonical checksummed receipt and atomic
`latest.json` publication are the checkpoint completion marker. Retention
keeps the most recent2 registered complete checkpoints and does not adopt or
delete unrelated/unregistered experiment files. Failed publication preserves
earlier complete receipts; retry the same in-memory boundary or stop. Do not
rewind consumed optimizer work.

For a fresh load, construct the same policy/LoRA structure and Adam parameter
groups, create fresh selection/data objects and `SmallVideoResumeState` at the
original persistent root, then call `load_small_video_checkpoint`. Supply the
same named extra generator keys. The helper restores complete state and returns
`Progress`; use its counter directly. Do not reinitialize the selection RNG,
data stream or step budget after loading. Treat any failed load as terminal
for those model/optimizer objects, matching the canonical loader's contract.

After the10th actual optimizer step and a final complete checkpoint,
`export_small_video_final` publishes a named `final` directory by staging files,
hashing and writing `complete.json`, then renaming the completed directory.
Existing named exports are never overwritten. Full mode exports `policy.pt`;
LoRA mode exports a PEFT adapter directory. The manifest contains immutable
identity, policy/LoRA structure, source checkpoint receipt and file hashes.
This inference export is separate from the complete resumable checkpoints;
continue training only from the latter. Its creation does not perform an
optimizer update or reset any counter.

The convenience save/load/export helpers are local, unsharded, single-writer
primitives. The state envelope additionally supports an explicit `rank`,
`world_size` and `distributed=True` layout with one owned journal directory
per rank; membership validation then uses64/world_size local window rows.
Pass policy identity captured before DDP/FSDP wrapping to preserve the original
PEFT/parameter schema. An8GPU runner must coordinate those envelopes, per-rank
RNG/journals and replicated identities through the repository's existing
distributed checkpoint adapters. Local CPU tests do not establish DDP/FSDP
checkpoint correctness or pretrained GPU training parity. No distributed
checkpoint validation is claimed here.

The runner's `SMALL_VIDEO_PAUSE_AFTER_UPDATE=4` is an operational pause after a
complete update4 checkpoint; it leaves `training.max_updates=10` unchanged.
Unset it before resuming that checkpoint to execute exactly6 remaining updates.
It never authorizes a new budget or checkpoint rewind.

For a separately requested general held-out evaluation, `run_native` accepts
`evaluation_dataset` and `evaluation_label` only in evaluation-only mode with
an explicit complete checkpoint. Load that dataset with a verified manifest
SHA256. Normalized text overlap with training prompts is rejected, and all
ranks agree on the extra dataset hash and safe single-component output label.
The original paired8 identity and scientific config stay unchanged. Full
model/Adam/RNG state is restored before evaluation; no training window,
selection draw or data cursor is advanced. The parent evaluation helper keeps
collective forward counts equal even when the extra population needs dummy
rank inputs. This capability does not enlarge the acceptance run's paired8
evaluation budget.

Evaluation uses an explicit read-only envelope. It may load an older complete
checkpoint whose counter is at or below the original journal highwater, while
leaving that journal byte-for-byte unchanged. Every optimizer step is rejected
and the read-only envelope cannot publish a training checkpoint. Training
resume continues to require exact journal/checkpoint agreement; this exception
does not authorize rollback or a fresh budget.

The optional phased VideoReward adapter uses a separate pinned canonical reward
interpreter for each scoring request. It retains the canonical DanceGRPO scorer,
raw VQ/MQ/TA, reward4fps preprocessing and weights. The worker sees only the
parent rank's existing physical GPU, including UUID visibility mappings; no
additional scorer GPU or persistent service is used. Parent model/FSDP shards
may remain resident. The adapter waits for the owned child to exit before
returning scores and continuing backward.

Each phase retains bounded atomic request/response JSON, a bounded combined
worker log and a process audit with PID/start ticks/PPID/PGID, visibility and
safe runtime versions. Torchrun/Accelerate bootstrap variables are removed,
model access is offline, caches stay in the owned phase directory, and Linux
parent-death fencing precedes third-party/CUDA imports. Timeouts terminate and
reap only the owned child, preserving the trainer's process group.

Worker loading/scoring/total times, its own CUDA allocator peaks and the parent
resident allocation are recorded separately. Available NVML figures are actual
total-device snapshots after loading/scoring; they are not sampled peak usage.
The parent Torch allocator alone does not measure scorer-process memory, and
these metrics do not establish an8×80GB hardware result.


For a fresh **inference export** load, `visual_rl.small_video_export.verify_final`
checks the complete inventory and identity; `load_final_into_adapter(config,
final_directory, prepared_adapter)` loads only policy weights into the matching
fresh native base/LoRA structure. It restores no Adam, RNG or training counter.
Use the full resumable checkpoint interface above for training continuation.
For LTX full inference on a single device, cast the freshly loaded transformer
to BF16 and use its explicit native BF16 forward contract. This inference-only
cast is not the full training path: full training uses the LTX-specific FSDP
BF16 compute parameters, FP32 optimizer masters/reduction, and FP32 times and
position coordinates. LoRA keeps FP32 adapter masters and scopes BF16 autocast
only to its A/B GEMMs; do not cast all its trainable parameters to BF16.
Wan retains its separate native FP32-parameter/BF16-autocast time/rotary path.
These precision contracts are included in the resolved protocol and its hash.
