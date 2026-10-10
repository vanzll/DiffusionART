# Distributed Execution

Native SD3/FLUX ART, NFT, Flow-GRPO, Fast and Flow-DPPO, plus SD3 DanceGRPO,
support `runtime.distributed: ddp` through the same
configuration entrypoint. This is replicated DDP, not FSDP or model parallelism.
Image FSDP mode remains rejected. Hunyuan ART separately
integrates full-shard execution as described below. Pretrained GPU/paper acceptance is
pending; the tests below use tiny real model components on CPU.

## Launch

Prepare the model/reward assets using the relevant SD3 or FLUX guide first.
Every worker must see those assets and the same output/data paths. DDP keeps a
full model/optimizer replica per GPU; adding GPUs does not reduce that memory.
The two-rank ART/NFT examples keep P=2 and K=2; Fast examples use P=2 and K=4.
All run four successful global optimizer updates.

```bash
uv sync --locked --python 3.10 --extra diffusers --extra image-rewards
uv run --locked --extra diffusers --extra image-rewards visual-rl \
  --config configs/examples/sd3_art_ddp.yaml --validate
uv run --locked --extra diffusers --extra image-rewards torchrun \
  --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29500 \
  -m visual_rl.train \
  --config configs/examples/sd3_art_ddp.yaml
```

Other examples are `sd3_nft_ddp.yaml`, `flux_art_ddp.yaml`, `flux_nft_ddp.yaml`,
`sd3_fast_ddp.yaml` and `flux_fast_ddp.yaml`.
Use an available port; change 29500 if another job owns it. Explicit IPv4 avoids
a hostname/IPv6-resolution stall observed with standalone rendezvous on the
development Mac. The actual two-worker validation command succeeded with an
explicit loopback address; this is not a GPU training acceptance result.

Use the same normal entrypoint and change the configuration. For multiple nodes,
use torchrun's allocation-appropriate rendezvous/node settings and a reachable
master address, not loopback; no private launcher or agent skill is required. Each node needs
the same package/environment/assets. `VISUAL_RL_COLLECTIVE_TIMEOUT_SECONDS`
controls the collective timeout (default 300 seconds); it does not bound disk
operations. Do not launch a `distributed: none` configuration with multiple
workers: that combination is rejected before loading models or writing output.

Resume using the same command with `--resume latest`, a run/checkpoint
directory, or an explicit complete `manifest.json`. Increase `--max-updates`
to extend a finished run. History newer than the checkpoint requires explicit
`--resume-rewind`; discarded work is archived and its elapsed cost is retained.
See [checkpoint resume](checkpoint-resume.md).

## Native Semantics

P is global and must divide by DP degree. K is unchanged and each prompt slot's
K trajectories stay on one rank. ART requires an even local trajectory count
and makes two disjoint local windows, producing two global optimizer updates.
NFT defaults to one global update with independent old-policy and evaluation
EMA state. Its optional two-window recipe retains the old policy through both
updates and refreshes it only at the rollout boundary; evaluation EMA updates
after each successful step. Local rows must divide evenly into those windows.
Parameters/scorers initialize from the same seed; subsequent stochastic work
uses `seed + rank`. PromptStream selects one shared global order before sharding.
This ordering is explicitly recorded in run.json, not claimed bit-identical to
historical DistributedSampler runs or to a different world size.

Reward values and diagnostics are gathered in global row order before computing
NFT/GRPO advantages, optional ART multi-reward guidance or reported means. No rank
substitutes its local reward population for the configured global one. Losses
retain their original global denominator through DDP accumulation.

GRPO keeps `training_batches` logical batches **per rank** and the configured
global updates per rollout. Both logical training and sampling batches must
divide the local trajectory count. With `filter_pad`, all ranks pad to the
maximum nonzero-advantage row count, rounded up to a logical-batch multiple.
Padding selects existing zero-advantage trajectories, which remain in the
original loss denominator. A rank with no signal still participates; only a
globally empty rollout skips optimizer/EMA updates. Its sampling cost is retained.
The configured consecutive-empty-rollout limit stops an unproductive run.
Evaluation EMA retains its configured optimizer/logical-batch refresh cadence.

Rollout, old-policy predictions and evaluation use the unwrapped inference model;
gradient-bearing forwards use DDP. Evaluation is sharded by global sample index,
including unequal/empty shards, with seed `experiment seed + global index`.
Rank zero verifies exact identities and publishes one combined manifest/summary.
It alone owns metrics, run metadata and the output lock. Per-rank costs are
reduced by maximum (not sum); GPU-hours multiply by actual CUDA world size.
Elapsed cost includes eval/checkpoint since clock initialization after model
setup; training-only cost excludes them. CPU execution reports zero GPU-hours.

## Execution Primitives

`training.distributed.distributed_session` owns a fresh default process group.
It reads torchrun's `RANK`, `WORLD_SIZE`, `LOCAL_RANK` and `LOCAL_WORLD_SIZE`,
validates a homogeneous allocation, selects the local CUDA device before NCCL
initialization, and requires an explicit finite collective timeout. CPU tests
use Gloo. It never adopts or destroys a caller-owned group and does not add an
exit barrier that would wait for failed peers. Use a supervisor such as torchrun
to terminate peers after an unrecoverable rank failure; a collective timeout
does not bound a hung driver or filesystem operation.

All ranks can check a resolved scientific configuration or data-manifest SHA256
with `session.require_same_digest`. The check exchanges fixed-size numeric
tensors, not pickled objects. Call it on every rank in the same order, before
model-dependent work. Rank-local output paths must not enter that identity.

`RankInfo.local_prompt_count(P)` preserves the **global** prompt count and rejects
non-divisible allocations rather than rounding or increasing the batch. Each DP
rank owns complete K-sample prompt groups. For example, P=32 and DP=8 gives four
prompts per rank; K remains unchanged. This is pure DP, not a model-parallel mesh.

## Gradient Contract

`training.update.accumulated_step` uses the DDP model's actual process group for
validation, denominator reduction, loss reporting and world-size correction.
It does not assume DDP always covers the default world group.

Each rank supplies its original local window denominator. The loss sum is scaled
by `DP group size / sum(local denominators)` before backward, compensating for
standard DDP gradient averaging. Masked samples keep their original denominator
weight; explicitly inactive DDP placeholders have zero loss and no denominator
weight. GRPO's retained zero-advantage trajectories are not such placeholders:
they remain real rows in its original denominator. Ranks must
execute equal physical microbatch counts and the same model-collective sequence.
Custom DDP communication hooks that change gradient averaging are not covered.

Input preparation errors are coordinated before model collectives. A forward
or backward exception is terminal: do not attempt to resume that process group
or issue a new error-reporting collective while peers may be inside DDP.

## Evidence and Remaining Work

The DDP checkpoint primitive below is wired into the native image runner.

CPU multiprocess tests compare standard DDP accumulation with an independent
single-process update, including padding and unequal local denominator weights.
Additional tests create independent DDP subgroups with different window lengths
and compare each against a plain PyTorch AdamW update. They also exercise digest
mismatch, malformed identity and process-group cleanup. CUDA device mapping is
tested with mocks, not a GPU acceptance claim.

Native two-process tests exercise actual tiny SD3/FLUX modules, ART/NFT/GRPO updates,
parallel evaluation, simulated interruption, full-state resume and completed
no-op resume. They compare model, optimizer, shadows and sampling state against
uninterrupted execution at the same topology. FSDP, real pretrained GPU
memory/throughput and frozen paper-profile parity
remain to be integrated or accepted for the remaining backends.

## Hunyuan Full-Shard Execution

The [Hunyuan runner](hunyuan-runner.md) uses the same torchrun ownership and
configuration checks with `distributed: fsdp`. It does not reuse DDP's
unwrapped inference or `no_sync` behavior. All ranks participate in every model
forward and gradient reduce-scatter. The shared accumulator scales losses by
`world_size / sum(original local denominators)` and clips with FSDP's global
norm. Masked items retain denominator weight. FULL_SHARD with FP32 masters and
no CPU parameter offload is the supported video recipe.

Evaluation uses equal collective forward counts with unscored final-round
padding, then rank-zero identity validation and publication of all real videos.
Model exceptions terminate the group rather than entering an unrelated error
collective. Process-group timeouts and the torchrun supervisor own peer teardown.

`training.fsdp_checkpoint` uses PyTorch's LOCAL_STATE_DICT representation and
rank-local AdamW state. It never gathers full model/optimizer tensors. Layout,
original parameter identities, world/process-group membership, PyTorch version,
config and successful-update counters must agree. Each rank stores its own
RNG, generators, buffers and caller-owned state. This is same-topology resume,
not portable weight export or automatic elastic resharding.

New schema-1 publications explicitly declare `model_format: cpu-local-shards-v1`
in the manifest and each payload. Native LOCAL_STATE_DICT extraction keeps
ShardedTensor objects on their original device (`offload_to_cpu=False`): native
CUDA-to-CPU offload can lose `requires_grad` consistency. Only dense local shard
tensors are detached/copied to CPU, with plain size/offset/placement/tensor-property
metadata and module state-dict version metadata preserved. Loading validates the
same layout and every local shard before copying into live ShardedTensor objects;
it neither reconstructs sharded objects from pickle nor adds construction
collectives. Padding and ranks with no local shard remain supported.

Legacy schema-1 checkpoints without `model_format` retain their existing trusted
ShardedTensor loading path. Unknown formats or manifest/payload format mismatches
are rejected; new-format checkpoints require the updated loader (old readers
are not forward-compatible). Optimizer, rank-local RNG, generators, data cursor and NFT shadow
components retain their existing complete-resume ownership; this is not an
inference export. CPU two-rank tests cover exact continued training, legacy loads,
metadata/shape/dtype rejection, corruption and retention. A two-GPU CUDA/NCCL BF16
resume regression is included but skips without two CUDA devices; it has not been
executed on the CPU-only development machine.

Payload checksums and manifests are published only after all writers report
success. Failed payload publication does not advance `latest.json`. The storage
root must be shared and responsive; no claim of a hard filesystem deadline is
made. Resume stages and verifies all rank payloads before applying live state;
an application failure is terminal rather than a partial rollback.

## Full-State DDP Checkpoints

`training.distributed_checkpoint.save_ddp_checkpoint` and
`load_ddp_checkpoint` are collective entrypoints: every member of the model's
DDP process group calls them at the same completed optimizer boundary, after
gradient buffers have been cleared. The root must be the same shared,
responsive filesystem path on all participants. These primitives do not
provide a storage deadline and do not support FSDP or sharded optimizers.
The caller owns exclusive publication rights for the checkpoint root; unrelated
jobs or independent DDP groups must use different roots.

State ownership is explicit:

- Model parameters and the synchronized optimizer are stored once by group
  rank zero. Callers must preserve standard DDP optimizer synchronization.
- `replicated_stateful` is for synchronized components such as policy shadows.
  Callers guarantee these components are identical across ranks.
- `stateful` stores each rank's data cursor and other local components.
- Each rank separately stores its model buffers/nonparameter state, progress,
  named generators, Python/NumPy/CPU RNG and its own CUDA device RNG. Capturing
  RNG does not initialize contexts on every GPU visible to the process.
  PyTorch state-dict version metadata is preserved per rank as well.

Saving first agrees on configuration, successful-update counts, component names
and topology. Payload files are checksummed and atomically renamed. The manifest
is published only after every rank acknowledges successful writes; `latest.json`
then points to that manifest. Interrupted writes can leave an incomplete
directory, but no valid manifest is published for a failed payload set and the
previous latest pointer is preserved. If manifest publication succeeds but the
latest-pointer write fails, the explicit manifest may still be valid; the
operation reports failure rather than claiming the pointer advanced.

Pass the returned `manifest.json` path explicitly to load. Loading checks file
integrity, world size, config identity, step, rank, model shape/dtype and exact
component names on all ranks before applying any state. World-size changes are
not silently resharded. Component-specific load errors are terminal: some live
state may already have changed, so the caller must exit rather than continue.

CPU two-rank tests compare continuous training with save/recreate/restore and
further training. Model, AdamW, shadow, local buffer, data cursor and subsequent
random draws match exactly. Fault injection covers corrupt/missing payloads,
rank-step mismatch, wrong topology/components and one-rank storage failure.
Real CUDA resume and a slow/unresponsive shared filesystem remain separate
acceptance work.
