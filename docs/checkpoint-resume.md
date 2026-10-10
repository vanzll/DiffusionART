# Checkpoints and Resume

Checkpoints contain model, optimizer, random generators, data position and
algorithm state, including NFT old-policy/EMA state. Resume requires the same
experiment config and distributed topology; model-only weights are not a
training checkpoint.

Periodic checkpoints default to every 200 successful optimizer updates.
Explicit `checkpoint_at_updates` are additional save points; when checkpointing
is enabled, the final update is also saved. Set cadence and retention in YAML:

```yaml
runtime:
  checkpoint_every: 200
  checkpoint_keep: 2
```

`checkpoint_keep: null` retains all checkpoints. Distributed cleanup runs in
the background; delayed cleanup may temporarily leave extra checkpoints.

Use `latest` to resolve the most recent published checkpoint in the configured
output directory:

```bash
python -m visual_rl.train --config configs/my-experiment.yaml \
  --resume latest
```

For distributed training, append the same `--resume` argument to the original
`torchrun` command on every node. Keep the same world size and assets.
An explicit receipt/manifest, `latest.json`, run directory or checkpoint
directory is also accepted. Incomplete directories are not selected.

`--max-updates 1500` extends the total budget, not the number of additional
updates. Output location, checkpoint/eval cadence, retention and logging may
change on resume; algorithm, model, rewards, sampling, optimizer and physical
microbatch settings must still match. Learning rates remain those restored
with AdamW; no schedule is silently restarted or rescaled. The original
`resolved-config.json` stays immutable; `resume-config.json` records active
operational settings. Assets and checkpoint lineage retain their original
identity, including checkpoints made by earlier versions.

For DiffusionART, `runtime.max_consecutive_no_signal_rollouts` accepts a positive
integer to stop with an error on that many consecutive globally inactive
rollouts. Omission or `null` preserves unlimited skipping. Warnings occur at
streak lengths 1, 2, 4, 8, and so on, on rank zero for distributed runs. Existing
scalar/pairwise skipped metrics are retained; skipped rollouts still apply no
optimizer or EMA updates. A useful optimizer update resets the streak. Pairwise
restoration activity still counts as useful under its existing global rules.
This operational budget may change on resume without changing checkpoint
identity. The streak is process-local and starts at zero on resume; no new
checkpoint state is required.

If an interrupted run has metrics or evaluation artifacts newer than the saved
checkpoint, the default is to refuse resume. Add `--resume-rewind` to explicitly
archive that future history before continuing; recorded physical costs remain.
Only load checkpoints you trust.

To evaluate a checkpoint without updating parameters or training history:

```bash
python -m visual_rl.train --config configs/my-experiment.yaml \
  --resume /absolute/path/to/checkpoint-receipt.json --eval-only
```

Use the original config with evaluation enabled and the original launch
topology. Evaluation media and scores are saved locally. See
[distributed execution](distributed-execution.md) for DDP/FSDP checkpoint details.
For standard SD3/FLUX inference artifacts, see [policy export](model-export.md).
