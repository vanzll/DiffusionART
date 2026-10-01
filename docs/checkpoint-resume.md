# Checkpoints and Resume

Checkpoints contain model, optimizer, random generators, data position and
algorithm state, including NFT old-policy/EMA state. Resume requires the same
experiment config and distributed topology; model-only weights are not a
training checkpoint.

Set storage cadence and retention in YAML:

```yaml
runtime:
  checkpoint_every: 100
  checkpoint_keep: 2
```

`checkpoint_keep: null` retains all checkpoints. Distributed cleanup runs in
the background; delayed cleanup may temporarily leave extra checkpoints.

Read `checkpoints/latest.json` and resolve its `receipt` (single device) or
`manifest` (DDP/FSDP) relative to the checkpoint directory. Pass that complete
receipt/manifest, not the distributed `latest.json` pointer:

```bash
python -m visual_rl.train --config configs/my-experiment.yaml \
  --resume /absolute/path/to/checkpoint-receipt.json
```

For distributed training, append the same `--resume` argument to the original
`torchrun` command on every node. Keep the same world size and assets.

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
