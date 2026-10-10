# W&B and Local Metrics

Training always writes `metrics.jsonl`; complete evaluations publish their own
`evaluation/<step>/<id>/summary.json`. These files are the source of truth.
Full-budget paper configs enable live W&B by default. The SDK is installed with
the package. For a custom config, select:

```yaml
runtime:
  log: wandb
  wandb:
    project: my-diffusion-experiments
    name: my-run
    mode: online
```

```bash
# Use existing credentials or run `wandb login` interactively first.
# Alternatively, supply WANDB_API_KEY from your secret store/environment.
export WANDB_PROJECT=my-diffusion-experiments
export WANDB_ENTITY=my-team
export WANDB_MODE=online
python -m visual_rl.train --config configs/my-experiment.yaml
```

Without `WANDB_MODE`, export defaults to **online**. Missing local credentials
are reported before downloading assets or loading GPU models. No token is
written to experiment config or logs. Set `WANDB_MODE=offline` to record SDK
files without an account/network, or `runtime.log: jsonl` to disable W&B.
The same switch works with SD3/FLUX single-device/DDP and Hunyuan/Wan
single-device/FSDP. Only global rank zero starts an exporter. The tensor fixture
remains JSONL-only.

`runtime.wandb` accepts `project`, `entity`, `name` and `mode`. Corresponding
`WANDB_*` environment variables override these values. Credentials are never
configuration fields. Resume exports use the active `resume-config.json`.

## Axes and Scores

- Training panels default to `train/optimizer_step`, not W&B's internal `_step`.
- Evaluation panels use `eval/optimizer_step`; eval0 and delayed evaluation
  records never overwrite a training step.
- Select `cost/elapsed_gpu_hours` or `cost/training_gpu_hours` for cost plots.
  These are the trainer's recorded clocks. Exporter's W&B `_runtime` is **not**
  a training-cost metric, especially when replaying history.
- Raw and fixed-normalized reward names are unchanged. ReFL's differentiable
  optimization scores remain separate from the common MP4 reporting scores.

The exporter uploads numeric metrics only. Evaluation videos stay on disk;
it does not upload checkpoints, datasets, source code or videos.

## Ownership and Recovery

An independent, CPU-only process owns all SDK calls. The trainer only writes its
existing JSONL; there is no SDK call or upload wait inside an optimizer update.
On exit, export gets up to ten seconds to drain. A separate deadline guard also
handles a dead parent or SDK shutdown hang. An incomplete export never claims
successful upload and does not invalidate already saved local training data.
Exporter-only directory/status failures warn rather than stop training. This
does not make an unresponsive filesystem safe for the trainer's own files.

Each export has a fresh W&B ID and uses the experiment name as both name and
group. **Training resume creates a new display run containing the full available
history**, not an append to the prior cloud run. It preserves the actual
optimizer steps and cumulative cost clocks; use the latest complete export for
figures, not both duplicate histories. This also avoids replaying uncertain SDK
writes into the same run.

Inspect `wandb-exports/<export-id>/status.json` for the run URL, SDK directory,
record count and export state, and `<export-id>.owner.json` for process exit
status. `pending_or_unknown` means completion was not established, not that
the training itself failed. `finished` in offline mode means local SDK storage
completed, not cloud upload. Upload offline SDK files with `wandb sync` on the
recorded run directory.

To recreate a display run from an existing output directory without restarting
training:

```bash
WANDB_MODE=online python -m visual_rl.wandb_export \
  --root outputs/my-experiment \
  --destination outputs/my-experiment/wandb-exports/manual-replay-01 \
  --timeout-seconds 120
```

The destination must not already exist. This command reads local records and
does not load models, use GPUs or modify their scores. Its deadline includes SDK
initialization and finish. Give a large history a larger explicit timeout.

W&B's [custom metric axes](https://docs.wandb.ai/ref/python/experiments/run/)
separate displayed training steps from the SDK's append-only event counter.
