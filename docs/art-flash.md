# DiffusionART-Flash

DiffusionART-Flash concentrates policy updates in the high-noise part of a
complete denoising trajectory. For the 16-step HunyuanVideo recipe, sample two
of the first four positions for each rollout. This greatly reduces the number
of sample–timestep pairs processed during updates while achieving comparable
performance to training across the full denoising schedule.

## Enable

```yaml
extends: hunyuan_art.yaml
name: hunyuan-art-flash
sampling:
  steps: 16
training:
  timestep_selection: high_noise
runtime:
  output_dir: outputs/hunyuan-art-flash
```

`high_noise` draws two distinct zero-based positions from `[0, 1, 2, 3]` and
replays them in chronological order. One pair is shared across the global
rollout: every rank, trajectory, modality and both optimizer windows use the
same selection. The private selection RNG depends on the experiment seed and
rollout attempt, so retries and checkpoint continuation reproduce the pair
without consuming the generation RNG. Each new rollout resamples; a pair may
recur.

Keep the complete generation schedule. The original next sigma and cached
prediction at each chosen position remain in the ART coefficient and mask.
Prompt groups, rewards, covariance, batches and optimizer updates per rollout
remain unchanged. This is training replay selection, not shortened generation.

| Model | Example |
| --- | --- |
| SD3 | [sd3_art_flash.yaml](../configs/examples/sd3_art_flash.yaml) |
| FLUX | [flux_art_flash.yaml](../configs/examples/flux_art_flash.yaml) |
| HunyuanVideo | [hunyuan_art_flash.yaml](../configs/examples/hunyuan_art_flash.yaml) |
| MiniMax-H3, 8 GPUs | [minimax_h3_art_flash_full_8gpu.yaml](../configs/examples/minimax_h3_art_flash_full_8gpu.yaml) |
| MiniMax-H3, 32 GPUs | [minimax_h3_art_flash_full_32gpu.yaml](../configs/examples/minimax_h3_art_flash_full_32gpu.yaml) |

Set the model/data/asset environment bindings required by the parent recipe,
then use its normal launcher:

```bash
python -m visual_rl.train --config configs/examples/hunyuan_art_flash.yaml --validate
python -m visual_rl.train --config configs/examples/hunyuan_art_flash.yaml --nproc-per-node 8
```

These are inherited integration recipes. For an existing training profile,
override only the selection, name and output path. Keep its batch sizes,
optimizer, rewards and rollout schedule. The README plots show training VQ/MQ
for HunyuanVideo; support in another runner is not a quality claim for that model.

## Loss averaging and compatibility

Each update averages over the actual selected sample–timestep pairs. Two
selected positions give a temporal denominator of **2**, regardless of the
original pool size. Masked items remain in the denominator; microbatch and rank
boundaries do not change the global mean. For a 128-trajectory window, the
update processes 256 training pairs. Relative to 16 positions, selecting two
uses eight times fewer pairs; relative to a recipe excluding the final position,
it uses 7.5 times fewer. This reduction applies to update items, not generation.

`configured` remains the default and preserves the existing backend selection.
The previous `first` mode remains available for compatibility. The native
four-transition Wan recipe retains its [legacy Flash example](../configs/examples/wan_art_flash.yaml):
it has only three eligible ART positions and rejects `high_noise` rather than
changing its rollout or final-position policy. Other short schedules also fail
validation if they do not contain four eligible positions.

Default `configured` is omitted from serialized configs, preserving historical
checkpoint identity. Nondefault selections are recorded in the training
signature. Changing the selection requires a new experiment; do not bypass
resume checks.
