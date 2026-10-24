# DiffusionART-Flash

DiffusionART-Flash trains at the initial denoising position while retaining
the complete rollout for reward evaluation and covariance guidance. The switch
works with the SD3, FLUX, HunyuanVideo, distilled Wan and MiniMax-H3 ART runners,
including their supported DDP/FSDP and full/LoRA paths. The comparison shown in
the README is a HunyuanVideo experiment; it does not establish quality on every
model family.

## Enable

Add this field to an existing ART YAML, or inherit one:

```yaml
extends: hunyuan_art.yaml
name: hunyuan-art-flash
training:
  timestep_selection: first
runtime:
  output_dir: outputs/hunyuan-art-flash
```

`configured` is the default and preserves the backend's existing position
selection. `first` overrides it with rollout index `[0]`, including when the
configured pool/list does not start at zero. In the shipped schedules this is
sigma approximately one. This field changes training replay only: do not change
`sampling.steps` to one. The original next sigma remains in the ART coefficient
and mask calculation; it is not replaced by zero. Other algorithms reject the
Flash switch rather than silently ignoring it.

| Model | Example |
| --- | --- |
| SD3 | [sd3_art_flash.yaml](../configs/examples/sd3_art_flash.yaml) |
| FLUX | [flux_art_flash.yaml](../configs/examples/flux_art_flash.yaml) |
| HunyuanVideo | [hunyuan_art_flash.yaml](../configs/examples/hunyuan_art_flash.yaml) |
| Wan I2V | [wan_art_flash.yaml](../configs/examples/wan_art_flash.yaml) |
| MiniMax-H3, 8 GPUs | [minimax_h3_art_flash_full_8gpu.yaml](../configs/examples/minimax_h3_art_flash_full_8gpu.yaml) |
| MiniMax-H3, 32 GPUs | [minimax_h3_art_flash_full_32gpu.yaml](../configs/examples/minimax_h3_art_flash_full_32gpu.yaml) |

Set the same model/data/asset environment bindings as each parent recipe, then
use its normal launcher, e.g.:

```bash
python -m visual_rl.train --config configs/examples/hunyuan_art_flash.yaml --validate
python -m visual_rl.train --config configs/examples/hunyuan_art_flash.yaml --nproc-per-node 8
```

These are inherited integration recipes. For a paper profile, inherit that
profile and override only `training.timestep_selection`, name and output path.
Keep its batch, group size, optimizer, rewards, schedules and update windows.

## Normalization and checkpoints

Each update averages over its actual selected sample–timestep pairs. For
uniform selection, the denominator is `window_trajectories * selected_positions`;
distributed accumulation sums the participating ranks' denominators and corrects
the gradient averaging. Masked items contribute zero loss but remain in the
original denominator. Physical microbatch boundaries do not change this mean.

For the README comparison, each update contains 128 trajectories. Original ART
trains 9 positions per trajectory (1,152 training items); Flash trains 1 (128
items). Each rollout still generates 256 trajectories and makes 2 disjoint
updates. The temporal averaging denominator becomes 1; there is no extra
division by the original 9. This reduces backward work, not the number of
generated trajectories or complete prompt groups used for guidance.

Default `configured` is omitted from serialized configs, preserving historical
config/checkpoint hashes. `first` is recorded and changes the training signature;
resuming between Original and Flash requires a new experiment rather than
bypassing configuration checks.

## README evidence

The figure uses raw training VQ/MQ from these two seed 42, 32-GPU HunyuanVideo runs:

- [Original, 9 positions per trajectory](https://wandb.ai/vanzl3386-chinese-university-of-hong-kong-shenzhen/flow-grpo-algo/runs/4dnkgeto).
- [DiffusionART-Flash, initial position only](https://wandb.ai/vanzl3386-chinese-university-of-hong-kong-shenzhen/flow-grpo-algo/runs/gswb6l1m).

Both use 16-step generation, P32/K8, VQ:MQ=1:1, the same model/reward revisions,
mask settings, optimizer and physical microbatches. Execution commits and
asset locations differ. The figure uses raw measurements without extrapolation,
smoothing or invented error bars.

One raw reward is logged twice per rollout, once per update. The plot retains
the second record as the completed-rollout point. Final 10-rollout means are
VQ/MQ 15.071/11.133 for Original and 14.645/11.154 for Flash; these are training
scores from one seed, not held-out evaluation or multi-seed equivalence tests.
The 9× figure describes fewer update items, not an asserted 9× wall-clock speedup.

Frozen [tidy data](../assets/readme/art-flash/curves.csv) and
[metadata](../assets/readme/art-flash/metadata.json) include exact keys, identities,
cost scope and plotting choices. Regenerate the SVG, PDF and 600-DPI PNG with:

```bash
python scripts/plot_art_flash_readme.py
```
