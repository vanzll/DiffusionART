<h1 align="center">DiffusionART</h1>

<p align="center"><strong>A configurable RL framework for visual generation</strong></p>
<p align="center">Text-to-image &nbsp;&middot;&nbsp; Text-to-video &nbsp;&middot;&nbsp; Image-to-video</p>

<p align="center">
  <a href="https://vanzll.github.io/diffusionart-project/">Project Page</a> &nbsp;/&nbsp;
  <a href="#quick-start">Quick Start</a> &nbsp;/&nbsp;
  <a href="docs/README.md">Documentation</a> &nbsp;/&nbsp;
  <a href="#paper-experiments">Paper Experiments</a>
</p>

<p align="center">
  Zhenglin Wan, Jiayu Chen, Zhengfeiyang Zhang, Pengfei Zhou, Wangbo Zhao,<br>
  Fangyuan Kong, Pengfei Wan, Kun Gai, Yang You<br>
  <sub>HPC-AI Lab, National University of Singapore &middot; Kling AI &middot; HKUST</sub>
</p>

Train diffusion models with **DiffusionART, DiffusionNFT, Flow-GRPO, Flow-GRPO-Fast,
Flow-DPPO, DanceGRPO, and ReFL**. Choose the model, algorithm, reward, and training
recipe in YAML; use the same entrypoint for rollout, updates, evaluation, and resume.

- **Composable recipes.** Switch models and rewards without rewriting the training loop.
- **Explicit algorithms.** Preserve each method's sampling, update schedule, and policy state.
- **A complete lifecycle.** DDP/FSDP, full-state checkpoints, evaluation galleries, and W&B logging.
- **Comparable metrics.** Training steps, raw rewards, and training/elapsed GPU-hours.

## Visual Results

<table>
  <thead><tr><th></th><th>Pre-trained</th><th>DiffusionART</th></tr></thead>
  <tbody>
    <tr>
      <th>Text-to-image</th>
      <td><img src="assets/readme/image-chicago-before.jpg" width="320" alt="Pre-trained image: a snowy Chicago street at Christmas"></td>
      <td><img src="assets/readme/image-chicago-after.jpg" width="320" alt="DiffusionART image: a snowy Chicago street at Christmas"></td>
    </tr>
    <tr>
      <th>Text-to-video</th>
      <td><img src="assets/readme/video-106-before.jpg" width="320" alt="Pre-trained video frame: a river through an autumn valley"></td>
      <td><img src="assets/readme/video-106-after.jpg" width="320" alt="DiffusionART video frame: a river through an autumn valley"></td>
    </tr>
  </tbody>
</table>

Selected paper examples; video cells show one frame. Visit the
[project page](https://vanzll.github.io/diffusionart-project/) for playable
comparisons and quantitative results. [Sample provenance](assets/readme/provenance.json).

## Quick Start

### 1. Check Your Installation

Use Python 3.10+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).
This four-update CPU check needs no model downloads, GPU, or W&B account.

```bash
git clone https://github.com/vanzll/DiffusionART.git
cd DiffusionART
uv sync --locked --python 3.10
uv run --locked visual-rl --config configs/examples/cpu_smoke.yaml --validate
uv run --locked visual-rl --config configs/examples/cpu_smoke.yaml
```

Check `outputs/cpu-smoke/status.json`: `status` should be `completed` and
`optimizer_steps` should be `4`. To repeat, pass `--output-dir outputs/cpu-smoke-2`.
This checks the software lifecycle, not pretrained image generation.

### 2. Prepare a Model

Pretrained training runs on Linux with CUDA. Follow the matching guide for
dependencies, model access, pinned assets, and data preparation:

| Task | Models | Setup |
| --- | --- | --- |
| T2I | SD3.5-M, FLUX.1-dev | [SD3](docs/sd3-runner.md#install-and-prepare) / [FLUX](docs/flux-runner.md) |
| T2V | HunyuanVideo-13B | [Video environment](docs/video-installation.md) / [Hunyuan assets](docs/hunyuan-runner.md) |
| I2V | CFG-distilled Wan-14B | [Video environment](docs/video-installation.md) / [Wan assets](docs/wan-runner.md) |

Image environments use `uv --extra diffusers`; video environments use the pinned
installer in the video guide. Prepare the assets before launching training.

### 3. Run a Recipe

For SD3, after completing the setup above:

```bash
uv run --locked --extra diffusers visual-rl \
  --config configs/examples/sd3_pickscore.yaml
```

For eight-GPU image training, prepare the [paper profile](docs/paper-image.md) first:

```bash
uv run --locked --extra diffusers torchrun --standalone --nproc-per-node=8 \
  -m visual_rl.train --config configs/paper/image/sd3_pickscore_art.yaml
```

For video training, activate the prepared video environment and use the same CLI:

```bash
torchrun --standalone --nproc-per-node=8 -m visual_rl.train \
  --config configs/paper/video/hunyuan_art_721.yaml
```

See [distributed execution](docs/distributed-execution.md) for multi-node launches.
Add `--validate` to check a config before loading models. Use `--resume` with a
full-state checkpoint receipt to continue training.

## One Config, One Experiment

Start from an [example](configs/examples/) or a [paper profile](configs/paper/).
For instance, save this as `configs/my-pickscore.yaml`:

```yaml
extends: examples/sd3_pickscore.yaml
name: my-pickscore
training:
  max_updates: 1000
runtime:
  output_dir: outputs/my-pickscore
```

```bash
uv run --locked --extra diffusers visual-rl --config configs/my-pickscore.yaml
```

The config owns `algorithm`, `model`, `reward`, `sampling`, `training`, and
`evaluation`. Inheritance keeps recipes short; overrides select the experiment.
Changing a component resets its old options, and reward weights replace the
whole weight map. Required asset variables use `${ENV_VAR}`.

Local metrics are always written to `metrics.jsonl`. Enable [W&B](docs/logging.md)
with `runtime.log: wandb`; evaluation images and videos stay on disk with an
HTML gallery. [Prompt manifests](docs/prompt-data.md) preserve sample identity.

## Supported Methods

| Model | Training algorithms |
| --- | --- |
| SD3 / SD3.5 | DiffusionART, DiffusionNFT, Flow-GRPO, Flow-GRPO-Fast, Flow-DPPO, DanceGRPO |
| FLUX.1-dev | DiffusionART, DiffusionNFT, Flow-GRPO, Flow-GRPO-Fast, Flow-DPPO |
| HunyuanVideo / Wan I2V | DiffusionART, DiffusionNFT, Flow-GRPO-Fast, DanceGRPO, ReFL |

Image rewards include **PickScore, GenEval, OCR, Aesthetic, and CLIPScore**;
video recipes use **VideoReward (TA, VQ, MQ)**. Compose scalar rewards with
explicit weights, or use DiffusionART's binary pairwise feedback. ReFL has its
own differentiable reward path. See [algorithm contracts](docs/algorithm-provenance.md)
and [reward composition](docs/composite-rewards.md).

## Paper Experiments

| Task | Recipes | Selected DiffusionART Runs |
| --- | --- | --- |
| T2I | [Image profiles](docs/paper-image.md) | [GenEval](https://wandb.ai/vanzl3386-chinese-university-of-hong-kong-shenzhen/flow-grpo-algo/runs/11pq0iy3) / [PickScore](https://wandb.ai/vanzl3386-chinese-university-of-hong-kong-shenzhen/flow-grpo-algo/runs/mofumvbw) / [OCR](https://wandb.ai/vanzl3386-chinese-university-of-hong-kong-shenzhen/flow-grpo-algo/runs/9c0zrzlw) |
| T2V | [Video profiles](docs/paper-video.md) | [TA:VQ:MQ = 7:2:1](https://wandb.ai/vanzl3386-chinese-university-of-hong-kong-shenzhen/flow-grpo-algo/runs/gyvtocue) |
| I2V | [Video profiles](docs/paper-video.md) | [VQ:MQ = 1:1](https://wandb.ai/vanzl3386-chinese-university-of-hong-kong-shenzhen/flow-grpo-algo/runs/08ca4146a532752c473f1f37) |

The [run inventory](docs/reproduction-inventory.json) includes baseline runs,
source commits, and recipe provenance. W&B links may require access.

> **Development release.** CPU integration and distributed tests are in place;
> pretrained GPU acceptance and exact paper-result reproduction remain pending.
> See [release scope](docs/release-scope.md) for coverage and remaining work.

## Documentation & Development

[Documentation index](docs/README.md) &nbsp;/&nbsp;
[Development guide](docs/development.md) &nbsp;/&nbsp;
[Model interfaces](docs/model-contract.md) &nbsp;/&nbsp;
[Reward interfaces](docs/reward-contract.md)

We build on [Flow-GRPO](https://github.com/yifan123/flow_grpo),
[DiffusionNFT](https://github.com/NVlabs/DiffusionNFT),
[DanceGRPO](https://github.com/XueZeyue/DanceGRPO), and
[UniRL](https://github.com/Tencent-Hunyuan/UniRL). Attribution and asset terms
are recorded in [third-party notices](THIRD_PARTY_NOTICES.md); the project-wide
license is pending.
