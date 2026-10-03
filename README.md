<p align="center">
  <img src="assets/readme/logo.svg" width="560" alt="DiffusionART">
</p>

<p align="center"><strong>Efficient, stable and scalable reinforcement of diffusion models</strong></p>
<p align="center">Text-to-image &nbsp;&middot;&nbsp; Text-to-video &nbsp;&middot;&nbsp; Image-to-video</p>

<!-- Resource badges are intentionally unlinked until their destinations are supplied. -->
<p align="center">
  <img src="https://img.shields.io/badge/arXiv-b31b1b?style=flat-square" alt="arXiv">
  <img src="https://img.shields.io/badge/GitHub-24292f?style=flat-square&amp;logo=github&amp;logoColor=white" alt="GitHub">
  <img src="https://img.shields.io/badge/Hugging_Face_Models-ffd21e?style=flat-square&amp;logo=huggingface&amp;logoColor=black" alt="Hugging Face Models">
  <img src="https://img.shields.io/badge/Twitter_Channel-1d9bf0?style=flat-square" alt="Twitter Channel">
</p>

<p align="center">
  <a href="https://vanzll.github.io/diffusionart-project/">Project Page</a> &nbsp;/&nbsp;
  <a href="#method">Method</a> &nbsp;/&nbsp;
  <a href="#quick-start">Quick Start</a> &nbsp;/&nbsp;
  <a href="docs/README.md">Documentation</a> &nbsp;/&nbsp;
  <a href="#paper-experiments">Experiments</a>
</p>

**DiffusionART** is **Diffusion Reinforcement via Aggregation-First Reward Tilting
with Adaptive Restoration**. It combines the advantages of forward-process and
reverse-process reinforcement learning while avoiding their limitations.
DiffusionART is **efficient, stable and simple**, requires minimal hyperparameter
tuning and no differentiable reward model, and naturally supports pairwise
feedback and small rollout groups for scalable post-training.

This repository provides a comprehensive framework for fine-tuning visual
generation models, supporting widely used open-source **base models, reward
models and state-of-the-art algorithms** across text-to-image, text-to-video
and image-to-video generation.

<p align="center">
  <img src="assets/readme/t2v-gpu-hours.png" width="460" alt="Text-to-video training reward versus GPU hours; all methods use VQ:MQ = 1:1">
  <img src="assets/readme/i2v-gpu-hours.png" width="460" alt="Image-to-video training reward versus GPU hours; all methods use VQ:MQ = 1:1">
</p>

*Fine-tuning [HunyuanVideo (13B)](https://huggingface.co/tencent/HunyuanVideo)
(left) and [Wan2.1-I2V-14B-480P, CFG/step-distilled (14B)](https://huggingface.co/lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v)
(right) using [VideoReward](https://huggingface.co/KlingTeam/VideoReward), with
**Visual Quality (VQ):Motion Quality (MQ) = 1:1**. Mean train reward is the mean
of raw VQ and MQ scores; GPU hours include evaluation and waiting time.
[Curve sources](assets/readme/efficiency.json).*

## Method

### Motivations and Insights

1. NFT-style forward-process training contains a hidden **forward-reverse
   consistency force** in its objective, slowing and destabilizing training.
2. Off-policy strategies stabilize post-training of continuous-space generative
   models through a **strong restoration force**. Even a small output-space
   restoration MSE can produce a large aggregated parameter gradient.

### Aggregation-First Guidance and Adaptive Restoration

We call the covariance-form reward-improving direction **guidance**, bringing
a classifier-free guidance (CFG)-like idea into post-training: the model learns
to internalize reward guidance, much like distilling CFG. For a reward-tilted
endpoint distribution with tilt strength $\eta$,

$$
p_\eta(x_0\mid c)\propto p_{\mathrm{old}}(x_0\mid c)\exp(\eta R(x_0,c)),
\qquad
\left.\frac{\partial}{\partial\eta}\mathbb{E}_{p_\eta}[x_0]\right|_{\eta=0}
=\operatorname{Cov}_{p_{\mathrm{old}}}(R,x_0).
$$

Here $c$ is a prompt, $x_0$ is a clean latent endpoint, and $R$ is its reward.
DiffusionART inherits DiffusionNFT's reward-tilting principle while systematically
removing the self-consistency force: it aggregates reward evidence **before**
applying the normalized guidance to reverse-process rollout states, without
relying on an ideal forward-process velocity field. The second update retains
guidance while adaptively restoring velocity drift toward the rollout policy,
without an EMA off-policy strategy.

<table>
  <tr>
    <th width="50%">Guidance geometry and state assignment</th>
    <th width="50%">Paper pseudocode</th>
  </tr>
  <tr>
    <td align="center" valign="middle">
      <img src="assets/readme/method-overview.png" width="460" alt="Two stacked paper diagrams: reward guidance geometry above, covariance guidance assigned to ODE rollout states below">
    </td>
    <td valign="middle">
      <img src="assets/readme/paper-algorithm.png" width="460" alt="Paper Algorithm 1: ODE sampling, covariance guidance, frozen targets, and two disjoint MSE updates">
      <p>Implementation: <a href="src/visual_rl/algorithms/covariance.py">covariance guidance</a>, <a href="src/visual_rl/algorithms/art.py">ART loss</a>, and <a href="src/visual_rl/training/objectives.py">disjoint update objectives</a>.</p>
    </td>
  </tr>
</table>

### DiffusionART-Pairwise

**DiffusionART-Pairwise** is a variant in which the reward model gives preference
scores for two visual outputs: **1** for the winner, **0** for the loser, and
**0.5** for a tie. It accommodates pairwise feedback by simply replacing scalar
rewards with these labels when computing the shared guidance.
See the [preference-label and pairwise-guidance implementation](src/visual_rl/algorithms/covariance.py#L113).

<p align="center">
  <img src="assets/readme/pairwise-gpu-hours.png" width="460" alt="Scalar K=8 and pairwise K=2,4,8: VQ reward versus GPU hours">
  <img src="assets/readme/visual-improvements.png" width="460" alt="Selected paper examples: pre-trained versus DiffusionART images and four video frames">
</p>

*Left: pairwise T2V training with **K = 2, 4, 8** videos per prompt, alongside
scalar K = 8. Right: selected image and video improvements from the paper;
video strips show four frames. [Curve sources](assets/readme/efficiency.json)
and [visual provenance](assets/readme/provenance.json).*

### Why DiffusionART?

- **ODE solver:** deterministic rollouts, with no reverse-transition likelihood.
- **Rollout efficiency:** reuse rollout states and cached predictions, without
  adding noise again or recomputing predictions on newly noised states.
- **Sample efficiency:** smaller groups retain strong performance while using
  fewer samples and GPU hours.
- **Training efficiency:** remove forward-process RL's unstable self-consistency
  force and address reverse-process RL's inefficient credit assignment.
- **Simple and scalable:** two disjoint MSE updates, minimal tuning, and
  compatibility with black-box rewards and pairwise feedback.

## News

- 🎉 **[2026.10.01]** DiffusionART and six baselines are available in one framework for **T2I, T2V and I2V**. Explore the [training recipes](#paper-experiments)!

## Quick Feature Summary

| Feature Category | Supported Capabilities |
| --- | --- |
| **Tasks** | ✅ Text-to-image<br>✅ Text-to-video<br>✅ Image-to-video |
| **Algorithms** | ✅ DiffusionART<br>✅ DiffusionNFT (EMA / on-policy)<br>✅ Flow-GRPO / Flow-GRPO-Fast<br>✅ Flow-DPPO / DanceGRPO<br>✅ ReFL (video) |
| **Base Models** | ✅ SD3.5-M<br>✅ FLUX.1-dev<br>✅ HunyuanVideo-13B<br>✅ Wan2.1-I2V-14B (CFG/step-distilled) |
| **Benchmarks** | ✅ GenEval<br>✅ OCR<br>✅ Pick-a-Pic (PickScore) |
| **Reward Models** | ✅ GenEval / PickScore / OCR<br>✅ CLIPScore / Aesthetic<br>✅ VideoReward (TA / VQ / MQ)<br>✅ Configurable reward weights |
| **Feedback** | ✅ Scalar rewards<br>✅ Pairwise preferences (Hunyuan ART)<br>✅ Small rollout groups: K = 2, 4, 8 |
| **Training** | ✅ LoRA / full-parameter tuning<br>✅ DDP / FSDP<br>✅ Checkpoint resume / local evaluation artifacts |

## Quick Start

### 1. Install

Use Python 3.10+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).
Pretrained training requires Linux and CUDA.

```bash
git clone https://github.com/vanzll/DiffusionART.git
cd DiffusionART
uv sync --locked --python 3.10
```

<details>
<summary><strong>Optional: four-update CPU installation check</strong></summary>

No GPU, model downloads or W&B account needed:

```bash
uv run --locked visual-rl --config configs/examples/cpu_smoke.yaml --validate
uv run --locked visual-rl --config configs/examples/cpu_smoke.yaml
```

`outputs/cpu-smoke/status.json` should report `completed` and four optimizer
steps. To repeat, pass `--output-dir outputs/cpu-smoke-2`.

</details>

### 2. Prepare Assets

Follow the matching [model guide](#supported-models) for model access, pinned
weights, reward models and prompt data. Images use the `diffusers` extra;
videos use the [pinned video environment](docs/video-installation.md).

### 3. Train

After preparing SD3 and PickScore assets:

```bash
uv run --locked --extra diffusers visual-rl \
  --config configs/examples/sd3_pickscore.yaml
```

For an eight-GPU paper image recipe, prepare its [data and assets](docs/paper-image.md), then:

```bash
uv run --locked --extra diffusers torchrun --standalone --nproc-per-node=8 \
  -m visual_rl.train --config configs/paper/image/sd3_pickscore_art.yaml
```

Video recipes use the same entrypoint inside the prepared video environment:

```bash
torchrun --standalone --nproc-per-node=8 -m visual_rl.train \
  --config configs/paper/video/hunyuan_art_721.yaml
```

Add `--validate` for config checks before loading models. See
[distributed execution](docs/distributed-execution.md) for multi-node launches and
[checkpoint/resume](docs/checkpoint-resume.md) for recovery and evaluation.

## Results

**HunyuanVideo-13B &middot; text-to-video.** Final evaluation from the paper:

| Method | VQ &uarr; | MQ &uarr; |
| --- | ---: | ---: |
| Pre-trained | 5.195 | 1.943 |
| Flow-GRPO-Fast | 6.279 | 3.222 |
| DanceGRPO | 6.173 | 3.121 |
| DiffusionNFT (EMA) | 6.827 | 2.683 |
| DiffusionNFT (on-policy) | 5.543 | 1.839 |
| **DiffusionART** | **8.970** | **4.224** |

ART and NFT use TA:VQ:MQ = 7:2:1; GRPO baselines use VQ:MQ = 1:1.
See [experiment recipes](#paper-experiments) and the
[project page](https://vanzll.github.io/diffusionart-project/) for learning curves
and playable comparisons. [Table source](assets/readme/results.json).

<table>
  <thead><tr><th width="50%">Pre-trained</th><th width="50%">DiffusionART</th></tr></thead>
  <tbody>
    <tr><th colspan="2">Text-to-image &middot; A snowy Chicago street at Christmas</th></tr>
    <tr>
      <td><img src="assets/readme/image-chicago-before.jpg" width="360" alt="Pre-trained: snowy Chicago street"></td>
      <td><img src="assets/readme/image-chicago-after.jpg" width="360" alt="DiffusionART: snowy Chicago street"></td>
    </tr>
    <tr><th colspan="2">Text-to-video &middot; A river through an autumn valley</th></tr>
    <tr>
      <td><img src="assets/readme/video-106-before.jpg" width="360" alt="Pre-trained video frame: autumn valley"></td>
      <td><img src="assets/readme/video-106-after.jpg" width="360" alt="DiffusionART video frame: autumn valley"></td>
    </tr>
  </tbody>
</table>

Selected paper examples; video cells show individual frames.
[Sample provenance](assets/readme/provenance.json).

## Supported Models

| Task | Model | Algorithms | Setup |
| --- | --- | --- | --- |
| T2I | **SD3.5-M** | ART, NFT, GRPO, Fast, DPPO, Dance | [SD3 guide](docs/sd3-runner.md) |
| T2I | **FLUX.1-dev** | ART, NFT, GRPO, Fast, DPPO | [FLUX guide](docs/flux-runner.md) |
| T2V | **HunyuanVideo-13B** | ART, NFT, Fast, Dance, ReFL | [Hunyuan guide](docs/hunyuan-runner.md) |
| I2V | **Wan2.1-I2V-14B, CFG/step-distilled** | ART, NFT, Fast, Dance, ReFL | [Wan guide](docs/wan-runner.md) |

Fast = Flow-GRPO-Fast; DPPO = Flow-DPPO; Dance = DanceGRPO.

| Reward | Tasks | Feedback |
| --- | --- | --- |
| **PickScore, GenEval, OCR, Aesthetic, CLIPScore** | T2I | Scalar scores and weighted combinations |
| **VideoReward: TA, VQ, MQ** | T2V / I2V | Scalar scores and weighted combinations |
| **Pairwise preferences** | Hunyuan T2V / ART | Binary preference labels derived from scorer comparisons |
| **Differentiable VideoReward** | Video ReFL | Pixel-space gradients; common MP4 scores for reporting |

See [algorithm contracts](docs/algorithm-provenance.md) and
[reward composition](docs/composite-rewards.md) for method-specific behavior.

## Configure Your Experiment

Extend an [example](configs/examples/) or [paper recipe](configs/paper/).
Save this as `configs/my-pickscore.yaml`:

```yaml
extends: examples/sd3_pickscore.yaml
name: my-pickscore
training:
  max_updates: 1000
runtime:
  output_dir: outputs/my-pickscore
  checkpoint_keep: 2
```

```bash
uv run --locked --extra diffusers visual-rl --config configs/my-pickscore.yaml
```

Select components through `model.name`, `algorithm.name` and `reward.name`;
compose objectives with `reward.weights`. Sampling, microbatches, evaluation
and checkpoint cadence live in the same file. Required asset paths use `${ENV_VAR}`.

Metrics are saved to `metrics.jsonl`; evaluation images, videos and scores stay
on disk. Set `runtime.log: wandb` for [scalar logging](docs/logging.md).

## Paper Experiments

| Task | Benchmark / reward | Configs | Selected ART runs |
| --- | --- | --- | --- |
| T2I | GenEval, PickScore, OCR, joint rewards | [Image recipes](docs/paper-image.md) | [GenEval](https://wandb.ai/vanzl3386-chinese-university-of-hong-kong-shenzhen/flow-grpo-algo/runs/11pq0iy3) &middot; [PickScore](https://wandb.ai/vanzl3386-chinese-university-of-hong-kong-shenzhen/flow-grpo-algo/runs/mofumvbw) &middot; [OCR](https://wandb.ai/vanzl3386-chinese-university-of-hong-kong-shenzhen/flow-grpo-algo/runs/9c0zrzlw) |
| T2V | TA:VQ:MQ = 7:2:1; scalar / pairwise VQ | [Video recipes](docs/paper-video.md) | [Main](https://wandb.ai/vanzl3386-chinese-university-of-hong-kong-shenzhen/flow-grpo-algo/runs/gyvtocue) &middot; [Pairwise K=2](https://wandb.ai/vanzl3386-chinese-university-of-hong-kong-shenzhen/flow-grpo-algo/runs/x9mzbm4c) |
| I2V | VQ:MQ = 1:1 | [Video recipes](docs/paper-video.md) | [Main](https://wandb.ai/vanzl3386-chinese-university-of-hong-kong-shenzhen/flow-grpo-algo/runs/08ca4146a532752c473f1f37) |

Baseline runs and source commits are in the [run inventory](docs/reproduction-inventory.json).
W&B links may require access. Current validation covers CPU integration and
distributed tests; pretrained GPU acceptance and paper-number parity remain
pending. [Release scope](docs/release-scope.md).

## Learn More

[Documentation](docs/README.md) &nbsp;/&nbsp;
[Development](docs/development.md) &nbsp;/&nbsp;
[Model interfaces](docs/model-contract.md) &nbsp;/&nbsp;
[Reward interfaces](docs/reward-contract.md)

## Acknowledgements

Built on [Flow-GRPO](https://github.com/yifan123/flow_grpo),
[DiffusionNFT](https://github.com/NVlabs/DiffusionNFT),
[DanceGRPO](https://github.com/XueZeyue/DanceGRPO) and
[UniRL](https://github.com/Tencent-Hunyuan/UniRL).
README organization draws inspiration from
[verl-agent](https://github.com/langfengQ/verl-agent) and Flow-GRPO.
See [third-party notices](THIRD_PARTY_NOTICES.md) for attribution and asset terms;
the project-wide license is pending.
