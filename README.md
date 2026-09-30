# DiffusionART

**Efficient, Stable and Scalable Reinforcement of Diffusion Model**

[Project page](https://vanzll.github.io/diffusionart-project/) |
[Experiment configs](configs/paper/) |
[Installation](#install-and-check) |
[Algorithm contracts](docs/algorithm-provenance.md)

Zhenglin Wan, Jiayu Chen, Zhengfeiyang Zhang, Pengfei Zhou, Wangbo Zhao,
Fangyuan Kong, Pengfei Wan, Kun Gai, Yang You

HPC-AI Lab, National University of Singapore · Kling AI · HKUST

## Visual Diffusion RL

A configuration-driven framework for text-to-image, text-to-video, and
image-to-video post-training. Select the algorithm, base model, reward model,
and reward weights in one experiment configuration.

**Development build:** native T2I/T2V/I2V training, evaluation and full-state
resume are connected. CPU integration and distributed tests validate software
behavior; clean pretrained GPU acceptance and historical paper parity remain
pending. Short examples and full-budget paper candidates are kept separate.

### Selected Results

| | Pre-trained | DiffusionART |
| --- | --- | --- |
| Text-to-image | ![Chicago before RL](assets/readme/image-chicago-before.jpg) | ![Chicago after RL](assets/readme/image-chicago-after.jpg) |
| Text-to-video, one frame | ![Autumn river before RL](assets/readme/video-106-before.jpg) | ![Autumn river after RL](assets/readme/video-106-after.jpg) |

Selected qualitative examples from the paper's experiments. See the
[project page](https://vanzll.github.io/diffusionart-project/) for playable videos
and quantitative figures; sample identities are retained in
[asset provenance](assets/readme/provenance.json).

| Task | Base Model | Paper Configs | DiffusionART Run |
| --- | --- | --- | --- |
| T2I | SD3.5-M / FLUX.1-dev | [Image profiles](docs/paper-image.md) | [GenEval](https://wandb.ai/vanzl3386-chinese-university-of-hong-kong-shenzhen/flow-grpo-algo/runs/11pq0iy3), [PickScore](https://wandb.ai/vanzl3386-chinese-university-of-hong-kong-shenzhen/flow-grpo-algo/runs/mofumvbw), [OCR](https://wandb.ai/vanzl3386-chinese-university-of-hong-kong-shenzhen/flow-grpo-algo/runs/9c0zrzlw) |
| T2V | HunyuanVideo-13B | [Video profiles](docs/paper-video.md) | [TA/VQ/MQ = 7/2/1](https://wandb.ai/vanzl3386-chinese-university-of-hong-kong-shenzhen/flow-grpo-algo/runs/gyvtocue) |
| I2V | Distilled Wan-14B | [Video profiles](docs/paper-video.md) | [VQ/MQ = 1/1](https://wandb.ai/vanzl3386-chinese-university-of-hong-kong-shenzhen/flow-grpo-algo/runs/08ca4146a532752c473f1f37) |

The [full run/source inventory](docs/reproduction-inventory.json) includes
baseline evidence. W&B public access has not yet been audited.

The [distributed execution notes](docs/distributed-execution.md) distinguish the
tested native image DDP execution from pending pretrained GPU/paper acceptance.
Multi-worker launches of single-device configs fail before training.

### Install and Check

Prerequisites: Linux or macOS, Python 3.10+,
[uv](https://docs.astral.sh/uv/getting-started/installation/), and network access
for the initial package installation. Pretrained CUDA training targets Linux;
Windows is not supported by the filesystem ownership layer. No GPU, model access
token, private paths or W&B account is needed for this check.

```bash
git clone https://github.com/vanzll/DiffusionART.git
cd DiffusionART
uv sync --locked --python 3.10 --extra dev
uv run --locked visual-rl --config configs/examples/cpu_smoke.yaml --validate
uv run --locked visual-rl --config configs/examples/cpu_smoke.yaml
```

The fixture completes four optimizer updates and writes:

```text
outputs/cpu-smoke/
  resolved-config.json
  run.json
  metrics.jsonl
  status.json
  checkpoints/latest.json
```

`status.json` must report `completed` with `optimizer_steps: 4`. CPU GPU-hours
are correctly zero. Use a new output directory for another run; existing
results are never silently overwritten:

```bash
uv run --locked visual-rl --config configs/examples/cpu_smoke.yaml --output-dir outputs/cpu-smoke-2
uv run --locked --extra dev pytest -q
```

Optional model-adapter tests use tiny randomly initialized Diffusers models,
not downloaded pretrained weights:

```bash
uv sync --locked --python 3.10 --extra dev --extra diffusers
uv run --locked --extra dev --extra diffusers pytest -q tests/test_models_diffusers.py
uv run --locked --extra dev --extra diffusers pytest -q tests/test_sd3_text.py tests/test_pickscore.py tests/test_image_training.py
```

Gallery JavaScript tests additionally require Node.js. Tests report explicit
skips when optional dependencies are absent; report those skips with results.

### Native SD3 Integration

The [SD3 runner guide](docs/sd3-runner.md) covers pinned asset preparation,
training, evaluation PNGs, score definitions and full-state local resume.
Its [example config](configs/examples/sd3_pickscore.yaml) is a short native-model
integration check, not a paper reproduction profile. The actual loader/CLI path
has offline tiny-model tests; full pretrained GPU acceptance is still pending.
The [NFT guide](docs/sd3-nft.md) and [NFT config](configs/examples/sd3_nft_pickscore.yaml)
use that same entrypoint with old-policy sampling, one update per rollout by
default, an explicit two-window OCR recipe, and separate old-policy/evaluation
EMA state.
The [GRPO guide](docs/sd3-grpo.md) provides configs for Flow-GRPO, Fast,
Flow-DPPO and DanceGRPO, preserving logical sampling/update batches independently of physical
microbatching.
SD3 also supports [FP16 loss scaling](configs/examples/sd3_pickscore_fp16.yaml),
including coordinated DDP overflow recovery and checkpointed scaler state;
see the [precision contract](docs/sd3-runner.md#scientific-and-runtime-boundaries).
The [Aesthetic guide](docs/sd3-aesthetic.md) shows how to switch the reward
model through configuration, with a checksummed scoring head and unchanged
training/evaluation interfaces.
The [CLIPScore guide](docs/sd3-clipscore.md) adds the original floating-tensor
scoring path, with pinned preprocessing and explicit raw versus logit/30 scores.
The [OCR integration guide](docs/sd3-ocr.md) describes the historical text reward,
explicit local Paddle model bundles and its pending native Linux acceptance.
The [GenEval guide](docs/geneval.md) adds annotation-aware dense rewards, separate
accuracy diagnostics and an isolated legacy detector environment for SD3/FLUX.
The [composition guide](docs/composite-rewards.md) combines native scorers with
explicit weights and once-only fixed normalization in the same entrypoint.
It also documents ART's separate optional group-normalized advantage mixture;
reporting weights and algorithm-side weights are not conflated.

### Native FLUX Integration

The [FLUX guide](docs/flux-runner.md) uses the same CLI for DiffusionART with
FLUX text conditioning, resolution-dependent scheduling and explicit
rollout/replay precision. Its [short example](configs/examples/flux_pickscore.yaml)
is an integration check, not a paper profile. The [FLUX NFT guide](docs/flux-nft.md)
adds the distinct fresh-noising mirror update and independent policy shadows.
The [FLUX GRPO guide](docs/flux-grpo.md) adds full-SDE and windowed Fast-CPS
sampling with explicit cache/replay precision. Pretrained GPU reproduction
and remaining task backends remain pending.

### Native Video Integration

The [native video installation guide](docs/video-installation.md) provides
pinned Linux environment installers, asset downloads, Wan checkpoint conversion
and preflight commands. Full pretrained CUDA acceptance is still pending.

The [Hunyuan guide](docs/hunyuan-runner.md) describes the first connected video
path: native conditioning and asset loading, single-device/FSDP ART, NFT, Fast-SDE,
DanceGRPO and final-step ReFL updates,
MP4 reward scoring, full-retention evaluation HTML and full-state resume.
Real FFmpeg and two-process FSDP orchestration are tested with small CPU
components. The same entrypoint also supports [binary pairwise feedback](docs/hunyuan-runner.md#pairwise-feedback)
with K=2/4/8 config examples.

The [Wan I2V guide](docs/wan-runner.md) adds four-step CFG-distilled Wan ART,
NFT, Fast-SDE, Dance and final-step ReFL through the same lifecycle, with online FLUX training
references and fixed-image evaluation. Five configs select their distinct
sampling/update rules. Pretrained CUDA loading, clean Linux environment validation
and paper reproduction acceptance remain in progress.

### One Config, One Experiment

[W&B logging](docs/logging.md) is enabled with `runtime.log: wandb`. A separate
rank-zero exporter publishes the existing train/eval metrics, with training-step
and GPU-hour axes; JSONL remains available without an account or network.

The [dataset guide](docs/prompt-data.md) describes hashed prompt manifests,
annotation identity and reference-image inputs.

Full-budget recipes are indexed in [paper-image.md](docs/paper-image.md) and
[paper-video.md](docs/paper-video.md). Once the documented assets and manifest
variables are prepared, the launch interface is the same for every algorithm:

```bash
# Eight-GPU image training, executed on an authorized training node.
uv run --locked --extra diffusers torchrun --standalone --nproc-per-node=8 \
  -m visual_rl.train --config configs/paper/image/sd3_pickscore_art.yaml

# Video training inside the separately installed native video environment.
torchrun --standalone --nproc-per-node=8 -m visual_rl.train \
  --config configs/paper/video/hunyuan_art_721.yaml
```

Use the distributed guide for multi-node launches. `--resume` accepts a
complete checkpoint receipt; it never guesses continuation from a run name.
No private mount, cluster launcher, GPU-hold utility or agent plugin is needed.

```yaml
# Configuration shape; see complete native integration examples above.
task: t2v
algorithm: diffusionart
model:
  name: hunyuan_public
  checkpoint: ${MODEL_PATH}
reward:
  name: video_reward
  normalization: fixed
  weights: {TA: 0.7, VQ: 0.2, MQ: 0.1}
```

Experiment configs can `extend` common presets using the `extends:` key. A
component-name change resets its old options, and a reward-weight override
replaces the entire weight map. Misspellings, duplicate keys and unset required
environment variables are errors. Unsupported backends fail before loading
models; they never fall back to a different algorithm.

### Implementation Status

| Layer | Implemented now | Remaining acceptance |
| --- | --- | --- |
| Algorithms | ART covariance/MSE/mask, NFT mirror loss/EMA schedule, GRPO/DPPO losses, ReFL reward loss | Full task-specific sampling/update integration and paper-run parity |
| Models | SD3/FLUX loaders; Hunyuan/Wan native assets and conditioning; full-weight/LoRA setup; differentiable VAEs; pinned video installation/conversion | Clean Linux/CUDA acceptance and pretrained GPU parity |
| Training | SD3 ART/NFT/GRPO/Fast/DPPO/Dance and FLUX ART/NFT/GRPO/Fast/DPPO single-device/DDP; Hunyuan/Wan ART/NFT/Fast-SDE/Dance/ReFL single-device/FSDP | Pretrained GPU acceptance |
| Checkpoints | Local/DDP/FSDP full-state integration; optimizer/shadow/per-rank RNG resume tests | Pretrained large-model lifecycle validation |
| Rewards/evaluation | PickScore/Aesthetic/CLIPScore, OCR and isolated GenEval loaders, scalar composition and ART joint-advantage path; identity manifests and HTML | Native legacy-environment/Linux acceptance, other scorers, frozen paper profiles, pretrained parity, production browser verification |

The framework does not force algorithms into the same update schedule. NFT
re-noising, ART disjoint windows, stochastic likelihood replay and ReFL's
differentiable endpoint path remain explicit algorithm behavior.

### Reproduction and Extension

- [Release scope and acceptance gates](docs/release-scope.md)
- [Paper run/source inventory](docs/reproduction-inventory.json)
- [Algorithm equations and provenance](docs/algorithm-provenance.md)
- [Model adapter contracts](docs/model-contract.md)
- [Reward and evaluation contracts](docs/reward-contract.md)

Paper candidates record asset revisions, data hashes, reward preprocessing,
sampling/update schedules and metric definitions. Outstanding runtime identities
and dataset/order parity are listed per profile. Historical results and new
recommended recipes stay separate; fixes never silently rewrite a result.
W&B links in the inventory identify evidence, not yet-verified public access.

Credentials belong in the user's environment or credential store, never YAML.
Model weights, datasets, private launchers and research Git history are not
bundled. See [third-party notices](THIRD_PARTY_NOTICES.md).

## Project page

The project page includes the core insights, quantitative results, and selected
before/after images and playable videos. Its source and assets are hosted in the
separate [website repository](https://github.com/vanzll/diffusionart-project).
This repository is reserved for the code release; its visibility does not
control the website.

## Release history

The future code release will be imported as a clean source snapshot. Private
experiment-repository history, logs, credentials, and training artifacts will
not be carried into this repository. Third-party notices and applicable
licenses will be preserved in the release.

The implementation and its license will be announced with the code release.
