# Native Video Setup

The video backends use the research runtime's older APIs, not the generic image
development environment. Install into a **new environment**. Do not upgrade an
environment that is running experiments.

| Profile | PyTorch / torchvision | Transformers | Diffusers | HF Hub |
| --- | --- | --- | --- | --- |
| Hunyuan T2V | 2.6.0 / 0.21.0 | 4.46.1 | 0.32.0 | 0.26.1 |
| Distilled Wan I2V | 2.6.0 / 0.21.0 | 4.46.1 | 0.33.1 | 0.30.2 |

Both use Python 3.10, PEFT 0.13.2, TRL 0.8.6 and FlashAttention 2.7.4.post1.
Direct requirements and resolved Python-3.10/Linux dependency locks live in
`environments/`; the installer uses the resolved locks, not latest transitive
dependencies. FlashAttention is built separately against the installed Torch.
The two environment files preserve the native model APIs; algorithm selection
within each task is still a config change, not a separate installation.

## Install

Prerequisites: Linux x86_64, Python 3.10 with `venv`, Git, GNU `timeout`, FFmpeg
and ffprobe, a C++ compiler, and the CUDA 12.4 toolkit (`nvcc` on PATH).
The CUDA toolkit is needed to build FlashAttention; a driver alone is not enough.
Use a CUDA-compatible driver and BF16-capable GPUs. No particular H200 SKU is
required; H800 is not rejected by a model-name check.

From the repository root:

```bash
export VENV="$HOME/envs/visual-rl-wan"
bash scripts/install_video_environment.sh wan "$VENV"
source "$VENV/bin/activate"
export DANCEGRPO_ROOT="$VENV/third_party/DanceGRPO"
```

Use `hunyuan` and a different environment path for T2V. The installer refuses
existing destinations, installs the [official CUDA 12.4 PyTorch wheels](https://pytorch.org/get-started/previous-versions/),
then the pinned dependencies, [FlashAttention](https://github.com/Dao-AILab/flash-attention/tree/v2.7.4.post1),
the exact DanceGRPO source checkout, and this repository. Native source is not
installed with its broad legacy dependency list, which would overwrite Wan's
Diffusers version. `MAX_JOBS=4` limits build parallelism by default.

Installation prints phase progress and has per-command deadlines. If a build
fails, keep its output for diagnosis and use a new destination; the installer
does not delete or repair an existing environment automatically. CUDA build and
large downloads can take minutes. Pip cache and temporary build files stay
under the selected environment directory.

## Check Before Loading Models

The installer runs a version/import/FFmpeg check with CUDA hidden and downloads
disabled. Repeat it explicitly with:

```bash
python -m visual_rl.check_environment --profile wan \
  --native-root "$DANCEGRPO_ROOT" --timeout-seconds 120
```

On an **authorized training GPU**, also test the compiled attention kernel:

```bash
python -m visual_rl.check_environment --profile wan \
  --native-root "$DANCEGRPO_ROOT" --cuda
```

This allocates a small tensor and checks FlashAttention forward/backward. It is
not full-model memory or multi-node acceptance. It never downloads weights.
Preserve the emitted versions in the experiment record.

The dependency recipes resolve for Linux/Python 3.10. CPU native tests run on
Torch 2.6.0, Diffusers 0.33.1 and Transformers 4.46.1; a clean Linux/CUDA build
and pretrained run are still pending. `pip check` alone cannot establish runtime
compatibility, so the import and kernel checks are explicit.

## Prepare Pinned Assets

Use a writable asset root with space for model weights, their downloads and
conversion output. Set `HF_HOME` to that storage before downloading. Accept any
gated model licenses through Hugging Face and use existing credentials or
`huggingface-cli login`; never put access tokens in configs or scripts.

```bash
export ASSET_ROOT=/absolute/path/to/video-assets
export HF_HOME="$ASSET_ROOT/hf-cache"
mkdir -p "$ASSET_ROOT"

huggingface-cli download KwaiVGI/VideoReward \
  --revision 4f26600130683e6f1de9f5d463887f28e8ef995c \
  --local-dir "$ASSET_ROOT/VideoReward"
huggingface-cli download Qwen/Qwen2-VL-2B-Instruct \
  --revision 895c3a49bc3fa70a340399125c650a463535e71c \
  --local-dir "$ASSET_ROOT/Qwen2-VL-2B-Instruct"
export VIDEO_REWARD_MODEL="$ASSET_ROOT/VideoReward"
export VIDEO_REWARD_REVISION=4f26600130683e6f1de9f5d463887f28e8ef995c
export VIDEO_REWARD_BACKBONE="$ASSET_ROOT/Qwen2-VL-2B-Instruct"
```

The reward directory must include `model_config.json` and
`checkpoint-11352/model.pth` (SHA256
`48375908e6112de9f0248402db156a23b480709a6960b091c598c6f4c88d21b9`).
It is not the Qwen backbone directory. Model weights remain separately licensed
assets, not part of this repository.

### Hunyuan

```bash
export HUNYUAN_REVISION=e8c2aaa66fe3742a32c11a6766aecbf07c56e773
export HUNYUAN_MODEL="$ASSET_ROOT/HunyuanVideo"
huggingface-cli download hunyuanvideo-community/HunyuanVideo \
  --revision "$HUNYUAN_REVISION" --local-dir "$HUNYUAN_MODEL"

export HUNYUAN_TEXT_MODEL="$ASSET_ROOT/hunyuan-native-text"
python -m visual_rl.models.prepare_hunyuan \
  --model-root "$HUNYUAN_MODEL" --output-dir "$HUNYUAN_TEXT_MODEL"
```

The pinned native encoder expects tokenizer files beside encoder weights,
whereas this snapshot stores them separately. The command creates a symlink
view for conditioning preparation; it does not copy weights or modify the
download. Keep its hidden sibling directory and original assets in place.
Existing outputs are never replaced. Policy/VAE loading still uses
`HUNYUAN_MODEL`; only conditioning preparation uses `HUNYUAN_TEXT_MODEL`.

Prepare hashed train/eval manifests and cached text conditioning using the
[Hunyuan guide](hunyuan-runner.md#cached-conditioning). The dataset schema and
sample identities are in [prompt data](prompt-data.md). Bind the returned
manifest paths and SHA256 values; do not hand-match cached tensors by ordering.

### Distilled Wan

```bash
export WAN_BASE_REVISION=b184e23a8a16b20f108f727c902e769e873ffc73
export WAN_BASE_MODEL="$ASSET_ROOT/Wan2.1-I2V-14B-480P-Diffusers"
export FLUX_REFERENCE_REVISION=3de623fc3c33e44ffbe2bad470d0f45bccf2eb21
export FLUX_REFERENCE_MODEL="$ASSET_ROOT/FLUX.1-dev"
export WAN_DISTILLED_REVISION=fef288b326f4fed6d2983b9800c35363da31fcfe

huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P-Diffusers \
  --revision "$WAN_BASE_REVISION" --local-dir "$WAN_BASE_MODEL"
huggingface-cli download black-forest-labs/FLUX.1-dev \
  --revision "$FLUX_REFERENCE_REVISION" --local-dir "$FLUX_REFERENCE_MODEL"
huggingface-cli download lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v \
  distill_models/distill_model.safetensors \
  --revision "$WAN_DISTILLED_REVISION" --local-dir "$ASSET_ROOT/LightX2V"

export WAN_DISTILLED_TRANSFORMER="$ASSET_ROOT/wan-distilled-transformer"
timeout -k 10s 3600s python -m visual_rl.models.convert_wan \
  --source "$ASSET_ROOT/LightX2V/distill_models/distill_model.safetensors" \
  --original-config "$WAN_BASE_MODEL/transformer/config.json" \
  --expected-source-sha256 12869d6856085d97cbf502fecbed85d1a67ce108174f95d82547497522479599 \
  --output-dir "$WAN_DISTILLED_TRANSFORMER"
```

Conversion is CPU-only, uses Diffusers 0.33.1's official name mapping and checks
all tensor names/shapes without allocating the full model. It does not change
tensor values or dtypes. The source is about 32.8 GB and output is similar in
size; hashing and conversion time depend on storage throughput.

The output is an atomically published relative symlink to a hidden sibling
directory. Keep that backing directory when moving assets, or copy the
dereferenced directory. Existing outputs are never replaced; failed staging is
preserved and named in stderr. `conversion-receipt.json` records source and
config hashes, converter identity and output-file hashes. Do not load an
unpublished staging directory.

Bind Wan's hashed text training manifest and fixed-image evaluation manifest as
described in [the Wan guide](wan-runner.md#data-and-reference-images). Training
reference images are generated online by FLUX; eval reference images are fixed.

## Start Through the Shared Entrypoint

Once asset and manifest variables are set:

```bash
python -m visual_rl.train --config configs/examples/wan_art.yaml --validate
torchrun --standalone --nproc-per-node=8 -m visual_rl.train \
  --config configs/examples/wan_art.yaml --output-dir outputs/wan-art-check
```

Use `hunyuan_art.yaml` in the Hunyuan environment. For four nodes, run torchrun
on every node with a shared rendezvous address and distinct `--node-rank`;
global prompt count must divide the total rank count. Algorithm configs change
the algorithm; they do not replace the launcher or runtime.

The supplied examples are short integration runs, not frozen paper recipes.
Full pretrained memory, historical prompt order and paper-result reproduction
remain separate acceptance steps. Logging and cost-axis setup are described in
[W&B and local metrics](logging.md).
