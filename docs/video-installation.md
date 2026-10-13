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

## Automatic Paper Assets

All 18 `configs/paper/video/*.yaml` profiles inherit
`configs/paper/hunyuan-assets.yaml` or `configs/paper/wan-assets.yaml`.
Setup remains separate: bind `DANCEGRPO_ROOT` to the installer's pinned checkout,
accept model licenses and use existing HF credentials. FLUX.1-dev is gated.
Preparation does not install software, modify native source, or require a W&B
token; training's W&B authentication remains separate.

The shared entrypoint prepares in the **parent before torchrun**, then reloads
the configuration with generated hashes. `--validate` and `--print-config`
never download or preprocess. In the Hunyuan environment, on an **authorized
training allocation**, not Mac or development `remote`:

```bash
export DANCEGRPO_ROOT="$VENV/third_party/DanceGRPO"
# Optional; default is ~/.cache/visual-rl. Select writable shared storage first.
export VISUAL_RL_ASSET_ROOT=/absolute/writable/shared/video-assets
export HF_HOME="$VISUAL_RL_ASSET_ROOT/hf-cache"
python -m visual_rl.train --config configs/paper/video/hunyuan_art_721.yaml \
  --prepare-only
python -m visual_rl.train --config configs/paper/video/hunyuan_art_721.yaml \
  --nproc-per-node 8
```

Omit the separate preparation command for preparation followed by training.
Hunyuan downloads its model, VideoReward and Qwen backbone, creates the native
text-encoder view, and computes conditioning once. Encoding 50,000 training
prompts can take hours; stderr reports rows every100. Wan downloads the base,
online FLUX training reference model, reward/backbone and only the nonquantized
LightX2V source, then runs the existing CPU converter with per-shard progress.
The approximately32.8GB source plus similarly sized output need substantial space.
Pretrained GPU execution and memory acceptance have **not** been verified.

### Exact Public Data And Blocker

All dataset downloads below now use our public
[HF mirror](https://huggingface.co/datasets/vanzl/DiffusionART-Prompts), revision
`9d947fb510d67a8e5334ec1d8c7d8927f7658c08`. Source bytes, converted hashes and
frozen selections are unchanged. Upstream attribution is retained in the dataset.

On 2026-10-04, the following pinned URLs were fetched and converted on CPU.
All source and output SHA256 values matched the asset YAML pins:

| Binding | Exact source | Rows |
| --- | --- | --- |
| Hunyuan train raw | [DanceGRPO video_prompts.txt at15cc71d](https://huggingface.co/datasets/vanzl/DiffusionART-Prompts/resolve/9d947fb510d67a8e5334ec1d8c7d8927f7658c08/sources/e22c29dba55a-video_prompts.txt) | 50,000 |
| Hunyuan eval raw | [VidProM example CSV at0c0900e](https://huggingface.co/datasets/vanzl/DiffusionART-Prompts/resolve/9d947fb510d67a8e5334ec1d8c7d8927f7658c08/sources/1e07955405d5-VidProM_unique_example.csv) | Frozen100 |
| Wan train | [DanceGRPO consist-id.txt at15cc71d](https://huggingface.co/datasets/vanzl/DiffusionART-Prompts/resolve/9d947fb510d67a8e5334ec1d8c7d8927f7658c08/sources/91943875adfe-consist-id.txt) | 27,815 |

Hunyuan eval extracts the original ordered UUIDs of `t2v-vidprom-val100-v1`,
inspected read-only in research commit `3ca2a36f6b20c53a06813013234666d9c033a660`.
It preserves whitespace-consolidated texts and NFKC/casefold prompt IDs, checks
the converted manifest pin, and never resamples. VidProM is **CC-BY-NC-4.0**,
separate from the code license. This is validation100, not the official
DanceGRPO test set or later author-written showcase150. Corrected common
evaluation and generic ordering still differ from historical paper evaluation.

The archived **Wan fixed evaluation JPEG bytes have no verified public
download** in the inspected provenance. Wan fails before model downloads unless
`WAN_EVAL_MANIFEST` (ready native JSONL) or `WAN_EVAL_RAW_MANIFEST` (raw fixed-image
JSONL) is supplied. It never generates replacement evaluation images or
substitutes cached images for online FLUX training references. A default Wan
command is therefore not one-command paper-ready. In the Wan environment:

For a new evaluation protocol that needs only public HF prompts, use
`configs/examples/wan_art_online_eval.yaml` (or the NFT/Fast/Dance/ReFL variants).
FLUX generates fixed-seed evaluation references lazily and caches their exact
pixels. These full-budget recipes leave training unchanged and do not require
the old JPEGs. See [reference caching and comparisons](wan-runner.md#automatic-evaluation-references).

A training-only short config may explicitly set `evaluation.every_updates: 0`
and `evaluation.at_updates: []`. Preparation then defers only the fixed eval
images; online FLUX training references, all training prompts, policy and reward
assets remain required. No substitute eval images or manifest are created.
Re-enabling evaluation restores the original fixed-image requirement, so this
route does not establish complete I2V performance or lifecycle acceptance.

```bash
export DANCEGRPO_ROOT="$VENV/third_party/DanceGRPO"
export WAN_EVAL_RAW_MANIFEST=/absolute/path/to/fixed-eval.jsonl
python -m visual_rl.train --config configs/paper/video/wan_art_11.yaml \
  --nproc-per-node 8
```

### Precomputed I2V Train And Test

The preparation tool freezes all 27,815 ConsisID training rows and the fixed
VidProM test1000 selection. The first 100 test prompts and images remain the
historical evaluation100. It generates training images and only the 900 new
test images. Each reference uses pinned FLUX.1-dev, 400x640, 30 steps, guidance
3.5 and seed312+split-index. Original JPEG bytes are copied without re-encoding.

This is a new fixed-reference training recipe, not an exact reproduction of
online FLUX references. Existing paper profiles remain unchanged. The dataset
is **not yet uploaded**: original eval100 images must be accessible and an
authorized GPU allocation must complete generation first.

In the Wan environment, select writable asset storage and download the small
public source files (CPU only):

```bash
export DATA_ROOT=/absolute/writable/shared/i2v-assets
mkdir -p "$DATA_ROOT/sources"
curl --fail --location --max-time 120 \
  https://huggingface.co/datasets/vanzl/DiffusionART-Prompts/resolve/9d947fb510d67a8e5334ec1d8c7d8927f7658c08/sources/91943875adfe-consist-id.txt \
  --output "$DATA_ROOT/sources/consist-id.txt"
curl --fail --location --max-time 120 \
  https://huggingface.co/datasets/vanzl/DiffusionART-Prompts/resolve/9d947fb510d67a8e5334ec1d8c7d8927f7658c08/sources/1e07955405d5-VidProM_unique_example.csv \
  --output "$DATA_ROOT/sources/vidprom.csv"
python -m visual_rl.evaluation.prepare_i2v plan \
  --train-prompts "$DATA_ROOT/sources/consist-id.txt" \
  --eval-source "$DATA_ROOT/sources/vidprom.csv" \
  --eval-assets configs/paper/hunyuan-eval1000-assets.yaml \
  --output "$DATA_ROOT/dataset"
python -m visual_rl.evaluation.prepare_i2v import-original \
  --source /path/to/original/manifest.json \
  --image-root /path/to/original/images \
  --output "$DATA_ROOT/original-eval100"
```

The plan verifies source hashes, train/test lexical separation and the unchanged
test100 prefix. Original import checks the archived manifest digest and every
image hash. A private copy of the original manifest anchors these checks during
generation and publication; it is never uploaded.

On an **authorized GPU node**, with no other job using the selected GPUs:

```bash
export HF_HOME="$DATA_ROOT/hf-cache"
torchrun --standalone --nproc-per-node=8 \
  -m visual_rl.evaluation.prepare_i2v generate \
  --output "$DATA_ROOT/dataset" \
  --original-bundle "$DATA_ROOT/original-eval100"
python -m visual_rl.evaluation.prepare_i2v finalize --output "$DATA_ROOT/dataset"
```

Each worker owns a disjoint image shard and prints completed indices. Generation
is a one-time multi-hour job, not part of training startup. Repeating the same
command verifies completed image receipts and generates only missing work;
changed receipts or nonidentical unreceipted regeneration fail explicitly.
If one FLUX instance cannot fit on a GPU, use a larger-memory GPU; reducing worker
count reduces parallelism but does not reduce per-GPU model memory. Do not change
precision, model or image settings silently.

After complete generation, use existing HF credentials to publish and write a
training recipe with the **returned immutable dataset revision**:

```bash
python -m visual_rl.evaluation.prepare_i2v publish \
  --output "$DATA_ROOT/dataset" --repo-id YOUR_ACCOUNT/diffusionart-i2v \
  --base-config configs/paper/video/wan_art_11.yaml \
  --recipe configs/examples/wan_art_fixed_11.yaml
python -m visual_rl.train --config configs/examples/wan_art_fixed_11.yaml \
  --nproc-per-node 8
```

The generated recipe automatically downloads the dataset, policy and reward
assets. It excludes the FLUX training snapshot, binds relative train/test image
manifests and uses fixed reference mode. Its default training budget remains400
updates. Use another `configs/paper/video/wan_*.yaml` as `--base-config` to retain
that algorithm's settings. Recipe filenames must be new; publication never
places credentials, preparation receipts or internal paths in the HF dataset.
Images are sharded into directories of at most 1,000 files. Large uploads resume
through the Hub client; the selected immutable revision is checked against every
manifest and image hash before a successful publication is recorded.

### Larger T2V Evaluation

`configs/examples/hunyuan_art_eval1000.yaml` selects 1,000 frozen prompts from
the same pinned VidProM CSV. The first 100 UUIDs, texts, IDs and generation-seed
positions remain exactly the paper's validation100. The additional 900 use the
same score-blind seed20260908 selection, basic text filters and lexical overlap
exclusion against all50,000 training prompts. No model outputs or rewards were
used for selection. The paper recipes continue to default to100; the larger
recipe uses a separate output directory and is not a historical paper result.

```bash
python -m visual_rl.train --config configs/examples/hunyuan_art_eval1000.yaml \
  --nproc-per-node 8
```

Preparation downloads the byte-identical VidProM CSV from our HF dataset
and extracts the frozen ordered UUIDs in `configs/paper/hunyuan-eval1000-assets.yaml`.
The raw JSONL SHA256 is `3b3392989d399ec0b840fc1161d83858872d3843806f4800162dd728958f7e69`.
To reproduce the selection itself with local copies of the two pinned sources:

```bash
python -m visual_rl.evaluation.select_prompts \
  --base-config configs/paper/hunyuan-assets.yaml \
  --source /path/to/VidProM_unique_example.csv \
  --training /path/to/video_prompts.txt \
  --output /path/to/new-eval1000-assets.yaml
```

The selector refuses changed sources, a changed original100 prefix, or an
existing output. This is a validation extension, not an untouched test set.

### Publishing Fixed I2V Assets

The original fixed images must be recovered before a public dataset can be
declared in the default Wan recipe. Do not regenerate them to fill missing files.
The packaging command verifies the source checksum,100 original image hashes,
and image decoding; copies unchanged bytes; and removes private machine paths
from public provenance. It does not load models or use a GPU.

```bash
python -m visual_rl.evaluation.package_i2v \
  --source /path/to/wan_i2v_validation100_v1/manifest.json \
  --output /path/to/i2v-eval100
# After confirming the destination account and redistribution terms:
hf auth login
python -m visual_rl.evaluation.package_i2v \
  --output /path/to/i2v-eval100 --publish NAMESPACE/DATASET
```

For a copied legacy manifest containing old absolute paths, add
`--image-root /path/to/copied/images`. Every relocated file must still match the
original image digest; the original manifest is never rewritten.

Publication prints the actual Hub commit and manifest digest. A dataset snapshot
supports `repo_type: dataset` and `manifest_file: prompts.jsonl`; binding its
`env` to `WAN_EVAL_RAW_MANIFEST` lets the existing `fixed_images` transformation
prepare evaluation automatically. Set `for_env: WAN_EVAL_MANIFEST` so a user-supplied
ready manifest still skips the download. Pin the returned40-character revision,
not `main`. Retain the original dataset/output terms and source attribution.

### Custom Inputs And Reuse

Explicit asset paths always win. Bind `HUNYUAN_TRAIN_RAW_MANIFEST` and/or
`HUNYUAN_EVAL_RAW_MANIFEST` to text-only native JSONL to retain automatic
conditioning for custom raw inputs. Rows require `sample_id` and `prompt`;
IDs, order and metadata are preserved. Ready conditioned
`HUNYUAN_TRAIN_MANIFEST`/`HUNYUAN_EVAL_MANIFEST` skip the corresponding raw
download and encoding; with both ready, the text view is also skipped.
Bind `WAN_TRAIN_MANIFEST` to custom text-only JSONL or
`WAN_DISTILLED_TRANSFORMER` to an existing converted policy to skip that work.
Custom data is not automatically certified as historical paper data.

Raw Wan eval rows require `sample_id`, `prompt`, `image`; optional
`image_sha256` pins remain strict. Relative images resolve beside the raw manifest.
Preparation records image hashes and validates supplied pins,
copies **unchanged bytes**, and publishes relative image paths. The
`fixed_images` action also supports `format: wan_validation` for original
checksummed `wan-i2v-formal-validation/v1` JSON with `samples[].id`; retain other
transformation entries when adapting that asset YAML. Arbitrary commands,
callables, services and environment copies are not supported.

User manifest SHA bindings are optional; preparation fills them after
conversion/conditioning. Supplied hashes must match; automatic public pins stay
strict. Output names use `env`/`sha_env`, other dependencies use validated
`*_env` references, and defaults are in `assets.defaults`.
`prepare_assets.environment_bindings(spec)` returns these non-secret bindings
in deterministic name order for the parent launcher's shared readiness record.

Under the actual asset root (default `~/.cache/visual-rl`):

- `models/<env-lowercase>/<HF-commit>/`: pinned snapshots.
- `prompts/<manifest-SHA256>/prompts.jsonl`: converted public raw manifests.
- `prepared/hunyuan_text_view/<recipe-key>/`: native text view.
- `prepared/hunyuan_conditioning/<recipe-key>/prompts.jsonl`: conditioning.
- `prepared/wan_transformer/<recipe-key>/`: converted policy.
- `prepared/fixed_images/<recipe-key>/prompts.jsonl`: fixed-image data.

Recipe keys are deterministic hashes of recipes and bound inputs. Data and
conditioning validate content against `preparation-receipt.json`, recording
actual source/output paths, source digest and file inventory. Wan reuses
`conversion-receipt.json`, checking source identity, config and output paths/sizes
without rehashing multi-GB model weights. Initial conversion still checks its
source SHA. HF models use commit pins and standard Hub integrity. The text view
uses `asset-view.json` and symlink target checks without scanning weights.
Corrupt/changed inputs fail rather than overwrite outputs. Keep hidden sibling
backing directories when moving symlink outputs; failed staging paths are
printed and are not reusable final outputs. Ready user assets need no receipt.
There is no automatic storage failover: select an available shared root before
preparation, record actual paths, and switch explicitly if storage is unavailable.

Verification on 2026-10-04: asset automation, paper-video profiles, native text
view and CPU Wan conversion tests reported **115 passed, 1 skipped** (5.69s).
The skip is pinned Diffusers0.33.1 integration in the different local development
version. Download/encoder mocks and real small CPU serialization/conversion
fixtures were used; no large model download or GPU job was performed.

## Manual Pinned Assets

These explicit commands remain available for offline/custom preparation.

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

For automatic preparation on a shared filesystem, the `visual_rl.train`
launcher accepts `--preparation-run-id ID`. Generate a fresh ID for each launch
and supply the same ID to every node. One node fully prepares and verifies the
assets before publishing this launch's bindings; peers then reuse that completed
verification. Failed preparation does not publish readiness. Reusing an old ID
would reuse its validation, so never carry it across launches or asset changes.
Without this option, cached transformations are reverified by each node.
`--rendezvous-timeout-seconds` controls the subsequent torchrun wait (default
900 seconds); it changes no sampling or optimizer settings.

The supplied examples are short integration runs, not frozen paper recipes.
Full pretrained memory, historical prompt order and paper-result reproduction
remain separate acceptance steps. Logging and cost-axis setup are described in
[W&B and local metrics](logging.md).
