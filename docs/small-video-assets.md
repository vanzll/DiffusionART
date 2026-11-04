# Small-video assets and preflight

This entrypoint prepares pinned assets for **Wan2.1 T2V1.3B** and
**LTX-Video2B0.9.6 distilled**. It verifies downloaded bytes, source checkouts and
prompt splits without importing pretrained models or allocating GPU tensors.
Native ART replay/mask semantics still require an explicit scientific decision;
successful asset preflight is not permission or evidence to claim a validated
training recipe.

The checked-in manifest is
[`configs/assets/small_video.json`](../configs/assets/small_video.json). It has
51 required files totaling52,440,650,102 bytes for both models, their text
encoders, VideoReward/Qwen and the selected prompt files. Allow additional
space for HF caches, temporary files, videos, optimizer checkpoints and exports.
The LTX allowlist contains only the exact0.9.6 distilled checkpoint, its model
card and license; it excludes other LTX model weights and prompt-enhancement
language models. Wan uses the native1.3B checkpoint and bundled UMT5 assets.

## Install into new environments

Use Linux x86_64, Python3.10, Git, GNU `timeout`, FFmpeg/ffprobe and writable
persistent storage. CUDA12.4 and a compiler are needed for the optional native
FlashAttention build. Existing training environments should remain untouched.
The following paths are examples chosen by the operator; no site-specific
directory or SSH configuration is required.

From this repository root, create a new **model** environment:

```bash
set -euo pipefail
export SV_ROOT=/absolute/persistent/small-video
export SV_REPO="$PWD"
export SV_MODEL_ENV="$SV_ROOT/env-model"
export SV_REWARD_ENV="$SV_ROOT/env-reward"
export SMALL_VIDEO_ASSETS="$SV_ROOT/assets"
export HF_HOME="$SV_ROOT/hf-cache"
export PIP_CACHE_DIR="$SV_ROOT/pip-cache"
export TMPDIR="$SV_ROOT/tmp"
test ! -e "$SV_MODEL_ENV"
mkdir -p "$SV_ROOT" "$TMPDIR" "$PIP_CACHE_DIR"
timeout -k 5s 120s python3.10 -m venv "$SV_MODEL_ENV"
timeout -k 5s 120s "$SV_MODEL_ENV/bin/python" -m pip install pip==25.3
# Large wheel download; terminal output reports progress, deadline two hours.
timeout -k 10s 7200s "$SV_MODEL_ENV/bin/python" -m pip install \
  torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124 \
  --extra-index-url https://pypi.org/simple
timeout -k 10s 1800s "$SV_MODEL_ENV/bin/python" -m pip install \
  -r "$SV_REPO/environments/small-video-py310-linux.lock"
timeout -k 5s 120s "$SV_MODEL_ENV/bin/python" -m pip install --no-deps -e "$SV_REPO"
timeout -k 5s 120s "$SV_MODEL_ENV/bin/python" -m pip check
```

The model lock pins the scoped dependency closure; `small-video.txt` lists its direct inputs, including Diffusers0.33.1,
Transformers4.51.3 and HF Hub0.30.2; a successful dependency resolution does not
establish model runtime or GPU compatibility. The model environment uses the headless OpenCV4.10.0.84 build and does not require desktop libGL. Decord is not used by the native T2V generation paths and stays out of this model environment. The separate canonical reward environment retains its pinned Decord reader; reward decoding is unchanged. Do not install upstream native
package metadata with its broader dependency list into this environment.

For **native Wan T2V**, FlashAttention2 is mandatory: the pinned official
transformer calls `flash_attention` directly. Its generic SDPA helper is not a
fallback for these model forwards. The shared model lock deliberately leaves
FA optional for LTX; install the separate Wan-only add-on before native Wan
factory or parity execution. Use the dedicated model environment above, not an
existing training/reward environment. Torch2.6.0+cu124 must be installed first.

```bash
# CPU build tools only; do not resolve/replace the pinned model dependency closure.
timeout -k 5s 120s "$SV_MODEL_ENV/bin/python" -m pip install --no-deps \
  ninja==1.11.1.3 wheel==0.45.1
timeout -k 5s 30s nvcc --version
export MAX_JOBS=${MAX_JOBS:-4}
# May download a matching wheel or compile for up to an hour; pip prints progress.
timeout -k 10s 3600s "$SV_MODEL_ENV/bin/python" -m pip install \
  --no-build-isolation --no-deps \
  -r "$SV_REPO/environments/wan-t2v-flash-attn.txt"
timeout -k 5s 120s "$SV_MODEL_ENV/bin/python" -m pip check
# Import/binary/ABI check only; no CUDA query, context initialization or kernel.
CUDA_VISIBLE_DEVICES='' timeout -k 5s 60s "$SV_MODEL_ENV/bin/python" \
  -m visual_rl.models.wan_t2v_dependencies
```

The add-on freezes `flash-attn==2.7.4.post1` and checks Torch2.6.0, CUDA build12.4
and C++ ABI=False. It reports actual package/module versions, binary origin and
Torch ABI before loading native models. A discoverable `flash_attn_interface`
(FA3) is rejected because the unmodified official default would select it over
FA2. Use an FA2-only dedicated Wan environment; the preflight does not patch
the native attention function or silently select another kernel. Record the
actual `nvcc --version` output separately from Torch's CUDA build; no driver or
Torch replacement is implied by a differing compiler minor version. Successful
binary import is not GPU kernel or inference parity evidence. The canonical
reward interpreter remains separate and uses its own installer below.

Clone the two native model sources at their exact commits. Each command works
only in a new operator-selected directory; fetching source may take minutes.

```bash
mkdir -p "$SMALL_VIDEO_ASSETS/sources"
while read -r name url revision; do
  target="$SMALL_VIDEO_ASSETS/sources/$name"
  test ! -e "$target"
  timeout -k 5s 30s git init -q "$target"
  timeout -k 5s 30s git -C "$target" remote add origin "$url"
  timeout -k 5s 600s git -C "$target" fetch --depth 1 origin "$revision"
  timeout -k 5s 30s git -C "$target" checkout -q --detach FETCH_HEAD
done <<'SOURCES'
wan https://github.com/Wan-Video/Wan2.1.git 9737cba9c1c3c4d04b33fcad41c111989865d315
ltx https://github.com/Lightricks/LTX-Video.git 50f0aef1297574a4b29acbdb874e3a363a6d5f2e
SOURCES
```

The reward stays on the **canonical DanceGRPO** implementation at
`15cc71d53cc2e6e18a68ee607d5fb6ba9a99e344`, with4fps sampling and
`use_norm=False`. It uses the existing VideoReward assets and Qwen2-VL2B
backbone, without substituting the H3 vendored24fps VideoAlign implementation.
[`environments/small-video-reward.txt`](../environments/small-video-reward.txt)
inherits the canonical Wan/video requirements. Reuse the existing installer
and its resolved dependency lock for a separate new reward environment:

```bash
# Includes dependency checks, source checkout and FlashAttention build.
# The build can take up to an hour; the installer prints phase progress.
bash scripts/install_video_environment.sh wan "$SV_REWARD_ENV"
export SMALL_VIDEO_REWARD_PYTHON="$SV_REWARD_ENV/bin/python"
test ! -e "$SMALL_VIDEO_ASSETS/sources/dancegrpo"
test ! -L "$SMALL_VIDEO_ASSETS/sources/dancegrpo"
ln -sT "$SV_REWARD_ENV/third_party/DanceGRPO" "$SMALL_VIDEO_ASSETS/sources/dancegrpo"
export SV_REWARD_SOURCE="$SMALL_VIDEO_ASSETS/sources/dancegrpo"
```

Canonical reward Transformers4.46.1 and model Transformers4.51.3 differ. Their
shared-environment compatibility has not been established by real GPU tests.
Asset preparation does not silently upgrade either environment. The current
small-video runner uses a separate canonical-reward subprocess, selected by
`SMALL_VIDEO_REWARD_PYTHON`, during the scoring phase on the same rank GPU.
Generation and scoring therefore keep their separate Transformers environments
without assuming shared-environment compatibility or adding a ninth GPU.
This phased worker implementation still requires real GPU validation; scorer
memory/time belongs in the eight-GPU acceptance evidence.

## Plan, download and verify

All commands use paths supplied by the operator. `plan` performs no downloads
or filesystem writes and prints selected revisions, required source bindings,
destinations and total bytes. Choose `--profile wan`, `ltx` or `all`:

```bash
"$SV_MODEL_ENV/bin/python" -m visual_rl.small_video_assets plan \
  --manifest configs/assets/small_video.json --root "$SMALL_VIDEO_ASSETS" --profile all
```

`download` delegates the existing `visual_rl.prepare_assets` snapshot interface,
with exact revisions and file allowlists. It does not install source, encode
prompts, select a sampler or launch training. Use existing HF login/credentials;
never include tokens in a command, manifest, receipt or log.

```bash
# About52.4GB for all profiles; HF output reports per-file progress.
timeout -k 15s 14400s "$SV_MODEL_ENV/bin/python" -m visual_rl.small_video_assets download \
  --manifest configs/assets/small_video.json --root "$SMALL_VIDEO_ASSETS" --profile all
```

Initial `preflight` streams every required file once. Large scans may take
several minutes; stderr reports the file and progress every15seconds while
blocks are read. Run this once in the launch master before distributed workers:

```bash
timeout -k 10s 7200s "$SV_MODEL_ENV/bin/python" -m visual_rl.small_video_assets preflight \
  --manifest configs/assets/small_video.json --root "$SMALL_VIDEO_ASSETS" --profile wan \
  --source wan="$SMALL_VIDEO_ASSETS/sources/wan" \
  --source reward="$SV_REWARD_SOURCE"
timeout -k 10s 7200s "$SV_MODEL_ENV/bin/python" -m visual_rl.small_video_assets preflight \
  --manifest configs/assets/small_video.json --root "$SMALL_VIDEO_ASSETS" --profile ltx \
  --source ltx="$SMALL_VIDEO_ASSETS/sources/ltx" \
  --source reward="$SV_REWARD_SOURCE"
```

Supply exactly the sources required by a profile: `wan`+`reward` or
`ltx`+`reward`, and all three for `all`. Native Git HEAD must match the manifest
and tracked source must be clean. Adapter loading independently verifies actual
import origins. Existing asset locations can be bound explicitly, for example
`--asset wan=/absolute/existing/native-wan`; the same immutable file pins still
apply. Repeat identical bindings for download, preflight and receipt verification.
Missing/corrupt files, nonfinite JSON values, incomplete shard indexes, changed
source or wrong data identity fail closed. Weights are checked against pinned
LFS SHA256; non-LFS files are checked against pinned Git blob IDs and their
observed SHA256 is recorded. The preflight does not unpickle weights or claim
that tensor values or model/reward outputs have passed finiteness tests.

Workers use the completed receipt without repeating multi-GB content scans:

```bash
timeout -k 5s 120s "$SV_MODEL_ENV/bin/python" -m visual_rl.small_video_assets verify \
  --manifest configs/assets/small_video.json --root "$SMALL_VIDEO_ASSETS" --profile wan
timeout -k 5s 120s "$SV_MODEL_ENV/bin/python" -m visual_rl.small_video_assets verify \
  --manifest configs/assets/small_video.json --root "$SMALL_VIDEO_ASSETS" --profile ltx
```

Receipts contain only selected paths, revisions, file hashes/stat identities and
data identities. Publication uses a file lock and atomic replacement. Fast
verification checks manifest/profile/root/inventory/checksum, file targets,
sizes, inode and nanosecond modification/change times, plus native source
identity. Changed files fail and require another full master preflight. This
trusted local receipt cache detects later changes; it is not an authenticated
signature or protection against an adversary rewriting both assets and receipts.
Moving/copying assets or changing filesystem metadata can require rebuilding
the receipt. There is no automatic fallback to unverified assets.

The default layout under`SMALL_VIDEO_ASSETS` is:

```text
models/wan/37ec512624d61f7aa208f7ea8140a131f93afc9a/
models/ltx/8984fa25007f376c1a299016d0957a37a2f797bb/
models/ltx_text/b89adadeccd9ead2adcb9fa2825d3fabec48d404/
models/reward/4f26600130683e6f1de9f5d463887f28e8ef995c/
models/reward_backbone/895c3a49bc3fa70a340399125c650a463535e71c/
datasets/prompts/9d947fb510d67a8e5334ec1d8c7d8927f7658c08/t2v/train.jsonl
datasets/prompts/9d947fb510d67a8e5334ec1d8c7d8927f7658c08/t2v/validation.jsonl
derived/eval8.jsonl
sources/wan/
sources/ltx/
sources/dancegrpo/
preflight-wan.json
preflight-ltx.json
```

Each profile has a separate receipt;`verify --profile wan` does not accept an
`all` or`ltx` receipt. Source paths recorded in receipts are resolved targets,
so the canonical reward-source symlink and its target must remain available.

## Frozen prompts and paired evaluation

The dataset is
[`vanzl/DiffusionART-Prompts@9d947fb`](https://huggingface.co/datasets/vanzl/DiffusionART-Prompts/tree/9d947fb510d67a8e5334ec1d8c7d8927f7658c08),
configuration`t2v`. Keep its model-independent subset attribution and source
terms: DanceGRPO train and VidProM validation have separate provenance; VidProM
is CC-BY-NC-4.0.

| File | Rows | SHA256 |
| --- | --- | --- |
|`t2v/train.jsonl`|50,000|`b54fe7c10e7530d24e5f928f3f40f9d6c160191301631235463945575b499194`|
|`t2v/validation.jsonl`|100|`af1cecf6f794db3bbcd85d90cbca0e10cb196bc8f043a2a3724c00635383e70f`|
|derived paired subset|8|`06fb7106c1fa00cbe91c5f4b6e79cb3eb99c90d13f3c7995fb62d8486148c9ff`|

Preflight verifies source bytes and exact row counts, rejects duplicate/empty
sample identities, and checks **all** train/validation text intersection after
NFKC normalization, case folding and whitespace consolidation. It reconstructs
`random.Random(42).sample(validation,8)` in source order and verifies both the
ordered sample IDs and serialized bytes against the frozen manifest. The output
is`<root>/derived/eval8.jsonl`; a different existing output is rejected rather
than overwritten. Base/post evaluation must pair these same eight prompts and
generation seeds/noise. Generic larger evaluation capability does not expand
this run's eight-prompt budget.

The current experiment contract permits at most10 cumulative actual global
optimizer steps/five rollouts per combination, including resumed updates, and
the same eight prompts for base/post. Asset receipt success does not authorize
longer runs or establish reward improvement, training stability or8×H80080GB
hardware acceptance. No pretrained GPU validation is claimed here.


The prepared acceptance environment pins FFmpeg/ffprobe7.0.2 static, with identical codecs for
all paired samples. One reproducible binary source is the
[7.0.2 static archive](https://johnvansickle.com/ffmpeg/releases/ffmpeg-7.0.2-amd64-static.tar.xz),
SHA256 `abda8d77ce8309141f83ab8edf0596834087c52467f6badf376a6a2a4c87cf67`.
Verify this digest before extracting the two binaries into an operator-selected
persistent tools directory and adding it to PATH. No binary is redistributed
by this repository. Record actual versions/hashes in acceptance evidence.
