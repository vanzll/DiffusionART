#!/usr/bin/env bash
# Install into a NEW environment; never upgrade an existing training environment.
set -euo pipefail
profile=${1:?Usage: bash scripts/install_video_environment.sh hunyuan|wan /absolute/new/venv}
destination=${2:?Supply an absolute path for a new environment}
case "$profile" in hunyuan|wan) ;; *) echo "Unknown profile: $profile" >&2; exit 2 ;; esac
case "$destination" in /*) ;; *) echo "Environment path must be absolute" >&2; exit 2 ;; esac
if [[ $(uname -s) != Linux || $(uname -m) != x86_64 ]]; then
  echo "Native video installation requires Linux x86_64" >&2
  exit 2
fi
if [[ -e "$destination" || -L "$destination" ]]; then
  echo "Refusing to modify existing environment: $destination" >&2
  exit 2
fi
for binary in python3.10 git timeout nvcc ffmpeg ffprobe; do
  command -v "$binary" >/dev/null || { echo "Required executable missing: $binary" >&2; exit 2; }
done
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
echo "Creating Python 3.10 environment: $destination"
timeout -k 5s 120s python3.10 -m venv "$destination"
python="$destination/bin/python"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$destination/cache/pip}"
export TMPDIR="$destination/cache/tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_DEFAULT_TIMEOUT=60
export MAX_JOBS=${MAX_JOBS:-4}
echo "Upgrading pip for normalized wheel metadata before resolving CUDA dependencies"
timeout -k 5s 120s "$python" -m pip install --upgrade 'pip>=25,<26' --index-url https://pypi.org/simple
echo "Installing CUDA 12.4 PyTorch and pinned native dependencies"
timeout -k 10s 7200s "$python" -m pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple
timeout -k 10s 1800s "$python" -m pip install -r "$repo/environments/$profile-py310-linux.lock"
echo "Applying source-preserving Decord wheel tag repair"
timeout -k 5s 120s "$python" -m pip install --no-deps --force-reinstall \
  "$repo/third_party/decord/decord-0.6.0-1-py3-none-manylinux2010_x86_64.whl"
echo "Building FlashAttention 2.7.4.post1 (up to 60 minutes, MAX_JOBS=$MAX_JOBS)"
timeout -k 10s 3600s "$python" -m pip install --no-deps --no-build-isolation flash-attn==2.7.4.post1
native="$destination/third_party/DanceGRPO"
mkdir -p "$destination/third_party"
git init -q "$native"
git -C "$native" remote add origin https://github.com/XueZeyue/DanceGRPO.git
timeout -k 5s 600s git -C "$native" fetch --depth 1 origin 15cc71d53cc2e6e18a68ee607d5fb6ba9a99e344
git -C "$native" checkout -q --detach FETCH_HEAD
# Import native source directly. Installing its package metadata would pull an
# incompatible Diffusers pin into the Wan environment and many unused tools.
timeout -k 5s 120s "$python" -m pip install --no-deps -e "$repo"
timeout -k 5s 60s "$python" -m pip check
"$python" -m visual_rl.check_environment --profile "$profile" --native-root "$native"
printf '\nEnvironment ready for GPU/asset validation.\nPython: %s\nDANCEGRPO_ROOT=%s\n' "$python" "$native"
