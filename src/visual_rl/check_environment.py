"""Bounded native video dependency/import checks, without downloading models."""

import argparse
import importlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys

CORE = {
    "torch": "2.6.0", "torchvision": "0.21.0", "transformers": "4.46.1",
    "tokenizers": "0.20.1", "accelerate": "1.0.1", "peft": "0.13.2",
    "trl": "0.8.6", "datasets": "2.18.0", "numpy": "1.26.3",
    "flash-attn": "2.7.4.post1", "av": "13.1.0",
}
PROFILES = {
    "hunyuan": {"diffusers": "0.32.0", "huggingface-hub": "0.26.1"},
    "wan": {"diffusers": "0.33.1", "huggingface-hub": "0.30.2"},
}


def check_versions(profile, *, version=importlib.metadata.version):
    expected = {**CORE, **PROFILES[profile]}
    observed = {}
    for name, pin in expected.items():
        actual = version(name)
        if actual.split("+")[0] != pin:
            raise RuntimeError(f"{name}: expected {pin}, installed {actual}")
        observed[name] = actual
    return observed


def probe(profile, native_root, *, cuda=False):
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("native video profiles require Python 3.10")
    observed = check_versions(profile)
    from .models.hunyuan_assets import native_checkout

    root = native_checkout(native_root)
    names = ["torchvision", "datasets", "trl", "flash_attn",
             "fastvideo.models.videoalign.inference"]
    if profile == "hunyuan":
        names += ["fastvideo.models.hunyuan_hf.modeling_hunyuan", "fastvideo.utils.load"]
    else:
        names += ["diffusers.models.transformers.transformer_wan",
                  "diffusers.pipelines.wan.pipeline_wan_i2v", "diffusers.pipelines.flux.pipeline_flux"]
    for name in names:
        print(json.dumps({"phase": "import", "module": name}), flush=True)
        importlib.import_module(name)
    for binary in ("ffmpeg", "ffprobe"):
        result = subprocess.run([binary, "-version"], check=True, capture_output=True,
                                text=True, timeout=10)
        observed[binary] = result.stdout.splitlines()[0]
    if cuda:
        import torch
        from flash_attn import flash_attn_func

        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("a visible BF16-capable CUDA GPU is required")
        if not torch.distributed.is_nccl_available():
            raise RuntimeError("NCCL is unavailable")
        # A small real kernel checks the compiled extension, not just its import.
        q = torch.ones(1, 16, 2, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        value = flash_attn_func(q, q, q, dropout_p=0., causal=False)
        value.float().sum().backward()
        torch.cuda.synchronize()
        if not torch.isfinite(value).all() or q.grad is None or not torch.isfinite(q.grad).all():
            raise RuntimeError("FlashAttention forward/backward produced invalid values")
        observed.update(cuda=torch.version.cuda, gpu=torch.cuda.get_device_name())
    print(json.dumps({"passed": True, "profile": profile, "cuda_kernel_checked": cuda,
                      "native_root": str(root), "versions": observed}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--native-root", required=True)
    parser.add_argument("--cuda", action="store_true", help="run a small kernel on an authorized GPU")
    parser.add_argument("--timeout-seconds", type=float, default=120)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        parser.error("timeout must be positive and finite")
    if args.worker:
        probe(args.profile, args.native_root, cuda=args.cuda)
        return 0
    command = [sys.executable, "-u", "-m", "visual_rl.check_environment", "--worker",
               "--profile", args.profile, "--native-root", args.native_root]
    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                   "PYTHONDONTWRITEBYTECODE": "1"}
    if args.cuda:
        command.append("--cuda")
    else:
        environment["CUDA_VISIBLE_DEVICES"] = ""
    try:
        return subprocess.run(command, env=environment, timeout=args.timeout_seconds).returncode
    except subprocess.TimeoutExpired:
        print("native import/kernel check timed out; no model loading was attempted", file=sys.stderr)
        return 124


if __name__ == "__main__":
    raise SystemExit(main())
