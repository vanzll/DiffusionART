"""Bounded H3 environment probe. No checkpoints, Hub calls or license actions.

Default CPU import checks distinguish the model-only stack from all five native
scorers (plus ImageBind when present in the manifest). Optional tiny forward/backward uses random local weights and is not a
real-weight or distributed acceptance test.
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .minimax_h3 import DIFFUSERS_REVISION

NATIVE_SOURCE_SHA256 = {
    "models/transformers/transformer_minimax_h3.py": "92d665e9fa10b1417088341dce5977663a180f9e6fa404fb9330062f353fe7ef",
    "models/autoencoders/autoencoder_kl_minimax_h3.py": "58e0ed47fdee8227d738eec5bb1e7a47acc21c65ee3de01aa3795c97d679d015",
    "models/autoencoders/autoencoder_kl_minimax_h3_audio.py": "b56c5bb95a7dc99ace03df192544615c89ec01ac3b134a7f1dd7348b03731d57",
    "modular_pipelines/minimax_h3/before_denoise.py": "c0c3484b238b7cf529fb494bcfd02c286db8b34ce3e23fd78a17eeda257764d1",
    "modular_pipelines/minimax_h3/encoders.py": "594345d3619198c15fadf42b0fc8c13cec324e546ba10ee4c60c3d06c4b8e2fb",
    "modular_pipelines/minimax_h3/decoders.py": "17580a0335d870cf98870050186ac9ef8551a5da9f7c09b8fe8e21aed3853784",
    "schedulers/scheduling_minimax_h3.py": "307d5bf755337ef00c47237f9ac8be116e627d26e1df3b5f0bd504a80f9de8dd",
}
MODEL_IMPORTS = (
    ("torch", "torch"), ("diffusers", "diffusers"), ("transformers", "transformers"),
    ("peft", "peft"), ("accelerate", "accelerate"), ("safetensors", "safetensors"),
    ("diffusers.models.transformers.transformer_minimax_h3", "diffusers"),
    ("diffusers.modular_pipelines.minimax_h3.encoders", "diffusers"),
)
SCORER_IMPORTS = (
    ("torchaudio", "torchaudio"), ("torchvision", "torchvision"),
    ("librosa", "librosa"), ("scipy", "scipy"),
    ("audiobox_aesthetics.model.aes", "audiobox-aesthetics"),
    ("visual_rl.rewards._native_av.videoalign", "visual-diffusion-rl"),
    ("visual_rl.rewards._native_av.hpsv3_reward", "visual-diffusion-rl"),
    ("visual_rl.rewards._native_av.audiobox", "visual-diffusion-rl"),
    ("visual_rl.rewards._native_av.clap", "visual-diffusion-rl"),
    ("visual_rl.rewards._native_av.desync", "visual-diffusion-rl"),
)

IMAGEBIND_IMPORTS = (
    ("timm", "timm"), ("ftfy", "ftfy"), ("regex", "regex"),
    ("einops", "einops"), ("iopath", "iopath"),
    ("visual_rl.rewards._native_av.imagebind", "visual-diffusion-rl"),
)


def probe_imagebind(manifest_path):
    """Check optional supplied implementation imports/bytes without weights."""
    path = Path(manifest_path).resolve(strict=True)
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("reward manifest exceeds probe size bound")
    data = json.loads(path.read_text())
    names = set(data["paths"]) & {"imagebind", "imagebind_source"}
    if not names:
        return {}, {}, []
    if names != {"imagebind", "imagebind_source"}:
        raise ValueError("ImageBind requires both checkpoint and implementation source")
    from ..rewards._native_av.imagebind import import_model_source
    from ..rewards.audio_video import validate_asset_inventory
    paths = {}
    for name in names:
        root = Path(data["paths"][name])
        paths[name] = root if root.is_absolute() else path.parent / root
    # Import probe verifies source bytes only; checkpoint byte validation belongs
    # to preparation/runtime, avoiding multi-GB reads in this bounded probe.
    validate_asset_inventory({"imagebind_source": paths["imagebind_source"]},
                             {"imagebind_source": data["sha256"]["imagebind_source"]})
    if not paths["imagebind"].is_file():
        raise ValueError("ImageBind checkpoint must be an explicit local file")
    packages, sources, errors = probe_imports(IMAGEBIND_IMPORTS)
    if not errors:
        sources["supplied_imagebind"] = _source_record(import_model_source(paths["imagebind_source"]))
    return packages, sources, errors


def _source_record(module):
    file = getattr(module, "__file__", None)
    if not file:
        raise ValueError("imported module has no inspectable source path")
    path = Path(file).resolve(strict=True)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def probe_imports(imports):
    packages, sources, errors = {}, {}, []
    for module_name, distribution in imports:
        try:
            module = importlib.import_module(module_name)
            try:
                packages[distribution] = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                packages[distribution] = "source checkout (distribution metadata absent)"
            sources[module_name] = _source_record(module)
        except Exception as exc:
            errors.append(f"{module_name}: {type(exc).__name__}: {str(exc)[:400]}")
    return packages, sources, errors


def check_native_sources(root):
    actual, errors = {}, []
    for name, expected in NATIVE_SOURCE_SHA256.items():
        path = Path(root) / name
        if not path.is_file():
            errors.append(f"native source missing: {name}")
            continue
        actual[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual[name] != expected:
            errors.append(f"native source differs from Diffusers@{DIFFUSERS_REVISION}: {name}")
    return actual, errors


def probe_tools():
    tools, errors = {}, []
    for name in ("ffmpeg", "ffprobe"):
        path = shutil.which(name)
        if not path:
            errors.append(f"required executable missing: {name}")
            continue
        try:
            result = subprocess.run([path, "-version"], capture_output=True, text=True,
                                    timeout=10, check=True)
            tools[name] = {"path": path, "version": result.stdout.splitlines()[0][:300]}
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {str(exc)[:300]}")
    return tools, errors


def probe_synchformer(manifest_path):
    if not manifest_path:
        raise ValueError("all-scorers requires --reward-manifest for the actual Synchformer source")
    path = Path(manifest_path).resolve(strict=True)
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("reward manifest exceeds probe size bound")
    data = json.loads(path.read_text())
    root = Path(data["paths"]["desync_source"])
    if not root.is_absolute():
        root = path.parent / root
    root = root.resolve(strict=True)
    from visual_rl.rewards._native_av.desync import _import_synchformer
    module_path = root / "flow_grpo/audio_video_align/synchformer/synchformer.py"
    module = _import_synchformer(root, module_path)
    return _source_record(module)


def tiny_forward_backward(device):
    import torch
    from diffusers import MiniMaxH3Transformer3DModel

    from ..training.audio_video_types import AVState
    from .minimax_h3 import MiniMaxH3Adapter, MiniMaxH3Conditioning
    torch.manual_seed(17)
    model = MiniMaxH3Transformer3DModel(
        num_attention_heads=2, attention_head_dim=12, hidden_size=24, num_layers=1,
        num_refiner_layers=1, ffn_dim=32, in_channels=2, audio_in_channels=3,
        patch_size=(1, 2, 2), text_dim=8, freq_dim=8,
        time_embed_hidden_dim=16, time_embed_dim=12, rope_freq_dim=2,
    ).to(device)
    state = AVState(torch.randn(1, 2, 2, 4, 4, device=device),
                    torch.randn(1, 2, 3, 3, device=device))
    conditioning = MiniMaxH3Conditioning(torch.randn(1, 4, 8, device=device),
                                        torch.ones(1, 4, dtype=torch.bool, device=device))
    out = MiniMaxH3Adapter(model).predict_velocity(state, 0.8, 0.3, conditioning)
    loss = out.video.square().mean() + out.audio.square().mean()
    loss.backward()
    gradients = [p.grad for p in model.parameters()]
    if (not torch.isfinite(loss) or any(g is None or not torch.isfinite(g).all() for g in gradients)
            or model.proj_out.weight.grad.norm() == 0 or model.audio_proj_out.weight.grad.norm() == 0):
        raise RuntimeError("tiny H3 has missing/nonfinite/zero modality gradients")
    return {"device": str(device), "loss": float(loss.detach()), "both_modality_gradients": True,
            "pretrained_weights": False, "distributed_acceptance": False}


def run_probe(mode="model-only", *, tiny_forward=False, cuda=False, reward_manifest=None):
    imports = MODEL_IMPORTS + (SCORER_IMPORTS if mode == "all-scorers" else ())
    packages, sources, errors = probe_imports(imports)
    tools, tool_errors = probe_tools()
    errors.extend(tool_errors)
    native = {}
    try:
        import diffusers
        native, mismatches = check_native_sources(Path(diffusers.__file__).parent)
        errors.extend(mismatches)
    except Exception as exc:
        errors.append(f"native sources: {type(exc).__name__}: {str(exc)[:400]}")
    if mode == "all-scorers":
        try:
            sources["supplied_synchformer"] = probe_synchformer(reward_manifest)
        except Exception as exc:
            errors.append(f"Synchformer: {type(exc).__name__}: {str(exc)[:400]}")
        try:
            extra_packages, extra_sources, extra_errors = probe_imagebind(reward_manifest)
            packages.update(extra_packages)
            sources.update(extra_sources)
            errors.extend(extra_errors)
        except Exception as exc:
            errors.append(f"ImageBind: {type(exc).__name__}: {str(exc)[:400]}")
    result = {"ok": not errors, "mode": mode, "python": sys.version.split()[0],
              "packages": packages, "source_files": sources, "native_source_sha256": native,
              "expected_diffusers_revision": DIFFUSERS_REVISION, "tools": tools, "errors": errors,
              "downloads": False, "pretrained_or_scorer_weights_loaded": False}
    if (tiny_forward or cuda) and not errors:
        try:
            if cuda:
                import torch
                if not torch.cuda.is_available():
                    raise RuntimeError("authorized CUDA probe requested but CUDA is unavailable")
                result["cuda"] = {"torch_cuda": torch.version.cuda,
                                  "device": torch.cuda.get_device_name(0)}
            result["tiny_forward_backward"] = tiny_forward_backward("cuda" if cuda else "cpu")
        except Exception as exc:
            result["errors"].append(f"tiny forward: {type(exc).__name__}: {str(exc)[:400]}")
            result["ok"] = False
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("model-only", "all-scorers"), default="model-only")
    parser.add_argument("--tiny-forward", action="store_true", help="local random-weight CPU forward/backward")
    parser.add_argument("--cuda", action="store_true", help="tiny CUDA forward/backward on an authorized allocation")
    parser.add_argument("--authorized-gpu-allocation", action="store_true")
    parser.add_argument("--reward-manifest")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.cuda and not args.authorized_gpu_allocation:
        parser.error("--cuda requires explicit --authorized-gpu-allocation; never use on a development host")
    if not 1 <= args.timeout <= 600:
        parser.error("--timeout must be between 1 and600 seconds")
    if args._worker:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        result = run_probe(args.mode, tiny_forward=args.tiny_forward, cuda=args.cuda,
                           reward_manifest=args.reward_manifest)
        print(json.dumps(result, sort_keys=True))
        return 0 if result["ok"] else 1
    forwarded = list(sys.argv[1:] if argv is None else argv)
    command = [sys.executable, "-m", __name__, *forwarded, "--_worker"]
    # __main__ is not an importable module in the child interpreter.
    command[2] = "visual_rl.models.check_minimax_h3_environment"
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
    except subprocess.TimeoutExpired:
        print(json.dumps({"ok": False, "mode": args.mode, "errors": ["environment probe timed out"]}))
        return 1
    if completed.stdout.strip():
        print(completed.stdout.strip())
    else:
        print(json.dumps({"ok": False, "mode": args.mode,
                          "errors": [f"probe worker exited {completed.returncode}", completed.stderr[-1000:]]}))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
