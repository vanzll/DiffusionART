"""Pinned native FastVideo loading and identity-checked cached conditioning."""

import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from ..evaluation.manifest import file_sha256
from ..rewards.video_reward import VideoReward
from .finetune import FineTuneConfig, prepare_policy
from .hunyuan import HunyuanVideoAdapter

NATIVE_REVISION = "15cc71d53cc2e6e18a68ee607d5fb6ba9a99e344"


def native_checkout(root):
    root = Path(root).resolve(strict=True)
    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], check=True,
                              capture_output=True, text=True, timeout=20).stdout.strip()
    if git("rev-parse", "HEAD") != NATIVE_REVISION or git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("native DanceGRPO checkout must be clean at the pinned revision")
    existing = sys.modules.get("fastvideo")
    if existing is not None and [Path(p).resolve() for p in existing.__path__] != [root / "fastvideo"]:
        raise ValueError("another fastvideo installation is already imported")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    native = importlib.import_module("fastvideo")
    if [Path(p).resolve() for p in native.__path__] != [root / "fastvideo"]:
        raise ValueError("fastvideo import origin differs from pinned checkout")
    return root


def cached_conditioning(rows, manifest):
    from safetensors import safe_open

    embeddings, masks = [], []
    for row in rows:
        entry = (row.metadata or {}).get("hunyuan_conditioning")
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            raise ValueError("prepare Hunyuan conditioning before training")
        path = Path(manifest).resolve().parent / entry["path"]
        if file_sha256(path) != entry["sha256"]:
            raise ValueError(f"conditioning hash differs: {row.sample_id}")
        with safe_open(path, framework="pt", device="cpu") as tensors:
            metadata = tensors.metadata() or {}
            if (metadata.get("prompt_sha256") != hashlib.sha256(row.text.encode()).hexdigest()
                    or metadata.get("native_revision") != NATIVE_REVISION
                    or set(tensors.keys()) != {"prompt_embeds", "attention_mask"}):
                raise ValueError("conditioning prompt/native identity differs")
            embed, mask = tensors.get_tensor("prompt_embeds"), tensors.get_tensor("attention_mask")
        if (embed.ndim != 2 or mask.ndim != 1 or embed.shape[0] != mask.shape[0] + 1
                or not embed.is_floating_point() or not torch.isfinite(embed).all()
                or not ((mask == 0) | (mask == 1)).all()):
            raise ValueError("conditioning must contain pooled token plus masked text tokens")
        embeddings.append(embed)
        masks.append(mask)
    return torch.stack(embeddings), torch.stack(masks)


def load_adapter(c, device):
    native_checkout(c.model.options["native_root"])
    from diffusers import AutoencoderKLHunyuanVideo
    from fastvideo.models.hunyuan_hf.modeling_hunyuan import HunyuanVideoTransformer3DModel
    from fastvideo.utils.parallel_states import initialize_sequence_parallel_state

    initialize_sequence_parallel_state(1)
    common = dict(revision=c.model.revision, local_files_only=True, torch_dtype=torch.float32)
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(c.model.checkpoint, subfolder="transformer", **common)
    if c.model.options["gradient_checkpointing"] and c.runtime.distributed != "fsdp":
        # This pinned native model owns its non-reentrant checkpoint calls;
        # newer Diffusers enable_gradient_checkpointing uses an incompatible hook.
        transformer.gradient_checkpointing = True
    transformer = prepare_policy(transformer, FineTuneConfig(**c.model.options["finetune"]))
    vae = AutoencoderKLHunyuanVideo.from_pretrained(c.model.checkpoint, subfolder="vae", **common)
    vae.enable_tiling()
    adapter = HunyuanVideoAdapter(transformer, vae).eval()
    if c.runtime.distributed != "fsdp":
        adapter.transformer.to(device)
    return adapter


def load_reward(c):
    native_checkout(c.model.options["native_root"])
    from fastvideo.models.videoalign.inference import VideoVLMRewardInference
    from fastvideo.models.videoalign.prompt_template import build_prompt

    base = Path(c.reward.options["backbone"]).resolve(strict=True)
    reward = Path(c.reward.checkpoint).resolve(strict=True)
    # Native VideoAlign hardcodes this relative backbone name. Scope its working
    # directory to initialization only; do not modify the upstream source/assets.
    previous = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="video-reward-load-") as directory:
        (Path(directory) / "Qwen2-VL-2B-Instruct").symlink_to(base, target_is_directory=True)
        try:
            os.chdir(directory)
            inferencer = VideoVLMRewardInference(str(reward), device="cpu", dtype=torch.bfloat16)
        finally:
            os.chdir(previous)
    return VideoReward(inferencer, build_prompt, weights=c.reward.weights,
                       space="fixed_normalized" if c.reward.normalization == "fixed" else "raw",
                       normalization_source=f"VideoReward@{c.reward.revision}", video_fps=c.sampling.fps,
                       reward_fps=c.evaluation.reward_fps, ffprobe=c.model.options["ffprobe"])


def prepare_conditioning(dataset, *, source_manifest, model_root, native_root, output, device="cuda"):
    """Generate a new manifest; never mutate a user's source prompt file."""
    from safetensors.torch import save_file

    native_checkout(native_root)
    from fastvideo.utils.load import HunyuanTextEncoderWrapper

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    encoder = HunyuanTextEncoderWrapper(str(Path(model_root).resolve()), device).requires_grad_(False).eval()
    records = []
    with torch.no_grad():
        for index, row in enumerate(dataset.rows):
            embed, mask = encoder.encode_prompt([row.text])
            path = output / f"conditioning-{index:06d}.safetensors"
            save_file({"prompt_embeds": embed[0].contiguous().cpu(), "attention_mask": mask[0].contiguous().cpu()},
                      str(path), metadata={"prompt_sha256": hashlib.sha256(row.text.encode()).hexdigest(),
                                           "native_revision": NATIVE_REVISION})
            records.append({"sample_id": row.sample_id, "prompt": row.text,
                            "metadata": {**(row.metadata or {}), "hunyuan_conditioning": {
                                "path": path.name, "sha256": file_sha256(path)}}})
    manifest = output / "prompts.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in records))
    return {"manifest": str(manifest), "sha256": file_sha256(manifest), "source_manifest": str(source_manifest)}


def main():
    import argparse

    from ..data import PromptDataset

    parser = argparse.ArgumentParser(description="Prepare native Hunyuan text conditioning")
    for option in ("manifest", "sha256", "model-root", "native-root", "output"):
        parser.add_argument("--" + option, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    dataset = PromptDataset.load(args.manifest, expected_sha256=args.sha256, task="t2v")
    print(json.dumps(prepare_conditioning(dataset, source_manifest=args.manifest, model_root=args.model_root,
        native_root=args.native_root, output=args.output, device=args.device), sort_keys=True))


if __name__ == "__main__":
    main()
