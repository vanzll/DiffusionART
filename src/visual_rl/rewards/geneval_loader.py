"""Checksummed local GenEval assets and legacy-environment native loading."""

import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path

PACKAGES = ("torch", "torchvision", "mmcv-full", "mmdet", "open-clip-torch",
            "clip-benchmark", "numpy", "Pillow")
FILES = {"detector.pth": 2 * 1024**3, "clip.pt": 2 * 1024**3,
         "detector_config.py": 2 * 1024**2, "object_names.txt": 65536}
CLASSES_SHA256 = "608f6a0e5c8ca1c7a92b818430141fe268c85b7930b50f4160f1d65893b5bacd"


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate GenEval manifest key")
        result[key] = value
    return result


def _sha256(path, limit):
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"GenEval asset must be a local regular file: {path.name}")
    digest, count = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            count += len(block)
            if count > limit:
                raise ValueError("GenEval asset exceeds byte limit")
            digest.update(block)
    if not count:
        raise ValueError("empty GenEval asset")
    return digest.hexdigest()


def verify_bundle(manifest_path, manifest_sha256):
    if not isinstance(manifest_path, str) or not manifest_path.strip() or not _digest(manifest_sha256):
        raise ValueError("explicit GenEval manifest path and SHA256 required")
    path = Path(manifest_path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("GenEval manifest must be a local regular file")
    with path.open("rb") as stream:
        raw = stream.read(65537)
    if len(raw) > 65536 or hashlib.sha256(raw).hexdigest() != manifest_sha256:
        raise ValueError("GenEval manifest limit or SHA256 mismatch")
    spec = json.loads(raw, object_pairs_hook=_unique)
    if (not isinstance(spec, dict) or set(spec) != {"schema_version", "versions", "files"}
            or type(spec["schema_version"]) is not int or spec["schema_version"] != 1
            or not isinstance(spec["versions"], dict) or set(spec["versions"]) != set(PACKAGES)
            or any(not isinstance(v, str) or not v.strip() for v in spec["versions"].values())
            or not isinstance(spec["files"], dict) or set(spec["files"]) != set(FILES)
            or any(not _digest(v) for v in spec["files"].values())):
        raise ValueError("invalid GenEval asset manifest")
    if not spec["versions"]["mmdet"].startswith("2."):
        raise ValueError("GenEval requires the MMDetection 2.x result interface")
    root = path.parent.resolve()
    for name, limit in FILES.items():
        if _sha256(root / name, limit) != spec["files"][name]:
            raise ValueError(f"GenEval asset SHA256 mismatch: {name}")
    if spec["files"]["object_names.txt"] != CLASSES_SHA256:
        raise ValueError("GenEval class vocabulary/order differs from reference")
    return root, spec


def _load_native_engine(manifest_path, manifest_sha256, *, device, detector_microbatch, color_workers):
    root, spec = verify_bundle(manifest_path, manifest_sha256)
    for package, expected in spec["versions"].items():
        installed = importlib.metadata.version(package)
        if installed != expected:
            raise RuntimeError(f"GenEval requires {package}=={expected}, found {installed}")
    import torch
    from clip_benchmark.metrics import zeroshot_classification as zsc
    from mmcv import Config
    from mmdet.apis import inference_detector, init_detector
    from open_clip import get_tokenizer
    from open_clip.openai import load_openai_model
    from open_clip.transform import image_transform

    from .geneval_engine import build_evaluator

    # Only flattened, explicitly hashed configs are supported; no untracked bases.
    tree = ast.parse((root / "detector_config.py").read_text())
    if any(isinstance(node, ast.Name) and node.id == "_base_" for node in ast.walk(tree)):
        raise ValueError("GenEval detector config must be flattened before hashing")
    config = Config.fromfile(str(root / "detector_config.py"))
    detector = init_detector(config, str(root / "detector.pth"), device=device)
    clip_model = load_openai_model(str(root / "clip.pt"), precision="fp16", device=device)
    transform = image_transform(224, is_train=False, mean=clip_model.visual.image_mean,
                                std=clip_model.visual.image_std)
    tokenizer = get_tokenizer("ViT-L-14")
    classnames = (root / "object_names.txt").read_text().splitlines()

    def infer(model, images):
        results = []
        for start in range(0, len(images), detector_microbatch):
            results.extend(inference_detector(model, images[start:start + detector_microbatch]))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if len(results) != len(images):
            raise RuntimeError("GenEval detector returned an incomplete batch")
        return results

    evaluate = build_evaluator(
        detector=detector, clip_model=clip_model, transform=transform, tokenizer=tokenizer,
        classnames=classnames, device=device, inference_detector=infer, zsc=zsc,
        color_workers=color_workers,
    )

    def score(images, metadata, only_strict):
        for row in metadata:
            for clause in row.get("include", []) + row.get("exclude", []):
                if clause["class"] not in classnames:
                    raise ValueError("unknown GenEval object class")
        return evaluate(images, metadata, only_strict=only_strict)

    return score


def main(argv=None):
    """Record prepared assets using the intended worker's Python environment."""
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--bundle", required=True)
    args = parser.parse_args(argv)
    root = Path(args.bundle).resolve()
    spec = {"schema_version": 1,
            "versions": {name: importlib.metadata.version(name) for name in PACKAGES},
            "files": {name: _sha256(root / name, limit) for name, limit in FILES.items()}}
    if spec["files"]["object_names.txt"] != CLASSES_SHA256:
        raise ValueError("GenEval class vocabulary/order differs from reference")
    payload = (json.dumps(spec, sort_keys=True, indent=2) + "\n").encode()
    path = root / "manifest.json"
    with path.open("xb") as output:
        output.write(payload)
    print(json.dumps({"manifest": str(path), "sha256": hashlib.sha256(payload).hexdigest()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
