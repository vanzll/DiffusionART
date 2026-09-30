"""Explicit local assets for the PaddleOCR 2.x research scoring interface."""

import hashlib
import importlib.metadata
import json
from pathlib import Path

VERSIONS = {"paddleocr": "2.9.1", "paddlepaddle": "2.6.2"}
MODEL_FILES = frozenset(
    f"{kind}/inference.{suffix}"
    for kind in ("det", "rec", "cls")
    for suffix in ("pdmodel", "pdiparams")
) | {"en_dict.txt"}


def _digest(value):
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate OCR manifest key: {key}")
        result[key] = value
    return result


def _sha256(path, limit):
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"OCR asset must be a local regular file: {path}")
    total, digest = 0, hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            total += len(chunk)
            if total > limit:
                raise ValueError(f"OCR asset exceeds byte limit: {path}")
            digest.update(chunk)
    if not total:
        raise ValueError(f"empty OCR asset: {path}")
    return digest.hexdigest()


def verify_bundle(manifest_path, manifest_sha256):
    """Verify fixed filenames and bytes before importing/loading PaddleOCR.

    This is a synchronous local asset check, not a shared-storage I/O deadline.
    The bundle must remain immutable throughout model construction.
    """
    if (
        not isinstance(manifest_path, str)
        or not manifest_path.strip()
        or not _digest(manifest_sha256)
    ):
        raise ValueError("explicit OCR manifest path and lowercase SHA256 are required")
    path = Path(manifest_path)
    if not path.is_file() or path.is_symlink():
        raise ValueError("OCR manifest must be a local regular file")
    with path.open("rb") as stream:
        payload = stream.read(65537)
    if len(payload) > 65536:
        raise ValueError("OCR manifest exceeds byte limit")
    if hashlib.sha256(payload).hexdigest() != manifest_sha256:
        raise ValueError("OCR manifest SHA256 mismatch")
    document = json.loads(payload, object_pairs_hook=_unique)
    if (
        not isinstance(document, dict)
        or set(document) != {"schema_version", "versions", "ocr_version", "files"}
        or type(document["schema_version"]) is not int
        or document["schema_version"] != 1
        or document["versions"] != VERSIONS
        or document["ocr_version"] not in ("PP-OCRv3", "PP-OCRv4")
        or not isinstance(document["files"], dict)
        or set(document["files"]) != MODEL_FILES
        or any(not _digest(value) for value in document["files"].values())
    ):
        raise ValueError("invalid or unsupported OCR asset manifest")
    root = path.parent.resolve()
    for kind in ("det", "rec", "cls"):
        if (root / kind).is_symlink():
            raise ValueError("OCR model directories must not be symlinks")
    for name, expected in document["files"].items():
        limit = 1024 * 1024 if name == "en_dict.txt" else 512 * 1024 * 1024
        if _sha256(root / name, limit) != expected:
            raise ValueError(f"OCR asset SHA256 mismatch: {name}")
    return root, document


def _load_native_engine(manifest_path, manifest_sha256):
    root, document = verify_bundle(manifest_path, manifest_sha256)
    for package, expected in VERSIONS.items():
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"OCR requires {package}=={expected}; see docs/sd3-ocr.md") from exc
        if installed != expected:
            raise RuntimeError(f"OCR requires {package}=={expected}, found {installed}")
    from paddleocr import PaddleOCR

    # PaddleOCR 2.x checks all three model directories even with cls disabled.
    return PaddleOCR(
        use_angle_cls=False,
        lang="en",
        use_gpu=False,
        show_log=False,
        ocr_version=document["ocr_version"],
        det_model_dir=str(root / "det"),
        rec_model_dir=str(root / "rec"),
        cls_model_dir=str(root / "cls"),
        rec_char_dict_path=str(root / "en_dict.txt"),
    )


def load_ocr_reward(*, manifest_path, manifest_sha256, error_policy, device):
    if error_policy not in ("raise", "zero"):
        raise ValueError("OCR error_policy must be raise or zero")
    # Cheap, actionable failures precede worker startup; the worker rechecks bytes.
    verify_bundle(manifest_path, manifest_sha256)
    from .ocr import OCRScore
    from .ocr_process import PaddleWorker

    engine = PaddleWorker(manifest_path, manifest_sha256)
    try:
        return OCRScore(engine, error_policy=error_policy, device=device)
    except BaseException:
        engine.close()
        raise


def main(argv=None):
    """Record a prepared local bundle; this neither downloads nor certifies assets."""
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--ocr-version", required=True, choices=("PP-OCRv3", "PP-OCRv4"))
    args = parser.parse_args(argv)
    root = Path(args.bundle).resolve()
    document = {
        "schema_version": 1,
        "versions": VERSIONS,
        "ocr_version": args.ocr_version,
        "files": {
            name: _sha256(root / name, 1024 * 1024 if name == "en_dict.txt" else 512 * 1024 * 1024)
            for name in sorted(MODEL_FILES)
        },
    }
    for kind in ("det", "rec", "cls"):
        if (root / kind).is_symlink():
            raise ValueError("OCR model directories must not be symlinks")
    payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    path = root / "manifest.json"
    with path.open("xb") as stream:
        stream.write(payload)
    print(json.dumps({"manifest": str(path), "sha256": hashlib.sha256(payload).hexdigest()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
