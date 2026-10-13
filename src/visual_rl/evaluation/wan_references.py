"""Frozen, content-checked FLUX evaluation references, separate from training."""

import errno
import hashlib
import json
import platform
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from PIL import Image, features

from ..models.wan_assets import reference_mode


def evaluation_reference_mode(options):
    if "evaluation_reference_cache" in options:
        path = options["evaluation_reference_cache"]
        if not isinstance(path, str) or not path.strip():
            raise ValueError("Wan evaluation_reference_cache must be a nonempty string")
    mode = options.get("evaluation_reference_mode", "fixed_images")
    if mode not in ("fixed_images", "online_flux"):
        raise ValueError("Wan evaluation_reference_mode must be fixed_images or online_flux")
    if mode == "online_flux" and reference_mode(options) != "online_flux":
        raise ValueError("Wan evaluation_reference_mode=online_flux requires reference_mode=online_flux")
    return mode


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _runtime_versions():
    # An explicit allowlist, never an environment/config dump (which may hold tokens).
    result = {"python": platform.python_version(), "cuda": torch.version.cuda,
              "jpeg": features.version_codec("jpg"), "zlib": features.version_codec("zlib")}
    for name in ("torch", "diffusers", "transformers", "accelerate", "numpy", "Pillow", "safetensors"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def _identity(c, row, device):
    base_seed = c.seed if c.evaluation.seed is None else c.evaluation.seed
    digest = hashlib.sha256(_json(["wan-evaluation-reference-v1", base_seed, row.sample_id])).digest()
    seed = int.from_bytes(digest[:8], "little") % (2**63 - 1)
    return {
        "schema_version": 1, "prompt": row.text, "sample_id": row.sample_id,
        "evaluation_seed": base_seed, "seed": seed,
        "checkpoint": c.model.options["reference_checkpoint"],
        "revision": c.model.options["reference_revision"],
        "height": c.evaluation.height or c.sampling.height,
        "width": c.evaluation.width or c.sampling.width,
        "generator": {"helper": "wan_assets.generate_reference-v1", "num_inference_steps": 30,
                      "guidance_scale": 3.5, "max_sequence_length": 512, "dtype": "bfloat16",
                      "rng_device": "cpu", "transport": "Pillow-default-JPEG-decode-RGB",
                      "cache_format": "PNG", "device_type": torch.device(device).type,
                      "allow_tf32": c.model.options["allow_tf32"]},
        "runtime_versions": _runtime_versions(),
    }


def _load(directory, identity):
    try:
        metadata = json.loads((directory / "metadata.json").read_bytes())
        if (not isinstance(metadata, dict) or set(metadata) != {"identity", "png_sha256"}
                or _json(metadata["identity"]) != _json(identity)):
            raise ValueError("metadata identity differs")
        payload = (directory / "reference.png").read_bytes()
        if hashlib.sha256(payload).hexdigest() != metadata["png_sha256"]:
            raise ValueError("PNG hash differs")
        with Image.open(BytesIO(payload)) as source:
            if (source.format != "PNG" or source.mode != "RGB"
                    or source.size != (identity["width"], identity["height"])):
                raise ValueError("PNG format/mode/geometry differs")
            source.verify()
        with Image.open(BytesIO(payload)) as source:
            return source.copy()
    except (OSError, ValueError, SyntaxError) as error:
        raise ValueError(f"invalid Wan evaluation reference cache at {directory}: {error}") from error


def cached_reference(c, row, flux, device):
    """Reuse exact pixels across evaluations/resume; never regenerate corrupt entries.

    A complete image/metadata pair is published by one same-filesystem rename.
    Concurrent publishers use the winning entry, including its exact pixels.
    """
    from ..models.wan_assets import generate_reference

    identity = _identity(c, row, device)
    root = Path(c.model.options.get("evaluation_reference_cache",
                                   Path(c.runtime.output_dir) / "evaluation" / "wan_references")).expanduser()
    directory = root / hashlib.sha256(_json(identity)).hexdigest()
    if directory.exists() or directory.is_symlink():
        return _load(directory, identity)
    root.mkdir(parents=True, exist_ok=True)
    image = generate_reference(flux, row.text, height=identity["height"], width=identity["width"],
                               device=device, seed=identity["seed"])
    if (not isinstance(image, Image.Image) or image.mode != "RGB"
            or image.size != (identity["width"], identity["height"])):
        raise ValueError("FLUX evaluation reference must be RGB with the requested geometry")
    with TemporaryDirectory(prefix=".pending-", dir=root) as temporary:
        pending = Path(temporary)
        image.save(pending / "reference.png", format="PNG")
        payload = (pending / "reference.png").read_bytes()
        (pending / "metadata.json").write_bytes(_json({
            "identity": identity, "png_sha256": hashlib.sha256(payload).hexdigest()}))
        _load(pending, identity)
        try:
            pending.rename(directory)
        except OSError as error:
            if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
    return _load(directory, identity)
