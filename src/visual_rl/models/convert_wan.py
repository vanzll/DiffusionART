"""Offline, CPU-only Wan checkpoint preparation with Diffusers 0.33.1 naming.

Run with ``python -m visual_rl.models.convert_wan --help``. The original config
is the local Diffusers transformer config.json, not the original Wan repo config.
Tensor payload per shard is capped by shard_bytes; one larger tensor is allowed
alone. Live tensor references are bounded by one shard plus one incoming tensor
(in addition to safetensors serialization overhead and source mmap/page cache).
No tensor arithmetic, dtype conversion, full state-dict load, or GPU is used.

Publication atomically creates a relative directory symlink, never replacing any
existing entry, even an empty directory or dangling symlink. Keep its sibling
backing directory when moving the output, or copy the dereferenced directory.
This requires a filesystem supporting directory symlinks. Failed staging is
never deleted or resumed automatically; its absolute path is reported. Atomic
publication is not a guarantee against power loss or external file modification.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
import sys
import tempfile
from pathlib import Path

DIFFUSERS_VERSION = "0.33.1"
SOURCE = {
    "repo_id": "lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v",
    "revision": "fef288b326f4fed6d2983b9800c35363da31fcfe",
    "filename": "distill_models/distill_model.safetensors",
    "sha256": "12869d6856085d97cbf502fecbed85d1a67ce108174f95d82547497522479599",
    "size": 32789894024,
}
INDEX_NAME = "diffusion_pytorch_model.safetensors.index.json"
RECEIPT_NAME = "conversion-receipt.json"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _write_json(path, value):
    with path.open("xb") as stream:
        stream.write(_json_bytes(value) + b"\n")


def _native_converter():
    import diffusers

    if diffusers.__version__ != DIFFUSERS_VERSION:
        raise RuntimeError(f"conversion requires diffusers=={DIFFUSERS_VERSION}; "
                           f"installed {diffusers.__version__}")
    from diffusers import WanTransformer3DModel
    from diffusers.loaders.single_file_utils import convert_wan_transformer_to_diffusers

    return WanTransformer3DModel, convert_wan_transformer_to_diffusers


def rename_keys(keys, converter):
    """Use the official converter on name tokens, including norm2/norm3 swaps."""
    keys = list(keys)
    converted = converter({key: key for key in keys})
    if (len(set(keys)) != len(keys) or not isinstance(converted, dict)
            or len(converted) != len(keys)
            or any(not isinstance(k, str) or not isinstance(v, str)
                   for k, v in converted.items())
            or set(converted.values()) != set(keys)):
        raise ValueError("lossy/colliding Wan conversion map")
    return converted


def _expected_shapes(model_cls, config):
    import torch

    with torch.device("meta"):
        model = model_cls.from_config(config)
    state = model.state_dict()
    if not state or any(value.device.type != "meta" for value in state.values()):
        raise ValueError("Wan architecture validation requires all parameters/buffers on meta")
    return {key: tuple(value.shape) for key, value in state.items()}


def convert_checkpoint(source, original_config, output_dir, *, expected_source_sha256,
                       shard_bytes=1024**3):
    """Return and save a portable receipt; source/config must remain immutable.

    The receipt inventories config, index and all shards (not itself), with no
    machine-specific paths. Source provenance pins are included only when both
    size and digest match SOURCE. A failed attempt leaves its staging directory.
    """
    if type(shard_bytes) is not int or shard_bytes <= 0:
        raise ValueError("shard_bytes must be a positive integer")
    if (not isinstance(expected_source_sha256, str)
            or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_source_sha256)):
        raise ValueError("expected_source_sha256 must be a 64-digit SHA256")
    source, original_config = Path(source), Path(original_config)
    # Do not resolve the final component: a dangling destination link exists too.
    output = Path(os.path.abspath(output_dir))
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite output: {output}")
    before = source.stat()
    digest = file_sha256(source)
    if digest != expected_source_sha256.lower():
        raise ValueError("source SHA256 mismatch; conversion not started")
    config_bytes = original_config.read_bytes()
    config = json.loads(config_bytes)
    if not isinstance(config, dict):
        raise ValueError("original config must be a Diffusers transformer JSON object")
    model_cls, converter = _native_converter()
    expected = _expected_shapes(model_cls, config)
    converter_identity = {
        "diffusers": DIFFUSERS_VERSION,
        "name": f"{converter.__module__}.{converter.__name__}",
        "source_sha256": hashlib.sha256(inspect.getsource(converter).encode()).hexdigest(),
        "implementation_sha256": file_sha256(__file__),
    }
    from safetensors import safe_open
    from safetensors.torch import save_file

    # Parent must already exist; never create unrelated directory trees.
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    marker = stage / "INCOMPLETE"
    print(f"Wan conversion staging: {stage}", file=sys.stderr, flush=True)
    try:
        marker.write_text("Unpublished conversion. Preserve for inspection; do not load.\n")
        (stage / "config.json").write_bytes(config_bytes)
        weight_map, chunk, total, chunk_bytes, shard = {}, {}, 0, 0, 0

        def flush():
            nonlocal chunk, chunk_bytes, shard
            if not chunk:
                return
            shard += 1
            filename = f"diffusion_pytorch_model-{shard:05d}.safetensors"
            save_file(chunk, str(stage / filename), metadata={"format": "pt"})
            weight_map.update({key: filename for key in chunk})
            print(f"Wan shard {shard}: {chunk_bytes} tensor bytes", file=sys.stderr, flush=True)
            chunk, chunk_bytes = {}, 0

        with safe_open(str(source), framework="pt", device="cpu") as checkpoint:
            mapping = rename_keys(checkpoint.keys(), converter)
            missing, extra = set(expected) - set(mapping), set(mapping) - set(expected)
            if missing or extra:
                raise ValueError(f"Wan architecture key mismatch: missing={sorted(missing)} "
                                 f"extra={sorted(extra)}")
            # Validate ALL shapes from metadata before loading the first tensor.
            for name, old in mapping.items():
                shape = tuple(checkpoint.get_slice(old).get_shape())
                if shape != expected[name]:
                    raise ValueError(f"Wan shape mismatch: {old} -> {name}: "
                                     f"{shape} != {expected[name]}")
            for name, old in sorted(mapping.items()):
                tensor = checkpoint.get_tensor(old)
                size = tensor.numel() * tensor.element_size()
                if chunk and chunk_bytes + size > shard_bytes:
                    flush()
                chunk[name] = tensor
                chunk_bytes += size
                total += size
                del tensor
                if chunk_bytes >= shard_bytes:
                    flush()
            flush()

        after = source.stat()
        if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)):
            raise ValueError("source changed during conversion; output not published")
        _write_json(stage / INDEX_NAME, {"metadata": {"total_size": total},
                                       "weight_map": weight_map})
        source_identity = {"sha256": digest, "size": before.st_size}
        if digest == SOURCE["sha256"] and before.st_size == SOURCE["size"]:
            source_identity.update(SOURCE)
        receipt = {
            "schema_version": "wan-checkpoint-conversion/v1",
            "source": source_identity,
            "config": {"sha256": hashlib.sha256(config_bytes).hexdigest(),
                       "size": len(config_bytes)},
            "converter": converter_identity,
            "mapping_sha256": hashlib.sha256(_json_bytes(mapping)).hexdigest(),
            "value_transform": "none", "dtype_transform": "none",
            "tensor_count": len(weight_map), "tensor_bytes": total,
            "shard_bytes": shard_bytes,
            "files": [{"path": file.name, "size": file.stat().st_size,
                       "sha256": file_sha256(file)}
                      for file in sorted(stage.iterdir()) if file.name != "INCOMPLETE"],
        }
        _write_json(stage / RECEIPT_NAME, receipt)
        marker.unlink()
        # symlink() is atomic and exclusive, unlike rename() over an empty dir.
        os.symlink(stage.name, output, target_is_directory=True)
    except BaseException:
        try:
            marker.write_text("Unpublished conversion. Preserve for inspection; do not load.\n")
        except OSError:
            pass  # The staging name and stderr still identify an incomplete attempt.
        print(f"Wan conversion failed; unpublished staging preserved at {stage}",
              file=sys.stderr, flush=True)
        raise
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--original-config", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-bytes", type=int, default=1024**3)
    args = parser.parse_args(argv)
    receipt = convert_checkpoint(args.source, args.original_config, args.output_dir,
                                 expected_source_sha256=args.expected_source_sha256,
                                 shard_bytes=args.shard_bytes)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
