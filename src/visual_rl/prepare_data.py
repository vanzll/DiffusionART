"""Convert public prompt files to hashed native manifests without changing order."""

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path


def prepare(source, output, *, format, prefix="sample"):
    source, output = Path(source), Path(output)
    if format not in ("text", "geneval") or not re.fullmatch(r"[A-Za-z0-9_.-]+", prefix):
        raise ValueError("format must be text/geneval and prefix must be a safe identifier")
    if source.stat().st_size > 512 * 1024 * 1024:
        raise ValueError("source exceeds 512 MiB")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    source_hash, manifest_hash = hashlib.sha256(), hashlib.sha256()
    rows = 0
    source_bytes = 0
    stage = None
    try:
        with source.open("rb") as inputs, tempfile.NamedTemporaryFile(
            dir=output.parent, prefix=f".{output.name}.", delete=False
        ) as destination:
            stage = Path(destination.name)
            for index in range(1_000_001):
                line = inputs.readline(1024 * 1024 + 1)
                if not line:
                    break
                source_bytes += len(line)
                if source_bytes > 512 * 1024 * 1024:
                    raise ValueError("source exceeds 512 MiB")
                if index == 1_000_000 or len(line) > 1024 * 1024:
                    raise ValueError("source exceeds row/line limit")
                source_hash.update(line)
                text = line.decode("utf-8").strip()
                if not text:
                    raise ValueError(f"blank source row at line {index + 1}")
                metadata = None
                if format == "geneval":
                    metadata = json.loads(text)
                    if (not isinstance(metadata, dict) or not isinstance(metadata.get("tag"), str)
                            or not isinstance(metadata.get("include"), list) or not metadata["include"]):
                        raise ValueError(f"invalid GenEval annotations at line {index + 1}")
                    text = metadata.get("prompt")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"invalid prompt at line {index + 1}")
                row = {"sample_id": f"{prefix}-{index:06d}", "prompt": text}
                if metadata is not None:
                    row["metadata"] = metadata
                block = (json.dumps(row, ensure_ascii=True, allow_nan=False, sort_keys=True) + "\n").encode()
                destination.write(block)
                manifest_hash.update(block)
                rows += 1
            if not rows:
                raise ValueError("empty prompt source")
            destination.flush()
            os.fsync(destination.fileno())
        # Exclusive same-filesystem publication never replaces an existing file.
        os.link(stage, output)
        return {"manifest": str(output.absolute()), "sha256": manifest_hash.hexdigest(),
                "samples": rows, "source_sha256": source_hash.hexdigest(), "format": format}
    finally:
        if stage is not None:
            stage.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--format", required=True, choices=("text", "geneval"))
    parser.add_argument("--prefix", default="sample")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(prepare(args.source, args.output, format=args.format, prefix=args.prefix),
                         sort_keys=True, allow_nan=False))
    except (OSError, ValueError) as error:
        parser.exit(2, f"data preparation failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
