"""Create a native text-encoder view without copying or modifying model assets."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


def prepare_text_encoder_view(model_root, output_dir):
    source = Path(model_root).resolve(strict=True)
    output = Path(os.path.abspath(output_dir))
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite output: {output}")
    parent = output.parent.resolve(strict=True)
    if parent == source or source in parent.parents:
        raise ValueError("asset view must be outside the source model directory")
    required = {
        "tokenizer": ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"),
        "tokenizer_2": ("vocab.json", "merges.txt", "tokenizer_config.json", "special_tokens_map.json"),
    }
    plans = {}
    for encoder, tokenizer in (("text_encoder", "tokenizer"), ("text_encoder_2", "tokenizer_2")):
        if not (source / encoder / "config.json").is_file():
            raise ValueError(f"missing encoder file: {source / encoder / 'config.json'}")
        for name in required[tokenizer]:
            if not (source / tokenizer / name).is_file():
                raise ValueError(f"missing tokenizer file: {source / tokenizer / name}")
        entries = {}
        for folder in (encoder, tokenizer):
            for entry in sorted((source / folder).iterdir()):
                if entry.name in entries:
                    raise ValueError(f"encoder/tokenizer name conflict: {encoder}/{entry.name}")
                entries[entry.name] = entry.resolve(strict=True)
        plans[encoder] = entries

    # Publish exclusively; concurrent preparations cannot replace an existing view.
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        for encoder, entries in plans.items():
            directory = stage / encoder
            directory.mkdir()
            for name, target in entries.items():
                (directory / name).symlink_to(target, target_is_directory=target.is_dir())
        receipt = {"source": str(source), "output": str(output), "weights_copied": False,
                   "files": {encoder: {name: str(target) for name, target in entries.items()}
                             for encoder, entries in plans.items()}}
        (stage / "asset-view.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
        output.symlink_to(stage.name, target_is_directory=True)
    except BaseException:
        print(f"Unpublished asset view preserved at {stage}", file=sys.stderr)
        raise
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_text_encoder_view(args.model_root, args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
