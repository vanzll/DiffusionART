"""Package unchanged fixed evaluation images for a portable Hub dataset."""

import argparse
import json
import math
import re
import shutil
import tempfile
from pathlib import Path

from ..config import ConfigError
from ..data import PromptDataset, file_sha256
from ..prepare_assets import _fixed_images

_CARD = """---
license: cc-by-nc-4.0
task_categories:
  - image-to-video
configs:
  - config_name: default
    data_files:
      - split: validation
        path: prompts.jsonl
---
# DiffusionART I2V Validation

Fixed reference images and prompts used for the Wan I2V evaluation recipes.
This is a custom validation split, not the official DanceGRPO test set.
Images retain the original JPEG bytes; no images were regenerated or re-encoded.

Prompts come from [VidProM](https://huggingface.co/datasets/WenhaoWang/VidProM),
licensed under [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/).
Reference images were generated with
[FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev).
Its [output terms](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md)
also apply; the dataset license does not replace them.

`prompts.jsonl` preserves ordered sample IDs, prompt text, relative image paths,
and image SHA256 values. `provenance.json` records the source manifest digest
and reference-image generation settings. Model weights, training videos,
machine paths, credentials, and private run logs are not included.

Download with `snapshot_download(repo_id=..., repo_type="dataset", revision=...)`.
Point `WAN_EVAL_RAW_MANIFEST` at the downloaded `prompts.jsonl`, or use a recipe
declaring this pinned dataset in `assets.snapshots`.
"""


def _public_generation(generation):
    numbers = {"height", "width", "num_inference_steps", "guidance_scale", "max_sequence_length",
               "seed", "jpeg_quality", "jpeg_subsampling", "batch_size"}
    choices = {"dtype": {"bfloat16", "float16", "float32"}, "generator_device": {"cpu", "cuda"},
               "image_format": {"JPEG", "PNG"}}
    result = {}
    for key in numbers & generation.keys():
        value = generation[key]
        if type(value) not in {int, float} or not math.isfinite(value):
            raise ConfigError("invalid public image-generation parameter")
        result[key] = value
    for key, allowed in choices.items():
        if key in generation:
            if not isinstance(generation[key], str) or generation[key] not in allowed:
                raise ConfigError("invalid public image-generation setting")
            result[key] = generation[key]
    return result


def package(source, output, *, expected_count=100, image_root=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise ConfigError("refusing to overwrite an evaluation bundle")
    if type(expected_count) is not int or expected_count < 1:
        raise ConfigError("expected_count must be positive")
    source_digest = file_sha256(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".i2v-eval-", dir=output.parent) as temporary:
        stage = Path(temporary) / "dataset"
        stage.mkdir()
        _fixed_images(source, stage, "wan_validation", image_root=image_root)
        manifest = stage / "prompts.jsonl"
        dataset = PromptDataset.load(manifest, task="i2v")
        if len(dataset.rows) != expected_count:
            raise ConfigError("fixed evaluation sample count differs")
        from PIL import Image

        for row in dataset.rows:
            with Image.open(row.image) as image:
                image.verify()
        provenance = {"source_manifest_sha256": source_digest,
                      "manifest_sha256": file_sha256(manifest), "sample_count": len(dataset.rows),
                      "reference_images": "unchanged original bytes", "image_generator": "FLUX.1-dev"}
        original = json.loads(source.read_text())
        provenance["generation"] = _public_generation(original.get("generation", {}))
        model = original.get("reference_model", {})
        provenance["reference_model"] = {
            key: value for key, value in model.items() if isinstance(value, str) and (
                key in {"repo_id", "repository"} and re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value)
                or key == "revision" and re.fullmatch(r"[0-9a-f]{40}", value))}
        prompt_digest = original.get("source", {}).get("prompts_sha256")
        if isinstance(prompt_digest, str) and re.fullmatch(r"[0-9a-f]{64}", prompt_digest):
            provenance["source_prompt_manifest_sha256"] = prompt_digest
        # Publish only scientific metadata, not arbitrary source/run dictionaries.
        rows = [json.loads(line) for line in manifest.read_text().splitlines()]
        for row, old in zip(rows, original["samples"]):
            if row["sample_id"] != old["id"]:
                raise ConfigError("fixed evaluation identity/order changed")
            if "image_seed" in old:
                if type(old["image_seed"]) is not int or not 0 <= old["image_seed"] < 2**63:
                    raise ConfigError("invalid fixed-image seed")
                row["metadata"] = {"image_seed": old["image_seed"]}
        manifest.write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=True) + "\n" for r in rows))
        provenance["manifest_sha256"] = file_sha256(manifest)
        if file_sha256(source) != source_digest:
            raise ConfigError("source evaluation manifest changed during packaging")
        (stage / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
        (stage / "README.md").write_text(_CARD)
        shutil.move(str(stage), str(output))
    return provenance


def publish(output, repo_id):
    from huggingface_hub import HfApi

    output = Path(output).resolve()
    provenance = json.loads((output / "provenance.json").read_text())
    dataset = PromptDataset.load(output / "prompts.jsonl", task="i2v",
                                 expected_sha256=provenance["manifest_sha256"])
    if len(dataset.rows) != provenance["sample_count"]:
        raise ConfigError("evaluation bundle count changed")
    api = HfApi()
    try:
        api.create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True)
        commit = api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(output),
                                   allow_patterns=["README.md", "prompts.jsonl", "provenance.json", "images/*.jpg",
                                                   "images/*.jpeg", "images/*.png"],
                                   commit_message="Publish fixed I2V validation prompts and reference images")
    except Exception as exc:
        raise ConfigError(f"HF dataset publication failed ({type(exc).__name__}); "
                          "check the account, write access and network; no successful publication recorded") from None
    return {"repo_id": repo_id, "revision": commit.oid,
            "manifest_sha256": provenance["manifest_sha256"], "sample_count": len(dataset.rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source")
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-count", type=int, default=100)
    parser.add_argument("--image-root", help="Directory containing unchanged original JPEGs after relocation")
    parser.add_argument("--publish", metavar="NAMESPACE/DATASET")
    args = parser.parse_args()
    if args.source:
        print(json.dumps(package(args.source, args.output, expected_count=args.expected_count,
                                 image_root=args.image_root), sort_keys=True))
    elif not args.publish:
        parser.error("--source or --publish is required")
    if args.publish:
        print(json.dumps(publish(args.output, args.publish), sort_keys=True))


if __name__ == "__main__":
    main()
