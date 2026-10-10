"""Freeze and publish ConsisID training / VidProM test reference images."""

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

import yaml
from PIL import Image

from ..config import ConfigError, _read
from ..data import PromptDataset, file_sha256
from ..prepare_assets import _vidprom_manifest
from .package_i2v import package
from .select_prompts import _Exclusions

TRAIN_SHA = "85096428c4aa4467345a1ddd0855f004d26c1ba39cffafd4a06d8b8d798a4f6f"
TRAIN_ROWS_SHA = "caa19dd080dd899f66c2356a69879dfebc952babc91485a145858f4936cb85bd"
EVAL_SHA = "3b3392989d399ec0b840fc1161d83858872d3843806f4800162dd728958f7e69"
PREFIX_SHA = "af1cecf6f794db3bbcd85d90cbca0e10cb196bc8f043a2a3724c00635383e70f"
ORIGINAL_SHA = "2a43b566a94418b7b260d967a34c40545d99bd0a8a66cce50dc89bd0ff872d4c"
MODEL = {"repo_id": "black-forest-labs/FLUX.1-dev",
         "revision": "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"}
GENERATION = {"height": 400, "width": 640, "num_inference_steps": 30,
              "guidance_scale": 3.5, "max_sequence_length": 512,
              "dtype": "bfloat16", "generator_device": "cpu",
              "jpeg_quality": 75, "jpeg_subsampling": 2}
TRAIN_COUNT, TEST_COUNT, PREFIX_COUNT = 27815, 1000, 100


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False)


def _atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _row_bytes(rows):
    return "".join(_json(row) + "\n" for row in rows).encode()


def create_plan(training, evaluation, output):
    """CPU-only admission of frozen public prompts; never filter training rows."""
    training, evaluation, output = Path(training), Path(evaluation), Path(output)
    if file_sha256(training) != TRAIN_SHA or file_sha256(evaluation) != EVAL_SHA:
        raise ConfigError("I2V source differs from the pinned prompt release")
    texts = [line.strip() for line in training.read_text().splitlines() if line.strip()]
    test = [json.loads(line) for line in evaluation.read_text().splitlines()]
    if len(texts) != TRAIN_COUNT or len(test) != TEST_COUNT:
        raise ConfigError("I2V requires all 27815 training prompts and 1000 fixed test prompts")
    if hashlib.sha256(_row_bytes(test[:PREFIX_COUNT])).hexdigest() != PREFIX_SHA:
        raise ConfigError("I2V test prefix differs from the original evaluation100")
    PromptDataset.load(evaluation, task="t2v", expected_sha256=EVAL_SHA)
    exclusions, unique = _Exclusions(texts), _Exclusions([])
    for row in test:
        if exclusions.matches(row["prompt"]) or unique.matches(row["prompt"]):
            raise ConfigError("I2V test overlaps training or duplicates another test prompt")
        unique.add(row["prompt"])
    plan = {"schema_version": 1, "reference_model": MODEL, "generation": GENERATION,
            "sources": {"train": {"dataset": "ConsisID", "sha256": TRAIN_SHA,
                                   "release": "DanceGRPO/15cc71d/assets/consist-id.txt"},
                        "test": {"dataset": "WenhaoWang/VidProM", "sha256": EVAL_SHA,
                                  "revision": "0c0900e6a037b058b1903bc1de9bcb6037d9fc24"}},
            "original_evaluation": {"count": PREFIX_COUNT, "manifest_sha256": ORIGINAL_SHA},
            "splits": {"train": [{"sample_id": f"train:{i}", "prompt": text,
                                   "image_seed": 312 + i} for i, text in enumerate(texts)],
                       "test": [{**row, "image_seed": 312 + i} for i, row in enumerate(test)]}}
    destination = output / "plan.json"
    content = (_json(plan) + "\n").encode()
    if destination.exists():
        if destination.read_bytes() != content:
            raise ConfigError("existing I2V preparation plan differs")
    elif output.exists():
        raise ConfigError("refusing to adopt an existing directory without an I2V plan")
    else:
        _atomic(destination, content)
    return {"plan_sha256": file_sha256(destination), "train_count": len(texts), "test_count": len(test)}


def _load_plan(output):
    plan = json.loads((Path(output) / "plan.json").read_text())
    if (plan.get("schema_version") != 1 or plan.get("reference_model") != MODEL
            or plan.get("generation") != GENERATION
            or set(plan) != {"schema_version", "reference_model", "generation", "sources",
                             "original_evaluation", "splits"}
            or plan.get("original_evaluation") != {"count": PREFIX_COUNT, "manifest_sha256": ORIGINAL_SHA}
            or set(plan.get("splits", {})) != {"train", "test"}):
        raise ConfigError("unsupported I2V reference-image plan")
    expected_sources = {"train": {"dataset": "ConsisID", "sha256": TRAIN_SHA,
                                 "release": "DanceGRPO/15cc71d/assets/consist-id.txt"},
                        "test": {"dataset": "WenhaoWang/VidProM", "sha256": EVAL_SHA,
                                 "revision": "0c0900e6a037b058b1903bc1de9bcb6037d9fc24"}}
    if plan["sources"] != expected_sources:
        raise ConfigError("I2V source provenance differs")
    for split, count in (("train", TRAIN_COUNT), ("test", TEST_COUNT)):
        rows = plan["splits"][split]
        if (not isinstance(rows, list) or len(rows) != count
                or any(set(row) != {"sample_id", "prompt", "image_seed"}
                       or type(row["image_seed"]) is not int or row["image_seed"] != 312 + i
                       or not isinstance(row["prompt"], str) or not row["prompt"].strip()
                       or not isinstance(row["sample_id"], str) or not row["sample_id"]
                       for i, row in enumerate(rows))):
            raise ConfigError("I2V split identity or seed plan differs")
    test_rows = [{"sample_id": row["sample_id"], "prompt": row["prompt"]}
                 for row in plan["splits"]["test"]]
    if hashlib.sha256(_row_bytes(plan["splits"]["train"])).hexdigest() != TRAIN_ROWS_SHA:
        raise ConfigError("I2V training selection differs")
    if hashlib.sha256(_row_bytes(test_rows)).hexdigest() != EVAL_SHA:
        raise ConfigError("I2V test selection differs")
    return plan


def _paths(output, split, index):
    root = Path(output) / split
    return (root / "images" / f"{index // 1000:04d}" / f"{index:06d}.jpg",
            root / "receipts" / f"{index // 1000:04d}" / f"{index:06d}.json")


def _image_hash(path):
    with Image.open(path) as image:
        image.load()
        if image.format != "JPEG" or image.mode != "RGB" or image.size != (640, 400):
            raise ConfigError("I2V reference must be a complete 640x400 RGB JPEG")
    return file_sha256(path)


def _receipt(output, plan, split, index):
    image, receipt = _paths(output, split, index)
    if not receipt.exists():
        return None
    row = json.loads(receipt.read_text())
    sample = plan["splits"][split][index]
    expected = {"sample_id": sample["sample_id"], "prompt": sample["prompt"],
                "image": image.relative_to(Path(output) / split).as_posix(),
                "metadata": {"image_seed": sample["image_seed"]}}
    if (set(row) != {*expected, "image_sha256"}
            or any(row.get(key) != value for key, value in expected.items())
            or row["image_sha256"] != _image_hash(image)):
        raise ConfigError("I2V image receipt, identity or bytes changed")
    return row


def _original_samples(source):
    if file_sha256(source) != ORIGINAL_SHA:
        raise ConfigError("original eval100 source digest differs")
    samples = json.loads(Path(source).read_text())["samples"]
    if len(samples) != PREFIX_COUNT:
        raise ConfigError("original eval100 images are incomplete")
    return samples


def generate_shard(output, generator, *, rank=0, world_size=1, original_bundle):
    """Resumable disjoint shards; first100 images can only be copied, never regenerated."""
    if (type(rank) is not int or type(world_size) is not int
            or world_size < 1 or not 0 <= rank < world_size):
        raise ConfigError("invalid reference generation shard")
    output, original_bundle = Path(output), Path(original_bundle)
    plan = _load_plan(output)
    source = original_bundle / "source-manifest.json"
    historical = _original_samples(source)
    original_provenance = json.loads((original_bundle / "provenance.json").read_text())
    if original_provenance["source_manifest_sha256"] != plan["original_evaluation"]["manifest_sha256"]:
        raise ConfigError("original eval100 manifest differs from the frozen history")
    original = PromptDataset.load(original_bundle / "prompts.jsonl", task="i2v",
                                  expected_sha256=original_provenance["manifest_sha256"])
    prefix_count = plan["original_evaluation"]["count"]
    if len(original.rows) != prefix_count:
        raise ConfigError("original eval100 images are incomplete")
    for row, sample, old in zip(original.rows, plan["splits"]["test"][:prefix_count], historical):
        if (row.sample_id != sample["sample_id"] or row.text != sample["prompt"]
                or row.metadata != {"image_seed": sample["image_seed"]}
                or (row.sample_id, row.text, row.image_sha256, sample["image_seed"])
                != (old["id"], old["prompt"], old["image_sha256"], old["image_seed"])):
            raise ConfigError("original eval100 prompt order or seed differs")
    # This private evidence is never part of the upload allowlist.
    _atomic(output / "original-source-manifest.json", source.read_bytes())
    processed = 0
    for split, samples in plan["splits"].items():
        for index in range(rank, len(samples), world_size):
            completed = _receipt(output, plan, split, index)
            if completed is not None:
                if (split == "test" and index < prefix_count
                        and completed["image_sha256"] != historical[index]["image_sha256"]):
                    raise ConfigError("original eval100 image bytes differ")
                continue
            sample = samples[index]
            destination, receipt = _paths(output, split, index)
            destination.parent.mkdir(parents=True, exist_ok=True)
            previous_hash = file_sha256(destination) if destination.exists() else None
            with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
                candidate = Path(temporary) / "image.jpg"
                if split == "test" and index < prefix_count:
                    shutil.copyfile(original.rows[index].image, candidate)
                    if file_sha256(candidate) != original.rows[index].image_sha256:
                        raise ConfigError("original image changed during copying")
                else:
                    image = generator(sample["prompt"], sample["image_seed"])
                    image.convert("RGB").save(candidate, format="JPEG", quality=75, subsampling=2)
                digest = _image_hash(candidate)
                if previous_hash is not None and previous_hash != digest:
                    raise ConfigError("unreceipted image differs from deterministic regeneration")
                _atomic(destination, candidate.read_bytes())
            record = {"sample_id": sample["sample_id"], "prompt": sample["prompt"],
                      "image": destination.relative_to(output / split).as_posix(), "image_sha256": digest,
                      "metadata": {"image_seed": sample["image_seed"]}}
            _atomic(receipt, (_json(record) + "\n").encode())
            processed += 1
            print(f"I2V_REFERENCE rank={rank} split={split} index={index} complete", flush=True)
    return processed


_CARD = """---
license: cc-by-nc-4.0
task_categories: [image-to-video]
configs:
  - config_name: default
    data_files:
      - split: train
        path: train/prompts.jsonl
      - split: test
        path: test/prompts.jsonl
---
# DiffusionART I2V Reference Dataset

27,815 training prompts from DanceGRPO's released ConsisID list and 1,000
held-out VidProM test prompts, paired with fixed FLUX.1-dev reference images.
The original evaluation100 is the unchanged prefix of the test split.
This is our cross-dataset evaluation protocol, not the official DanceGRPO split.
Exact normalized and token-set Jaccard>=0.8 train/test overlap is excluded;
semantic paraphrase separation is not guaranteed. Training duplicates and order
are preserved. Selection is independent of generated images and reward scores.

Training images are precomputed, unlike the online per-rollout FLUX images in
the historical paper runs. All algorithms use the same fixed references.
One-time image preparation cost is separate from subsequent training cost.

Sources: [DanceGRPO](https://github.com/XueZeyue/DanceGRPO),
[ConsisID](https://huggingface.co/datasets/BestWishYsh/ConsisID-preview-Data)
(CC-BY-4.0), [VidProM](https://huggingface.co/datasets/WenhaoWang/VidProM)
(CC-BY-NC-4.0), and
[FLUX.1-dev output terms](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md).
The dataset license does not replace upstream attribution or model output terms.

`provenance.json` records source pins, reference generation settings, counts and
manifest hashes. Each JSONL row includes relative image paths and image SHA256.
Model weights, internal machine paths, credentials and training logs are excluded.
"""


def finalize(output):
    output = Path(output)
    plan = _load_plan(output)
    source = output / "original-source-manifest.json"
    historical = _original_samples(source) if source.exists() else None
    manifests = {}
    for split, samples in plan["splits"].items():
        rows = [_receipt(output, plan, split, i) for i in range(len(samples))]
        if any(row is None for row in rows):
            raise ConfigError(f"I2V {split} image preparation is incomplete")
        if split == "test":
            if historical is None:
                raise ConfigError("original eval100 source evidence is missing")
            for row, old in zip(rows[:PREFIX_COUNT], historical):
                if (row["sample_id"], row["prompt"], row["image_sha256"], row["metadata"]["image_seed"]) != (
                    old["id"], old["prompt"], old["image_sha256"], old["image_seed"]
                ):
                    raise ConfigError("original eval100 image bytes differ")
        destination = output / split / "prompts.jsonl"
        content = _row_bytes(rows)
        if destination.exists() and destination.read_bytes() != content:
            raise ConfigError("published I2V manifest differs from preparation receipts")
        _atomic(destination, content)
        PromptDataset.load(destination, task="i2v", expected_sha256=file_sha256(destination))
        manifests[split] = {"count": len(rows), "sha256": file_sha256(destination)}
    provenance = {key: plan[key] for key in ("sources", "reference_model", "generation", "original_evaluation")}
    provenance.update({"splits": manifests, "plan_sha256": file_sha256(output / "plan.json")})
    _atomic(output / "README.md", _CARD.encode())
    _atomic(output / "provenance.json", (_json(provenance) + "\n").encode())
    return provenance


def publish(output, repo_id):
    from huggingface_hub import HfApi

    output = Path(output)
    provenance = finalize(output)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo_id):
        raise ConfigError("invalid Hub dataset name")
    expected = {"README.md", "provenance.json", "train/prompts.jsonl", "test/prompts.jsonl"}
    hashes = {name: file_sha256(output / name) for name in expected}
    for split in ("train", "test"):
        for line in (output / split / "prompts.jsonl").read_text().splitlines():
            row = json.loads(line)
            name = f"{split}/{row['image']}"
            expected.add(name)
            hashes[name] = row["image_sha256"]
    try:
        api = HfApi()
        api.create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True)
        existing = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
        if existing - expected - {".gitattributes"}:
            raise ConfigError("Hub dataset has unrelated files; use a clean destination")
        api.upload_large_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(output), num_workers=4,
                                allow_patterns=sorted(expected))
        revision = api.repo_info(repo_id=repo_id, repo_type="dataset").sha
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ConfigError("Hub publication has no immutable revision")
        files = {entry.path: entry for entry in api.list_repo_tree(
            repo_id=repo_id, repo_type="dataset", revision=revision, recursive=True
        ) if hasattr(entry, "lfs")}
        if expected != set(files) - {".gitattributes"}:
            raise ConfigError("published I2V dataset has missing or unrelated files")
        for name in expected:
            remote = files[name]
            if remote.lfs is not None:
                matches = remote.lfs.sha256 == hashes[name]
            else:
                data = (output / name).read_bytes()
                matches = remote.blob_id == hashlib.sha1(
                    f"blob {len(data)}\0".encode() + data
                ).hexdigest()
            if not matches:
                raise ConfigError("published I2V bytes differ at the selected revision")
    except Exception as exc:
        raise ConfigError(f"HF I2V publication failed ({type(exc).__name__})") from None
    result = {"repo_id": repo_id, "revision": revision, "splits": provenance["splits"]}
    _atomic(Path(output) / "publication.json", (_json(result) + "\n").encode())
    return result


def write_recipe(base_config, output, publication):
    """Bind a real published commit; no invented URLs, revisions or image hashes."""
    base_config, output = Path(base_config).resolve(), Path(output).resolve()
    if (not re.fullmatch(r"[0-9a-f]{40}", publication["revision"])
            or any(not re.fullmatch(r"[0-9a-f]{64}", publication["splits"][split]["sha256"])
                   for split in ("train", "test"))):
        raise ConfigError("published I2V dataset requires immutable revision and manifest pins")
    base = _read(base_config)
    if base["model"]["name"] != "wan_i2v_distilled":
        raise ConfigError("fixed I2V dataset recipes require the Wan backend")
    spec = base["assets"]
    snapshots = [entry for entry in spec["snapshots"] if entry["env"] != "FLUX_REFERENCE_MODEL"]
    snapshots.append({"env": "WAN_FIXED_DATASET", "repo_id": publication["repo_id"],
                      "revision": publication["revision"], "repo_type": "dataset"})
    name = base["name"] + "-fixed-eval1000"
    document = {"extends": os.path.relpath(base_config, output.parent), "name": name,
                "assets": {"snapshots": snapshots, "prompts": [],
                           "transformations": [e for e in spec["transformations"] if e["action"] != "fixed_images"]},
                "model": {"options": {"reference_mode": "fixed_images", "reference_checkpoint": None,
                                      "reference_revision": None}},
                "data": {"manifest": "${WAN_FIXED_DATASET}/train/prompts.jsonl",
                         "sha256": publication["splits"]["train"]["sha256"]},
                "evaluation": {"manifest": "${WAN_FIXED_DATASET}/test/prompts.jsonl",
                               "manifest_sha256": publication["splits"]["test"]["sha256"]},
                "runtime": {"output_dir": "${PAPER_VIDEO_OUTPUT_ROOT}/" + name},
                "provenance": {"kind": "fixed_reference_dataset_extension", "run_id": None}}
    if output.exists():
        raise ConfigError("refusing to overwrite a training recipe")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        yaml.safe_dump(document, stream, sort_keys=False)


def _flux_generator():
    import torch

    from ..models.wan_assets import isolated_rng, require_native_version

    require_native_version()
    if not torch.cuda.is_available():
        raise ConfigError("reference generation requires an authorized CUDA training node")
    from diffusers import FluxPipeline

    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    pipe = FluxPipeline.from_pretrained(MODEL["repo_id"], revision=MODEL["revision"],
                                        torch_dtype=torch.bfloat16).to(device)
    for component in pipe.components.values():
        if isinstance(component, torch.nn.Module):
            component.requires_grad_(False).eval()

    def generate(prompt, seed):
        with isolated_rng(seed, device), torch.no_grad():
            return pipe(prompt, height=400, width=640, num_inference_steps=30,
                        guidance_scale=3.5, max_sequence_length=512,
                        generator=torch.Generator(device="cpu").manual_seed(seed)).images[0]
    return generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--train-prompts", required=True)
    evaluation = plan.add_mutually_exclusive_group(required=True)
    evaluation.add_argument("--eval-prompts", help="Prepared pinned eval1000 JSONL")
    evaluation.add_argument("--eval-source", help="Pinned VidProM CSV; select the frozen UUIDs")
    plan.add_argument("--eval-assets", help="configs/paper/hunyuan-eval1000-assets.yaml")
    plan.add_argument("--output", required=True)
    original = commands.add_parser("import-original")
    original.add_argument("--source", required=True)
    original.add_argument("--image-root")
    original.add_argument("--output", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--output", required=True)
    generate.add_argument("--original-bundle", required=True)
    finish = commands.add_parser("finalize")
    finish.add_argument("--output", required=True)
    upload = commands.add_parser("publish")
    upload.add_argument("--output", required=True)
    upload.add_argument("--repo-id", required=True)
    upload.add_argument("--base-config", required=True)
    upload.add_argument("--recipe", required=True)
    recipe = commands.add_parser("recipe")
    recipe.add_argument("--publication", required=True)
    recipe.add_argument("--base-config", required=True)
    recipe.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "plan":
        if args.eval_prompts:
            result = create_plan(args.train_prompts, args.eval_prompts, args.output)
        else:
            if not args.eval_assets:
                parser.error("--eval-source requires --eval-assets")
            entry = next(e for e in _read(Path(args.eval_assets))["assets"]["prompts"]
                         if e["env"] == "HUNYUAN_EVAL_RAW_MANIFEST")
            if file_sha256(args.eval_source) != entry["source_sha256"]:
                raise ConfigError("VidProM source CSV differs from its pinned release")
            with tempfile.TemporaryDirectory() as temporary:
                manifest = Path(temporary) / "test.jsonl"
                _vidprom_manifest(Path(args.eval_source), manifest, entry["ids"])
                result = create_plan(args.train_prompts, manifest, args.output)
    elif args.command == "import-original":
        if file_sha256(args.source) != ORIGINAL_SHA:
            raise ConfigError("original eval100 source digest differs")
        result = package(args.source, args.output, image_root=args.image_root)
        _atomic(Path(args.output) / "source-manifest.json", Path(args.source).read_bytes())
    elif args.command == "generate":
        # Lazy model loading: complete shards only verify bytes, never allocate CUDA.
        generator = None

        def generate_image(prompt, seed):
            nonlocal generator
            if generator is None:
                generator = _flux_generator()
            return generator(prompt, seed)
        result = {"new_images": generate_shard(args.output, generate_image,
                  original_bundle=args.original_bundle, rank=int(os.environ.get("RANK", "0")),
                  world_size=int(os.environ.get("WORLD_SIZE", "1")))}
    elif args.command == "finalize":
        result = finalize(args.output)
    elif args.command == "recipe":
        result = json.loads(Path(args.publication).read_text())
        write_recipe(args.base_config, args.output, result)
    else:
        if Path(args.recipe).exists():
            raise ConfigError("refusing to overwrite a training recipe")
        result = publish(args.output, args.repo_id)
        write_recipe(args.base_config, args.recipe, result)
    print(_json(result), flush=True)


if __name__ == "__main__":
    main()
