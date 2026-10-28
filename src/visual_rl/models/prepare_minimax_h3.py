"""Prepare or validate the pinned H3 component snapshot before launching ranks.

This CLI never accepts a model license, changes a Hub account, or starts a model.
Download is an explicit coordinator operation; validation is entirely offline.
"""

import argparse
import fnmatch
import hashlib
import json
import os
import re
import stat
import tempfile
from pathlib import Path, PurePosixPath

from ..rewards.av_profiles import THREE_REWARD_ASSETS, validate_pickscore_provenance
from .minimax_h3 import DIFFUSERS_REVISION, MODEL_ID, MODEL_REVISION
from .minimax_h3_assets import H3_ASSET_ALLOW_PATTERNS

MANIFEST_NAME = "h3_asset_manifest.json"
COMPONENT_CLASSES = {
    "transformer": "MiniMaxH3Transformer3DModel",
    "vae": "AutoencoderKLMiniMaxH3", "audio_vae": "AutoencoderKLMiniMaxH3Audio",
    "text_encoder": "Qwen3VLForConditionalGeneration", "tokenizer": "Qwen2TokenizerFast",
    "processor": "Qwen3VLProcessor",
}
WEIGHT_COMPONENTS = ("transformer", "vae", "audio_vae", "text_encoder")
SOURCE_REVISIONS = {
    "diffusers": DIFFUSERS_REVISION,
    "verl_omni": "ca61e80c42c883e1efa161f6b609cf897fba1787",
    "diffusion_art_pr744": "93c8a747252d27e4549fba4e4b23abc19d6114bb",
}
REWARD_ASSETS = frozenset((
    "video_align", "hpsv3", "audiobox", "clap", "desync", "video_align_base", "desync_source",
))

SIX_REWARD_ASSETS = REWARD_ASSETS | {"imagebind", "imagebind_source"}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_blob_sha1(path):
    path = Path(path)
    digest = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_name(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"unsafe repository path: {name}")
    return name


def _selected(name):
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in H3_ASSET_ALLOW_PATTERNS)


def fetch_repository_metadata():
    """Read metadata only, with a bounded Hub request and immutable revision."""
    from huggingface_hub import HfApi
    info = HfApi().model_info(MODEL_ID, revision=MODEL_REVISION, files_metadata=True, timeout=30)
    if info.sha != MODEL_REVISION:
        raise ValueError("Hub resolved a different H3 commit")
    files = []
    for sibling in info.siblings:
        name = _safe_name(sibling.rfilename)
        if not _selected(name):
            continue
        lfs = sibling.lfs
        files.append({
            "path": name, "size": sibling.size, "git_blob_id": sibling.blob_id,
            "hub_lfs_sha256": None if lfs is None else lfs.sha256,
        })
    return {"model_id": MODEL_ID, "revision": info.sha, "files": files}


def _read_small_json(path):
    if Path(path).stat().st_size > 32 * 1024 * 1024:
        raise ValueError(f"unexpectedly large configuration: {path}")
    return json.loads(Path(path).read_text())


def build_manifest(root, metadata, *, hash_weights=False, require_weights=True):
    """Check actual component bytes against pinned Hub metadata.

    Without --hash-weights this checks weight sizes, not weight byte integrity;
    the distinction is persisted so an audit cannot mistake the two levels.
    """
    root = Path(root).resolve(strict=True)
    if metadata.get("model_id") != MODEL_ID or metadata.get("revision") != MODEL_REVISION:
        raise ValueError("asset metadata must identify the fixed H3 repository and commit")
    index = _read_small_json(root / "model_index.json")
    expected = {}
    for component, cls in COMPONENT_CLASSES.items():
        entry = index.get(component)
        if not isinstance(entry, list) or len(entry) < 2 or entry[1] != cls:
            raise ValueError(f"unsupported component class: {component}")
        source = entry[2] if len(entry) > 2 else {}
        if (source.get("pretrained_model_name_or_path") not in (None, MODEL_ID)
                or source.get("revision") not in (None, MODEL_REVISION)
                or source.get("subfolder", component) != component):
            raise ValueError(f"component resolves to foreign assets: {component}")
        expected[component] = {"class": cls, "subfolder": component,
                               "model_id": MODEL_ID, "revision": MODEL_REVISION,
                               "config_sha256": {}, "weight_files": []}
    files, missing = [], []
    seen = set()
    for item in metadata["files"]:
        name = _safe_name(item["path"])
        if name in seen or not _selected(name):
            raise ValueError(f"duplicate or non-T2AV metadata entry: {name}")
        seen.add(name)
        file = root / name
        record = dict(item)
        is_weight = name.endswith(".safetensors")
        if not file.is_file():
            if require_weights or not is_weight:
                raise FileNotFoundError(f"required H3 asset missing: {file}")
            missing.append(name)
            record["present"] = False
        else:
            size = file.stat().st_size
            if type(item.get("size")) is not int or item["size"] != size:
                raise ValueError(f"Hub/local file size mismatch: {name}")
            record["present"] = True
            if not is_weight or hash_weights:
                record["sha256"] = _sha256(file)
                lfs_sha = item.get("hub_lfs_sha256")
                if is_weight or lfs_sha is not None:
                    if not isinstance(lfs_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", lfs_sha):
                        raise ValueError(f"authoritative Hub LFS SHA256 missing: {name}")
                    if lfs_sha != record["sha256"]:
                        raise ValueError(f"asset SHA256 mismatch: {name}")
                else:
                    blob = item.get("git_blob_id")
                    if (not isinstance(blob, str) or not re.fullmatch(r"[0-9a-f]{40}", blob)
                            or _git_blob_sha1(file) != blob):
                        raise ValueError(f"Hub Git blob identity mismatch: {name}")
        files.append(record)
        component = name.split("/", 1)[0]
        if component in expected:
            if is_weight:
                expected[component]["weight_files"].append(name)
            elif name.endswith(".json"):
                expected[component]["config_sha256"][name] = record.get("sha256")
                if ".safetensors.index.json" in name and record.get("present"):
                    index_data = _read_small_json(file)
                    for shard in set(index_data.get("weight_map", {}).values()):
                        shard_name = _safe_name(f"{component}/{shard}")
                        if not any(entry["path"] == shard_name for entry in metadata["files"]):
                            raise ValueError(f"weight index shard missing from Hub manifest: {shard_name}")
    if "model_index.json" not in seen:
        raise ValueError("Hub metadata is missing model_index.json")
    for component in COMPONENT_CLASSES:
        if not expected[component]["config_sha256"]:
            raise ValueError(f"component configuration identity missing: {component}")
    for component in WEIGHT_COMPONENTS:
        if not expected[component]["weight_files"]:
            raise ValueError(f"no native safetensors weights listed for {component}")
    return {
        "schema": 1, "model_id": MODEL_ID, "revision": MODEL_REVISION,
        "diffusers_revision": DIFFUSERS_REVISION, "source_revisions": SOURCE_REVISIONS,
        "root": str(root), "components": expected, "files": files,
        "ready": not missing, "missing_weights": missing,
        "weight_verification": "sha256" if hash_weights else "size_only",
        "gpu_validated": False,
    }


def write_manifest(path, manifest):
    """Publish one complete JSON receipt atomically, without touching model files."""
    path = Path(path)
    descriptor, stage = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(stage, path)
    finally:
        if os.path.exists(stage):
            os.unlink(stage)


def validate_reward_provenance(source, *, name):
    required = {"repo", "revision", "license"}
    if (not isinstance(source, dict) or not required <= set(source)
            or any(not isinstance(source[key], str) or not source[key].strip() for key in required)
            or not re.fullmatch(r"[0-9a-f]{40}", source["revision"])):
        raise ValueError(f"reward asset needs repository, immutable revision and license: {name}")
    for key, value in source.items():
        if isinstance(value, dict) and "repo" in value:
            validate_reward_provenance(value, name=f"{name}/{key}")


def build_reward_manifest(paths, provenance, *, base_dir="."):
    """Hash explicitly supplied local native scorer assets, without loading them.

    This is a complete recursive ordinary-file inventory. Symlinked files and
    directories are rejected instead of following mutable external locations.
    """
    if set(paths) not in (REWARD_ASSETS, SIX_REWARD_ASSETS, THREE_REWARD_ASSETS) or set(provenance) != set(paths):
        raise ValueError("reward paths/provenance require seven legacy, nine six-reward or five three-reward assets")
    if set(paths) == THREE_REWARD_ASSETS:
        validate_pickscore_provenance(provenance)
    if "imagebind" in paths:
        from ..rewards._native_av.imagebind import (
            IMAGEBIND_LICENSE,
            IMAGEBIND_REPOSITORY,
            IMAGEBIND_REVISION,
            validate_source,
        )
        source = provenance["imagebind_source"]
        validate_reward_provenance(source, name="imagebind_source")
        if (source["repo"] != IMAGEBIND_REPOSITORY or source["revision"] != IMAGEBIND_REVISION
                or source["license"] != IMAGEBIND_LICENSE):
            raise ValueError("ImageBind source provenance must identify the fixed upstream revision/license")
    resolved, hashes = {}, {}
    for name in sorted(paths):
        source = provenance[name]
        validate_reward_provenance(source, name=name)
        if not isinstance(paths[name], str) or not paths[name]:
            raise ValueError(f"reward asset path must be a nonempty string: {name}")
        path = Path(paths[name]).expanduser()
        if not path.is_absolute():
            path = Path(base_dir) / path
        # Reject symlinks anywhere in the spelling, including parent aliases.
        absolute = Path(os.path.abspath(path))
        if any(p.is_symlink() for p in (absolute, *absolute.parents)):
            raise ValueError(f"reward asset may not use symlinks: {name}")
        path = absolute.resolve(strict=True)
        inventory = {}
        if path.is_file():
            inventory["."] = _sha256(path)
        elif path.is_dir():
            for folder, directories, filenames in os.walk(path, followlinks=False):
                for child in (*directories, *filenames):
                    item = Path(folder) / child
                    if item.is_symlink():
                        raise ValueError(f"reward inventory contains symlink: {item}")
                for filename in sorted(filenames):
                    item = Path(folder) / filename
                    if not stat.S_ISREG(item.stat().st_mode):
                        raise ValueError(f"reward inventory needs ordinary files: {item}")
                    inventory[item.relative_to(path).as_posix()] = _sha256(item)
        else:
            raise ValueError(f"reward asset must be a regular file or directory: {name}")
        if not inventory:
            raise ValueError(f"reward asset directory is empty: {name}")
        if name == "imagebind_source":
            validate_source(path)
        if name == "imagebind" and not path.is_file():
            raise ValueError("ImageBind checkpoint must be an explicit local file")
        resolved[name] = str(path)
        hashes[name] = dict(sorted(inventory.items()))
    return {"schema_version": 1, "paths": resolved, "sha256": hashes,
            "provenance": provenance}


def validate_assets(root, *, manifest_path=None, hash_weights=False):
    """Offline validation: this function performs no Hub/API/download calls."""
    root = Path(root).resolve(strict=True)
    old = _read_small_json(manifest_path or root / MANIFEST_NAME)
    if old.get("schema") != 1 or not old.get("ready"):
        raise ValueError("H3 manifest is incomplete or has an unknown schema")
    if old.get("diffusers_revision") != DIFFUSERS_REVISION:
        raise ValueError("H3 manifest has a different Diffusers source revision")
    # Preserve stronger verification when the original receipt included hashes.
    verify_hashes = hash_weights or old.get("weight_verification") == "sha256"
    actual = build_manifest(root, old, hash_weights=verify_hashes, require_weights=True)
    old_files = {entry["path"]: entry for entry in old["files"]}
    for entry in actual["files"]:
        old_hash = old_files[entry["path"]].get("sha256")
        if old_hash is not None and old_hash != entry.get("sha256"):
            raise ValueError(f"prepared asset bytes changed: {entry['path']}")
    if actual["components"] != old["components"]:
        raise ValueError("prepared component identities changed")
    return actual


def prepare_assets(cache_dir, *, hash_weights=False, inspect_only=False, manifest_path=None):
    from huggingface_hub import snapshot_download
    metadata = fetch_repository_metadata()
    patterns = list(H3_ASSET_ALLOW_PATTERNS)
    if inspect_only:
        patterns = [pattern for pattern in patterns if "safetensors" not in pattern]
    root = Path(snapshot_download(
        repo_id=MODEL_ID, revision=MODEL_REVISION, cache_dir=cache_dir,
        allow_patterns=patterns, max_workers=4,
    ))
    receipt = build_manifest(root, metadata, hash_weights=hash_weights,
                             require_weights=not inspect_only)
    path = Path(manifest_path) if manifest_path else root / (
        "h3_asset_manifest.inspect.json" if inspect_only else MANIFEST_NAME)
    write_manifest(path, receipt)
    return receipt, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--download", action="store_true", help="download required native components")
    action.add_argument("--inspect", action="store_true", help="metadata/configs only; NOT training-ready")
    action.add_argument("--validate-only", action="store_true", help="offline local manifest validation")
    action.add_argument("--reward-paths", help="JSON mapping the complete legacy, six-reward or three-reward assets to paths")
    parser.add_argument("--cache-dir")
    parser.add_argument("--model-root")
    parser.add_argument("--manifest")
    parser.add_argument("--hash-weights", action="store_true", help="read/hash all weight bytes; can take minutes")
    parser.add_argument("--reward-provenance", help="JSON file with repo/revision/license for each reward asset")
    args = parser.parse_args()
    if args.reward_paths:
        if not args.reward_provenance or not args.manifest:
            parser.error("--reward-paths requires --reward-provenance and --manifest")
        paths_file = Path(args.reward_paths)
        receipt = build_reward_manifest(_read_small_json(paths_file),
                                        _read_small_json(args.reward_provenance),
                                        base_dir=paths_file.parent)
        write_manifest(args.manifest, receipt)
        print(json.dumps({"manifest": str(Path(args.manifest).resolve()),
                          "assets": len(receipt["paths"]),
                          "files": sum(len(v) for v in receipt["sha256"].values())}, sort_keys=True))
        return
    if args.validate_only:
        if not args.model_root:
            parser.error("--validate-only requires --model-root")
        receipt = validate_assets(args.model_root, manifest_path=args.manifest,
                                   hash_weights=args.hash_weights)
        path = Path(args.manifest) if args.manifest else Path(args.model_root) / MANIFEST_NAME
    else:
        if not args.cache_dir:
            parser.error("online preparation requires an explicit --cache-dir")
        receipt, path = prepare_assets(args.cache_dir, hash_weights=args.hash_weights,
                                       inspect_only=args.inspect, manifest_path=args.manifest)
    # Deliberately bounded output: full per-file records live in the manifest.
    print(json.dumps({"root": receipt["root"], "manifest": str(path), "ready": receipt["ready"],
                      "revision": receipt["revision"], "files": len(receipt["files"]),
                      "weight_verification": receipt["weight_verification"]}, sort_keys=True))


if __name__ == "__main__":
    main()
