"""Prepare declared, pinned assets before starting distributed GPU workers."""

import csv
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import unicodedata
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

from .config import ConfigError, _read
from .prepare_data import prepare

_ENV = re.compile(r"[A-Z][A-Z0-9_]*\Z")
_MAX_SOURCE_BYTES = 16 * 1024 * 1024
_TRANSFORMS = {
    "hunyuan_text_view": {"model_env"},
    "hunyuan_conditioning": {"sha_env", "source_env", "source_sha_env", "model_env", "native_env", "device"},
    "wan_transformer": {"source_env", "source_file", "config_env", "config_file", "source_sha256", "revision"},
    "fixed_images": {"sha_env", "source_env", "format"},
}


def _relative_file(value):
    return (isinstance(value, str) and bool(value) and not Path(value).is_absolute()
            and ".." not in Path(value).parts and value != ".")


def validate_assets(spec):
    if not isinstance(spec, dict) or set(spec) - {"defaults", "snapshots", "prompts", "transformations"}:
        raise ConfigError("assets supports defaults, snapshots, prompts and transformations")
    defaults = spec.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ConfigError("assets.defaults must be a mapping")
    names = set()
    for name, value in defaults.items():
        if (not isinstance(name, str) or not _ENV.fullmatch(name) or not isinstance(value, str) or not value
                or any(secret in name for secret in ("TOKEN", "API_KEY", "PASSWORD"))):
            raise ConfigError("asset defaults require nonempty environment values")
        names.add(name)
    schemas = {
        "snapshots": {"env", "repo_id", "revision"},
        "prompts": {"env", "sha_env", "url", "source_sha256", "sha256", "prefix", "format"},
    }
    optional = {"snapshots": {"allow_patterns", "for_env", "repo_type", "manifest_file"},
                "prompts": {"ids", "for_env"}}
    for kind, fields in schemas.items():
        entries = spec.get(kind, [])
        if not isinstance(entries, list):
            raise ConfigError(f"assets.{kind} must be a list")
        for entry in entries:
            if (not isinstance(entry, dict) or not fields <= set(entry)
                    or set(entry) - fields - optional[kind]
                    or any(not isinstance(entry[key], str) or not entry[key] for key in fields)):
                raise ConfigError(f"invalid assets.{kind} entry")
            variables = [entry["env"]] + ([entry["sha_env"]] if kind == "prompts" else [])
            for name in variables:
                if not _ENV.fullmatch(name) or name in names or any(
                    secret in name for secret in ("TOKEN", "API_KEY", "PASSWORD")
                ):
                    raise ConfigError("asset bindings require unique, non-secret environment names")
                names.add(name)
            if kind == "snapshots":
                if (not isinstance(entry.get("repo_type", "model"), str)
                        or entry.get("repo_type", "model") not in {"model", "dataset"}):
                    raise ConfigError("asset snapshot repo_type must be model or dataset")
                if "manifest_file" in entry and (
                    entry.get("repo_type") != "dataset" or not _relative_file(entry["manifest_file"])
                ):
                    raise ConfigError("dataset manifest_file must be a relative file path")
                if not re.fullmatch(r"[0-9a-f]{40}", entry["revision"]):
                    raise ConfigError("asset snapshot revision must be a full commit SHA")
                if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", entry["repo_id"]):
                    raise ConfigError("asset snapshot requires a Hub repository ID")
                if "allow_patterns" in entry and (
                    not isinstance(entry["allow_patterns"], list) or not entry["allow_patterns"]
                    or any(not _relative_file(p) for p in entry["allow_patterns"])
                ):
                    raise ConfigError("snapshot allow_patterns must be relative file patterns")
            else:
                url = urlsplit(entry["url"])
                if (url.scheme != "https" or not url.hostname or url.username or url.password
                        or url.query or url.fragment):
                    raise ConfigError("prompt sources must be credential-free HTTPS URLs")
                if any(not re.fullmatch(r"[0-9a-f]{64}", entry[key])
                       for key in ("sha256", "source_sha256")):
                    raise ConfigError("prompt sources and manifests require SHA256 pins")
                if (entry["format"] not in {"text", "geneval", "vidprom_ids"}
                        or not re.fullmatch(r"[A-Za-z0-9_.-]+", entry["prefix"])):
                    raise ConfigError("invalid prompt conversion format or prefix")
                if entry["format"] == "vidprom_ids":
                    ids = entry.get("ids")
                    if (not isinstance(ids, list) or not ids or len(ids) > 10000
                            or any(not isinstance(i, str) or not re.fullmatch(r"[0-9a-f-]{36}", i) for i in ids)
                            or len(set(ids)) != len(ids)):
                        raise ConfigError("vidprom_ids requires unique ordered source UUIDs")
                elif "ids" in entry:
                    raise ConfigError("ids is only supported for vidprom_ids")
            if "for_env" in entry and (not isinstance(entry["for_env"], str)
                                       or not _ENV.fullmatch(entry["for_env"])):
                raise ConfigError("for_env must name a transformation output")
    transformations = spec.get("transformations", [])
    if not isinstance(transformations, list):
        raise ConfigError("assets.transformations must be a list")
    for entry in transformations:
        if (not isinstance(entry, dict) or not isinstance(entry.get("action"), str)
                or entry["action"] not in _TRANSFORMS
                or set(entry) != {"env", "action"} | _TRANSFORMS[entry["action"]]
                or any(not isinstance(v, str) or not v for v in entry.values())):
            raise ConfigError("invalid or unsupported asset transformation")
        for key, value in entry.items():
            if key == "env" or key.endswith("_env"):
                if not _ENV.fullmatch(value) or any(s in value for s in ("TOKEN", "API_KEY", "PASSWORD")):
                    raise ConfigError("transformations require non-secret environment names")
                if key in {"env", "sha_env"}:
                    if value in names:
                        raise ConfigError("asset bindings require unique environment names")
                    names.add(value)
        if entry["action"] == "hunyuan_conditioning" and entry["device"] not in {"cuda", "cpu"}:
            raise ConfigError("conditioning device must be cuda or cpu")
        if entry["action"] == "fixed_images" and entry["format"] not in {"jsonl", "wan_validation"}:
            raise ConfigError("fixed_images requires jsonl or wan_validation")
        if entry["action"] == "wan_transformer" and (
            not re.fullmatch(r"[0-9a-f]{40}", entry["revision"])
            or not re.fullmatch(r"[0-9a-f]{64}", entry["source_sha256"])
            or not all(_relative_file(entry[k]) for k in ("source_file", "config_file"))
        ):
            raise ConfigError("Wan transformation requires pinned source and relative files")
    outputs = {e["env"] for e in transformations}
    for kind in schemas:
        for entry in spec.get(kind, []):
            if "for_env" in entry and entry["for_env"] not in outputs:
                raise ConfigError("for_env must name a transformation output")
    available = set(defaults) | {e["env"] for k in schemas for e in spec.get(k, [])}
    for entry in transformations:
        references = {v for k, v in entry.items() if k.endswith("_env") and k != "sha_env"}
        if references & (outputs - available):
            raise ConfigError("transformations must be in dependency order, without cycles")
        available.add(entry["env"])


def plan_environment(path):
    """Plan asset bindings without downloads or preprocessing."""
    document = _read(Path(path))
    spec = document.get("assets", {})
    validate_assets(spec)
    root = Path(os.environ.get("VISUAL_RL_ASSET_ROOT", "~/.cache/visual-rl")).expanduser().absolute()
    supplied = set(os.environ)
    for name, value in spec.get("defaults", {}).items():
        os.environ.setdefault(name, value)
    snapshots, prompts = [], []
    for entry in spec.get("snapshots", []):
        if (document.get("model", {}).get("options", {}).get("reference_mode") == "fixed_images"
                and entry["env"] == "FLUX_REFERENCE_MODEL"):
            continue
        if entry.get("for_env") in supplied:
            continue
        name = entry["env"]
        if name not in os.environ:
            category = "datasets" if entry.get("repo_type") == "dataset" else "models"
            destination = root / category / name.lower() / entry["revision"]
            os.environ[name] = str(destination / entry["manifest_file"]
                                   if "manifest_file" in entry else destination)
            snapshots.append((entry, destination))
    for entry in spec.get("prompts", []):
        if entry.get("for_env") in supplied:
            continue
        name, sha_name = entry["env"], entry["sha_env"]
        if name in os.environ:
            if sha_name not in os.environ:
                prompts.append(({"action": "manifest_binding", **entry}, Path(os.environ[name])))
            continue
        if sha_name in os.environ and os.environ[sha_name] != entry["sha256"]:
            raise ConfigError(f"{sha_name} differs from the automatic dataset")
        destination = root / "prompts" / entry["sha256"] / "prompts.jsonl"
        os.environ[name] = str(destination)
        os.environ[sha_name] = entry["sha256"]
        prompts.append((entry, destination))
    transformations = spec.get("transformations", [])
    for entry in transformations:
        name = entry["env"]
        if name in supplied:
            if "sha_env" in entry and entry["sha_env"] not in os.environ:
                prompts.append(({"action": "manifest_binding", "env": name,
                                 "sha_env": entry["sha_env"]}, Path(os.environ[name])))
            continue
        if entry["action"] == "hunyuan_text_view":
            consumers = [e for e in transformations
                         if e["action"] == "hunyuan_conditioning" and e["model_env"] == name]
            if consumers and all(e["env"] in supplied for e in consumers):
                continue
        inputs = {key: os.environ.get(value) for key, value in entry.items()
                  if key.endswith("_env") and key != "sha_env"}
        identity = hashlib.sha256(json.dumps({"recipe": entry, "inputs": inputs},
                                             sort_keys=True).encode()).hexdigest()
        directory = root / "prepared" / entry["action"] / identity
        destination = directory / "prompts.jsonl" if "sha_env" in entry else directory
        os.environ[name] = str(destination)
        prompts.append((entry, destination))
    evaluation = document.get("evaluation", {})
    if evaluation.get("every_updates") == 0 and not evaluation.get("at_updates", []):
        # A training-only check keeps the planned evaluation binding dormant.
        # Missing historical eval images must not prevent online-reference I2V
        # training; re-enabling cadence restores the required fixed-image gate.
        deferred = {entry["env"] for entry in transformations
                    if entry["action"] == "fixed_images"
                    and evaluation.get("manifest") == "${" + entry["env"] + "}"}
        if deferred:
            prompts = [(entry, destination) for entry, destination in prompts
                       if entry.get("env") not in deferred]
            print("Evaluation paused; deferring fixed eval assets: " + ", ".join(sorted(deferred)),
                  file=sys.stderr, flush=True)
    return snapshots, prompts


def environment_bindings(spec):
    """Deterministic, credential-free bindings for the parent launcher's shared record."""
    validate_assets(spec)
    names = set(spec.get("defaults", {}))
    for kind in ("snapshots", "prompts", "transformations"):
        for entry in spec.get(kind, []):
            names.update(value for key, value in entry.items() if key == "env" or key.endswith("_env"))
    return {name: os.environ[name] for name in sorted(names) if name in os.environ}


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_bindings(config, plan):
    """A local download path must not hide a conflicting configured revision."""
    revisions = {str(destination): entry["revision"] for entry, destination in plan[0]}
    revisions.update({str(destination): entry["revision"] for entry, destination in plan[1]
                      if entry.get("action") == "wan_transformer"})
    components = [config.model, config.reward]
    if config.reward.name == "composite":
        components.extend(SimpleNamespace(**spec)
                          for spec in config.reward.options["components"].values())
    for component in components:
        pairs = [(component.checkpoint, component.revision)]
        for path_key, revision_key in (("processor", "processor_revision"),
                                       ("base_checkpoint", "base_revision"),
                                       ("reference_checkpoint", "reference_revision")):
            pairs.append((component.options.get(path_key), component.options.get(revision_key)))
        for path, revision in pairs:
            if not isinstance(path, str):
                continue
            for destination, pin in revisions.items():
                if (Path(path).is_relative_to(destination) and revision != pin):
                    raise ConfigError("configured component revision differs from its automatic asset pin")


def _snapshot_options(entry):
    if "allow_patterns" in entry:
        return {"allow_patterns": entry["allow_patterns"]}
    # CLIPProcessor/PIL preprocessing needs tokenizer/config bytes here.
    # PickScore's CLIPModel loads its separate pinned model snapshot.
    if (entry.get("env"), entry.get("repo_id"), entry.get("revision")) == (
        "PICKSCORE_PROCESSOR", "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        "1c2b8495b28150b8a4922ee1c8edee224c284c0c",
    ):
        return {"allow_patterns": ["*.json", "*.txt", "*.model", "README*", "LICENSE*", ".gitattributes"]}
    return {}


def prepare_assets(plan):
    snapshots, prompts = plan
    # Reject missing inputs before downloading large weights.
    for entry, _ in prompts:
        if entry.get("action") == "fixed_images" and entry["source_env"] not in os.environ:
            raise ConfigError(f"no public fixed Wan evaluation images are declared; bind {entry['env']} "
                              f"or {entry['source_env']} to an existing fixed-image manifest; "
                              "never substitute newly generated FLUX images")
    planned_paths = {str(path) for _, path in [*snapshots, *prompts]}
    planned_paths.update(str(path / entry["manifest_file"]) for entry, path in snapshots
                         if "manifest_file" in entry)
    for entry, destination in prompts:
        if entry.get("action") == "manifest_binding":
            if not destination.is_file():
                raise ConfigError(f"custom manifest does not exist: {destination}")
        elif "action" in entry:
            for key in entry:
                if key.endswith("_env") and key not in {"sha_env", "source_sha_env"}:
                    path = _required_path(entry, key)
                    if str(path) not in planned_paths and not path.exists():
                        raise ConfigError(f"{entry['action']} input does not exist: {path}")
                    if (key == "source_env" and entry["action"] in {"fixed_images", "hunyuan_conditioning"}
                            and str(path) not in planned_paths and not path.is_file()):
                        raise ConfigError(f"{entry['action']} requires a raw manifest file: {path}")
    if snapshots:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ConfigError("asset downloads require the diffusers extra") from exc
        for entry, destination in snapshots:
            print(f"Preparing {entry['repo_id']}@{entry['revision']}", file=sys.stderr, flush=True)
            try:
                snapshot_download(repo_id=entry["repo_id"], revision=entry["revision"],
                                  local_dir=str(destination), max_workers=4,
                                  **({"repo_type": entry["repo_type"]}
                                     if "repo_type" in entry else {}),
                                  **_snapshot_options(entry))
            except Exception as exc:
                # HTTP exception text can expose signed URLs or credentials.
                raise ConfigError(f"download failed for {entry['repo_id']} ({type(exc).__name__}); "
                                  "check HF_TOKEN/login, model license access and cache space") from None
            if "manifest_file" in entry and not (destination / entry["manifest_file"]).is_file():
                raise ConfigError(f"dataset snapshot lacks its manifest: {entry['manifest_file']}")
    for entry, destination in prompts:
        if "action" in entry:
            _prepare_transformation(entry, destination)
            continue
        if destination.exists():
            if _hash(destination) != entry["sha256"]:
                raise ConfigError(f"prepared prompt manifest is corrupted: {destination}")
            continue
        print(f"Preparing {entry['env']}", file=sys.stderr, flush=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".prepare-", dir=destination.parent) as temporary:
            source = Path(temporary) / "source"
            try:
                with urllib.request.urlopen(entry["url"], timeout=30) as response, source.open("wb") as out:
                    total = 0
                    while block := response.read(1024 * 1024):
                        total += len(block)
                        if total > _MAX_SOURCE_BYTES:
                            raise ConfigError("prompt download exceeds 16 MiB")
                        out.write(block)
            except OSError as exc:
                raise ConfigError(f"prompt download failed ({type(exc).__name__})") from None
            if _hash(source) != entry["source_sha256"]:
                raise ConfigError("downloaded prompt source SHA256 differs")
            staged = Path(temporary) / "prompts.jsonl"
            if entry["format"] == "vidprom_ids":
                _vidprom_manifest(source, staged, entry["ids"])
                record = {"sha256": _hash(staged)}
            else:
                record = prepare(source, staged, format=entry["format"], prefix=entry["prefix"])
            if record["sha256"] != entry["sha256"]:
                raise ConfigError("converted prompt manifest SHA256 differs")
            try:
                os.link(staged, destination)
            except FileExistsError:
                if _hash(destination) != entry["sha256"]:
                    raise ConfigError("concurrent prompt preparation published a different manifest")


def _vidprom_manifest(source, output, ids):
    """Extract the already-frozen UUID order, never rerun data selection."""
    selected = {}
    with source.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("uuid") in ids:
                if row["uuid"] in selected or not row.get("prompt", "").strip():
                    raise ConfigError("duplicate or empty selected VidProM source row")
                text = " ".join(row["prompt"].split())
                normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
                selected[row["uuid"]] = {
                    "sample_id": hashlib.sha256(normalized.encode()).hexdigest(), "prompt": text,
                }
    if set(selected) != set(ids):
        raise ConfigError("pinned VidProM source lacks frozen evaluation UUIDs")
    _write_jsonl(output, [selected[i] for i in ids])


def _write_jsonl(output, rows):
    with output.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=True, sort_keys=True, allow_nan=False) + "\n")


def _required_path(entry, key):
    name = entry[key]
    if name not in os.environ:
        raise ConfigError(f"{entry['action']} requires {name}")
    return Path(os.environ[name]).expanduser().absolute()


def _bind_hash(entry, path):
    digest = _hash(path)
    name = entry["sha_env"]
    if name in os.environ and os.environ[name] != digest:
        raise ConfigError(f"{name} differs from the prepared/custom manifest: {path}")
    os.environ[name] = digest
    return digest


def _check_files(directory, files, *, hashes=True):
    if not isinstance(files, list) or not files:
        raise ConfigError(f"missing preparation inventory: {directory}")
    seen = set()
    for index, row in enumerate(files):
        if index % 100 == 0:
            print(f"Validating prepared files {index}/{len(files)}: {directory}", file=sys.stderr, flush=True)
        if (not isinstance(row, dict) or not _relative_file(row.get("path"))
                or row["path"] in seen):
            raise ConfigError(f"invalid preparation inventory: {directory}")
        seen.add(row["path"])
        path = directory / row["path"]
        if (path.stat().st_size != row["size"] or (hashes and _hash(path) != row["sha256"])):
            raise ConfigError(f"prepared asset is corrupted: {path}")


def _check_text_view(directory, model):
    receipt = json.loads((directory / "asset-view.json").read_text())
    if receipt["source"] != str(model.resolve(strict=True)) or receipt["weights_copied"] is not False:
        raise ConfigError("native text view source differs")
    for encoder, files in receipt["files"].items():
        if encoder not in {"text_encoder", "text_encoder_2"}:
            raise ConfigError("invalid native text view inventory")
        for name, target in files.items():
            link = directory / encoder / name
            if not _relative_file(name) or not link.is_symlink() or link.resolve(strict=True) != Path(target):
                raise ConfigError(f"native text view link differs: {link}")


def _fixed_images(source, stage, format, *, image_root=None):
    """Copy fixed bytes and preserve prompt IDs/order; no image synthesis or recoding."""
    from .data import PromptDataset

    if source.stat().st_size > _MAX_SOURCE_BYTES:
        raise ConfigError("fixed-image manifest exceeds 16 MiB")
    if format == "wan_validation":
        payload = json.loads(source.read_text())
        checksum = hashlib.sha256(json.dumps(
            {k: v for k, v in payload.items() if k != "checksum"}, sort_keys=True,
            separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()).hexdigest()
        if (payload.get("schema_version") != "wan-i2v-formal-validation/v1"
                or payload.get("checksum") != checksum):
            raise ConfigError("Wan validation provenance checksum differs")
        rows = [{"sample_id": row["id"], "prompt": row["prompt"],
                 "image": str((Path(image_root) / Path(row["image"]).name
                               if image_root is not None else source.parent / row["image"]).resolve()),
                 "image_sha256": row["image_sha256"]} for row in payload["samples"]]
        converted = stage / "raw.jsonl"
        _write_jsonl(converted, rows)
        dataset = PromptDataset.load(converted, expected_sha256=_hash(converted), task="i2v")
        converted.unlink()
    else:
        if image_root is not None:
            raise ConfigError("image_root relocation requires a checksummed Wan validation manifest")
        dataset = PromptDataset.load(source, expected_sha256=_hash(source), task="i2v")
    images = stage / "images"
    images.mkdir()
    records = []
    for index, row in enumerate(dataset.rows):
        target = images / f"{index:06d}{row.image.suffix.lower()}"
        shutil.copyfile(row.image, target)
        # Source could change during copy; validate against the loaded dataset, not fresh source bytes.
        expected = row.image_sha256
        if _hash(target) != expected:
            raise ConfigError("fixed reference image changed during copying")
        record = {"sample_id": row.sample_id, "prompt": row.text,
                  "image": f"images/{target.name}", "image_sha256": expected}
        if row.metadata is not None:
            record["metadata"] = row.metadata
        records.append(record)
    _write_jsonl(stage / "prompts.jsonl", records)
    PromptDataset.load(stage / "prompts.jsonl", expected_sha256=_hash(stage / "prompts.jsonl"), task="i2v")


def _progress_rows(rows):
    for index, row in enumerate(rows):
        if index % 100 == 0:
            print(f"Hunyuan conditioning {index}/{len(rows)}", file=sys.stderr, flush=True)
        yield row
    print(f"Hunyuan conditioning {len(rows)}/{len(rows)}", file=sys.stderr, flush=True)


def _prepare_transformation(entry, destination):
    action = entry["action"]
    if action == "manifest_binding":
        _bind_hash(entry, destination)
        return
    directory = destination.parent if "sha_env" in entry else destination
    print(f"Preparing {action}: {destination}", file=sys.stderr, flush=True)
    directory.parent.mkdir(parents=True, exist_ok=True)
    try:
        if action == "hunyuan_text_view":
            from .models.prepare_hunyuan import prepare_text_encoder_view

            model = _required_path(entry, "model_env")
            if not os.path.lexists(directory):
                prepare_text_encoder_view(model, directory)
            _check_text_view(directory, model)
            return
        if action == "wan_transformer":
            from .models.convert_wan import DIFFUSERS_VERSION, convert_checkpoint

            source = _required_path(entry, "source_env") / entry["source_file"]
            config = _required_path(entry, "config_env") / entry["config_file"]
            if not os.path.lexists(directory):
                convert_checkpoint(source, config, directory,
                                   expected_source_sha256=entry["source_sha256"])
            receipt = json.loads((directory / "conversion-receipt.json").read_text())
            if (receipt.get("schema_version") != "wan-checkpoint-conversion/v1"
                    or receipt["source"]["sha256"] != entry["source_sha256"]
                    or receipt["config"]["sha256"] != _hash(config)
                    or receipt["config"]["sha256"] != _hash(directory / "config.json")
                    or receipt.get("converter", {}).get("diffusers") != DIFFUSERS_VERSION
                    or receipt["value_transform"] != "none" or receipt["dtype_transform"] != "none"):
                raise ConfigError("Wan conversion receipt differs from pinned source/config")
            # Initial conversion verifies source/tensor bytes. Cached model startup
            # checks inventory/size, not another multi-GB weight scan.
            _check_files(directory, receipt["files"], hashes=False)
            return
        source = _required_path(entry, "source_env")
        source_digest = _hash(source)
        identity = {"recipe": entry, "source": str(source), "source_sha256": source_digest}
        if action == "hunyuan_conditioning":
            from .data import PromptDataset
            from .models.hunyuan_assets import NATIVE_REVISION, prepare_conditioning

            sha_name = entry["source_sha_env"]
            if sha_name in os.environ and os.environ[sha_name] != source_digest:
                raise ConfigError(f"{sha_name} differs from raw conditioning input")
            os.environ[sha_name] = source_digest
            model, native = _required_path(entry, "model_env"), _required_path(entry, "native_env")
            identity.update(model=str(model.resolve(strict=True)), native=str(native), native_revision=NATIVE_REVISION)
        receipt_path = directory / "preparation-receipt.json"
        if os.path.lexists(directory):
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("schema_version") != "video-assets/v1" or receipt["identity"] != identity:
                raise ConfigError(f"prepared asset inputs differ: {directory}; select a new cache/raw path")
            _check_files(directory, receipt["files"])
            _bind_hash(entry, destination)
            return
        stage = Path(tempfile.mkdtemp(prefix=f".{directory.name}.staging-", dir=directory.parent))
        try:
            if action == "hunyuan_conditioning":
                dataset = PromptDataset.load(source, expected_sha256=source_digest, task="t2v")
                # Existing preparer creates its own output directory exclusively.
                stage.rmdir()
                prepare_conditioning(SimpleNamespace(rows=_progress_rows(dataset.rows)),
                                     source_manifest=source, model_root=model,
                                     native_root=native, output=stage, device=entry["device"])
            else:
                _fixed_images(source, stage, entry["format"])
            if _hash(source) != source_digest:
                raise ConfigError("raw manifest changed during preparation")
            files = [{"path": str(file.relative_to(stage)), "size": file.stat().st_size, "sha256": _hash(file)}
                     for file in sorted(stage.rglob("*")) if file.is_file()]
            receipt = {"schema_version": "video-assets/v1", "identity": identity,
                       "output": str(destination), "files": files}
            (stage / receipt_path.name).write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
            directory.symlink_to(stage.name, target_is_directory=True)
        except BaseException:
            print(f"Unpublished video preparation preserved at {stage}", file=sys.stderr, flush=True)
            raise
        _bind_hash(entry, destination)
    except ConfigError:
        raise
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        raise ConfigError(f"{action} failed ({type(exc).__name__}): {destination}; "
                          "check input files, native environment and preparation receipts") from exc
