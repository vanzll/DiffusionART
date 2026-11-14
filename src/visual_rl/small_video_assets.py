"""Pinned small-video asset preparation, without model imports or GPU work.

Downloads reuse prepare_assets. Full content verification runs once in the
master; workers can revalidate the resulting receipt and immutable source/stat
identities without scanning multi-GB weights again. This does not certify tensor
finiteness, native inference parity or the unresolved ART replay/mask contract.
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import time
import unicodedata
from pathlib import Path

from .prepare_assets import prepare_assets, validate_assets

SCHEMA = "small-video-assets/v1"
RECEIPT_SCHEMA = "small-video-assets-receipt/v1"
DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "configs/assets/small_video.json"


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite asset JSON value")
    if isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, list):
        for item in value:
            _finite(item)
    return value


def _json(text):
    return _finite(json.loads(text))


def _encoded(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":")
    ).encode()


def _digest(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _relative(value):
    return (
        isinstance(value, str)
        and bool(value)
        and value != "."
        and not Path(value).is_absolute()
        and ".." not in Path(value).parts
    )


def load_manifest(path=DEFAULT_MANIFEST):
    spec = _json(Path(path).read_text())
    validate_manifest(spec)
    return spec


def validate_manifest(spec):
    _finite(spec)
    if not isinstance(spec, dict) or spec.get("schema_version") != SCHEMA:
        raise ValueError("invalid small-video asset manifest schema")
    ids = set()
    for entry in spec.get("snapshots", []):
        name = entry.get("id")
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name) or name in ids:
            raise ValueError("asset IDs must be unique safe names")
        ids.add(name)
        files = entry.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError("every asset requires a complete pinned file inventory")
        seen = set()
        for row in files:
            if (
                not isinstance(row, dict)
                or not _relative(row.get("path"))
                or row["path"] in seen
                or type(row.get("size")) is not int
                or row["size"] <= 0
                or not set(row) <= {"path", "size", "sha256", "git_blob_sha1"}
                or not {"sha256", "git_blob_sha1"} & set(row)
            ):
                raise ValueError("invalid or unpinned asset file inventory")
            seen.add(row["path"])
            for key, length in (("sha256", 64), ("git_blob_sha1", 40)):
                if key in row and not re.fullmatch(r"[a-f0-9]{" + str(length) + "}", row[key]):
                    raise ValueError("invalid pinned asset file digest")
        _profiles(entry)
        validate_assets({"snapshots": [_download_entry(entry)]})
    if not ids:
        raise ValueError("empty asset snapshot inventory")
    validate_assets({"snapshots": [_download_entry(entry) for entry in spec["snapshots"]]})
    source_ids = set()
    for source in spec.get("sources", []):
        if (
            not isinstance(source, dict)
            or not isinstance(source.get("id"), str)
            or not re.fullmatch(r"[a-z][a-z0-9_]*", source["id"])
            or source["id"] in source_ids
            or not _relative(source.get("package"))
            or not isinstance(source.get("revision"), str)
            or not re.fullmatch(r"[a-f0-9]{40}", source["revision"])
            or not isinstance(source.get("repo"), str)
            or not re.fullmatch(
                r"https://github.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\.git", source["repo"]
            )
        ):
            raise ValueError("invalid source revision/package inventory")
        source_ids.add(source["id"])
        _profiles(source)
    data = spec.get("data", {})
    if data.get("asset") not in ids or any(
        not _relative(data.get(k)) for k in ("train", "validation")
    ):
        raise ValueError("data must bind declared train/validation files")
    for key in ("train_count", "validation_count", "eval_count"):
        if type(data.get(key)) is not int or data[key] <= 0:
            raise ValueError("data counts must be positive integers")
    if (
        type(data.get("eval_seed")) is not int
        or data["eval_count"] > data["validation_count"]
        or data.get("expected_train_validation_overlap") != 0
        or data.get("normalization") != "NFKC/casefold/whitespace"
    ):
        raise ValueError("invalid frozen evaluation/leakage contract")
    for key in ("train_sha256", "validation_sha256", "eval_sha256"):
        if not isinstance(data.get(key), str) or not re.fullmatch(r"[a-f0-9]{64}", data[key]):
            raise ValueError("data files and evaluation require SHA256 pins")
    if (
        not isinstance(data.get("eval_sample_ids"), list)
        or len(data["eval_sample_ids"]) != data["eval_count"]
        or len(set(data["eval_sample_ids"])) != data["eval_count"]
    ):
        raise ValueError("frozen evaluation requires ordered distinct sample IDs")


def _profiles(entry):
    profiles = entry.get("profiles")
    if not isinstance(profiles, list) or not profiles or not set(profiles) <= {"wan", "ltx"}:
        raise ValueError("asset/source profiles must name wan and/or ltx")
    return profiles


def _selected(entries, profile):
    if profile not in {"wan", "ltx", "all"}:
        raise ValueError("profile must be wan, ltx or all")
    return [entry for entry in entries if profile == "all" or profile in entry["profiles"]]


def _download_entry(entry):
    return {key: entry[key] for key in ("env", "repo_id", "revision", "repo_type")} | {
        "allow_patterns": [row["path"] for row in entry["files"]]
    }


def snapshot_plan(spec, root, *, profile="all", bindings=None):
    """A public root or explicit per-asset bindings; never inspect site paths."""
    validate_manifest(spec)
    bindings = {} if bindings is None else bindings
    if set(bindings) - {e["id"] for e in spec["snapshots"]}:
        raise ValueError("unknown explicit asset binding")
    root = Path(root).expanduser().resolve()
    plan = []
    for entry in _selected(spec["snapshots"], profile):
        category = "datasets" if entry["repo_type"] == "dataset" else "models"
        path = bindings.get(entry["id"], root / category / entry["id"] / entry["revision"])
        plan.append((entry, Path(path).expanduser().resolve()))
    return plan


def download_assets(spec, root, *, profile="all", bindings=None):
    """Explicit potentially large download; no source install or preprocessing."""
    plan = snapshot_plan(spec, root, profile=profile, bindings=bindings)
    prepare_assets(([(_download_entry(e), path) for e, path in plan], []))
    return {entry["id"]: str(path) for entry, path in plan}


def _stamp(path):
    path = Path(path)
    stat = path.stat()
    if not path.is_file():
        raise ValueError(f"asset is not a regular file: {path}")
    return {
        "resolved": str(path.resolve(strict=True)),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "inode": stat.st_ino,
    }


def _hash_file(path, *, git_blob=False):
    """One streaming scan with progress, and stable pre/post file identity."""
    try:
        before = _stamp(path)
    except OSError:
        raise ValueError(f"required asset is missing or inaccessible: {path}") from None
    sha = hashlib.sha256()
    blob = hashlib.sha1(f"blob {before['size']}\0".encode()) if git_blob else None
    count, reported = 0, time.monotonic()
    print(f"Verifying {path} ({before['size']} bytes)", file=sys.stderr, flush=True)
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            sha.update(block)
            if blob is not None:
                blob.update(block)
            count += len(block)
            if time.monotonic() - reported >= 15:
                print(
                    f"Verified {count}/{before['size']} bytes: {path}", file=sys.stderr, flush=True
                )
                reported = time.monotonic()
    if _stamp(path) != before or count != before["size"]:
        raise ValueError("asset changed during verification")
    return {
        "sha256": sha.hexdigest(),
        **({"git_blob_sha1": blob.hexdigest()} if blob else {}),
        "stat": before,
    }


def _source_record(source, path):
    binding = Path(path).expanduser().absolute()
    path = binding.resolve(strict=True)
    if not (path / source["package"]).is_dir():
        raise ValueError("declared native source package is absent")

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(path), *args], check=True, capture_output=True, text=True, timeout=20
        ).stdout.strip()

    if git("rev-parse", "HEAD") != source["revision"] or git(
        "status", "--porcelain", "--untracked-files=no"
    ):
        raise ValueError(f"{source['id']} source must be clean at its pinned Git revision")
    return {
        "id": source["id"],
        "binding": str(binding),
        "path": str(path),
        "revision": source["revision"],
        "package": source["package"],
    }


def _json_file_checks(entry, directory):
    required = {row["path"] for row in entry["files"]}
    for name in required:
        if not name.endswith(".json"):
            continue
        path = directory / name
        if path.stat().st_size > 32 * 1024 * 1024:
            raise ValueError("JSON asset exceeds bounded validation size")
        document = _json(path.read_text())
        if name.endswith(".index.json"):
            mapping = document.get("weight_map")
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError("model shard index has no weight map")
            for shard in mapping.values():
                if not _relative(shard) or str(Path(name).parent / shard) not in required:
                    raise ValueError("model shard index references an absent/unpinned shard")


def _rows(path, count):
    rows, seen = [], set()
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if len(line) > 1024 * 1024 or len(rows) >= count:
                raise ValueError("prompt manifest exceeds its frozen row/line bound")
            row = _json(line)
            if (
                not isinstance(row, dict)
                or set(row) != {"sample_id", "prompt"}
                or not isinstance(row["sample_id"], str)
                or not row["sample_id"].strip()
                or row["sample_id"] in seen
                or not isinstance(row["prompt"], str)
                or not row["prompt"].strip()
            ):
                raise ValueError("invalid/duplicate/empty text-only prompt row")
            seen.add(row["sample_id"])
            rows.append(row)
    if len(rows) != count:
        raise ValueError("prompt count differs from frozen data contract")
    return rows


def _normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_encoded(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def prepare_eval_subset(data, directory, output):
    """Verify source bytes and all split leakage before publishing paired eval."""
    train_path, val_path = directory / data["train"], directory / data["validation"]
    for path, key in ((train_path, "train_sha256"), (val_path, "validation_sha256")):
        if _hash_file(path)["sha256"] != data[key]:
            raise ValueError("frozen prompt source SHA256 differs")
    train, validation = (
        _rows(train_path, data["train_count"]),
        _rows(val_path, data["validation_count"]),
    )
    train_texts = {_normalized(row["prompt"]) for row in train}
    if train_texts & {_normalized(row["prompt"]) for row in validation}:
        raise ValueError("normalized train/validation prompt leakage")
    rows = random.Random(data["eval_seed"]).sample(validation, data["eval_count"])
    payload = "".join(
        json.dumps(row, ensure_ascii=True, sort_keys=True, allow_nan=False) + "\n" for row in rows
    ).encode()
    if [row["sample_id"] for row in rows] != data["eval_sample_ids"] or hashlib.sha256(
        payload
    ).hexdigest() != data["eval_sha256"]:
        raise ValueError("paired evaluation order/hash differs from frozen Random(seed) subset")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Atomic exclusive publish: a corrupt existing artifact is never overwritten.
    if output.exists():
        if output.read_bytes() != payload:
            raise ValueError("existing paired evaluation subset differs")
    else:
        descriptor, temporary = tempfile.mkstemp(prefix=".eval-", dir=output.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, output)
            except FileExistsError:
                if output.read_bytes() != payload:
                    raise ValueError("concurrent evaluation subset differs")
        finally:
            os.unlink(temporary)
    return {
        "path": str(output.resolve()),
        "sha256": data["eval_sha256"],
        "sample_ids": data["eval_sample_ids"],
        "seed": data["eval_seed"],
        "stat": _stamp(output),
        "normalized_train_validation_overlap": 0,
    }


def preflight_assets(spec, root, *, source_roots, profile="all", bindings=None, receipt=None):
    """Master-only full scan; no downloads and no pretrained model evaluation."""
    root = Path(root).expanduser().resolve()
    receipt = Path(receipt) if receipt else root / f"preflight-{profile}.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.with_suffix(receipt.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        plan = snapshot_plan(spec, root, profile=profile, bindings=bindings)
        selected_sources = _selected(spec["sources"], profile)
        if set(source_roots) != {e["id"] for e in selected_sources}:
            raise ValueError("supply exactly the required native source bindings for this profile")
        sources = [_source_record(e, source_roots[e["id"]]) for e in selected_sources]
        files = []
        for entry, directory in plan:
            for row in entry["files"]:
                path = directory / row["path"]
                actual = _hash_file(path, git_blob="git_blob_sha1" in row)
                if actual["stat"]["size"] != row["size"] or any(
                    actual[key] != row[key] for key in ("sha256", "git_blob_sha1") if key in row
                ):
                    raise ValueError(f"pinned asset bytes differ: {entry['id']}/{row['path']}")
                files.append(
                    {"asset": entry["id"], "relative": row["path"], "path": str(path), **actual}
                )
            _json_file_checks(entry, directory)
        locations = {entry["id"]: directory for entry, directory in plan}
        data = prepare_eval_subset(
            spec["data"], locations[spec["data"]["asset"]], root / "derived/eval8.jsonl"
        )
        payload = {
            "schema_version": RECEIPT_SCHEMA,
            "manifest_sha256": _digest(spec),
            "profile": profile,
            "root": str(root),
            "sources": sources,
            "assets": {key: str(value) for key, value in locations.items()},
            "files": files,
            "data": data,
            "acceptance": "asset bytes/source/data only; native model/ART/GPU acceptance pending",
        }
        payload["checksum"] = _digest(payload)
        _atomic_json(receipt, payload)
        return payload


def verify_receipt(spec, root, receipt, *, profile="all", bindings=None, source_roots=None):
    """Worker fast path: no weight hashing; reject any changed receipt/file/source."""
    plan = snapshot_plan(spec, root, profile=profile, bindings=bindings)
    record = _json(Path(receipt).read_text())
    body = {key: value for key, value in record.items() if key != "checksum"}
    if (
        record.get("schema_version") != RECEIPT_SCHEMA
        or record.get("checksum") != _digest(body)
        or record.get("manifest_sha256") != _digest(spec)
        or record.get("profile") != profile
        or record.get("root") != str(Path(root).expanduser().resolve())
        or record.get("assets") != {e["id"]: str(p) for e, p in plan}
    ):
        raise ValueError("asset receipt manifest/profile/root/checksum identity differs")
    expected = {(entry["id"], row["path"]): row for entry, _ in plan for row in entry["files"]}
    actual = record.get("files", [])
    if len(actual) != len(expected) or {(f["asset"], f["relative"]) for f in actual} != set(
        expected
    ):
        raise ValueError("asset receipt file inventory differs")
    locations = {e["id"]: p for e, p in plan}
    for file in actual:
        pinned = expected[(file["asset"], file["relative"])]
        path = locations[file["asset"]] / file["relative"]
        if (
            file["path"] != str(path)
            or _stamp(path) != file["stat"]
            or any(file.get(k) != pinned[k] for k in ("sha256", "git_blob_sha1") if k in pinned)
        ):
            raise ValueError("asset changed since master verification; rerun full preflight")
    sources = {row["id"]: row for row in record.get("sources", [])}
    selected = _selected(spec["sources"], profile)
    if set(sources) != {e["id"] for e in selected}:
        raise ValueError("source receipt inventory differs")
    if source_roots is not None and set(source_roots) != set(sources):
        raise ValueError("source binding inventory differs")
    for source in selected:
        path = (
            source_roots[source["id"]]
            if source_roots is not None
            else sources[source["id"]]["binding"]
        )
        if _source_record(source, path) != sources[source["id"]]:
            raise ValueError("source binding changed since master verification")
    data = record["data"]
    expected_eval = str((Path(root).expanduser().resolve() / "derived/eval8.jsonl").resolve())
    if (
        data.get("path") != expected_eval
        or data.get("sha256") != spec["data"]["eval_sha256"]
        or data.get("sample_ids") != spec["data"]["eval_sample_ids"]
        or _stamp(data["path"]) != data["stat"]
    ):
        raise ValueError("paired evaluation subset changed since master verification")
    return record


def verify_bound_receipt(spec, root, receipt, *, profile="all"):
    """Runtime fast path using the master's explicit, content-verified bindings.

    All original manifest/checksum/source/stat checks remain in verify_receipt;
    custom locations do not get silently replaced by the default layout.
    """
    path = Path(receipt)
    if path.stat().st_size > 1024 * 1024:
        raise ValueError("asset receipt exceeds runtime byte limit")
    record = _json(path.read_text())
    bindings = record.get("assets")
    if (
        not isinstance(bindings, dict)
        or not bindings
        or any(
            not isinstance(key, str) or not isinstance(value, str) or not Path(value).is_absolute()
            for key, value in bindings.items()
        )
    ):
        raise ValueError("asset receipt requires explicit absolute asset bindings")
    return verify_receipt(spec, root, receipt, profile=profile, bindings=bindings)


def _bindings(values):
    result = {}
    for value in values:
        key, separator, path = value.partition("=")
        if not separator or not key or not path or key in result:
            raise ValueError("bindings require unique ID=/absolute/or/relative/path values")
        result[key] = path
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "download", "preflight", "verify"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--profile", choices=("wan", "ltx", "all"), default="all")
    parser.add_argument("--asset", action="append", default=[], metavar="ID=PATH")
    parser.add_argument("--source", action="append", default=[], metavar="ID=PATH")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args(argv)
    try:
        spec = load_manifest(args.manifest)
        bindings, sources = _bindings(args.asset), _bindings(args.source)
        receipt = args.receipt or args.root / f"preflight-{args.profile}.json"
        if args.command == "plan":
            plan = snapshot_plan(spec, args.root, profile=args.profile, bindings=bindings)
            result = {
                "manifest_sha256": _digest(spec),
                "profile": args.profile,
                "total_bytes": sum(f["size"] for e, _ in plan for f in e["files"]),
                "assets": {e["id"]: str(p) for e, p in plan},
                "required_sources": _selected(spec["sources"], args.profile),
                "runtime_policy": spec.get("runtime_policy", {}),
            }
        elif args.command == "download":
            result = download_assets(spec, args.root, profile=args.profile, bindings=bindings)
        elif args.command == "preflight":
            result = preflight_assets(
                spec,
                args.root,
                profile=args.profile,
                bindings=bindings,
                source_roots=sources,
                receipt=receipt,
            )
        else:
            result = verify_receipt(
                spec,
                args.root,
                receipt,
                profile=args.profile,
                bindings=bindings,
                source_roots=sources or None,
            )
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        # Do not print HTTP exception bodies, signed URLs, credentials or env values.
        detail = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
        parser.exit(2, f"small-video assets failed: {detail}; check manifest, paths and receipts\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
