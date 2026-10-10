"""Reproduce the local HPSv2 wheel repair from two SHA256-pinned official inputs."""

import argparse
import ast
import base64
import csv
import hashlib
import io
import json
import platform
import sys
import urllib.request
import zipfile
import zlib
from pathlib import Path

UPSTREAM_URL = (
    "https://files.pythonhosted.org/packages/46/f3/"
    "4ff37b7e5c5079bdc2880c9a72300c5b962e62ac0ebdcda2af358c3ddc4f/"
    "hpsv2-1.2.0-py3-none-any.whl"
)
UPSTREAM_SHA256 = "3ee638b06847bef24e1de43d0b5e3890a4a850034b4bf618e4bb3e78e2d806a1"
GIT_REVISION = "866735ecaae999fa714bd9edfa05aa2672669ee3"
BPE_URL = (
    f"https://raw.githubusercontent.com/tgxs002/HPSv2/{GIT_REVISION}/"
    "hpsv2/src/open_clip/bpe_simple_vocab_16e6.txt.gz"
)
BPE_SHA256 = "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a"
BPE_MEMBER = "hpsv2/src/open_clip/bpe_simple_vocab_16e6.txt.gz"
FACTORY_MEMBER = "hpsv2/src/open_clip/factory.py"
DIST_INFO = "hpsv2-1.2.0.dist-info"
RECORD_MEMBER = f"{DIST_INFO}/RECORD"
WHEEL_MEMBER = f"{DIST_INFO}/WHEEL"
WHEEL_NAME = "hpsv2-1.2.0-2-py3-none-any.whl"
REMOVED_LINE = b"from turtle import forward\r\n"
ZIP_TIME = (2026, 10, 10, 0, 0, 0)


def sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def pinned_input(path, url, expected_sha256, max_bytes):
    if path is None:
        with urllib.request.urlopen(url, timeout=20) as stream:
            payload = stream.read(max_bytes + 1)
    else:
        with Path(path).open("rb") as stream:
            payload = stream.read(max_bytes + 1)
    if len(payload) > max_bytes or sha256(payload) != expected_sha256:
        raise ValueError(f"input exceeds size bound or SHA256 mismatch: {url}")
    return payload


def repaired_factory(original):
    if original.count(REMOVED_LINE) != 1:
        raise ValueError("expected exactly one upstream turtle import with CRLF")
    tree = ast.parse(original.decode("utf-8"))
    imports = [node for node in ast.walk(tree)
               if isinstance(node, ast.ImportFrom) and node.module == "turtle"]
    if (len(imports) != 1 or len(imports[0].names) != 1
            or imports[0].names[0].name != "forward" or imports[0].names[0].asname is not None
            or any(isinstance(node, ast.Name) and node.id == "forward"
                   for node in ast.walk(tree))):
        raise ValueError("the turtle import is not the expected unused name")
    return original.replace(REMOVED_LINE, b"", 1)


def record_bytes(entries, order):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    for name in order:
        value = entries[name]
        digest = base64.urlsafe_b64encode(hashlib.sha256(value).digest()).decode().rstrip("=")
        writer.writerow((name, f"sha256={digest}", len(value)))
    writer.writerow((RECORD_MEMBER, "", ""))
    return stream.getvalue().encode()


def inspect_repair(upstream_payload, repaired_payload):
    with zipfile.ZipFile(io.BytesIO(upstream_payload)) as original:
        with zipfile.ZipFile(io.BytesIO(repaired_payload)) as repaired:
            before, after = set(original.namelist()), set(repaired.namelist())
            if len(before) != len(original.infolist()) or len(after) != len(repaired.infolist()):
                raise ValueError("duplicate wheel members")
            if after - before != {BPE_MEMBER} or before - after:
                raise ValueError("unexpected added/removed wheel members")
            changed = sorted(name for name in before if original.read(name) != repaired.read(name))
            if set(changed) != {FACTORY_MEMBER, WHEEL_MEMBER, RECORD_MEMBER}:
                raise ValueError(f"unexpected changed wheel members: {changed}")
            if repaired.read(FACTORY_MEMBER) != repaired_factory(original.read(FACTORY_MEMBER)):
                raise ValueError("factory differs beyond the single unused import removal")
            if sha256(repaired.read(BPE_MEMBER)) != BPE_SHA256:
                raise ValueError("repaired BPE mismatch")
            rows = list(csv.reader(io.StringIO(repaired.read(RECORD_MEMBER).decode())))
            if len(rows) != len(after) or {row[0] for row in rows} != after:
                raise ValueError("incomplete or duplicate RECORD")
            for name, digest, size in rows:
                if name == RECORD_MEMBER:
                    if digest or size:
                        raise ValueError("RECORD must not hash itself")
                    continue
                value = repaired.read(name)
                expected = base64.urlsafe_b64encode(hashlib.sha256(value).digest()).decode().rstrip("=")
                if digest != f"sha256={expected}" or size != str(len(value)):
                    raise ValueError(f"invalid RECORD hash/size: {name}")
            return {
                "changed_members": changed,
                "added_members": [BPE_MEMBER],
                "unchanged_python_files": sum(name.endswith(".py") and name != FACTORY_MEMBER
                                              for name in before),
                "factory_only_change": "removed exact unused `from turtle import forward` line",
                "factory_before_sha256": sha256(original.read(FACTORY_MEMBER)),
                "factory_after_sha256": sha256(repaired.read(FACTORY_MEMBER)),
                "metadata_byte_identical": True,
                "license_byte_identical": True,
                "record_hashes_and_sizes_valid": True,
            }


def build(upstream_payload, bpe_payload):
    with zipfile.ZipFile(io.BytesIO(upstream_payload)) as source:
        entries = {name: source.read(name) for name in source.namelist()}
    if BPE_MEMBER in entries:
        raise ValueError("upstream wheel unexpectedly already contains BPE")
    entries.pop(RECORD_MEMBER)
    entries[BPE_MEMBER] = bpe_payload
    entries[FACTORY_MEMBER] = repaired_factory(entries[FACTORY_MEMBER])
    original_wheel = entries[WHEEL_MEMBER]
    if b"Build:" in original_wheel or not original_wheel.endswith(b"\n\n"):
        raise ValueError("unexpected upstream WHEEL metadata")
    entries[WHEEL_MEMBER] = original_wheel[:-1] + b"Build: 2\n\n"
    order = sorted(entries)
    entries[RECORD_MEMBER] = record_bytes(entries, order)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        for name in [*order, RECORD_MEMBER]:
            info = zipfile.ZipInfo(name, date_time=ZIP_TIME)
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            output.writestr(info, entries[name], compresslevel=6)
    payload = stream.getvalue()
    inspect_repair(upstream_payload, payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-wheel", type=Path, help="verified local upstream input; otherwise download")
    parser.add_argument("--bpe", type=Path, help="verified local BPE input; otherwise download")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--verify-only", action="store_true", help="verify existing output without rewriting it")
    args = parser.parse_args()
    original = pinned_input(args.upstream_wheel, UPSTREAM_URL, UPSTREAM_SHA256, 8 * 1024 * 1024)
    bpe = pinned_input(args.bpe, BPE_URL, BPE_SHA256, 2 * 1024 * 1024)
    expected = build(original, bpe)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / WHEEL_NAME
    if args.verify_only:
        payload = target.read_bytes()
        if payload != expected:
            raise ValueError("existing wheel does not match the deterministic rebuild")
    else:
        payload = expected
        temporary = target.with_suffix(".whl.tmp")
        temporary.write_bytes(payload)
        temporary.replace(target)
    report = {
        "wheel": target.name, "wheel_sha256": sha256(payload), "wheel_bytes": len(payload),
        "upstream_url": UPSTREAM_URL, "upstream_sha256": UPSTREAM_SHA256,
        "bpe_url": BPE_URL, "bpe_sha256": BPE_SHA256, "official_git_revision": GIT_REVISION,
        "python": sys.version.split()[0], "platform": platform.platform(),
        "zlib_runtime": zlib.ZLIB_RUNTIME_VERSION,
        "zip_timestamp": list(ZIP_TIME), "zip_compression_level": 6,
        "rebuild_byte_identical": True,
        **inspect_repair(original, payload),
    }
    # Generated verification evidence, not a manually maintained experiment ledger.
    (args.output_dir / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
