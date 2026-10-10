"""Offline checks for the pinned, asset-complete HPS dependency wheel."""

import base64
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "third_party/hpsv2"
BPE = "hpsv2/src/open_clip/bpe_simple_vocab_16e6.txt.gz"


def test_lock_and_source_select_complete_wheel():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    path = project["tool"]["uv"]["sources"]["hpsv2"]["path"]
    assert path == "third_party/hpsv2/hpsv2-1.2.0-2-py3-none-any.whl"
    dependency = next(item for item in lock["package"] if item["name"] == "hpsv2")
    assert dependency["version"] == "1.2.0"
    assert dependency["source"] == {"path": path}
    report = json.loads((ARTIFACTS / "verification.json").read_text())
    payload = (ROOT / path).read_bytes()
    assert len(payload) == report["wheel_bytes"]
    assert hashlib.sha256(payload).hexdigest() == report["wheel_sha256"]
    assert report["metadata_byte_identical"] and report["license_byte_identical"]


def test_vocabulary_headless_import_and_record():
    with zipfile.ZipFile(ARTIFACTS / "hpsv2-1.2.0-2-py3-none-any.whl") as wheel:
        assert hashlib.sha256(wheel.read(BPE)).hexdigest() == (
            "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a"
        )
        assert b"from turtle import forward" not in wheel.read(
            "hpsv2/src/open_clip/factory.py"
        )
        assert "hpsv2/src/open_clip/model_configs/ViT-H-14.json" in wheel.namelist()
        assert b"Apache License" in wheel.read("hpsv2-1.2.0.dist-info/LICENSE")
        record = "hpsv2-1.2.0.dist-info/RECORD"
        rows = list(csv.reader(io.StringIO(wheel.read(record).decode())))
        assert len(rows) == len(wheel.namelist())
        assert {row[0] for row in rows} == set(wheel.namelist())
        for name, digest, size in rows:
            if name == record:
                assert digest == size == ""
                continue
            payload = wheel.read(name)
            expected = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).decode().rstrip("=")
            assert digest == f"sha256={expected}" and size == str(len(payload))
