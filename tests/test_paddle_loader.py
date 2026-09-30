"""Asset/version and config boundary tests; no Paddle model inference claim."""

import hashlib
import importlib.metadata
import json
import sys
from types import SimpleNamespace

import pytest

from visual_rl.config import ConfigError, Reward
from visual_rl.rewards.image_factory import load_image_reward, validate_image_reward
from visual_rl.rewards.paddle_loader import (
    MODEL_FILES,
    VERSIONS,
    _load_native_engine,
    load_ocr_reward,
    verify_bundle,
)


@pytest.fixture
def bundle(tmp_path):
    files = {}
    for name in MODEL_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    document = {
        "schema_version": 1,
        "versions": VERSIONS,
        "ocr_version": "PP-OCRv4",
        "files": files,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(document))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def reward(bundle):
    path, digest = bundle
    return Reward(
        name="ocr",
        checkpoint=str(path),
        revision=digest,
        weights={"ocr": 1},
        options={"microbatch": 2, "error_policy": "zero"},
    )


def test_verified_files_include_unused_classifier_to_prevent_constructor_download(bundle):
    path, digest = bundle
    root, doc = verify_bundle(str(path), digest)
    assert root == path.parent and doc["versions"] == VERSIONS
    (root / "cls/inference.pdmodel").unlink()
    with pytest.raises(ValueError, match="regular file"):
        verify_bundle(str(path), digest)


@pytest.mark.parametrize("name", sorted(MODEL_FILES))
def test_corrupt_asset_rejected_before_paddle_import(bundle, name, monkeypatch):
    path, digest = bundle
    (path.parent / name).write_bytes(b"changed")
    monkeypatch.setitem(sys.modules, "paddleocr", None)
    with pytest.raises(ValueError, match="asset SHA256"):
        load_ocr_reward(
            manifest_path=str(path), manifest_sha256=digest, error_policy="zero", device="cpu"
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.update(schema_version=True),
        lambda d: d.update(extra=True),
        lambda d: d.update(versions={"paddleocr": "3.0"}),
        lambda d: d.update(ocr_version="latest"),
        lambda d: d["files"].update({"../outside": "a" * 64}),
        lambda d: d["files"].update({"en_dict.txt": "bad"}),
    ],
)
def test_manifest_schema_and_versions_do_not_silently_fall_back(bundle, mutation):
    path, _ = bundle
    document = json.loads(path.read_text())
    mutation(document)
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="manifest"):
        verify_bundle(str(path), hashlib.sha256(path.read_bytes()).hexdigest())


def test_manifest_digest_and_duplicate_keys_checked(bundle):
    path, digest = bundle
    with pytest.raises(ValueError, match="SHA256"):
        verify_bundle(str(path), "a" * 64)
    path.write_text('{"schema_version": 1, "schema_version": 1}')
    with pytest.raises(ValueError, match="duplicate"):
        verify_bundle(str(path), hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.mark.parametrize("version", ["3.0.0", None])
def test_runtime_dependency_drift_fails_before_engine_construction(bundle, monkeypatch, version):
    path, digest = bundle

    def installed(package):
        if version is None:
            raise importlib.metadata.PackageNotFoundError(package)
        return version

    monkeypatch.setattr(importlib.metadata, "version", installed)
    monkeypatch.setitem(sys.modules, "paddleocr", None)
    with pytest.raises(RuntimeError, match="requires paddleocr"):
        _load_native_engine(str(path), digest)


def test_native_factory_constructs_local_cpu_engine_with_original_language_and_cls(
    bundle, monkeypatch
):
    pytest.importorskip("rapidfuzz")
    from visual_rl.rewards.ocr import OCRScore

    calls = []
    engine = SimpleNamespace(ocr=lambda *a, **kw: [None])

    def construct(**kwargs):
        calls.append(kwargs)
        return engine

    monkeypatch.setattr(importlib.metadata, "version", VERSIONS.__getitem__)
    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PaddleOCR=construct))
    monkeypatch.setattr("visual_rl.rewards.ocr_process.PaddleWorker", _load_native_engine)
    scorer = load_image_reward(reward(bundle), device="cpu")
    assert isinstance(scorer, OCRScore)
    root = bundle[0].parent
    assert calls == [
        {
            "use_angle_cls": False,
            "lang": "en",
            "use_gpu": False,
            "show_log": False,
            "ocr_version": "PP-OCRv4",
            "det_model_dir": str(root / "det"),
            "rec_model_dir": str(root / "rec"),
            "cls_model_dir": str(root / "cls"),
            "rec_char_dict_path": str(root / "en_dict.txt"),
        }
    ]


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"microbatch": 1},
        {"microbatch": 1, "error_policy": "ignore"},
        {"microbatch": 0, "error_policy": "zero"},
        {"microbatch": 1, "error_policy": "zero", "use_gpu": True},
    ],
)
def test_invalid_options_rejected_before_loading(bundle, options):
    from dataclasses import replace

    with pytest.raises(ConfigError):
        validate_image_reward(replace(reward(bundle), options=options))


def test_no_implicit_ocr_normalization_or_asset_revision(bundle):
    from dataclasses import replace

    c = reward(bundle)
    validate_image_reward(c)
    for overrides in ({"normalization": "fixed"}, {"revision": "main"}, {"weights": {"ocr": 2}}):
        with pytest.raises(ConfigError):
            validate_image_reward(replace(c, **overrides))


def test_manifest_helper_records_bundle_and_refuses_overwrite(bundle, capsys):
    from visual_rl.rewards.paddle_loader import main

    path, _ = bundle
    path.unlink()
    assert main(["--bundle", str(path.parent), "--ocr-version", "PP-OCRv4"]) == 0
    recorded = json.loads(capsys.readouterr().out)
    root, document = verify_bundle(recorded["manifest"], recorded["sha256"])
    assert root == path.parent and set(document["files"]) == MODEL_FILES
    previous = path.read_bytes()
    with pytest.raises(FileExistsError):
        main(["--bundle", str(path.parent), "--ocr-version", "PP-OCRv3"])
    assert path.read_bytes() == previous


def test_example_config_resolves_without_pickscore_residue(monkeypatch, bundle):
    from visual_rl.config import load_config
    from visual_rl.registry import get_backend

    monkeypatch.setenv("OCR_MANIFEST", str(bundle[0]))
    monkeypatch.setenv("OCR_MANIFEST_SHA256", bundle[1])
    c = load_config("configs/examples/sd3_ocr.yaml")
    assert c.reward.weights == {"ocr": 1} and c.reward.normalization == "raw"
    assert set(c.reward.options) == {"microbatch", "error_policy"}
    assert get_backend(c).name == "sd3"
