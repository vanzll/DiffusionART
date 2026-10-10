import hashlib
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from visual_rl.rewards import geneval_loader as loader


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    versions = dict.fromkeys(loader.PACKAGES, "1.0")
    versions["mmdet"] = "2.28.2"
    for name in loader.FILES:
        (tmp_path / name).write_bytes(b"fixture model")
    (tmp_path / "detector_config.py").write_text("model = dict(type='fixture')\n")
    (tmp_path / "object_names.txt").write_text("cat\ndog\n")
    files = {name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() for name in loader.FILES}
    monkeypatch.setattr(loader, "CLASSES_SHA256", files["object_names.txt"])
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema_version": 1, "versions": versions, "files": files}))
    return path, hashlib.sha256(path.read_bytes()).hexdigest(), versions


def test_bundle_verifies_every_byte_and_versions(bundle):
    path, sha, versions = bundle
    root, spec = loader.verify_bundle(str(path), sha)
    assert spec["versions"] == versions
    (root / "clip.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="asset SHA256"):
        loader.verify_bundle(str(path), sha)


def test_bundle_rejects_symlink_and_hash_mismatch(bundle):
    path, sha, _ = bundle
    with pytest.raises(ValueError, match="SHA256"):
        loader.verify_bundle(str(path), "a" * 64)
    target = path.parent / "real.pt"
    (path.parent / "clip.pt").rename(target)
    (path.parent / "clip.pt").symlink_to(target)
    with pytest.raises(ValueError, match="regular file"):
        loader.verify_bundle(str(path), sha)


def test_native_loader_preserves_openai_half_precision_and_detector_chunks(bundle, monkeypatch):
    import torch

    from visual_rl.rewards import geneval_engine

    path, sha, versions = bundle
    monkeypatch.setattr(loader.importlib.metadata, "version", versions.__getitem__)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    seen = {}
    for name in ("clip_benchmark", "clip_benchmark.metrics", "mmcv", "mmdet", "mmdet.apis",
                 "open_clip", "open_clip.openai", "open_clip.transform"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules["clip_benchmark.metrics"].zeroshot_classification = "zsc"
    sys.modules["mmcv"].Config = SimpleNamespace(fromfile=lambda p: {"loaded": p})
    sys.modules["mmdet.apis"].init_detector = lambda cfg, p, device: seen.update(detector=(cfg, p, device))
    calls = []
    sys.modules["mmdet.apis"].inference_detector = lambda model, images: calls.append(len(images)) or images
    sys.modules["open_clip"].get_tokenizer = lambda name: name

    def clip(path, **kwargs):
        seen["clip"] = (path, kwargs)
        return SimpleNamespace(visual=SimpleNamespace(image_mean="mean", image_std="std"))

    sys.modules["open_clip.openai"].load_openai_model = clip
    sys.modules["open_clip.transform"].image_transform = lambda *a, **kw: seen.update(transform=(a, kw))

    def build(**kwargs):
        seen["build"] = kwargs
        return lambda images, metadata, only_strict: kwargs["inference_detector"](None, images)

    monkeypatch.setattr(geneval_engine, "build_evaluator", build)
    engine = loader._load_native_engine(str(path), sha, device="cpu", detector_microbatch=2, color_workers=0)
    assert engine([1, 2, 3], [{"include": [{"class": "cat"}]}], True) == [1, 2, 3]
    assert calls == [2, 1]
    assert seen["clip"][1] == {"precision": "fp16", "device": "cpu"}
    assert seen["transform"] == ((224,), {"is_train": False, "mean": "mean", "std": "std"})
    assert seen["build"]["classnames"] == ["cat", "dog"]
    with pytest.raises(ValueError, match="unknown GenEval object class"):
        engine([1], [{"include": [{"class": "unknown"}]}], True)


def test_environment_mismatch_fails_before_native_import(bundle, monkeypatch):
    path, sha, _ = bundle
    monkeypatch.setattr(loader.importlib.metadata, "version", lambda _: "wrong")
    with pytest.raises(RuntimeError, match="GenEval requires"):
        loader._load_native_engine(str(path), sha, device="cpu", detector_microbatch=1, color_workers=0)


def test_composite_load_failure_closes_already_created_worker(monkeypatch):
    from visual_rl.config import Reward
    from visual_rl.rewards import geneval_process, image_factory, paddle_loader

    closed = []
    monkeypatch.setattr(loader, "verify_bundle", lambda *args: None)
    monkeypatch.setattr(geneval_process, "GenEvalWorker",
                        lambda *a, **kw: SimpleNamespace(close=lambda: closed.append(True)))

    def fail(**kwargs):
        raise RuntimeError("next component failed")

    monkeypatch.setattr(paddle_loader, "load_ocr_reward", fail)
    reward = Reward(name="composite", weights={"geneval": 1, "ocr": 1}, normalization="raw",
                    options={"microbatch": 1, "components": {
                        "geneval": {"checkpoint": "fixture", "revision": "a" * 64, "options": {
                            "worker_python": sys.executable, "microbatch": 1,
                            "detector_microbatch": 1, "color_workers": 0}},
                        "ocr": {"checkpoint": "fixture", "revision": "a" * 64, "options": {
                            "microbatch": 1, "error_policy": "zero"}},
                    }})
    with pytest.raises(RuntimeError, match="next component failed"):
        image_factory.load_image_reward(reward, device="cpu")
    assert closed == [True]
