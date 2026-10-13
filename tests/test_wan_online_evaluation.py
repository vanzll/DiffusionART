import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from threading import Barrier

import pytest
import torch
from PIL import Image
from test_wan_fixed import fixed_config, prediction_spy
from test_wan_runner import config_file

from visual_rl.backends import wan
from visual_rl.config import ConfigError, load_config
from visual_rl.data import Prompt, file_sha256
from visual_rl.evaluation import wan_references as references
from visual_rl.models import wan_assets as assets


def online_config(tmp_path, algorithm="diffusionart"):
    c = load_config(config_file(tmp_path, algorithm))
    c.model.options["evaluation_reference_mode"] = "online_flux"
    manifest = tmp_path / "text-eval.jsonl"
    manifest.write_text(''.join(json.dumps({"sample_id": str(i), "prompt": f"scene {i}"}) + "\n"
                                for i in range(2)))
    return replace(c, evaluation=replace(c.evaluation, manifest=str(manifest),
                                         manifest_sha256=file_sha256(manifest)))


@pytest.fixture
def generated(monkeypatch):
    calls = []

    def generate(flux, prompt, *, height, width, device, seed):
        image = Image.new("RGB", (width, height))
        image.putdata([((seed + i * 17) % 256, i % 256, (i * 31) % 256)
                       for i in range(width * height)])
        with BytesIO() as buffer:
            image.save(buffer, format="JPEG")
            buffer.seek(0)
            with Image.open(buffer) as source:
                decoded = source.convert("RGB")
        calls.append((prompt, seed, decoded.tobytes()))
        return decoded

    monkeypatch.setattr(assets, "generate_reference", generate)
    return calls


def cache_root(c):
    return Path(c.runtime.output_dir) / "evaluation" / "wan_references"


def test_opt_in_text_dataset_and_lazy_generation(tmp_path, generated):
    c = online_config(tmp_path)
    before = c.to_dict(), c.fingerprint()
    wan.validate_config(c)
    data, evaluation = wan.load_datasets(c)
    assert data.rows and all(row.image is None for row in evaluation.rows)
    assert not generated and not cache_root(c).exists()
    assert (c.to_dict(), c.fingerprint()) == before
    del c.model.options["evaluation_reference_mode"]
    with pytest.raises(ValueError, match="I2V rows require an image"):
        wan.load_datasets(c)
    c.model.options["evaluation_reference_mode"] = "fixed_images"
    wan.validate_config(c)
    with pytest.raises(ValueError, match="I2V rows require an image"):
        wan.load_datasets(c)


@pytest.mark.parametrize("mode", [None, "", "online", False, 1, [], {}])
def test_invalid_evaluation_mode(tmp_path, mode):
    c = online_config(tmp_path)
    c.model.options["evaluation_reference_mode"] = mode
    with pytest.raises(ConfigError, match="evaluation_reference_mode"):
        wan.validate_config(c)


def test_fixed_training_online_evaluation_fails_clearly(tmp_path):
    c = fixed_config(tmp_path)
    c.model.options["evaluation_reference_mode"] = "online_flux"
    with pytest.raises(ConfigError, match="requires reference_mode=online_flux"):
        wan.validate_config(c)
    with pytest.raises(ValueError, match="requires reference_mode=online_flux"):
        wan.load_datasets(c)


@pytest.mark.parametrize("path", [None, "", "  ", False, 1, [], {}])
def test_invalid_cache_path(tmp_path, path):
    c = online_config(tmp_path)
    c.model.options["evaluation_reference_cache"] = path
    with pytest.raises(ConfigError, match="evaluation_reference_cache must be a nonempty string"):
        wan.validate_config(c)


def test_shared_cache_across_algorithms_and_run_directories(tmp_path, monkeypatch, generated):
    shared = tmp_path / "shared-references"
    for algorithm in ("diffusionart", "diffusionnft", "flow_grpo_fast", "dancegrpo", "refl"):
        run = tmp_path / algorithm
        run.mkdir()
        c = online_config(run, algorithm)
        c.model.options["evaluation_reference_cache"] = str(shared)
        wan.validate_config(c)
        _, dataset = wan.load_datasets(c)
        parts, conditioned, _ = prediction_spy(monkeypatch)
        wan._predict(c, parts, dataset.rows[::-1], c.evaluation.manifest, 1, torch.device("cpu"),
                     evaluation=True, attempt=100, rank=2)
        assert len(generated) == 2
        pixels = {call[0]: call[2] for call in generated}
        assert all(call[1] == pixels[call[0]] for call in conditioned)
        assert not cache_root(c).exists()
    assert len(list(shared.glob("*/reference.png"))) == 2


@pytest.mark.parametrize("mode", [None, "fixed_images"])
def test_historical_evaluation_unchanged(tmp_path, monkeypatch, mode):
    c = load_config(config_file(tmp_path))
    if mode is not None:
        c.model.options["evaluation_reference_mode"] = mode
    wan.validate_config(c)
    _, dataset = wan.load_datasets(c)
    parts, conditioned, _ = prediction_spy(monkeypatch)
    monkeypatch.setattr(assets, "generate_reference", lambda *a, **kw: pytest.fail("unexpected FLUX"))
    wan._predict(c, parts, dataset.rows, c.evaluation.manifest, 1, torch.device("cpu"), evaluation=True)
    for row, call in zip(dataset.rows, conditioned):
        with Image.open(row.image) as image:
            assert call[1] == image.convert("RGB").tobytes()
    assert not cache_root(c).exists()


def test_repeat_and_resume_reuse_exact_png_bytes(tmp_path, monkeypatch, generated):
    c = online_config(tmp_path)
    _, dataset = wan.load_datasets(c)
    parts, conditioned, _ = prediction_spy(monkeypatch)
    before = torch.get_rng_state().clone()
    wan._predict(c, parts, dataset.rows, c.evaluation.manifest, 3, torch.device("cpu"),
                 evaluation=True, attempt=7, rank=2)
    assert len(generated) == 2
    assert [call[1] for call in conditioned] == [call[2] for call in generated]
    saved = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in cache_root(c).glob("*/*")}
    assert len(saved) == 4
    monkeypatch.setattr(assets, "generate_reference", lambda *a, **kw: pytest.fail("cache miss on resume"))
    resumed = online_config(tmp_path)  # Fresh config and components; no in-memory cache.
    parts, conditioned, _ = prediction_spy(monkeypatch)
    for rows, attempt, rank in ((dataset.rows[::-1], 90, 0), (dataset.rows, 0, 4)):
        wan._predict(resumed, parts, rows, c.evaluation.manifest, 1, torch.device("cpu"),
                     evaluation=True, attempt=attempt, rank=rank)
    pixels = {call[0]: call[2] for call in generated}
    assert all(call[1] == pixels[call[0]] for call in conditioned)
    assert all((path.read_bytes(), path.stat().st_mtime_ns) == value for path, value in saved.items())
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("seed", [None, 0, 937])
def test_cold_cache_seeds_ignore_order_rank_and_step(tmp_path, monkeypatch, generated, seed):
    c = online_config(tmp_path)
    c = replace(c, evaluation=replace(c.evaluation, seed=seed))
    _, dataset = wan.load_datasets(c)
    parts, _, _ = prediction_spy(monkeypatch)
    for index, rows in enumerate((dataset.rows, dataset.rows[::-1])):
        current = replace(c, runtime=replace(c.runtime, output_dir=str(tmp_path / f"run-{index}")))
        wan._predict(current, parts, rows, c.evaluation.manifest, 1, torch.device("cpu"),
                     evaluation=True, attempt=index * 100, rank=index * 3)
    assert {p: (s, b) for p, s, b in generated[:2]} == {p: (s, b) for p, s, b in generated[2:]}
    assert generated[0][1] != generated[1][1]
    identity = references._identity(c, dataset.rows[0], "cpu")
    assert identity["evaluation_seed"] == (c.seed if seed is None else seed)
    fallback = replace(c, evaluation=replace(c.evaluation, seed=None))
    explicit = replace(c, evaluation=replace(c.evaluation, seed=c.seed))
    assert references._identity(fallback, dataset.rows[0], "cpu") == references._identity(explicit, dataset.rows[0], "cpu")


@pytest.mark.parametrize("field", ["prompt", "sample_id", "seed", "revision", "checkpoint", "geometry", "tf32", "versions"])
def test_identity_changes_invalidate_cache(tmp_path, monkeypatch, generated, field):
    c = online_config(tmp_path)
    row = Prompt("id", "scene")
    references.cached_reference(c, row, None, "cpu")
    if field == "prompt":
        row = replace(row, text="different scene")
    elif field == "sample_id":
        row = replace(row, sample_id="other")
    elif field == "seed":
        c = replace(c, evaluation=replace(c.evaluation, seed=c.seed + 1))
    elif field in ("revision", "checkpoint"):
        c.model.options[f"reference_{field}"] = "changed"
    elif field == "geometry":
        c = replace(c, evaluation=replace(c.evaluation, width=32))
    elif field == "tf32":
        c.model.options["allow_tf32"] = not c.model.options["allow_tf32"]
    else:
        monkeypatch.setattr(references, "_runtime_versions", lambda: {"test": "changed"})
    references.cached_reference(c, row, None, "cpu")
    assert len(generated) == 2
    assert len(list(cache_root(c).glob("*/metadata.json"))) == 2


@pytest.mark.parametrize("corruption", ["missing_png", "missing_metadata", "bad_json", "identity", "hash",
                                        "truncated", "not_image", "jpeg", "size", "mode"])
def test_corrupt_cache_rejected_without_regeneration(tmp_path, monkeypatch, generated, corruption):
    c, row = online_config(tmp_path), Prompt("id", "scene")
    references.cached_reference(c, row, None, "cpu")
    directory = next(cache_root(c).iterdir())
    png, metadata_path = directory / "reference.png", directory / "metadata.json"
    metadata = json.loads(metadata_path.read_bytes())
    if corruption == "missing_png":
        png.unlink()
    elif corruption == "missing_metadata":
        metadata_path.unlink()
    elif corruption == "bad_json":
        metadata_path.write_text("{")
    elif corruption == "identity":
        metadata["identity"]["prompt"] = "tampered"
        metadata_path.write_text(json.dumps(metadata))
    elif corruption == "hash":
        png.write_bytes(png.read_bytes() + b"changed")
    else:
        if corruption == "truncated":
            png.write_bytes(png.read_bytes()[:50])
        elif corruption == "not_image":
            png.write_bytes(b"not an image")
        else:
            Image.new("L" if corruption == "mode" else "RGB",
                      (32, 16) if corruption == "size" else (16, 16)).save(
                          png, format="JPEG" if corruption == "jpeg" else "PNG")
        metadata["png_sha256"] = file_sha256(png)
        metadata_path.write_text(json.dumps(metadata))
    monkeypatch.setattr(assets, "generate_reference", lambda *a, **kw: pytest.fail("corrupt cache regenerated"))
    with pytest.raises(ValueError, match="invalid Wan evaluation reference cache"):
        references.cached_reference(c, row, None, "cpu")


def test_training_reference_still_online_and_attempt_scoped(tmp_path, monkeypatch, generated):
    c = online_config(tmp_path)
    data, _ = wan.load_datasets(c)
    parts, _, _ = prediction_spy(monkeypatch)
    for attempt in (1, 2):
        wan._predict(c, parts, data.rows, c.data.manifest, 1, torch.device("cpu"), attempt=attempt, rank=3)
    assert len(generated) == 4 and not cache_root(c).exists()
    for index, call in enumerate(generated):
        stream = "flux" if index % 2 == 0 else "flux:1"
        assert call[1] == assets.scoped_seed(c.seed, 3, index // 2 + 1, stream)


def test_metadata_contains_only_explicit_identity(tmp_path, monkeypatch, generated):
    monkeypatch.setenv("HF_TOKEN", "test-secret-never-persist")
    c = online_config(tmp_path)
    references.cached_reference(c, Prompt("id", "scene"), None, "cpu")
    metadata = next(cache_root(c).glob("*/metadata.json")).read_text()
    assert "test-secret-never-persist" not in metadata and "HF_TOKEN" not in metadata
    record = json.loads(metadata)
    assert record["identity"]["generator"]["num_inference_steps"] == 30
    assert "Pillow" in record["identity"]["runtime_versions"]
    assert record["png_sha256"] == hashlib.sha256(next(cache_root(c).glob("*/reference.png")).read_bytes()).hexdigest()


def test_concurrent_publishers_reuse_winning_bytes(tmp_path, monkeypatch):
    c, row = online_config(tmp_path), Prompt("id", "scene")
    barrier = Barrier(2, timeout=5)

    def generate(*args, **kwargs):
        index = barrier.wait()
        return Image.new("RGB", (16, 16), (index * 100, 20, 30))

    monkeypatch.setattr(assets, "generate_reference", generate)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(references.cached_reference, c, row, None, "cpu") for _ in range(2)]
        images = [future.result(timeout=10) for future in futures]
    assert images[0].tobytes() == images[1].tobytes()
    assert len(list(cache_root(c).iterdir())) == 1
    assert references.cached_reference(c, row, None, "cpu").tobytes() == images[0].tobytes()


def test_failed_generation_does_not_publish_cache(tmp_path, monkeypatch):
    c, row = online_config(tmp_path), Prompt("id", "scene")

    def fail(*args, **kwargs):
        raise RuntimeError("injected generation failure")

    monkeypatch.setattr(assets, "generate_reference", fail)
    with pytest.raises(RuntimeError, match="injected generation failure"):
        references.cached_reference(c, row, None, "cpu")
    assert not list(cache_root(c).iterdir())
