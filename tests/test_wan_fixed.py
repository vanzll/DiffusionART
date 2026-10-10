"""Fixed Wan references opt in without changing the shared algorithm/RNG contracts."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import test_hunyuan_runner as video_test
import torch
import yaml
from PIL import Image
from test_wan_runner import components, condition, config_file, reference

from visual_rl.backends import wan
from visual_rl.config import ConfigError, load_config
from visual_rl.data import Prompt, PromptDataset, file_sha256
from visual_rl.models import wan_assets as assets

ALGORITHMS = ("diffusionart", "diffusionnft", "flow_grpo_fast", "dancegrpo", "refl")


def fixed_config(tmp_path, algorithm="diffusionart", references="omitted"):
    path = config_file(tmp_path, algorithm)
    values = yaml.safe_load(path.read_text())
    options = values["model"]["options"]
    options["reference_mode"] = "fixed_images"
    for key in wan.ASSET_KEYS[2:]:
        if references == "omitted":
            del options[key]
        elif references == "null":
            options[key] = None
    records = []
    for index in range(2):
        image = tmp_path / f"train-{index}.jpg"
        pixels = Image.new("RGB", (16, 16))
        pixels.putdata([(i % 256, index * 60 + i % 100, (i * 17) % 256) for i in range(256)])
        pixels.save(image, format="JPEG")
        records.append(dict(sample_id=str(index), prompt=f"scene {index}",
                            image=image.name, image_sha256=file_sha256(image)))
    manifest = tmp_path / "train.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in records))
    values["data"].update(manifest=str(manifest), sha256=file_sha256(manifest))
    if algorithm != "refl":
        values["sampling"]["samples_per_prompt"] = 8
    path.write_text(yaml.safe_dump(values))
    return load_config(path)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("references", ["omitted", "null", "present"])
def test_fixed_option_schema_and_hashed_datasets(tmp_path, algorithm, references):
    c = fixed_config(tmp_path, algorithm, references)
    before, identity = c.to_dict(), c.fingerprint()
    wan.validate_config(c)
    data, evaluation = wan.load_datasets(c)
    assert c.to_dict() == before and c.fingerprint() == identity
    assert data.sha256 == c.data.sha256
    assert all(row.image.suffix == ".jpg" and row.image_sha256 for row in data.rows)
    assert evaluation == PromptDataset.load(c.evaluation.manifest,
        expected_sha256=c.evaluation.manifest_sha256, task="i2v")


@pytest.mark.parametrize("mode", [None, "online_flux", "fixed_images"])
def test_strict_model_options_not_relaxed(tmp_path, mode):
    c = fixed_config(tmp_path) if mode == "fixed_images" else load_config(config_file(tmp_path))
    if mode is not None:
        c.model.options["reference_mode"] = mode
    c.model.options["unexpected"] = True
    with pytest.raises(ConfigError, match="requires native_root"):
        wan.validate_config(c)
    del c.model.options["unexpected"], c.model.options["base_revision"]
    with pytest.raises(ConfigError, match="requires native_root"):
        wan.validate_config(c)


@pytest.mark.parametrize("mode", [None, "fixed", "", 1, False, []])
def test_invalid_mode_rejected(tmp_path, mode):
    c = load_config(config_file(tmp_path))
    c.model.options["reference_mode"] = mode
    with pytest.raises(ConfigError, match="reference_mode"):
        wan.validate_config(c)


def test_fixed_requires_i2v_and_base_identities(tmp_path):
    c = fixed_config(tmp_path)
    with pytest.raises(ConfigError):
        wan.validate_config(replace(c, task="t2v"))
    c.model.options["base_checkpoint"] = None
    with pytest.raises(ConfigError, match="identities required"):
        wan.validate_config(c)


@pytest.mark.parametrize("value", ["", False, 1, {}])
def test_fixed_optional_flux_identities_have_strict_types(tmp_path, value):
    c = fixed_config(tmp_path)
    c.model.options["reference_checkpoint"] = value
    with pytest.raises(ConfigError, match="nonempty strings or null"):
        wan.validate_config(c)


@pytest.mark.parametrize("mode", [None, "online_flux"])
@pytest.mark.parametrize("key", wan.ASSET_KEYS[2:])
def test_online_still_requires_flux_identities(tmp_path, mode, key):
    c = load_config(config_file(tmp_path))
    if mode is not None:
        c.model.options["reference_mode"] = mode
    c.model.options[key] = None
    with pytest.raises(ConfigError, match="identities required"):
        wan.validate_config(c)
    del c.model.options[key]
    with pytest.raises(ConfigError, match="requires native_root"):
        wan.validate_config(c)


@pytest.mark.parametrize("problem", ["text_only", "no_hash", "bad_hash", "missing", "corrupt"])
def test_fixed_training_requires_valid_pinned_images(tmp_path, problem, monkeypatch):
    c = fixed_config(tmp_path)
    manifest = tmp_path / "train.jsonl"
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    row = records[0]
    if problem == "text_only":
        del row["image"], row["image_sha256"]
    elif problem == "no_hash":
        del row["image_sha256"]
    elif problem == "bad_hash":
        row["image_sha256"] = "0" * 64
    elif problem == "missing":
        (tmp_path / row["image"]).unlink()
    else:
        (tmp_path / row["image"]).write_bytes(b"not an image")
        row["image_sha256"] = file_sha256(tmp_path / row["image"])
    manifest.write_text("".join(json.dumps(row) + "\n" for row in records))
    c = replace(c, data=replace(c.data, sha256=file_sha256(manifest)))
    monkeypatch.setattr(assets, "generate_reference", lambda *a, **kw: pytest.fail("FLUX fallback"))
    with pytest.raises((ValueError, OSError)):
        wan.load_datasets(c)


def prediction_spy(monkeypatch):
    calls, selected = [], []

    def encode(pipe, prompt, image, **kwargs):
        calls.append((prompt, image.tobytes(), kwargs.copy()))
        return condition(pipe, prompt, image, **kwargs)

    def predict(state, times, conditioning):
        selected.append(conditioning)
        return state

    monkeypatch.setattr(assets, "condition_image", encode)
    parts = SimpleNamespace(pipe=object(), flux=object(),
                            adapter=SimpleNamespace(predict_velocity=predict))
    return parts, calls, selected


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_fixed_k_repetitions_share_decoded_source_and_preserve_rng(tmp_path, monkeypatch, algorithm):
    c = fixed_config(tmp_path, algorithm)
    wan.validate_config(c)
    data, _ = wan.load_datasets(c)
    parts, calls, selected = prediction_spy(monkeypatch)
    monkeypatch.setattr(assets, "generate_reference", lambda *a, **kw: pytest.fail("FLUX called"))
    rank, attempt, repeats = 2, 7, c.sampling.samples_per_prompt
    online = replace(c, model=replace(c.model, options={**c.model.options, "reference_mode": "online_flux"}))
    baseline = wan.training_inputs(online, attempt, rank)
    before = torch.get_rng_state().clone()
    for current_attempt in (attempt, attempt + 1):
        predict = wan._predict(c, parts, data.rows, c.data.manifest, repeats, torch.device("cpu"),
                               rank=rank, attempt=current_attempt)
        predict(torch.zeros(2 * repeats, 1), torch.ones(2 * repeats), torch.arange(2 * repeats))
    assert torch.equal(before, torch.get_rng_state())
    for actual, expected in zip(wan.training_inputs(c, attempt, rank), baseline):
        assert actual is expected is None or torch.equal(actual, expected)
    assert len(calls) == 4  # once per prompt/rollout, not once per trajectory
    for index, row in enumerate(data.rows):
        with Image.open(row.image) as source:
            expected = source.convert("RGB").tobytes()
        assert calls[index][1] == calls[index + 2][1] == expected
        assert calls[index][2]["seed"] == assets.scoped_seed(c.seed, rank, attempt,
            "image_vae" if index == 0 else f"image_vae:{index}")
        assert calls[index + 2][2]["seed"] == assets.scoped_seed(c.seed, rank, attempt + 1,
            "image_vae" if index == 0 else f"image_vae:{index}")
        assert calls[index][2]["cfg"] is False
        for conditioning in selected:
            for name in ("prompt_embeds", "image_embeds", "image_condition"):
                group = getattr(conditioning, name)[index * repeats:(index + 1) * repeats]
                assert torch.equal(group, group[:1].expand_as(group))
    torch.testing.assert_close(selected[0].image_condition, selected[1].image_condition, rtol=0, atol=0)


@pytest.mark.parametrize("problem", ["no_image", "no_hash", "changed", "missing"])
def test_fixed_predict_fails_without_online_fallback(tmp_path, monkeypatch, problem):
    c = fixed_config(tmp_path)
    data, _ = wan.load_datasets(c)
    row = data.rows[0]
    if problem == "no_image":
        row = replace(row, image=None)
    elif problem == "no_hash":
        row = replace(row, image_sha256=None)
    elif problem == "changed":
        Image.new("RGB", (16, 16), "red").save(row.image)
    else:
        row.image.unlink()
    monkeypatch.setattr(assets, "generate_reference", lambda *a, **kw: pytest.fail("FLUX fallback"))
    monkeypatch.setattr(assets, "condition_image", lambda *a, **kw: pytest.fail("invalid image conditioned"))
    with pytest.raises((ValueError, OSError)):
        wan._predict(c, SimpleNamespace(), (row,), c.data.manifest, 8, torch.device("cpu"))


@pytest.mark.parametrize("mode", [None, "online_flux"])
def test_online_default_behavior_seeds_and_fingerprint_unchanged(tmp_path, monkeypatch, mode):
    c = load_config(config_file(tmp_path))
    if mode is not None:
        c.model.options["reference_mode"] = mode
    before, identity = c.to_dict(), c.fingerprint()
    wan.validate_config(c)
    data, _ = wan.load_datasets(c)
    assert data == PromptDataset.load(c.data.manifest, expected_sha256=c.data.sha256, task="t2v")
    parts, calls, _ = prediction_spy(monkeypatch)
    generated = []

    def generate(flux, prompt, **kwargs):
        assert flux is parts.flux
        generated.append((prompt, kwargs.copy()))
        return reference(flux, prompt, **kwargs)

    monkeypatch.setattr(assets, "generate_reference", generate)
    for attempt in (7, 8):
        wan._predict(c, parts, data.rows, c.data.manifest, 8, torch.device("cpu"), rank=2, attempt=attempt)
    assert len(generated) == len(calls) == 4
    for position, (prompt, kwargs) in enumerate(generated):
        index, attempt = position % 2, 7 + position // 2
        assert prompt == data.rows[index].text
        assert kwargs == dict(height=16, width=16, device=torch.device("cpu"),
            seed=assets.scoped_seed(c.seed, 2, attempt, "flux" if index == 0 else f"flux:{index}"))
        assert calls[position][1] == reference(parts.flux, prompt, **kwargs).tobytes()
        assert calls[position][2]["seed"] == assets.scoped_seed(c.seed, 2, attempt,
            "image_vae" if index == 0 else f"image_vae:{index}")
    assert c.to_dict() == before and c.fingerprint() == identity
    assert ("reference_mode" in c.model.options) == (mode is not None)


@pytest.mark.parametrize("mode", [None, "online_flux", "fixed_images"])
def test_evaluation_remains_fixed_in_every_mode(tmp_path, monkeypatch, mode):
    c = fixed_config(tmp_path) if mode == "fixed_images" else load_config(config_file(tmp_path))
    if mode is not None:
        c.model.options["reference_mode"] = mode
    _, evaluation = wan.load_datasets(c)
    parts, calls, _ = prediction_spy(monkeypatch)
    monkeypatch.setattr(assets, "generate_reference", lambda *a, **kw: pytest.fail("eval FLUX called"))
    wan._predict(c, parts, evaluation.rows, c.evaluation.manifest, 1, torch.device("cpu"), evaluation=True)
    for call, row in zip(calls, evaluation.rows):
        with Image.open(row.image) as source:
            assert call[1] == source.convert("RGB").tobytes()
    with pytest.raises(ValueError, match="evaluation requires a fixed reference"):
        wan._predict(c, parts, (Prompt("missing", "scene"),), c.evaluation.manifest,
                     1, torch.device("cpu"), evaluation=True)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_fixed_cli_training_eval_and_exact_resume(tmp_path, monkeypatch, algorithm):
    def config(directory, name):
        fixed_config(directory, name)
        return directory / "config.yaml"

    monkeypatch.setattr(wan, "load_components", components)
    monkeypatch.setattr(assets, "generate_reference", lambda *a, **kw: pytest.fail("FLUX called"))
    monkeypatch.setattr(assets, "condition_image", condition)
    monkeypatch.setattr(video_test, "config_file", config)
    video_test.test_native_video_cli_encoding_eval_and_exact_resume(tmp_path, monkeypatch, algorithm)
