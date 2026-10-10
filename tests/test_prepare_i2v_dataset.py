"""Fixed I2V dataset preparation with tiny CPU images, never FLUX downloads."""

import hashlib
import json
import os
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

from visual_rl.config import ConfigError, _read, load_config
from visual_rl.data import PromptDataset, file_sha256
from visual_rl.evaluation import prepare_i2v as module
from visual_rl.evaluation.package_i2v import package
from visual_rl.prepare_assets import plan_environment, validate_assets, validate_bindings


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    training = tmp_path / "train.txt"
    training.write_text("Snow falls on an alpine mountain.\nA chef serves hot tea.\nSnow falls on an alpine mountain.\n")
    rows = [{"sample_id": "test-id0", "prompt": "Purple butterflies circle a crystal sculpture."},
            {"sample_id": "test-id1", "prompt": "A silver robot walks between towering skyscrapers."}]
    evaluation = tmp_path / "test.jsonl"
    evaluation.write_bytes(module._row_bytes(rows))
    image = tmp_path / "original.jpg"
    Image.new("RGB", (640, 400), "green").save(image)
    original_body = {"schema_version": "wan-i2v-formal-validation/v1",
                     "source": {"training_json": "/private/training.json"},
                     "reference_model": module.MODEL, "generation": module.GENERATION,
                     "samples": [{"id": rows[0]["sample_id"], "prompt": rows[0]["prompt"],
                                  "image": str(image), "image_sha256": file_sha256(image), "image_seed": 312}]}
    checksum = hashlib.sha256(json.dumps(original_body, sort_keys=True, separators=(",", ":"),
                                         ensure_ascii=True).encode()).hexdigest()
    original = tmp_path / "original.json"
    original.write_text(json.dumps({**original_body, "checksum": checksum}))
    for name, value in {"TRAIN_COUNT": 3, "TEST_COUNT": 2, "PREFIX_COUNT": 1,
                        "TRAIN_SHA": file_sha256(training), "EVAL_SHA": file_sha256(evaluation),
                        "TRAIN_ROWS_SHA": hashlib.sha256(module._row_bytes([
                            {"sample_id": f"train:{i}", "prompt": text, "image_seed": 312 + i}
                            for i, text in enumerate(training.read_text().splitlines())
                        ])).hexdigest(),
                        "PREFIX_SHA": hashlib.sha256(module._row_bytes(rows[:1])).hexdigest(),
                        "ORIGINAL_SHA": file_sha256(original)}.items():
        monkeypatch.setattr(module, name, value)
    original_bundle = tmp_path / "original-bundle"
    package(original, original_bundle, expected_count=1)
    (original_bundle / "source-manifest.json").write_bytes(original.read_bytes())
    output = tmp_path / "dataset"
    module.create_plan(training, evaluation, output)
    calls = []

    def generator(prompt, seed):
        calls.append((prompt, seed))
        return Image.new("RGB", (640, 400), (seed % 255, 60, 80))

    return SimpleNamespace(output=output, training=training, evaluation=evaluation,
                           original=original_bundle, image=image, calls=calls, generator=generator)


def test_fixed_dataset_sharding_resume_and_unchanged_original(bundle):
    assert module.generate_shard(bundle.output, bundle.generator, rank=0, world_size=2,
                                 original_bundle=bundle.original) == 3
    with pytest.raises(ConfigError, match="incomplete"):
        module.finalize(bundle.output)
    assert module.generate_shard(bundle.output, bundle.generator, rank=1, world_size=2,
                                 original_bundle=bundle.original) == 2
    assert len(bundle.calls) == 4
    assert module.generate_shard(bundle.output, bundle.generator, original_bundle=bundle.original) == 0
    assert len(bundle.calls) == 4
    result = module.finalize(bundle.output)
    train = PromptDataset.load(bundle.output / "train/prompts.jsonl", task="i2v",
                               expected_sha256=result["splits"]["train"]["sha256"])
    test = PromptDataset.load(bundle.output / "test/prompts.jsonl", task="i2v")
    assert train.rows[0].text == train.rows[2].text
    assert [row.sample_id for row in train.rows] == ["train:0", "train:1", "train:2"]
    assert test.rows[0].image.read_bytes() == bundle.image.read_bytes()
    assert test.rows[0].metadata == {"image_seed": 312}
    assert "/private" not in (bundle.output / "provenance.json").read_text()
    assert str(bundle.output.parent) not in (bundle.output / "test/prompts.jsonl").read_text()
    assert module.finalize(bundle.output) == result


def test_plan_rejects_changed_source_prefix_and_overlap(bundle, monkeypatch, tmp_path):
    bundle.training.write_text("changed\n")
    with pytest.raises(ConfigError, match="pinned"):
        module.create_plan(bundle.training, bundle.evaluation, tmp_path / "changed")
    monkeypatch.setattr(module, "TRAIN_SHA", file_sha256(bundle.training))
    with pytest.raises(ConfigError, match="27815"):
        module.create_plan(bundle.training, bundle.evaluation, tmp_path / "count")
    monkeypatch.setattr(module, "TRAIN_COUNT", 1)
    monkeypatch.setattr(module, "PREFIX_SHA", "0" * 64)
    with pytest.raises(ConfigError, match="prefix"):
        module.create_plan(bundle.training, bundle.evaluation, tmp_path / "prefix")
    monkeypatch.setattr(module, "PREFIX_SHA", hashlib.sha256(bundle.evaluation.read_bytes().splitlines(keepends=True)[0]).hexdigest())
    bundle.training.write_text("Purple butterflies circle a crystal sculpture.\n")
    monkeypatch.setattr(module, "TRAIN_SHA", file_sha256(bundle.training))
    with pytest.raises(ConfigError, match="overlaps"):
        module.create_plan(bundle.training, bundle.evaluation, tmp_path / "overlap")


def test_original_and_receipt_corruption_fail_without_generation(bundle):
    bundle.image.write_bytes(b"corrupt")
    # The original bundle is a copy; corruption of an unrelated source does not alter it.
    module.generate_shard(bundle.output, bundle.generator, original_bundle=bundle.original)
    image, _ = module._paths(bundle.output, "test", 0)
    image.write_bytes(b"corrupt")
    with pytest.raises((ConfigError, OSError)):
        module.generate_shard(bundle.output, bundle.generator, original_bundle=bundle.original)
    assert len(bundle.calls) == 4


def test_unreceipted_regeneration_is_checked(bundle):
    image, _ = module._paths(bundle.output, "train", 0)
    image.parent.mkdir(parents=True)
    Image.new("RGB", (640, 400), "red").save(image)
    with pytest.raises(ConfigError, match="unreceipted"):
        module.generate_shard(bundle.output, bundle.generator, original_bundle=bundle.original)


def test_changed_plan_and_original_identity_rejected(bundle):
    path = bundle.output / "plan.json"
    plan = json.loads(path.read_text())
    plan["sources"]["train"]["private_path"] = "/private/data"
    path.write_text(json.dumps(plan))
    with pytest.raises(ConfigError, match="provenance"):
        module.finalize(bundle.output)
    plan["sources"]["train"].pop("private_path")
    plan["splits"]["test"][0]["prompt"] = "another prompt"
    path.write_text(json.dumps(plan))
    with pytest.raises(ConfigError, match="selection"):
        module.generate_shard(bundle.output, bundle.generator, original_bundle=bundle.original)
    assert not bundle.calls


def test_publish_only_complete_public_files_and_emit_real_pinned_recipe(bundle, monkeypatch, tmp_path):
    calls = []
    uploaded = []

    class Api:
        def create_repo(self, *args, **kwargs):
            calls.append((args, kwargs))

        def upload_large_folder(self, **kwargs):
            calls.append(kwargs)
            uploaded.extend(kwargs["allow_patterns"])

        def repo_info(self, **kwargs):
            return SimpleNamespace(sha="c" * 40)

        def list_repo_files(self, **kwargs):
            return uploaded

        def list_repo_tree(self, **kwargs):
            assert kwargs["revision"] == "c" * 40
            return [SimpleNamespace(path=name, lfs=None, blob_id=hashlib.sha1(
                f"blob {(bundle.output / name).stat().st_size}\0".encode()
                + (bundle.output / name).read_bytes()
            ).hexdigest()) for name in uploaded]

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=Api))
    with pytest.raises(ConfigError, match="incomplete"):
        module.publish(bundle.output, "account/dataset")
    assert not calls
    module.generate_shard(bundle.output, bundle.generator, original_bundle=bundle.original)
    extra = bundle.output / "train/images/0000/unrelated.jpg"
    extra.write_bytes(b"private unrelated image")
    result = module.publish(bundle.output, "account/dataset")
    assert "plan.json" not in calls[-1]["allow_patterns"]
    assert all("receipts" not in item for item in calls[-1]["allow_patterns"])
    assert "train/images/0000/unrelated.jpg" not in uploaded
    assert "original-source-manifest.json" not in uploaded
    recipe = tmp_path / "recipe.yaml"
    module.write_recipe("configs/paper/video/wan_art_11.yaml", recipe, result)
    config = _read(recipe)
    validate_assets(config["assets"])
    assert not any(row["env"] == "FLUX_REFERENCE_MODEL" for row in config["assets"]["snapshots"])
    assert config["model"]["options"]["reference_mode"] == "fixed_images"
    assert config["model"]["options"]["reference_checkpoint"] is None
    assert config["training"]["max_updates"] == 400
    assert config["sampling"]["samples_per_prompt"] == 8
    assert result["revision"] == "c" * 40
    assert json.loads((bundle.output / "publication.json").read_text()) == result
    with pytest.raises(ConfigError, match="overwrite"):
        module.write_recipe("configs/paper/video/wan_art_11.yaml", recipe, result)


def test_publish_failure_does_not_leak_credentials(bundle, monkeypatch):
    module.generate_shard(bundle.output, bundle.generator, original_bundle=bundle.original)

    class Api:
        def create_repo(self, *args, **kwargs):
            raise RuntimeError("hf_secret_token /private/path signed-url")

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=Api))
    with pytest.raises(ConfigError, match="RuntimeError") as error:
        module.publish(bundle.output, "account/dataset")
    assert "hf_secret_token" not in str(error.value)


@pytest.mark.parametrize("change", ["prompt", "order", "id"])
def test_training_plan_identity_is_independently_pinned(bundle, change):
    path = bundle.output / "plan.json"
    plan = json.loads(path.read_text())
    rows = plan["splits"]["train"]
    if change == "prompt":
        rows[0]["prompt"] = plan["splits"]["test"][0]["prompt"]
    elif change == "id":
        rows[0]["sample_id"] = "different"
    else:
        rows[0]["prompt"], rows[1]["prompt"] = rows[1]["prompt"], rows[0]["prompt"]
    path.write_text(json.dumps(plan))
    with pytest.raises(ConfigError, match="training selection"):
        module.generate_shard(bundle.output, bundle.generator, original_bundle=bundle.original)
    assert not bundle.calls


def test_original_hash_cannot_be_rebound_with_edited_bundle_metadata(bundle):
    path = bundle.original / "prompts.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    image = bundle.original / rows[0]["image"]
    Image.new("RGB", (640, 400), "red").save(image)
    rows[0]["image_sha256"] = file_sha256(image)
    path.write_bytes(module._row_bytes(rows))
    metadata = bundle.original / "provenance.json"
    provenance = json.loads(metadata.read_text())
    provenance["manifest_sha256"] = file_sha256(path)
    metadata.write_text(json.dumps(provenance))
    with pytest.raises(ConfigError, match="original eval100"):
        module.generate_shard(bundle.output, bundle.generator, original_bundle=bundle.original)
    assert not bundle.calls


def test_completed_prefix_must_match_historical_source(bundle):
    module.generate_shard(bundle.output, bundle.generator, original_bundle=bundle.original)
    image, receipt = module._paths(bundle.output, "test", 0)
    Image.new("RGB", (640, 400), "red").save(image)
    row = json.loads(receipt.read_text())
    row["image_sha256"] = file_sha256(image)
    receipt.write_text(json.dumps(row))
    with pytest.raises(ConfigError, match="original eval100 image bytes"):
        module.generate_shard(bundle.output, bundle.generator, original_bundle=bundle.original)
    with pytest.raises(ConfigError, match="original eval100 image bytes"):
        module.finalize(bundle.output)


@pytest.mark.parametrize("failure", ["extra", "changed", "missing"])
def test_publication_requires_exact_remote_revision_content(bundle, monkeypatch, failure):
    module.generate_shard(bundle.output, bundle.generator, original_bundle=bundle.original)
    uploaded = []

    class Api:
        def create_repo(self, *args, **kwargs):
            pass

        def list_repo_files(self, **kwargs):
            return ["private.txt"] if failure == "extra" else []

        def upload_large_folder(self, **kwargs):
            uploaded.extend(kwargs["allow_patterns"])

        def repo_info(self, **kwargs):
            return SimpleNamespace(sha="c" * 40)

        def list_repo_tree(self, **kwargs):
            files = [SimpleNamespace(path=name, lfs=SimpleNamespace(sha256=file_sha256(bundle.output / name)))
                     for name in uploaded]
            if failure == "changed":
                files[0].lfs.sha256 = "0" * 64
            elif failure == "missing":
                files.pop()
            return files

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=Api))
    with pytest.raises(ConfigError, match="publication failed"):
        module.publish(bundle.output, "account/dataset")
    assert not (bundle.output / "publication.json").exists()
    if failure == "extra":
        assert not uploaded


def test_sharded_image_directories_do_not_exceed_hub_limits(tmp_path):
    assert module._paths(tmp_path, "train", 999)[0].parent.name == "0000"
    assert module._paths(tmp_path, "train", 1000)[0].parent.name == "0001"


@pytest.mark.parametrize("profile", ["wan_art_11", "wan_nft_ema_11", "wan_fast_11", "wan_dance_11", "wan_refl_11"])
def test_generated_profile_plans_assets_and_preserves_algorithm(profile, bundle, tmp_path, monkeypatch):
    from visual_rl.backends.wan import validate_config

    monkeypatch.setattr(os, "environ", {})
    monkeypatch.setenv("DANCEGRPO_ROOT", "/installed/native")
    monkeypatch.setenv("VISUAL_RL_ASSET_ROOT", str(tmp_path / "assets"))
    base = f"configs/paper/video/{profile}.yaml"
    recipe = tmp_path / f"{profile}.yaml"
    module.write_recipe(base, recipe, {"repo_id": "account/dataset", "revision": "c" * 40,
                                     "splits": {"train": {"sha256": "d" * 64}, "test": {"sha256": "e" * 64}}})
    plan = plan_environment(recipe)
    c = load_config(recipe)
    validate_assets(c.assets)
    validate_bindings(c, plan)
    validate_config(c)
    assert c.algorithm.options == _read(module.Path(base))["algorithm"]["options"]
    assert c.training.max_updates == 400
    assert not any(e["env"] == "FLUX_REFERENCE_MODEL" for e, _ in plan[0])
    assert not any(e.get("action") == "fixed_images" for e, _ in plan[1])
    assert c.data.manifest.endswith("train/prompts.jsonl")
    assert c.evaluation.manifest.endswith("test/prompts.jsonl")
