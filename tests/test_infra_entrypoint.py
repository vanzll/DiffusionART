import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from visual_rl.config import ConfigError, load_config, resume_signature
from visual_rl.data import PromptDataset
from visual_rl.train import main
from visual_rl.training.resume import prepare_resume_config, resolve_checkpoint

EXAMPLE = Path(__file__).resolve().parents[1] / "configs/examples/cpu_smoke.yaml"


def test_custom_data_auto_fingerprint_and_explicit_pin_stays_strict(tmp_path):
    manifest = tmp_path / "prompts.jsonl"
    manifest.write_text('{"sample_id":"one","prompt":"A mountain"}\n')
    sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    c = load_config(EXAMPLE, {"data": {"manifest": str(manifest)}})
    assert c.data.sha256 == sha
    assert PromptDataset.load(manifest, task="t2v").sha256 == sha
    manifest.write_text('{"sample_id":"one","prompt":"A lake"}\n')
    with pytest.raises(ValueError, match="hash mismatch"):
        PromptDataset.load(manifest, task="t2v", expected_sha256=c.data.sha256)


def test_i2v_custom_images_are_auto_identified(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"first")
    manifest = tmp_path / "prompts.jsonl"
    manifest.write_text(json.dumps({"sample_id": "one", "prompt": "Move", "image": "image.png"}) + "\n")
    source_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    first = PromptDataset.load(manifest, task="i2v", expected_sha256=source_sha)
    image.write_bytes(b"second")
    second = PromptDataset.load(manifest, task="i2v", expected_sha256=source_sha)
    assert first.sha256 != second.sha256


def test_optional_hash_environment_does_not_require_manual_sha(tmp_path, monkeypatch):
    monkeypatch.delenv("USER_DATA_SHA", raising=False)
    manifest = tmp_path / "prompts.jsonl"
    manifest.write_text('{"sample_id":"one","prompt":"A mountain"}\n')
    c = load_config(EXAMPLE, {"data": {"manifest": str(manifest), "sha256": "${USER_DATA_SHA}"}})
    assert c.data.sha256 == hashlib.sha256(manifest.read_bytes()).hexdigest()


@pytest.mark.parametrize("resume_kind", ["latest", "run", "checkpoints", "pointer"])
def test_resume_latest_and_extension_keep_optimizer_and_lineage(tmp_path, resume_kind):
    from visual_rl.backends.tensor_fixture import run

    root = tmp_path / "run"
    c = load_config(EXAMPLE, {"runtime": {"output_dir": str(root)}})
    run(c)
    changed = replace(c, training=replace(c.training, max_updates=6),
                      runtime=replace(c.runtime, checkpoint_keep=1, checkpoint_every=2))
    options = {"latest": "latest", "run": root, "checkpoints": root / "checkpoints",
               "pointer": root / "checkpoints/latest.json"}
    bound, receipt = prepare_resume_config(changed, options[resume_kind])
    assert bound.fingerprint() == c.fingerprint()
    assert receipt.name != "latest.json"
    run(changed, resume=options[resume_kind])
    assert json.loads((root / "status.json").read_text())["optimizer_steps"] == 6
    assert json.loads((root / "resume-config.json").read_text())["training"]["max_updates"] == 6
    rows = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
    assert [row["train/optimizer_step"] for row in rows] == list(range(1, 7))
    assert all(a["cost/elapsed_seconds"] <= b["cost/elapsed_seconds"] for a, b in zip(rows, rows[1:]))


def test_resume_rejects_scientific_changes_and_unsafe_latest(tmp_path):
    from visual_rl.backends.tensor_fixture import run

    c = load_config(EXAMPLE, {"runtime": {"output_dir": str(tmp_path / "run")}})
    run(c)
    changed = replace(c, training=replace(c.training, learning_rate=2e-5))
    with pytest.raises(ConfigError, match="different run/config"):
        prepare_resume_config(changed, "latest")
    pointer = tmp_path / "latest.json"
    pointer.write_text('{"manifest":"../outside.json"}')
    with pytest.raises(ConfigError, match="invalid latest"):
        resolve_checkpoint(pointer, tmp_path)


def test_logging_is_operational_but_update_geometry_is_not():
    c = load_config(EXAMPLE)
    changed = replace(c, runtime=replace(c.runtime, log="wandb", wandb={"mode": "offline"}))
    assert resume_signature(c.to_dict()) == resume_signature(changed.to_dict())
    changed = replace(c, training=replace(c.training, microbatch=2))
    assert resume_signature(c.to_dict()) != resume_signature(changed.to_dict())


def test_multinode_launcher_preserves_global_prompt_count(tmp_path, monkeypatch):
    from visual_rl import train

    monkeypatch.setattr(os, "environ", dict(os.environ))
    for key in ("RANK", "WORLD_SIZE"):
        monkeypatch.delenv(key, raising=False)
    c = load_config(EXAMPLE).to_dict()
    c["runtime"]["distributed"] = "ddp"
    c["sampling"]["prompts_per_rollout"] = 32
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(c))
    monkeypatch.setattr(train, "get_backend", lambda c: SimpleNamespace(load=lambda: pytest.fail("GPU load")))
    monkeypatch.setattr("visual_rl.prepare_distributed.prepare_shared_assets", lambda *a: None)
    commands = []

    def execute(executable, command):
        commands.append(command)
        raise SystemExit(0)

    monkeypatch.setattr(train.os, "execv", execute)
    with pytest.raises(SystemExit):
        main(["--config", str(path), "--nproc-per-node", "8", "--nnodes", "4",
              "--node-rank", "2", "--master-addr", "10.0.0.1", "--max-updates", "6"])
    assert "--nnodes=4" in commands[0] and "--node-rank=2" in commands[0]
    assert commands[0][-2:] == ["--max-updates", "6"]
    assert load_config(path).sampling.prompts_per_rollout == 32


def test_shared_preparation_reuses_generated_manifest(tmp_path, monkeypatch):
    from visual_rl.prepare_distributed import prepare_shared_assets

    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"assets": {"prompts": [{"env": "TRAIN_MANIFEST",
                       "sha_env": "TRAIN_SHA"}]}}))
    destination = tmp_path / "generated.jsonl"
    monkeypatch.setenv("VISUAL_RL_ASSET_ROOT", str(tmp_path / "cache"))
    monkeypatch.setenv("TRAIN_MANIFEST", str(destination))
    monkeypatch.delenv("TRAIN_SHA", raising=False)
    plan = ([], [({"env": "TRAIN_MANIFEST"}, destination)])
    calls = []

    def prepare(plan):
        calls.append(plan)
        destination.write_text("generated")
        os.environ["TRAIN_SHA"] = "recorded"

    prepare_shared_assets(config, plan, prepare)
    monkeypatch.delenv("TRAIN_SHA")
    prepare_shared_assets(config, plan, prepare)
    assert len(calls) == 2 and os.environ["TRAIN_SHA"] == "recorded"
    assert calls[1] == ([], plan[1])
    destination.unlink()
    monkeypatch.delenv("TRAIN_SHA")
    prepare_shared_assets(config, plan, prepare)
    assert len(calls) == 3


def test_resume_execution_digest_checks_active_budget(tmp_path):
    from visual_rl.backends.tensor_fixture import run

    c = load_config(EXAMPLE, {"runtime": {"output_dir": str(tmp_path / "run")}})
    run(c)
    extended = replace(c, training=replace(c.training, max_updates=6))
    bound, _ = prepare_resume_config(extended, "latest")
    assert bound.fingerprint() == c.fingerprint()
    assert bound.execution_fingerprint() != c.execution_fingerprint()
    assert replace(bound, runtime=replace(bound.runtime, checkpoint_keep=1)).execution_fingerprint() != bound.execution_fingerprint()


def test_failed_shared_preparation_is_not_cached(tmp_path, monkeypatch):
    from visual_rl.prepare_distributed import prepare_shared_assets

    config = tmp_path / "config.yaml"
    config.write_text("assets:\n  defaults:\n    MODEL_PATH: /missing\n")
    monkeypatch.setenv("VISUAL_RL_ASSET_ROOT", str(tmp_path / "cache"))

    def fail(plan):
        raise ConfigError("download failed")

    with pytest.raises(ConfigError, match="download failed"):
        prepare_shared_assets(config, ([], []), fail)
    assert not list((tmp_path / "cache/prepared").glob("*.json"))
