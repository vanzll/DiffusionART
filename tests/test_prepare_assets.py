import hashlib
import io
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from visual_rl.config import ConfigError, load_config
from visual_rl.prepare_assets import (
    plan_environment,
    prepare_assets,
    validate_assets,
    validate_bindings,
)
from visual_rl.train import main


def manifest_bytes(source):
    return b"".join((json.dumps({"sample_id": f"train-{index:06d}", "prompt": line.strip()},
                              ensure_ascii=True, sort_keys=True) + "\n").encode()
                    for index, line in enumerate(source.decode().splitlines()))


def spec(source=b"one prompt\nsecond prompt\n"):
    return {"defaults": {"TEST_OUTPUT_ROOT": "outputs"},
            "snapshots": [{"env": "TEST_MODEL", "repo_id": "author/model", "revision": "a" * 40}],
            "prompts": [{"env": "TEST_PROMPTS", "sha_env": "TEST_PROMPTS_SHA256",
                         "url": "https://example.com/train.txt", "source_sha256": hashlib.sha256(source).hexdigest(),
                         "sha256": hashlib.sha256(manifest_bytes(source)).hexdigest(),
                         "prefix": "train", "format": "text"}]}


@pytest.fixture(autouse=True)
def environment(monkeypatch, tmp_path):
    # Isolate planner's environment assignments and never use real credentials.
    monkeypatch.setattr(os, "environ", dict(os.environ))
    for name in ("TEST_MODEL", "TEST_PROMPTS", "TEST_PROMPTS_SHA256", "TEST_OUTPUT_ROOT",
                 "RANK", "WORLD_SIZE", "WANDB_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VISUAL_RL_ASSET_ROOT", str(tmp_path / "assets"))


def write_config(tmp_path, assets=None):
    document = {"name": "prepare-test", "task": "t2i", "algorithm": {"name": "diffusionart"},
                "model": {"name": "test", "checkpoint": "${TEST_MODEL}", "revision": "a" * 40},
                "reward": {"name": "test", "weights": {"test": 1}}, "sampling": {"frames": 1},
                "data": {"manifest": "${TEST_PROMPTS}", "sha256": "${TEST_PROMPTS_SHA256}"},
                "runtime": {"distributed": "ddp"}, "assets": assets or spec()}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(document))
    return path


def test_planning_network_free_and_preserves_custom_bindings(tmp_path, monkeypatch):
    path = write_config(tmp_path)
    monkeypatch.setattr(Path, "mkdir", lambda *a, **k: pytest.fail("filesystem write"))
    monkeypatch.setenv("TEST_MODEL", "/user/model")
    snapshots, prompts = plan_environment(path)
    assert not snapshots and len(prompts) == 1
    c = load_config(path)
    assert c.model.checkpoint == "/user/model"
    assert c.runtime.distributed == "ddp"
    assert c.fingerprint() == load_config(path, {"assets": {}}).fingerprint()


def test_custom_prompt_pin_is_inferred_during_preparation(tmp_path, monkeypatch):
    manifest = tmp_path / "custom.jsonl"
    manifest.write_bytes(manifest_bytes(b"custom prompt\n"))
    monkeypatch.setenv("TEST_PROMPTS", str(manifest))
    monkeypatch.setenv("TEST_MODEL", "/custom/model")
    plan = plan_environment(write_config(tmp_path))
    assert "TEST_PROMPTS_SHA256" not in os.environ
    prepare_assets(plan)
    assert os.environ["TEST_PROMPTS_SHA256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()


def test_changed_revision_cannot_silently_load_a_different_snapshot(tmp_path):
    path = write_config(tmp_path)
    plan = plan_environment(path)
    validate_bindings(load_config(path), plan)
    with pytest.raises(ConfigError, match="revision differs"):
        validate_bindings(load_config(path, {"model": {"revision": "b" * 40}}), plan)


def test_download_conversion_cache_reuse_and_integrity(tmp_path, monkeypatch):
    source = b"one prompt\nsecond prompt\n"
    calls = []
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        snapshot_download=lambda **k: calls.append(k)))
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: io.BytesIO(source))
    plan = plan_environment(write_config(tmp_path))
    prepare_assets(plan)
    assert calls[0]["revision"] == "a" * 40
    assert calls[0]["repo_id"] == "author/model"
    destination = plan[1][0][1]
    assert destination.read_bytes() == manifest_bytes(source)
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: pytest.fail("redownloaded prompts"))
    prepare_assets(plan)
    destination.write_text("corruption")
    with pytest.raises(ConfigError, match="corrupted"):
        prepare_assets(plan)


def test_wrong_source_never_publishes_manifest(tmp_path, monkeypatch):
    plan = plan_environment(write_config(tmp_path))
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: io.BytesIO(b"wrong\n"))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=lambda **k: None))
    with pytest.raises(ConfigError, match="source SHA256 differs"):
        prepare_assets(plan)
    assert not plan[1][0][1].exists()


@pytest.mark.parametrize("change", [
    {"unknown": []}, {"defaults": {"WANDB_API_KEY": "not-allowed"}},
    {"snapshots": [{"env": "TEST_MODEL", "repo_id": "author/model", "revision": "main"}]},
])
def test_invalid_or_secret_asset_specs_rejected(change):
    with pytest.raises(ConfigError):
        validate_assets(change)


@pytest.mark.parametrize("resume_flags", [[], ["--resume", "receipt.json"],
    ["--resume", "receipt.json", "--resume-rewind"], ["--resume", "receipt.json", "--eval-only"]])
def test_launcher_prepares_once_and_execs_torchrun(tmp_path, monkeypatch, resume_flags):
    from visual_rl import train

    path = write_config(tmp_path)
    events = []
    monkeypatch.setattr(train, "get_backend", lambda c: SimpleNamespace(load=lambda: pytest.fail("GPU load")))
    monkeypatch.setattr("visual_rl.prepare_assets.prepare_assets", lambda plan: events.append("prepare"))

    def execute(executable, command):
        events.append(command)
        assert "--nproc-per-node=8" in command
        start = command.index("--output-dir")
        assert command[start:start + 2] == ["--output-dir", "custom-output"]
        if resume_flags:
            assert command[start + 2:] == resume_flags
        raise SystemExit(0)

    monkeypatch.setattr(train.os, "execv", execute)
    with pytest.raises(SystemExit) as exc:
        main(["--config", str(path), "--nproc-per-node", "8", "--output-dir", "custom-output",
              *resume_flags])
    assert exc.value.code == 0 and events[0] == "prepare" and len(events) == 2
    assert os.environ["TEST_PROMPTS_SHA256"] == spec()["prompts"][0]["sha256"]


def test_torchrun_workers_never_start_missing_asset_downloads(tmp_path, monkeypatch, capsys):
    from visual_rl import train

    path = write_config(tmp_path)
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setattr(train, "get_backend", lambda c: SimpleNamespace())
    monkeypatch.setattr("visual_rl.prepare_assets.prepare_assets", lambda plan: pytest.fail("download in worker"))
    assert main(["--config", str(path)]) == 2
    assert "prepare assets before torchrun" in capsys.readouterr().err


def test_print_only_and_invalid_backend_never_download(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("visual_rl.prepare_assets.prepare_assets", lambda plan: pytest.fail("download"))
    path = write_config(tmp_path)
    assert main(["--config", str(path), "--print-config"]) == 0
    assert json.loads(capsys.readouterr().out)["assets"]["snapshots"]
    assert main(["--config", str(path)]) == 2


def test_paper_pickscore_defaults_full_budget_and_no_download(tmp_path, monkeypatch):
    path = Path("configs/paper/image/sd3_pickscore_art.yaml")
    assets = yaml.safe_load(Path("configs/paper/pickscore-assets.yaml").read_text())["assets"]
    for entry in assets["snapshots"]:
        monkeypatch.delenv(entry["env"], raising=False)
    for entry in assets["prompts"]:
        monkeypatch.delenv(entry["env"], raising=False)
        monkeypatch.delenv(entry["sha_env"], raising=False)
    plan = plan_environment(path)
    c = load_config(path)
    assert len(plan[0]) == 3 and len(plan[1]) == 2
    assert c.training.max_updates == 1000 and c.runtime.log == "wandb"
    assert c.sampling.prompts_per_rollout == 64 and c.sampling.samples_per_prompt == 18
    assert c.evaluation.at_updates == (500, 800, 1000)


@pytest.mark.parametrize("mode, expects_flux", [("fixed_images", False), ("online_flux", True)])
def test_reference_mode_controls_inherited_flux_acquisition(tmp_path, monkeypatch, mode, expects_flux):
    monkeypatch.delenv("FLUX_REFERENCE_MODEL", raising=False)
    document = yaml.safe_load(write_config(tmp_path).read_text())
    document["model"]["options"] = {"reference_mode": mode}
    document["assets"]["snapshots"].append(
        {"env": "FLUX_REFERENCE_MODEL", "repo_id": "black-forest-labs/FLUX.1-dev", "revision": "b" * 40}
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(document))
    snapshots, _ = plan_environment(path)
    assert any(entry["env"] == "FLUX_REFERENCE_MODEL" for entry, _ in snapshots) == expects_flux
