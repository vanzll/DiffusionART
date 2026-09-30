import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from visual_rl import wandb_export as module


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_tail_partial_line_and_evaluation_once(tmp_path):
    source = tmp_path / "metrics.jsonl"
    source.write_text('{"train/optimizer_step":1,"train/reward":0.3}\n{"train/optimizer_step":')
    reader = module.Records(tmp_path)
    assert list(reader.read()) == [{"train/optimizer_step": 1, "train/reward": .3}]
    assert list(reader.read()) == []
    with source.open("a") as stream:
        stream.write('2}\n')
    write_json(tmp_path / "evaluation/step-000000000/id/summary.json",
               {"eval/optimizer_step": 0, "eval/reward": .1, "manifest": "/local/path"})
    assert list(reader.read()) == [{"train/optimizer_step": 2},
                                   {"eval/optimizer_step": 0, "eval/reward": .1}]
    assert list(reader.read()) == []
    source.write_text("")
    with pytest.raises(ValueError, match="truncated"):
        list(reader.read())


@pytest.mark.parametrize("row", [[], {}, {"train/reward": float("nan")},
                                 {"train/reward": float("inf")}])
def test_invalid_scalars_rejected(row):
    with pytest.raises(ValueError):
        module.Records._scalars(row)


def test_jsonl_no_sdk_import_or_subprocess(tmp_path, monkeypatch):
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **k: pytest.fail("spawned SDK"))
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda *a: pytest.fail("import lookup"))
    with module.open_metrics(SimpleNamespace(runtime=SimpleNamespace(log="jsonl")), tmp_path) as log:
        log.write('{"train/optimizer_step":1}\n')
    assert list(module.Records(tmp_path).read()) == [{"train/optimizer_step": 1}]


def test_hung_export_is_killed_without_masking_trainer_error(tmp_path, monkeypatch):
    import io

    calls = []

    class Hung:
        pid = 1234
        stdin = io.BytesIO()

        def wait(self, *, timeout):
            calls.append(timeout)
            raise subprocess.TimeoutExpired("fake-export", timeout)

    def spawn(*args, **kwargs):
        assert kwargs["start_new_session"]
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        return Hung()

    monkeypatch.setattr(module.subprocess, "Popen", spawn)
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda *a: object())
    monkeypatch.setattr(module.os, "killpg", lambda pid, sig: calls.append((pid, sig)))
    with pytest.warns(RuntimeWarning, match="incomplete"), pytest.raises(ValueError, match="trainer"):
        with module.open_metrics(SimpleNamespace(runtime=SimpleNamespace(log="wandb")), tmp_path) as log:
            log.write('{"train/optimizer_step":1}\n')
            raise ValueError("trainer")
    assert calls == [10, (1234, module.signal.SIGKILL), 2]
    owner = json.loads(next((tmp_path / "wandb-exports").glob("*.owner.json")).read_text())
    assert owner["state"] == "pending_or_unknown"
    assert list(module.Records(tmp_path).read()) == [{"train/optimizer_step": 1}]


@pytest.mark.parametrize("failure", ["directory", "status"])
def test_export_only_filesystem_failure_keeps_training(tmp_path, monkeypatch, failure):
    import io

    class Finished:
        pid = 1234
        stdin = io.BytesIO()

        def wait(self, **kwargs):
            return 0

    monkeypatch.setattr(module.importlib.util, "find_spec", lambda *a: object())
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **k: Finished())
    if failure == "directory":
        mkdir = Path.mkdir

        def unavailable(path, *args, **kwargs):
            if path.name == "wandb-exports":
                raise OSError("export storage unavailable")
            return mkdir(path, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", unavailable)
    else:
        def unavailable(*args, **kwargs):
            raise OSError("export status unavailable")

        monkeypatch.setattr(module, "_publish", unavailable)
    with pytest.warns(RuntimeWarning):
        with module.open_metrics(SimpleNamespace(runtime=SimpleNamespace(log="wandb")), tmp_path) as log:
            log.write('{"train/optimizer_step":1}\n')
    assert list(module.Records(tmp_path).read()) == [{"train/optimizer_step": 1}]


def test_standalone_deadline_interrupts_hung_sdk_import(tmp_path):
    write_json(tmp_path / "resolved-config.json", {"name": "hung-sdk"})
    fake = tmp_path / "fake"
    fake.mkdir()
    (fake / "wandb.py").write_text("import time\ntime.sleep(120)\n")
    process = subprocess.Popen([sys.executable, "-m", "visual_rl.wandb_export", "--root", str(tmp_path),
        "--destination", str(tmp_path / "export"), "--timeout-seconds", ".2"],
        env={**os.environ, "PYTHONPATH": str(fake)}, start_new_session=True)
    try:
        assert process.wait(timeout=5) != 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_follow_eof_interrupts_hung_sdk_import(tmp_path):
    write_json(tmp_path / "resolved-config.json", {"name": "orphaned-sdk"})
    fake = tmp_path / "fake"
    fake.mkdir()
    (fake / "wandb.py").write_text("import time\ntime.sleep(120)\n")
    process = subprocess.Popen([sys.executable, "-m", "visual_rl.wandb_export", "--root", str(tmp_path),
        "--destination", str(tmp_path / "export"), "--follow-stdin"], stdin=subprocess.PIPE,
        env={**os.environ, "PYTHONPATH": str(fake)}, start_new_session=True)
    process.stdin.close()
    try:
        assert process.wait(timeout=15) != 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_deadline_remains_armed_during_exception_atexit(tmp_path):
    write_json(tmp_path / "resolved-config.json", {"name": "hung-atexit"})
    fake = tmp_path / "fake"
    fake.mkdir()
    (fake / "wandb.py").write_text(
        "import time, atexit\natexit.register(time.sleep, 120)\nraise RuntimeError('injected')\n")
    process = subprocess.Popen([sys.executable, "-m", "visual_rl.wandb_export", "--root", str(tmp_path),
        "--destination", str(tmp_path / "export"), "--timeout-seconds", ".2"],
        env={**os.environ, "PYTHONPATH": str(fake)}, start_new_session=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        assert process.wait(timeout=5) in (-module.signal.SIGKILL, 124)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_sdk_metric_contract(tmp_path, monkeypatch):
    import sys

    monkeypatch.setattr(module.os, "environ", dict(module.os.environ))
    write_json(tmp_path / "resolved-config.json", {"name": "metric-contract"})
    train = {"train/optimizer_step": 1, "train/reward": .8, "cost/elapsed_gpu_hours": 2.}
    evaluation = {"eval/optimizer_step": 0, "eval/reward": .2}
    (tmp_path / "metrics.jsonl").write_text(json.dumps(train) + "\n")
    write_json(tmp_path / "evaluation/step-000000000/id/summary.json", evaluation)
    logged, definitions, finished = [], [], []
    run = SimpleNamespace(id="test", url=None, dir="test", log=logged.append,
                          define_metric=lambda *a, **k: definitions.append((a, k)),
                          finish=lambda: finished.append(True))
    init_args = []

    def init(**kwargs):
        init_args.append(kwargs)
        for key in ("WANDB_DIR", "WANDB_CACHE_DIR", "WANDB_DATA_DIR", "TMPDIR"):
            assert Path(module.os.environ[key]).is_dir()
        return run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init, Settings=lambda **k: k))
    module.export(tmp_path, tmp_path / "export")
    assert logged == [train, evaluation]
    assert finished == [True]
    assert init_args[0]["name"] == init_args[0]["group"] == "metric-contract"
    assert init_args[0]["resume"] == "never"
    assert (("train/*",), {"step_metric": "train/optimizer_step", "step_sync": False}) in definitions
    assert (("eval/*",), {"step_metric": "eval/optimizer_step", "step_sync": False}) in definitions


def test_real_offline_sdk_subprocess_and_replay(tmp_path, monkeypatch):
    pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("WANDB_RUN_ID", "must-not-resume")
    write_json(tmp_path / "resolved-config.json", {"name": "offline-replay-test"})
    write_json(tmp_path / "evaluation/step-000000000/id/summary.json",
               {"eval/optimizer_step": 0, "eval/reward": .25})
    config = SimpleNamespace(runtime=SimpleNamespace(log="wandb"))
    for step in (1, 2):
        with module.open_metrics(config, tmp_path) as log:
            log.write(json.dumps({"train/optimizer_step": step, "train/reward": step * .5,
                                  "cost/elapsed_gpu_hours": step * .25}) + "\n")
    destinations = sorted((tmp_path / "wandb-exports").glob("*/status.json"))
    assert len(destinations) == 2
    states = [json.loads(path.read_text()) for path in destinations]
    assert {state["state"] for state in states} == {"finished"}
    assert len({state["run_id"] for state in states}) == 2
    assert all(state["run_id"] != "must-not-resume" for state in states)
    assert sorted(state["records"] for state in states) == [2, 3]
    for path, state in zip(destinations, states):
        files = list(path.parent.rglob("*.wandb"))
        assert len(files) == 1 and files[0].stat().st_size > 1000


def test_image_cli_with_real_offline_export(tmp_path, monkeypatch):
    pytest.importorskip("wandb")
    import yaml
    from test_sd3_runner import config_file, tiny_parts

    from visual_rl.backends import sd3
    from visual_rl.train import main

    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setattr(sd3, "load_components", tiny_parts)
    path = config_file(tmp_path)
    config = yaml.safe_load(path.read_text())
    config["runtime"]["log"] = "wandb"
    path.write_text(yaml.safe_dump(config))
    assert main(["--config", str(path)]) == 0
    root = Path(config["runtime"]["output_dir"])
    status = next((root / "wandb-exports").glob("*/status.json"))
    exported = json.loads(status.read_text())
    assert exported["state"] == "finished" and exported["records"] == 7
    history = list(module.Records(root).read())
    assert [r["train/optimizer_step"] for r in history if "train/optimizer_step" in r] == [1, 2, 3, 4]
    assert sorted(r["eval/optimizer_step"] for r in history if "eval/optimizer_step" in r) == [0, 2, 4]


@pytest.mark.parametrize("strategy", ["ddp", "fsdp"])
def test_distributed_export_has_one_owner_and_preserves_resume(tmp_path, monkeypatch, strategy):
    pytest.importorskip("wandb")
    import torch.multiprocessing as mp
    import yaml

    if strategy == "ddp":
        from test_image_ddp_runner import worker
        from test_sd3_runner import config_file
    else:
        from test_hunyuan_fsdp import worker
        from test_hunyuan_runner import config_file

    monkeypatch.setenv("WANDB_MODE", "offline")
    path = config_file(tmp_path)
    c = yaml.safe_load(path.read_text())
    c["runtime"].update(log="wandb", distributed=strategy)
    c["sampling"]["prompts_per_rollout"] = 2
    path.write_text(yaml.safe_dump(c))
    context = mp.spawn(worker, args=(str(path), str(tmp_path)), nprocs=2, join=False)
    deadline = time.monotonic() + 150
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("distributed logger integration exceeded 150s")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    root = Path(c["runtime"]["output_dir"])
    owners = list((root / "wandb-exports").glob("*.owner.json"))
    assert len(owners) == 1
    assert json.loads(owners[0].read_text())["state"] == "exited"
    status = next((root / "wandb-exports").glob("*/status.json"))
    assert json.loads(status.read_text())["state"] == "finished"
    assert [row["train/optimizer_step"] for row in module.Records(root).read()
            if "train/optimizer_step" in row] == [1, 2, 3, 4]
