"""Scalar zero-signal accounting through the native optimizer and collectives."""

import copy
import json
import os
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp
import yaml
from test_distributed_checkpoint import assert_tree_equal

from visual_rl.backends import hunyuan, hunyuan_fsdp, image_ddp, sd3
from visual_rl.config import ConfigError, load_config
from visual_rl.rewards.scalar import weighted_score
from visual_rl.train import main
from visual_rl.training.rollout import collect_rollout


def install_checks(backend, family, rank, world):
    calls, updates, saved = [0], [0], {}
    seen_windows = []
    original_step = backend.accumulated_step

    def step(model, optimizer, items, objective, **kwargs):
        assert kwargs["denominator"] == len(items)
        assert not objective.row_anchors.any()
        assert len(items) == len(objective.cache.endpoints) // 2 * len(objective.cache.selected)
        seen_windows.append(set(items))
        if len(seen_windows) == 2:
            assert not seen_windows[0] & seen_windows[1]
            assert seen_windows[0] | seen_windows[1] == {
                (row, slot) for row in range(len(objective.cache.endpoints))
                for slot in range(len(objective.cache.selected))}
            seen_windows.clear()
        if calls[0] == 3:
            assert objective.has_guidance is (world == 1 or rank != 0)
        result = original_step(model, optimizer, items, objective, **kwargs)
        updates[0] += 1
        if updates[0] == 2:
            saved["optimizer"] = optimizer
            saved["parameters"] = list(model.parameters())
        return result

    def scores(count):
        calls[0] += 1
        if calls[0] == 2:
            saved["state"] = copy.deepcopy(saved["optimizer"].state_dict())
            saved["weights"] = [p.detach().clone() for p in saved["parameters"]]
            assert saved["state"]["state"]  # Skip after Adam has acquired momentum.
        if calls[0] == 3:
            assert updates[0] == 2
            assert_tree_equal(saved["state"], saved["optimizer"].state_dict())
            assert_tree_equal(saved["weights"], [p.detach().clone() for p in saved["parameters"]])
        values = torch.arange(count, dtype=torch.float32)
        if calls[0] == 2 or (calls[0] == 3 and world > 1 and rank == 0):
            values.fill_(.1)
        elif calls[0] == 3 and world == 1:
            values[:2] = .1
        key = "pickscore" if family in ("image", "flux") else "VQ"
        return weighted_score({key: values}, {key: 1.}, space="raw")

    backend.accumulated_step = step
    if family in ("video", "wan"):
        hunyuan.score_training_videos = lambda _b, _c, _p, x, *_args: scores(len(x))
    else:
        original_score = sd3._score

        def score(*args, **kwargs):
            result = original_score(*args, **kwargs)
            return replace(result, scores=scores(len(args[2])))

        sd3._score = score
    return calls, updates


def configured(tmp_path, family, distributed):
    if family == "flux":
        from test_image_ddp_runner import configured as image_config

        path = image_config(tmp_path, "diffusionart", "flux", "bf16")
    elif family == "wan":
        from test_wan_runner import config_file

        path = config_file(tmp_path)
    elif family == "image":
        from test_sd3_runner import config_file

        path = config_file(tmp_path)
    else:
        from test_hunyuan_runner import config_file

        path = config_file(tmp_path)
    values = yaml.safe_load(path.read_text())
    values["runtime"]["distributed"] = distributed
    values["evaluation"]["every_updates"] = 0
    values["evaluation"].pop("manifest")
    values["evaluation"].pop("manifest_sha256")
    values["sampling"]["prompts_per_rollout"] = 2
    path.write_text(yaml.safe_dump(values))
    return path


def assert_accounting(root):
    status = json.loads((root / "status.json").read_text())
    assert status["optimizer_steps"] == 4
    assert status["attempts"] == 3 and status["completed_rollouts"] == 2
    records = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
    assert [row["train/optimizer_step"] for row in records] == [1, 2, 2, 3, 4]
    assert records[2]["system/skipped_zero_guidance"] == 1
    assert records[2]["cost/training_seconds"] > records[1]["cost/training_seconds"]


@pytest.mark.parametrize("family", ["image", "flux", "video", "wan"])
def test_single_rank_scalar_zero_skip_preserves_adam(tmp_path, monkeypatch, family):
    from test_hunyuan_runner import components
    from test_sd3_runner import tiny_parts

    path = configured(tmp_path, family, "none")
    backend = sd3 if family in ("image", "flux") else hunyuan
    if family == "wan":
        from test_wan_runner import components as wan_parts
        from test_wan_runner import condition, reference

        from visual_rl.backends import wan
        from visual_rl.models import wan_assets

        monkeypatch.setattr(wan, "load_components", wan_parts)
        monkeypatch.setattr(wan_assets, "generate_reference", reference)
        monkeypatch.setattr(wan_assets, "condition_image", condition)
    elif family == "flux":
        from test_image_ddp_runner import parts_for_model

        monkeypatch.setattr(sd3, "load_components", parts_for_model)
    else:
        monkeypatch.setattr(backend, "load_components", tiny_parts if family == "image" else components)
    # The installer changes module attributes; register restoration with pytest.
    monkeypatch.setattr(backend, "accumulated_step", backend.accumulated_step)
    monkeypatch.setattr(sd3, "_score", sd3._score)
    monkeypatch.setattr(hunyuan, "score_training_videos", hunyuan.score_training_videos)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    calls, updates = install_checks(backend, family, 0, 1)
    try:
        assert main(["--config", str(path)]) == 0
    finally:
        torch.set_num_threads(threads)
    assert calls[0] == 3 and updates[0] == 4
    assert_accounting(tmp_path / "run")


def install_anchor_checks(backend, rank):
    calls, updates = [0], [0]
    original_step = backend.accumulated_step

    def scores(_backend, _config, _parts, endpoints, *_args):
        calls[0] += 1
        values = torch.arange(len(endpoints), dtype=torch.float32)
        if rank == 0 or calls[0] == 2:
            values.fill_(.1)
        return weighted_score({"VQ": values}, {"VQ": 1.}, space="raw")

    def step(model, optimizer, items, objective, **kwargs):
        assert kwargs["denominator"] == len(items)
        assert objective.has_training_signal
        if rank == 0 or calls[0] == 2:
            assert not objective.has_guidance
            assert objective.row_anchors.all()
            assert not objective.row_magnitudes.any()
        result = original_step(model, optimizer, items, objective, **kwargs)
        updates[0] += 1
        return result

    hunyuan.score_training_videos = scores
    backend.accumulated_step = step
    return calls, updates


def worker(rank, path, root, family, restore=False):
    from test_hunyuan_fsdp import wrap
    from test_hunyuan_runner import components
    from test_sd3_runner import tiny_parts

    torch.set_num_threads(1)
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE="2",
                      VISUAL_RL_COLLECTIVE_TIMEOUT_SECONDS="30")
    backend = image_ddp if family == "image" else hunyuan_fsdp
    session = backend.distributed_session
    backend.distributed_session = lambda *args, **kwargs: session(
        *args, **kwargs, init_method=f"file://{root}/rendezvous")
    sd3.load_components = tiny_parts
    hunyuan.load_components = components
    if family == "wan":
        from test_wan_runner import install_components

        install_components()
    hunyuan_fsdp.wrap_policy = wrap
    calls, updates = install_anchor_checks(backend, rank) if restore else install_checks(backend, family, rank, 2)
    assert main(["--config", path]) == 0
    assert calls[0] == (2 if restore else 3) and updates[0] == 4


@pytest.mark.parametrize("family,distributed,restore", [
    ("image", "ddp", False), ("video", "fsdp", False), ("wan", "fsdp", False),
    ("video", "fsdp", True), ("wan", "fsdp", True),
])
def test_scalar_collective_skip_and_local_zero_rank(tmp_path, family, distributed, restore):
    path = configured(tmp_path, family, distributed)
    if restore:
        values = yaml.safe_load(path.read_text())
        values["algorithm"]["options"]["restore_zero_signal"] = True
        path.write_text(yaml.safe_dump(values))
    ctx = mp.spawn(worker, args=(str(path), str(tmp_path), family, restore), nprocs=2, join=False)
    deadline = time.monotonic() + 90
    try:
        while not ctx.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("scalar zero-signal collective test exceeded 90s")
    finally:
        for process in ctx.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    if restore:
        status = json.loads((tmp_path / "run/status.json").read_text())
        assert status["optimizer_steps"] == 4
        assert status["attempts"] == status["completed_rollouts"] == 2
    else:
        assert_accounting(tmp_path / "run")


@pytest.mark.parametrize("restore", [False, True])
def test_video_zero_guidance_anchor_restores_drift_without_changing_windows(tmp_path, restore):
    c = load_config(configured(tmp_path, "video", "none"))
    c = replace(c, algorithm=replace(c.algorithm, options={
        **c.algorithm.options, "restore_zero_signal": restore}))
    parameter = torch.nn.Parameter(torch.tensor(.1))

    def predict(x, t, rows):
        return x * parameter

    cache = collect_rollout(predict, torch.arange(1., 9.).reshape(4, 2),
                            torch.tensor([1., .5, 0.]), selected=[0, 1])
    scores = weighted_score({"VQ": torch.full((4,), .1)}, {"VQ": 1.}, space="raw")
    parts = SimpleNamespace(adapter=SimpleNamespace(transformer=torch.nn.Linear(1, 1)))
    objective, windows = hunyuan.prepare_objective(c, parts, cache, predict, scores, scores, "cpu")
    assert not objective.has_guidance
    assert objective.has_training_signal is restore
    assert objective.row_anchors.tolist() == [restore] * 4
    assert len(windows) == 2 and all(len(window) == 4 for window in windows)
    assert not set(windows[0]) & set(windows[1])
    torch.testing.assert_close(objective(windows[0]), torch.zeros(4), atol=1e-12, rtol=0)
    with torch.no_grad():
        parameter.add_(.1)
    loss = objective(windows[1]).sum() / len(windows[1])
    gradient, = torch.autograd.grad(loss, parameter)
    assert (loss.item() > 0) is restore
    assert (gradient.item() > 0) is restore


def test_single_rank_video_anchor_only_rollouts_are_not_skipped(tmp_path, monkeypatch):
    from test_hunyuan_runner import components

    path = configured(tmp_path, "video", "none")
    values = yaml.safe_load(path.read_text())
    values["algorithm"]["options"]["restore_zero_signal"] = True
    path.write_text(yaml.safe_dump(values))
    monkeypatch.setattr(hunyuan, "load_components", components)
    monkeypatch.setattr(hunyuan, "accumulated_step", hunyuan.accumulated_step)
    monkeypatch.setattr(hunyuan, "score_training_videos", hunyuan.score_training_videos)
    calls, updates = install_anchor_checks(hunyuan, 0)
    assert main(["--config", str(path)]) == 0
    assert calls[0] == 2 and updates[0] == 4


@pytest.mark.parametrize("family", ["image", "flux", "video", "wan"])
@pytest.mark.parametrize("options", [
    {"restore_zero_signal": True}, {"covariance": {"zero_signal_mode": "historical"}},
    {"covariance": {"zero_signal_mode": "threshold", "zero_signal_atol": 1e-4}},
    {"covariance": None}, {"covariance": []},
])
def test_scalar_zero_signal_validator(tmp_path, family, options):
    path = configured(tmp_path, family, "none")
    values = yaml.safe_load(path.read_text())
    values["algorithm"]["options"].update(options)
    path.write_text(yaml.safe_dump(values))
    if family == "wan":
        from visual_rl.backends import wan

        backend = wan
    else:
        backend = sd3 if family in ("image", "flux") else hunyuan
    if family in ("video", "wan") and options == {"restore_zero_signal": True}:
        backend.validate_config(load_config(path))
        return
    with pytest.raises(ConfigError, match="scalar ART"):
        backend.validate_config(load_config(path))
