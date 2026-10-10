"""Tiny CPU rollout sequences exercise guard wiring without model downloads."""

import copy
import json
import os
import time
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from test_distributed_checkpoint import assert_tree_equal
from test_scalar_zero_signal_runner import configured

from visual_rl.backends import hunyuan, hunyuan_fsdp, image_ddp, sd3
from visual_rl.rewards.scalar import weighted_score
from visual_rl.train import main

SEQUENCES = {
    "default": (None, [False] * 4 + [True, False, True]),
    "reset": (3, [False, False, True, False, False, True]),
    "limit": (2, [True, False, False]),
}


def config_file(tmp_path, family, distributed, scenario):
    path = configured(tmp_path, "video" if family == "pairwise" else family, distributed)
    values = yaml.safe_load(path.read_text())
    values["runtime"]["max_consecutive_no_signal_rollouts"] = SEQUENCES[scenario][0]
    if family == "pairwise":
        values["algorithm"]["options"].update(
            pairwise={"seed": 42}, restore_zero_signal=True,
            covariance={"zero_signal_mode": "historical"})
        values["reward"].update(normalization="fixed", weights={"VQ": 1., "MQ": 0., "TA": 0.})
    path.write_text(yaml.safe_dump(values))
    return path


def install_checks(monkeypatch, backend, family, rank, world, scenario):
    calls, updates, saved = [0], [0], {}
    original_step = backend.accumulated_step

    def step(model, optimizer, items, objective, **kwargs):
        assert kwargs["denominator"] == len(items)
        if world > 1 and rank == 0:
            if family == "pairwise":
                assert objective.row_anchors.all() and not objective.row_magnitudes.any()
            else:
                assert not objective.has_guidance and not objective.row_anchors.any()
        result = original_step(model, optimizer, items, objective, **kwargs)
        updates[0] += 1
        saved.update(optimizer=optimizer, parameters=list(model.parameters()))
        saved["state"] = copy.deepcopy(optimizer.state_dict())
        saved["weights"] = [p.detach().clone() for p in saved["parameters"]]
        return result

    def scores(count):
        if saved:
            assert saved["state"]["state"]  # Exercise skipping with Adam momentum.
            assert_tree_equal(saved["state"], saved["optimizer"].state_dict())
            assert_tree_equal(saved["weights"], [p.detach() for p in saved["parameters"]])
        sequence = SEQUENCES[scenario][1]
        assert calls[0] < len(sequence), "runner exceeded the expected rollout budget"
        active = sequence[calls[0]] and (world == 1 or rank != 0)
        calls[0] += 1
        values = torch.arange(count, dtype=torch.float32) if active else torch.zeros(count)
        if family == "pairwise":
            from test_video_pairwise import scores as pairwise_scores

            return pairwise_scores(values)
        key = "pickscore" if family in ("image", "flux") else "VQ"
        return weighted_score({key: values}, {key: 1.}, space="raw")

    monkeypatch.setattr(backend, "accumulated_step", step)
    if family in ("image", "flux"):
        monkeypatch.setattr(sd3, "_score", lambda _c, _p, endpoints, *_a, **_kw:
                            SimpleNamespace(scores=scores(len(endpoints))))
    else:
        monkeypatch.setattr(hunyuan, "score_training_videos", lambda _b, _c, _p, endpoints, *_a:
                            scores(len(endpoints)))
    return calls, updates


def install_components(monkeypatch, family):
    from test_hunyuan_runner import components
    from test_sd3_runner import tiny_parts

    monkeypatch.setattr(sd3, "load_components", tiny_parts)
    monkeypatch.setattr(hunyuan, "load_components", components)
    if family == "flux":
        from test_image_ddp_runner import parts_for_model

        monkeypatch.setattr(sd3, "load_components", parts_for_model)
    elif family == "wan":
        from test_wan_runner import components as wan_parts
        from test_wan_runner import condition, reference

        from visual_rl.backends import wan
        from visual_rl.models import wan_assets

        monkeypatch.setattr(wan, "load_components", wan_parts)
        monkeypatch.setattr(wan_assets, "generate_reference", reference)
        monkeypatch.setattr(wan_assets, "condition_image", condition)


def assert_accounting(root, family, scenario):
    records = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
    sequence = SEQUENCES[scenario][1]
    skip_metric = "train/pairwise/inactive_rollout" if family == "pairwise" else "system/skipped_zero_guidance"
    assert sum(row.get(skip_metric, 0) for row in records) == sequence.count(False)
    cost = json.loads((root / "cost.json").read_text())["progress"]
    assert cost["attempts"] == len(sequence)
    assert cost["optimizer_steps"] == 2 * sequence.count(True)
    assert cost["completed_rollouts"] == sequence.count(True)
    assert records[-1]["train/optimizer_step"] == cost["optimizer_steps"]
    assert (root / "status.json").exists() is (scenario != "limit")


@pytest.mark.parametrize("family", ["image", "flux", "video", "wan", "pairwise"])
@pytest.mark.parametrize("scenario", SEQUENCES)
def test_single_rank_liveness(tmp_path, monkeypatch, caplog, capsys, family, scenario):
    path = config_file(tmp_path, family, "none", scenario)
    backend = sd3 if family in ("image", "flux") else hunyuan
    install_components(monkeypatch, family)
    calls, updates = install_checks(monkeypatch, backend, family, 0, 1, scenario)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        assert main(["--config", str(path)]) == (1 if scenario == "limit" else 0)
    finally:
        torch.set_num_threads(threads)
    assert calls[0] == len(SEQUENCES[scenario][1])
    assert updates[0] == 2 * SEQUENCES[scenario][1].count(True)
    if scenario == "limit":
        assert "max_consecutive_no_signal_rollouts=2 exhausted" in capsys.readouterr().err
    elif scenario == "reset":
        assert [record.args[0] for record in caplog.records
                if record.name == "visual_rl.training.no_signal"] == [1, 2, 1, 2]
    assert_accounting(tmp_path / "run", family, scenario)


def worker(rank, path, root, family, scenario):
    from test_hunyuan_fsdp import wrap

    torch.set_num_threads(1)
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE="2",
                      VISUAL_RL_COLLECTIVE_TIMEOUT_SECONDS="30")
    backend = image_ddp if family == "image" else hunyuan_fsdp
    session = backend.distributed_session
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(backend, "distributed_session", lambda *args, **kwargs: session(
            *args, **kwargs, init_method=f"file://{root}/rendezvous"))
        patch.setattr(hunyuan_fsdp, "wrap_policy", wrap)
        install_components(patch, family)
        calls, updates = install_checks(patch, backend, family, rank, 2, scenario)
        assert main(["--config", path]) == (1 if scenario == "limit" else 0)
        assert calls[0] == len(SEQUENCES[scenario][1])
        assert updates[0] == 2 * SEQUENCES[scenario][1].count(True)
        assert not dist.is_initialized()


@pytest.mark.parametrize("family,distributed", [("image", "ddp"), ("video", "fsdp"), ("pairwise", "fsdp")])
@pytest.mark.parametrize("scenario", ["reset", "limit"])
def test_global_liveness_and_local_inactive_rank(tmp_path, family, distributed, scenario):
    path = config_file(tmp_path, family, distributed, scenario)
    ctx = mp.spawn(worker, args=(str(path), str(tmp_path), family, scenario), nprocs=2, join=False)
    deadline = time.monotonic() + 90
    try:
        while not ctx.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("liveness collective test exceeded 90s")
    finally:
        for process in ctx.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    assert_accounting(tmp_path / "run", family, scenario)
