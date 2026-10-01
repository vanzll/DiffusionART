import faulthandler
import json
import os
import shutil
import time
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from test_distributed_checkpoint import assert_tree_equal
from test_hunyuan_runner import Transformer, components, config_file

from visual_rl.backends import hunyuan
from visual_rl.evaluation import EvaluationManifest, audit_manifest
from visual_rl.evaluation.manifest import file_sha256
from visual_rl.train import main
from visual_rl.training.sharding import wrap_video_policy


def wrap(c, parts, session):
    module = parts.adapter.trainable_module
    module.scale = torch.nn.Parameter(module.scale.detach().reshape(1))
    return wrap_video_policy(module, device=session.device,
                             block_types=(Transformer,), activation_checkpointing=False)


def worker(rank, path, root, family="hunyuan"):
    from visual_rl.backends import hunyuan_fsdp

    faulthandler.enable(all_threads=True)

    def phase(name):
        (Path(root) / f"rank-{rank}-phase.json").write_text(json.dumps({
            "rank": rank, "pid": os.getpid(), "family": family, "phase": name,
            "process_group_initialized": dist.is_initialized(),
        }))

    phase("worker-started")
    torch.set_num_threads(1)
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE="2",
                      VISUAL_RL_COLLECTIVE_TIMEOUT_SECONDS="30")
    hunyuan.load_components = components
    if family == "pairwise":
        from visual_rl.algorithms import art_pairwise_covariance
        from visual_rl.training import video_pairwise

        original_prepare = video_pairwise.prepare

        def feedback(c, cache, scores, *, attempt, group_offset=0):
            if attempt == 1 or (attempt == 2 and rank == 0):
                return art_pairwise_covariance(cache.endpoints, torch.zeros(len(cache.endpoints)),
                    c.sampling.samples_per_prompt, seed=42, round_index=attempt, group_offset=group_offset)
            return original_prepare(c, cache, scores, attempt=attempt, group_offset=group_offset)

        video_pairwise.prepare = feedback
    if family == "wan":
        from test_wan_runner import install_components

        install_components()
    hunyuan_fsdp.wrap_policy = wrap
    original_session = hunyuan_fsdp.distributed_session
    run_number = [0]

    def session(*args, **kwargs):
        run_number[0] += 1
        return original_session(*args, **kwargs, init_method=f"file://{root}/rendezvous-{run_number[0]}")

    hunyuan_fsdp.distributed_session = session
    original_save = hunyuan_fsdp.save_fsdp_checkpoint
    mode = ["continuous"]
    expected, observed = {}, {}
    receipt = [None]

    def save(*args, **kwargs):
        value = original_save(*args, **kwargs)
        receipt[0] = str(value)
        if kwargs["progress"].optimizer_steps == 4:
            import copy

            target = expected if mode[0] == "continuous" else observed
            target.update(copy.deepcopy({"parameters": [p.detach().clone() for p in kwargs["model"].parameters()],
                "optimizer": kwargs["optimizer"].state_dict(),
                "stateful": {name: item.state_dict() for name, item in kwargs["stateful"].items()}}))
        if mode[0] == "interrupted" and kwargs["progress"].optimizer_steps == 2:
            raise RuntimeError("test checkpoint interruption")
        return value

    hunyuan_fsdp.save_fsdp_checkpoint = save
    phase("continuous-run")
    assert main(["--config", path]) == 0
    assert not dist.is_initialized()
    mode[0] = "interrupted"
    target = str(Path(root) / "interrupted")
    phase("interruption-check")
    with pytest.raises(RuntimeError, match="test checkpoint interruption"):
        main(["--config", path, "--output-dir", target])
    assert not dist.is_initialized()
    mode[0] = "resumed"
    phase("resume-check")
    assert main(["--config", path, "--output-dir", target, "--resume", receipt[0]]) == 0
    assert_tree_equal(expected, observed)
    assert not dist.is_initialized()
    phase("completed-resume-check")
    assert main(["--config", path, "--output-dir", target, "--resume", receipt[0]]) == 0
    assert not dist.is_initialized()
    (Path(root) / f"rank-{rank}-passed.json").write_text(json.dumps({"passed": True}))
    phase("worker-completed")


@pytest.mark.parametrize("algorithm", ["diffusionart", "flow_grpo_fast", "dancegrpo", "diffusionnft", "refl", "pairwise"])
def test_native_hunyuan_fsdp_eval_and_resume(tmp_path, algorithm):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg/ffprobe required")
    path = config_file(tmp_path, algorithm)
    c = yaml.safe_load(path.read_text())
    c["runtime"]["distributed"] = "fsdp"
    c["sampling"]["prompts_per_rollout"] = 2
    records = [json.loads(line) for line in Path(c["data"]["manifest"]).read_text().splitlines()]
    records.append({**records[1], "sample_id": "third"})
    validation = tmp_path / "validation.jsonl"
    validation.write_text("".join(json.dumps(row) + "\n" for row in records))
    c["evaluation"].update(manifest=str(validation), manifest_sha256=file_sha256(validation))
    path.write_text(yaml.safe_dump(c))
    family = "pairwise" if algorithm == "pairwise" else "hunyuan"
    context = mp.spawn(worker, args=(str(path), str(tmp_path), family), nprocs=2, join=False)
    deadline = time.monotonic() + 120
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("Hunyuan FSDP regression exceeded 120s")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    for rank in range(2):
        assert json.loads((tmp_path / f"rank-{rank}-passed.json").read_text())["passed"]
    for name in ("run", "interrupted"):
        root = tmp_path / name
        manifests = list(root.glob("evaluation/*/*/manifest.json"))
        assert len(manifests) == 3
        for file in manifests:
            manifest = EvaluationManifest.from_json(file.read_text())
            assert len(manifest.samples) == 3
            assert [row.seed for row in manifest.samples] == [312, 313, 314]
            assert audit_manifest(manifest, media_root=root).complete
            assert (file.parent / "index.html").exists()
        metrics = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
        assert [row["train/optimizer_step"] for row in metrics] == ([0, 1, 2, 3, 4] if algorithm == "pairwise" else [1, 2, 3, 4])
        assert all(row["system/dp_world_size"] == 2 and row["cost/elapsed_gpu_hours"] == 0 for row in metrics)
        if algorithm == "pairwise":
            assert metrics[0]["train/pairwise/inactive_rollout"] == 1
            assert metrics[1]["system/rollout_attempts"] == 2
            assert metrics[1]["train/pairwise/inactive_rollout"] == 0
            assert metrics[2]["train/pairwise/inactive_rollout"] == 0
            assert json.loads((root / "status.json").read_text())["attempts"] == 3
        if algorithm == "diffusionnft":
            assert [row["nft/old_policy_decay"] for row in metrics] == pytest.approx([0., .1, .2, .3])
            assert json.loads((root / "status.json").read_text())["completed_rollouts"] == 4
        if algorithm == "refl":
            assert not any(key.startswith("train/refl/update_") for row in metrics for key in row)
            assert all(row["train/loss"] == pytest.approx(row["train/refl/loss"]) for row in metrics)
            assert json.loads((root / "status.json").read_text())["completed_rollouts"] == 4


def test_single_rank_fsdp_rejected_before_assets(tmp_path, monkeypatch):
    path = config_file(tmp_path)
    c = yaml.safe_load(path.read_text())
    c["runtime"]["distributed"] = "fsdp"
    path.write_text(yaml.safe_dump(c))
    for key, value in dict(RANK="0", LOCAL_RANK="0", WORLD_SIZE="1", LOCAL_WORLD_SIZE="1").items():
        monkeypatch.setenv(key, value)
    assert main(["--config", str(path)]) == 2
    assert not (tmp_path / "run").exists()
