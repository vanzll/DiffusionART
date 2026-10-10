import hashlib
import json
import os
import time
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp
import yaml
from test_sd3_runner import config_file, tiny_parts

from visual_rl.backends import image_ddp, sd3
from visual_rl.config import load_config
from visual_rl.evaluation import EvaluationManifest, audit_manifest
from visual_rl.train import main


def configured(tmp_path, algorithm, model="sd3", precision="fp32"):
    path = config_file(tmp_path)
    content = yaml.safe_load(path.read_text())
    content["runtime"]["distributed"] = "ddp"
    content["training"]["precision"] = precision
    if algorithm == "diffusionart_one_update":
        content["algorithm"]["options"]["updates_per_rollout"] = 1
        algorithm = "diffusionart"
    nft_updates = 2 if algorithm == "diffusionnft_two_updates" else 1
    if algorithm == "diffusionnft_two_updates":
        algorithm = "diffusionnft"
    # An uneven evaluation population must preserve identities without padding
    # duplicates or repeating the entire set on every worker.
    val = tmp_path / "validation.jsonl"
    val.write_text("".join(json.dumps({"sample_id": f"v{i}", "prompt": str(i + 2)}) + "\n" for i in range(3)))
    content["evaluation"].update(manifest=str(val), manifest_sha256=hashlib.sha256(val.read_bytes()).hexdigest())
    if algorithm == "diffusionnft":
        content["algorithm"] = {"name": algorithm, "options": {
            "loss": {}, "old_policy": {"rate": 0.001, "maximum": 0.5, "delay": 0},
            "evaluation_ema": {"enabled": True, "decay": 0.9},
        }}
        if nft_updates == 2:
            content["algorithm"]["options"]["updates_per_rollout"] = 2
        content["sampling"]["options"] = {"shift": 1.0, "solver": "euler",
            "train_timestep_count": 2, "timestep_selection": "random_per_row"}
    elif algorithm in ("flow_grpo", "flow_grpo_fast", "flow_dppo", "dancegrpo"):
        content["algorithm"] = {"name": algorithm, "options": {
            "loss": {"kl_mask_threshold": 1e-5, "advantage_clip": 5.0} if algorithm == "flow_dppo"
                    else {"ratio_clip": 1e-5, "advantage_clip": 5.0},
            "updates_per_rollout": 2, "training_batches": 2,
            "shuffle_rows": algorithm != "flow_grpo_fast", "zero_advantage_mode": "filter_pad",
            "max_empty_rollouts": 2,
            "evaluation_ema": {"enabled": True, "decay": 0.9, "refresh": "logical_batch"},
        }}
        content["sampling"]["options"] = {
            "shift": 1.0, "transition_kind": "sde" if algorithm == "flow_grpo" else "cps",
            "likelihood": "gaussian" if algorithm == "flow_grpo" else "mse",
            "noise_level": 0.8, "selection": "fixed", "pool": [0, 1], "count": 2,
            "logical_batch_size": 2, "unselected_noise": "all" if algorithm == "flow_grpo" else "ode",
        }
        if algorithm == "dancegrpo":
            content["algorithm"]["options"]["advantage_mode"] = "group_sample_std"
            content["sampling"]["options"].update(
                transition_kind="dance", likelihood="gaussian", unselected_noise="all",
                selection="random_per_row", pool=[0, 1, 2], count=2, same_latent=True,
                endpoint_kind="last_x0")
    if model == "flux":
        content["model"]["name"] = "flux"
        content["model"]["options"]["scheduler"] = dict(
            use_dynamic_shifting=True, shift=1.0, base_image_seq_len=256,
            max_image_seq_len=4096, base_shift=0.5, max_shift=1.15)
        content["sampling"]["options"].pop("shift")
        if algorithm == "flow_dppo":
            content["sampling"]["options"].update(transition_kind="sde", likelihood="gaussian", noise_level=.7)
            content["algorithm"]["options"]["evaluation_ema"]["enabled"] = False
    path.write_text(yaml.safe_dump(content))
    return path


def parts_for_model(c):
    parts = tiny_parts(c)
    if c.model.name == "sd3":
        return parts
    from diffusers import FluxTransformer2DModel

    from visual_rl.models import FluxAdapter, FluxConditioning, FluxConfig

    transformer = FluxTransformer2DModel(
        patch_size=1, in_channels=8, out_channels=8, num_layers=1, num_single_layers=1,
        attention_head_dim=16, num_attention_heads=2, joint_attention_dim=24,
        pooled_projection_dim=8, guidance_embeds=True, axes_dims_rope=(4, 6, 6))

    class Text:
        def encode(self, prompts, *, num_images_per_prompt, **kwargs):
            x = torch.tensor([float(prompt) for prompt in prompts])
            return FluxConditioning(
                x[:, None, None].expand(-1, 3, 24).repeat_interleave(num_images_per_prompt, 0),
                x[:, None].expand(-1, 8).repeat_interleave(num_images_per_prompt, 0),
                torch.zeros(3, 3),
            )

    adapter = FluxAdapter(transformer, parts.adapter.vae,
                          FluxConfig(2.0, 0.25, 2, embedded_guidance_scale=c.sampling.guidance_scale,
                                     vae_affine_dtype=torch.bfloat16 if c.training.precision == "bf16" else torch.float32)).eval()
    return sd3.Components(adapter, Text(), parts.scorer)


def saved_state(root):
    latest = json.loads((root / "checkpoints/latest.json").read_text())
    manifest = root / "checkpoints" / latest["manifest"]
    return manifest, torch.load(manifest.parent / "shared.pt", weights_only=True)


def worker(rank, path, root, verify_modes=False):
    torch.set_num_threads(1)
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE="2",
                      VISUAL_RL_COLLECTIVE_TIMEOUT_SECONDS="30")
    sd3.load_components = parts_for_model
    advantages = image_ddp.prompt_advantages

    def global_advantages(rewards, prompts):
        assert len(rewards) == len(prompts) == 4
        assert len(set(prompts)) == 2
        return advantages(rewards, prompts)

    image_ddp.prompt_advantages = global_advantages
    policy_advantages = sd3._policy_advantages

    def joint_advantages(c, scores, prompts):
        assert len(scores.weighted) == len(prompts) == 4
        assert len(set(prompts)) == 2
        return policy_advantages(c, scores, prompts)

    sd3._policy_advantages = joint_advantages
    session, counter = image_ddp.distributed_session, [0]

    def file_session(*args, **kwargs):
        counter[0] += 1
        return session(*args, **kwargs, init_method=f"file://{root}/rendezvous-{counter[0]}")

    image_ddp.distributed_session = file_session
    assert main(["--config", path]) == 0
    config = load_config(path)
    expected_manifest, expected = saved_state(Path(config.runtime.output_dir))
    interrupted = str(Path(root, "interrupted"))
    save = image_ddp.save_ddp_checkpoint

    def stop(*args, **kwargs):
        receipt = save(*args, **kwargs)
        if kwargs["progress"].optimizer_steps == 2:
            raise RuntimeError("test interruption after full checkpoint")
        return receipt

    image_ddp.save_ddp_checkpoint = stop
    with pytest.raises(RuntimeError, match="test interruption"):
        main(["--config", path, "--output-dir", interrupted])
    image_ddp.save_ddp_checkpoint = save
    receipt, _ = saved_state(Path(interrupted))
    assert main(["--config", path, "--output-dir", interrupted, "--resume", str(receipt)]) == 0
    actual_manifest, actual = saved_state(Path(interrupted))
    for key, tensor in expected["model"].items():
        torch.testing.assert_close(tensor, actual["model"][key], rtol=0, atol=0)
    for component, state in expected["replicated_stateful"].items():
        for key, tensor in state.items():
            torch.testing.assert_close(tensor, actual["replicated_stateful"][component][key], rtol=0, atol=0)
    from test_distributed_checkpoint import assert_tree_equal

    assert_tree_equal(expected["optimizer"], actual["optimizer"])
    expected_rank = torch.load(expected_manifest.parent / f"rank-{rank:05d}.pt", weights_only=True)
    actual_rank = torch.load(actual_manifest.parent / f"rank-{rank:05d}.pt", weights_only=True)
    assert expected_rank["stateful"] == actual_rank["stateful"]
    assert torch.equal(expected_rank["generators"]["sampling"], actual_rank["generators"]["sampling"])
    before = (Path(interrupted) / "metrics.jsonl").read_bytes()
    assert main(["--config", path, "--output-dir", interrupted, "--resume", str(actual_manifest)]) == 0
    assert (Path(interrupted) / "metrics.jsonl").read_bytes() == before
    if verify_modes:
        directory = Path(interrupted)
        records = {name: (directory / name).read_bytes() for name in ("metrics.jsonl", "cost.json", "status.json")}
        assert main(["--config", path, "--output-dir", interrupted, "--resume", str(actual_manifest), "--eval-only"]) == 0
        assert records == {name: (directory / name).read_bytes() for name in records}
        previous, = directory.glob("checkpoints/step-000000002-*/manifest.json")
        assert main(["--config", path, "--output-dir", interrupted, "--resume", str(previous), "--resume-rewind"]) == 0
        _, replayed = saved_state(directory)
        assert_tree_equal(expected["model"], replayed["model"])
        assert_tree_equal(expected["optimizer"], replayed["optimizer"])
        assert_tree_equal(expected["replicated_stateful"], replayed["replicated_stateful"])
        assert len(list(directory.glob("resume-history/*/resume.json"))) == 1


@pytest.mark.parametrize("algorithm,model,precision", [
    (algorithm, model, precision)
    for algorithm in ("diffusionart", "diffusionnft", "flow_grpo", "flow_grpo_fast")
    for model, precision in (("sd3", "fp32"), ("sd3", "fp16"), ("flux", "fp32"), ("flux", "bf16"))
] + [("flow_dppo", "sd3", "fp32"), ("flow_dppo", "sd3", "fp16"), ("flow_dppo", "flux", "fp32"), ("flow_dppo", "flux", "bf16"),
     ("dancegrpo", "sd3", "fp32"), ("dancegrpo", "sd3", "fp16"),
     ("diffusionart_one_update", "sd3", "fp32"),
     ("diffusionnft_two_updates", "sd3", "fp32"), ("diffusionnft_two_updates", "sd3", "fp16")])
def test_two_rank_native_training_eval_and_exact_resume(tmp_path, algorithm, model, precision):
    pytest.importorskip("diffusers")
    path = configured(tmp_path, algorithm, model, precision)
    ctx = mp.spawn(worker, args=(str(path), str(tmp_path)), nprocs=2, join=False)
    deadline = time.monotonic() + 120
    try:
        while not ctx.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("native DDP test exceeded 120s")
    finally:
        for process in ctx.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    root = tmp_path / "run"
    status = json.loads((root / "status.json").read_text())
    assert status["optimizer_steps"] == 4
    assert status["completed_rollouts"] == (4 if algorithm in ("diffusionnft", "diffusionart_one_update") else 2)
    logs = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
    assert [row["train/optimizer_step"] for row in logs] == [1, 2, 3, 4]
    assert all(row["system/dp_world_size"] == 2 and row["cost/gpu_count"] == 0 for row in logs)
    if algorithm == "diffusionnft_two_updates":
        assert [row["nft/old_policy_refreshed"] for row in logs] == [0, 1, 0, 1]
    for file in root.glob("evaluation/*/*/manifest.json"):
        manifest = EvaluationManifest.from_json(file.read_text())
        assert audit_manifest(manifest, media_root=root).complete
        assert [(sample.sample_id, sample.seed) for sample in manifest.samples] == [(f"v{i}", 17 + i) for i in range(3)]
        shards = [json.loads((file.parent / f"rank-{rank:05d}.json").read_text()) for rank in range(2)]
        assert [part["indices"] for part in shards] == [[0, 2], [1]]
    assert len(list(root.glob("evaluation/*/*/manifest.json"))) == 3


@pytest.mark.parametrize("algorithm", ["flow_grpo", "diffusionnft"])
def test_joint_advantages_two_rank_training_and_exact_resume(tmp_path, algorithm):
    pytest.importorskip("diffusers")
    path = configured(tmp_path, algorithm)
    content = yaml.safe_load(path.read_text())
    content["algorithm"]["options"]["multi_reward_advantage"] = {
        "weights": {"pickscore": 1}, "score_space": "fixed_normalized",
        "eps": 0.0001, "renormalize_combined": False,
    }
    path.write_text(yaml.safe_dump(content))
    ctx = mp.spawn(worker, args=(str(path), str(tmp_path)), nprocs=2, join=False)
    deadline = time.monotonic() + 120
    try:
        while not ctx.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("joint-advantage DDP test exceeded 120s")
    finally:
        for process in ctx.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    status = json.loads((tmp_path / "run/status.json").read_text())
    assert status["optimizer_steps"] == 4


def test_ddp_rejects_global_prompt_budget_that_cannot_shard(tmp_path, monkeypatch):
    from visual_rl.config import ConfigError

    path = configured(tmp_path, "diffusionart")
    for key, value in dict(RANK="0", WORLD_SIZE="4", LOCAL_RANK="0", LOCAL_WORLD_SIZE="4").items():
        monkeypatch.setenv(key, value)
    with pytest.raises(ConfigError, match="prompt groups"):
        sd3.run(load_config(path))
    assert not (tmp_path / "run").exists()


def test_ddp_checkpoint_evaluation_and_history_rewind(tmp_path):
    path = configured(tmp_path, "diffusionnft")
    ctx = mp.spawn(worker, args=(str(path), str(tmp_path), True), nprocs=2, join=False)
    deadline = time.monotonic() + 120
    try:
        while not ctx.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("DDP recovery regression exceeded 120s")
    finally:
        for process in ctx.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)


def score_worker(rank, root):
    import torch.distributed as dist

    from visual_rl.rewards.scalar import FixedNormalization, weighted_score
    from visual_rl.training.distributed import distributed_session

    torch.set_num_threads(1)
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE="2")
    with distributed_session("ddp", device="cpu", timeout_seconds=15,
                             init_method=f"file://{root}/rendezvous") as session:
        for space in ("raw", "fixed_normalized"):
            raw = torch.tensor([1., 2.]) + 2 * rank
            score = weighted_score({"a": raw}, {"a": 0.5}, space=space,
                                   normalization={"a": FixedNormalization(1, 2, "test")},
                                   diagnostics={"tag": torch.tensor([-10., float(rank)])})
            global_score = image_ddp._global_scores(session, score)
            expected = torch.tensor([1., 2., 3., 4.])
            assert torch.equal(global_score.raw["a"], expected)
            assert torch.equal(global_score.weighted, expected * 0.5 if space == "raw" else (expected - 1) / 4)
            assert global_score.diagnostics["tag"].tolist() == [-10., 0., -10., 1.]
        bad = weighted_score({"a": raw}, {"a": float(rank + 1)}, space="raw")
        with pytest.raises(Exception, match="identity differs"):
            image_ddp._global_scores(session, bad)
        dist.barrier()


def failure_worker(rank, path, root):
    torch.set_num_threads(1)
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE="2",
                      VISUAL_RL_COLLECTIVE_TIMEOUT_SECONDS="15")
    session = image_ddp.distributed_session
    image_ddp.distributed_session = lambda *args, **kwargs: session(
        *args, **kwargs, init_method=f"file://{root}/rendezvous")
    sd3.load_components = tiny_parts
    score = sd3._score

    def fail(*args, **kwargs):
        if rank == 1:
            raise ValueError("injected scorer failure")
        return score(*args, **kwargs)

    sd3._score = fail
    with pytest.raises(RuntimeError, match="evaluation generation/scoring"):
        main(["--config", path])
    Path(root, f"failed-{rank}").write_text("coordinated")


def grpo_edge_worker(rank, path, root):
    torch.set_num_threads(1)
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE="2",
                      VISUAL_RL_COLLECTIVE_TIMEOUT_SECONDS="20")
    session = image_ddp.distributed_session
    image_ddp.distributed_session = lambda *args, **kwargs: session(
        *args, **kwargs, init_method=f"file://{root}/rendezvous")
    sd3.load_components = tiny_parts
    calls = [0]

    def advantages(rewards, prompts):
        calls[0] += 1
        # First globally empty; subsequently rank zero has only padding rows.
        return torch.zeros(4) if calls[0] == 1 else torch.tensor([0., 0., -1., 1.])

    image_ddp.prompt_advantages = advantages
    assert main(["--config", path]) == 0


def test_grpo_collective_skip_and_zero_signal_rank_padding(tmp_path):
    path = configured(tmp_path, "flow_grpo_fast")
    ctx = mp.spawn(grpo_edge_worker, args=(str(path), str(tmp_path)), nprocs=2, join=False)
    deadline = time.monotonic() + 60
    try:
        while not ctx.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("GRPO padding test exceeded 60s")
    finally:
        for process in ctx.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    status = json.loads((tmp_path / "run/status.json").read_text())
    assert status["optimizer_steps"] == 4
    assert status["attempts"] == 3 and status["completed_rollouts"] == 2
    logs = [json.loads(line) for line in (tmp_path / "run/metrics.jsonl").read_text().splitlines()]
    assert [row["train/optimizer_step"] for row in logs] == [0, 1, 2, 3, 4]
    assert logs[0]["system/skipped_zero_advantage"] == 1
    assert all(row["grpo/nonzero_rows"] == 2 and row["grpo/retained_rows"] == 4 for row in logs[1:])


@pytest.mark.parametrize("case", ["scores", "scorer_failure"])
def test_global_reward_contract_and_coordinated_scorer_failure(tmp_path, case):
    if case == "scores":
        target, args = score_worker, (str(tmp_path),)
    else:
        path = configured(tmp_path, "diffusionart")
        target, args = failure_worker, (str(path), str(tmp_path))
    ctx = mp.spawn(target, args=args, nprocs=2, join=False)
    deadline = time.monotonic() + 60
    try:
        while not ctx.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("reward coordination test exceeded 60s")
    finally:
        for process in ctx.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    if case == "scorer_failure":
        assert all((tmp_path / f"failed-{rank}").exists() for rank in range(2))
        assert not (tmp_path / "run/status.json").exists()
        assert (tmp_path / "run/metrics.jsonl").read_text() == ""
        assert not list((tmp_path / "run").glob("evaluation/*/*/manifest.json"))
