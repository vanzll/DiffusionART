"""Real tiny offline Diffusers/PEFT round trips and native checkpoint writers."""

import json
from dataclasses import replace
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

from visual_rl.config import (
    Component,
    Evaluation,
    Experiment,
    Reward,
    Runtime,
    Sampling,
    Training,
)
from visual_rl.export import export_policy, main
from visual_rl.models.finetune import FineTuneConfig, prepare_policy
from visual_rl.training.checkpoint import _digest, save_checkpoint
from visual_rl.training.distributed_checkpoint import save_ddp_checkpoint
from visual_rl.training.state import Progress

diffusers = pytest.importorskip("diffusers")


@pytest.fixture(autouse=True)
def cpu_runtime(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(21)
        yield
    torch.set_num_threads(threads)


def fixture(tmp_path, family="sd3", mode="full", distributed="none"):
    if family == "sd3":
        base = diffusers.SD3Transformer2DModel(
            sample_size=4, patch_size=2, in_channels=2, out_channels=2,
            num_layers=1, attention_head_dim=8, num_attention_heads=2,
            joint_attention_dim=12, caption_projection_dim=16,
            pooled_projection_dim=8, pos_embed_max_size=4,
        )
    else:
        base = diffusers.FluxTransformer2DModel(
            patch_size=1, in_channels=8, out_channels=8, num_layers=1,
            num_single_layers=1, attention_head_dim=16, num_attention_heads=2,
            joint_attention_dim=12, pooled_projection_dim=8,
            guidance_embeds=False, axes_dims_rope=(4, 6, 6),
        )
    root = tmp_path / "base"
    base.save_pretrained(root / "transformer", safe_serialization=True)
    fine = {"mode": mode}
    if mode == "lora":
        pytest.importorskip("peft")
        fine.update(rank=2, alpha=4, target_modules=("to_q", "to_v"))
    config = Experiment(
        name="export-test", task="t2i", algorithm=Component("art"),
        model=Component(family, str(root), "test-revision", {"finetune": fine}),
        reward=Reward("test", {"score": 1}), sampling=Sampling(frames=1),
        training=Training(), evaluation=Evaluation(),
        runtime=Runtime(distributed=distributed, device="cpu"),
    )
    model = prepare_policy(base, FineTuneConfig(**fine)).eval()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
    # Real optimizer state and changed policy tensors, without a pipeline/download.
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.ones_like(p)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return config, model, optimizer


def save_local(tmp_path, config, model, optimizer):
    return save_checkpoint(
        tmp_path / "checkpoints", model=model, optimizer=optimizer,
        progress=Progress(attempts=1, completed_rollouts=1, optimizer_steps=1),
        config_sha256=config.fingerprint(),
    )


def assert_state_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        assert left[key].dtype == right[key].dtype
        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)


@pytest.mark.parametrize("family", ["sd3", "flux"])
@pytest.mark.parametrize("mode", ["full", "lora"])
def test_local_standard_round_trip(tmp_path, family, mode):
    config, model, optimizer = fixture(tmp_path, family, mode)
    receipt = save_local(tmp_path, config, model, optimizer)
    before = _digest(receipt)
    output = export_policy(config, receipt, tmp_path / "export")
    cls = getattr(diffusers, "SD3Transformer2DModel" if family == "sd3"
                  else "FluxTransformer2DModel")
    if mode == "full":
        restored = cls.from_pretrained(output, local_files_only=True)
        assert (output / "config.json").is_file()
    else:
        from peft import PeftModel

        base = cls.from_pretrained(config.model.checkpoint, subfolder="transformer",
                                   local_files_only=True)
        restored = PeftModel.from_pretrained(base, output, local_files_only=True)
        adapter = json.loads((output / "adapter_config.json").read_text())
        assert adapter["r"] == 2 and adapter["lora_alpha"] == 4
        assert adapter["revision"] == "test-revision"
        assert adapter["base_model_name_or_path"] == config.model.checkpoint
        assert not (output / "diffusion_pytorch_model.safetensors").exists()
    assert_state_equal(model.state_dict(), restored.state_dict())
    record = json.loads((output / "export.json").read_text())
    assert record["optimizer_step"] == 1 and record["checkpoint"] == str(receipt)
    assert record["output"] == str(output)
    assert not any(p.suffix == ".pt" for p in output.iterdir())
    assert _digest(receipt) == before
    assert torch.load(receipt.parent / json.loads(receipt.read_text())["file"],
                      weights_only=True)["optimizer"]["state"]


@pytest.fixture
def local(tmp_path):
    config, model, optimizer = fixture(tmp_path)
    return config, save_local(tmp_path, config, model, optimizer)


def rewrite_payload(receipt, mutate):
    metadata = json.loads(receipt.read_text())
    path = receipt.parent / metadata["file"]
    state = torch.load(path, weights_only=True)
    mutate(state)
    torch.save(state, path)
    metadata.update(sha256=_digest(path), bytes=path.stat().st_size)
    receipt.write_text(json.dumps(metadata))


@pytest.mark.parametrize("fault", ["corrupt", "missing", "partial", "identity", "keys", "progress",
                                   "shape", "dtype", "nan"])
def test_reject_invalid_local_without_output(tmp_path, local, fault):
    config, receipt = local
    path = receipt.parent / json.loads(receipt.read_text())["file"]
    if fault == "corrupt":
        path.write_bytes(path.read_bytes() + b"corrupt")
    elif fault == "missing":
        path.unlink()
    else:
        def mutate(state):
            key = next(iter(state["model"]))
            if fault == "partial":
                del state["optimizer"]
            elif fault == "progress":
                state["progress"] = None
            elif fault == "identity":
                state["progress"]["optimizer_steps"] += 1
            elif fault == "keys":
                del state["model"][key]
            elif fault == "shape":
                state["model"][key] = torch.ones(1)
            elif fault == "dtype":
                state["model"][key] = state["model"][key].double()
            else:
                state["model"][key].fill_(float("nan"))
        rewrite_payload(receipt, mutate)
    with pytest.raises(ValueError):
        export_policy(config, receipt, tmp_path / "export")
    assert not (tmp_path / "export").exists()


def test_no_overwrite_and_path_safety(tmp_path, local):
    config, receipt = local
    output = tmp_path / "export"
    output.mkdir()
    sentinel = output / "keep"
    sentinel.write_text("unchanged")
    with pytest.raises(FileExistsError):
        export_policy(config, receipt, output)
    assert sentinel.read_text() == "unchanged"
    link = tmp_path / "linked"
    link.symlink_to(receipt.parent, target_is_directory=True)
    for source, destination in [(link / receipt.name, tmp_path / "new"),
                                (receipt, link / "new"),
                                (receipt, tmp_path / ".." / "new"),
                                (receipt, receipt.parent / "new")]:
        with pytest.raises(ValueError):
            export_policy(config, source, destination)
    metadata = json.loads(receipt.read_text())
    metadata["file"] = "../escape.pt"
    receipt.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="file identity"):
        export_policy(config, receipt, tmp_path / "new")


def test_reject_wrong_config_latest_fsdp_and_video(tmp_path, local):
    config, receipt = local
    with pytest.raises(ValueError, match="schema/config"):
        export_policy(replace(config, seed=9), receipt, tmp_path / "new")
    with pytest.raises(ValueError, match="receipt"):
        export_policy(config, receipt.parent / "latest.json", tmp_path / "new")
    with pytest.raises(ValueError, match="model backend"):
        export_policy(replace(config, model=Component("hunyuan")), receipt, tmp_path / "new")
    metadata = json.loads(receipt.read_text())
    metadata["kind"] = "fsdp-local"
    receipt.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="no resharding"):
        export_policy(config, receipt, tmp_path / "new")


def test_lora_rejects_modified_frozen_base(tmp_path):
    config, model, optimizer = fixture(tmp_path, mode="lora")
    with torch.no_grad():
        next(p for p in model.parameters() if not p.requires_grad).add_(1)
    receipt = save_local(tmp_path, config, model, optimizer)
    with pytest.raises(ValueError, match="frozen base differs"):
        export_policy(config, receipt, tmp_path / "export")


@pytest.mark.parametrize("mode", ["full", "lora"])
def test_real_cpu_ddp_publication_and_all_rank_integrity(tmp_path, mode):
    if not dist.is_gloo_available():
        pytest.skip("CPU Gloo unavailable")
    config, model, optimizer = fixture(tmp_path, mode=mode, distributed="ddp")
    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/rendezvous",
                            rank=0, world_size=1, timeout=timedelta(seconds=20))
    try:
        wrapped = torch.nn.parallel.DistributedDataParallel(model)
        receipt = save_ddp_checkpoint(
            tmp_path / "checkpoints", model=wrapped, optimizer=optimizer,
            progress=Progress(attempts=1, completed_rollouts=1, optimizer_steps=1),
            config_sha256=config.fingerprint(),
        )
    finally:
        dist.destroy_process_group()
    export_policy(config, receipt, tmp_path / "export")
    # A second published rank exercises complete validation without launching GPUs.
    manifest = json.loads(receipt.read_text())
    for name in ("shared.pt", "rank-00000.pt"):
        path = receipt.parent / name
        state = torch.load(path, weights_only=True)
        state["world_size"] = 2
        torch.save(state, path)
    second = torch.load(receipt.parent / "rank-00000.pt", weights_only=True)
    second["rank"] = 1
    torch.save(second, receipt.parent / "rank-00001.pt")
    def record(name):
        path = receipt.parent / name
        return {"file": name, "bytes": path.stat().st_size, "sha256": _digest(path)}
    manifest.update(world_size=2, shared=record("shared.pt"),
                    ranks=[record("rank-00000.pt"), record("rank-00001.pt")])
    receipt.write_text(json.dumps(manifest))
    export_policy(config, receipt, tmp_path / "two-ranks")
    second["model_local"]["unexpected_buffer"] = torch.ones(1)
    torch.save(second, receipt.parent / "rank-00001.pt")
    manifest["ranks"][1] = record("rank-00001.pt")
    receipt.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="buffer names differ"):
        export_policy(config, receipt, tmp_path / "different-buffers")
    (receipt.parent / "rank-00001.pt").unlink()
    with pytest.raises(ValueError, match="integrity"):
        export_policy(config, receipt, tmp_path / "missing-rank")
    assert not (tmp_path / "missing-rank").exists()


def test_failed_save_cleans_output(tmp_path, local, monkeypatch):
    config, receipt = local
    def fail(*args, **kwargs):
        raise OSError("simulated disk full")
    monkeypatch.setattr(diffusers.SD3Transformer2DModel, "save_pretrained", fail)
    with pytest.raises(OSError, match="disk full"):
        export_policy(config, receipt, tmp_path / "export")
    assert not (tmp_path / "export").exists()


def test_cli_reports_failure(tmp_path, local, monkeypatch, capsys):
    config, receipt = local
    monkeypatch.setattr("visual_rl.export.load_config", lambda path: config)
    with pytest.raises(SystemExit) as error:
        main(["--config", "unused.yaml", "--checkpoint", str(receipt.parent / "latest.json"),
              "--output", str(tmp_path / "export")])
    assert error.value.code == 1
    assert "export failed" in capsys.readouterr().err
