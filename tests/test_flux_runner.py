"""Offline native FLUX ART CLI runs with real tiny model and reward assets."""

import hashlib
import inspect
import json
import math
import socket

import pytest
import torch
import yaml
from test_flux_text import real_flux_pipeline  # noqa: F401 -- shared offline asset fixture
from test_sd3_runner import config_file, payload

from visual_rl.backends import sd3
from visual_rl.config import load_config
from visual_rl.evaluation import EvaluationManifest, audit_manifest
from visual_rl.models import FluxAdapter
from visual_rl.models.flux_text import FluxTextEncoder
from visual_rl.registry import get_backend
from visual_rl.train import main


@pytest.fixture(autouse=True)
def offline_cpu(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("FLUX runner tests must not access the network")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(old)


@pytest.fixture(params=["fp32", "bf16"])
def precision(request):
    if request.param == "bf16":
        # Skip only an unsupported CPU primitive, never a failure in the runner.
        try:
            linear = torch.nn.Linear(8, 8, dtype=torch.bfloat16)
            query = linear(torch.ones(1, 2, 3, 8, dtype=torch.bfloat16))
            result = torch.nn.functional.scaled_dot_product_attention(query, query, query)
            result.float().square().mean().backward()
        except (RuntimeError, NotImplementedError) as exc:
            if "not implemented" in str(exc).lower() or "not support" in str(exc).lower():
                pytest.skip(f"CPU bf16 unavailable: {exc}")
            raise
        assert linear.weight.grad is not None and torch.isfinite(linear.weight.grad).all()
    return request.param


@pytest.fixture
def local_flux_config(tmp_path, request, precision):
    pipeline = request.getfixturevalue("real_flux_pipeline")
    diffusers = pytest.importorskip("diffusers")
    transformers = pytest.importorskip("transformers")
    scheduler = dict(use_dynamic_shifting=True, shift=1.0, base_image_seq_len=256,
                     max_image_seq_len=4096, base_shift=0.5, max_shift=1.15)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(456)
        pipeline.register_modules(
            transformer=diffusers.FluxTransformer2DModel(
                patch_size=1, in_channels=8, out_channels=8, num_layers=1, num_single_layers=1,
                attention_head_dim=16, num_attention_heads=2, joint_attention_dim=24,
                pooled_projection_dim=8, guidance_embeds=True, axes_dims_rope=(4, 6, 6)),
            vae=diffusers.AutoencoderKL(
                in_channels=3, out_channels=3, down_block_types=("DownEncoderBlock2D",),
                up_block_types=("UpDecoderBlock2D",), block_out_channels=(8,),
                layers_per_block=1, latent_channels=2, norm_num_groups=4,
                sample_size=4, scaling_factor=2.0, shift_factor=0.25),
            scheduler=diffusers.FlowMatchEulerDiscreteScheduler(**scheduler),
        )
        reward = transformers.CLIPModel(transformers.CLIPConfig(
            projection_dim=4,
            text_config=dict(vocab_size=8, hidden_size=8, intermediate_size=16,
                             num_hidden_layers=1, num_attention_heads=2,
                             max_position_embeddings=77, bos_token_id=0, eos_token_id=1),
            vision_config=dict(hidden_size=8, intermediate_size=16, num_hidden_layers=1,
                               num_attention_heads=2, image_size=4, patch_size=2)))
    base, rm = tmp_path / "flux-asset", tmp_path / "reward-asset"
    pipeline.save_pretrained(base, safe_serialization=True)
    reward.save_pretrained(rm)
    vocab = {"<|startoftext|>": 0, "<|endoftext|>": 1,
             "2</w>": 2, "3</w>": 3, "4</w>": 4, "5</w>": 5}
    if "vocab" in inspect.signature(transformers.CLIPTokenizer).parameters:
        tokenizer = transformers.CLIPTokenizer(vocab=vocab, merges=[], model_max_length=77)
    else:
        (rm / "vocab.json").write_text(json.dumps(vocab))
        (rm / "merges.txt").write_text("#version: 0.2\n")
        tokenizer = transformers.CLIPTokenizer(
            vocab_file=str(rm / "vocab.json"), merges_file=str(rm / "merges.txt"), model_max_length=77)
    image_cls = getattr(transformers, "CLIPImageProcessorPil", None)
    if image_cls is None:
        image_cls = transformers.CLIPImageProcessor
    transformers.CLIPProcessor(
        tokenizer=tokenizer,
        image_processor=image_cls(size={"shortest_edge": 4}, crop_size={"height": 4, "width": 4}),
    ).save_pretrained(rm)

    path = config_file(tmp_path)
    config = yaml.safe_load(path.read_text())
    config["name"] = "flux-native-test"
    config["model"].update(name="flux", checkpoint=str(base))
    config["model"]["options"]["scheduler"] = scheduler
    config["sampling"]["options"].pop("shift")
    config["reward"]["checkpoint"] = str(rm)
    config["reward"]["options"]["processor"] = str(rm)
    config["training"]["precision"] = precision
    path.write_text(yaml.safe_dump(config))
    return path


def records(root):
    return [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]


def assert_evaluations(root, steps=(0, 2, 4)):
    from PIL import Image

    manifests = [EvaluationManifest.from_json(path.read_text())
                 for path in sorted(root.glob("evaluation/step-*/*/manifest.json"))]
    assert [manifest.optimizer_step for manifest in manifests] == list(steps)
    comparable = {}
    for manifest in manifests:
        assert audit_manifest(manifest, media_root=root).complete
        assert len(manifest.samples) == 4
        assert [(sample.sample_id, sample.prompt, sample.seed) for sample in manifest.samples] == [
            (str(i), str(i + 2), 17 + i) for i in range(4)]
        for sample in manifest.samples:
            with Image.open(root / sample.path) as image:
                assert image.format == "PNG" and image.mode == "RGB" and image.size == (4, 4)
                image.load()
            assert sample.framecount == 1 and sample.fps == 1
            scores = sample.scores
            assert scores.space == "fixed_normalized" and scores.weights == {"pickscore": 1.0}
            assert math.isfinite(scores.raw["pickscore"]) and math.isfinite(scores.weighted)
            assert scores.weighted == pytest.approx(scores.raw["pickscore"] / 26, abs=1e-7)
        comparable[manifest.optimizer_step] = [
            (sample.sample_id, sample.seed, sample.content_sha256, sample.scores.to_dict())
            for sample in manifest.samples]
    assert len(list(root.glob("evaluation/step-*/*/*.png"))) == 4 * len(steps)
    return comparable


def assert_checkpoint(root, config, step):
    receipt = json.loads((root / "checkpoints/latest.json").read_text())
    checkpoint = root / "checkpoints" / receipt["file"]
    assert receipt["optimizer_step"] == step and receipt["bytes"] == checkpoint.stat().st_size
    assert receipt["sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    state = payload(root)
    assert set(state) == {"schema_version", "config_sha256", "model", "optimizer", "progress",
                          "rng", "stateful", "generators"}
    assert state["config_sha256"] == config.fingerprint()
    assert state["progress"]["optimizer_steps"] == step
    assert state["progress"]["completed_rollouts"] == step // 2
    assert set(state["stateful"]) == {"prompts"}
    assert set(state["generators"]) == {"sampling"}
    assert set(state["rng"]) == {"python", "numpy", "torch_cpu", "torch_cuda"}
    assert state["rng"]["torch_cuda"] == []
    assert state["optimizer"]["state"] and state["optimizer"]["param_groups"]
    assert all(torch.isfinite(value).all() for value in state["model"].values())
    assert all(value["step"].item() == step for value in state["optimizer"]["state"].values())
    assert any(value["exp_avg"].count_nonzero() > 0
               for value in state["optimizer"]["state"].values())
    return state


def assert_tree_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_tree_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_tree_equal(a, b)
    else:
        assert actual == expected


def test_offline_native_flux_cli_four_updates_eval_and_completed_resume(
        local_flux_config, tmp_path, request, precision):
    path = local_flux_config
    config = load_config(path)
    assert get_backend(config).name == "flux"
    assert main(["--config", str(path), "--validate"]) == 0
    assert not (tmp_path / "run").exists()
    # Exercise the production loader, including separate text-only loading.
    parts = sd3.load_components(config)
    assert isinstance(parts.adapter, FluxAdapter) and isinstance(parts.text, FluxTextEncoder)
    assert parts.text.pipeline.transformer is None and parts.text.pipeline.vae is None
    dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    assert all(p.dtype == dtype and not p.requires_grad
               for e in parts.text._encoders for p in e.parameters())
    del parts
    assert main(["--config", str(path)]) == 0
    root = tmp_path / "run"
    status = json.loads((root / "status.json").read_text())
    assert status["kind"] == "flux_native_development" and status["status"] == "completed"
    assert status["optimizer_steps"] == 4 and status["completed_rollouts"] == 2
    logs = records(root)
    assert [row["train/optimizer_step"] for row in logs] == [1, 2, 3, 4]
    assert [row["system/completed_rollouts"] for row in logs] == [0, 1, 1, 2]
    # Corrective masking may legitimately zero an entire disjoint update window.
    assert any(row["train/grad_norm/before_clip"] > 0 for row in logs)
    for row in logs:
        assert all(math.isfinite(value) for value in row.values() if isinstance(value, (int, float)))
        assert row["cost/elapsed_gpu_hours"] == 0
        assert row["train/grad_norm/before_clip"] >= 0
        assert row["train/reward"] == pytest.approx(row["train/reward/raw/pickscore"] / 26, abs=1e-7)
    assert_evaluations(root)
    before = assert_checkpoint(root, config, 4)
    initial = request.getfixturevalue("real_flux_pipeline").transformer.state_dict()
    assert initial.keys() == before["model"].keys()
    assert any(not torch.equal(initial[key], value) for key, value in before["model"].items())
    receipts = sorted(root.glob("checkpoints/step-*.json"))
    assert [json.loads(p.read_text())["optimizer_step"] for p in receipts] == [2, 4]
    existing = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert main(["--config", str(path), "--resume", str(root / "checkpoints/latest.json")]) == 0
    assert {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()} == existing


def test_offline_native_flux_interrupted_resume_is_exact(local_flux_config, tmp_path, monkeypatch):
    path = local_flux_config
    assert main(["--config", str(path)]) == 0
    expected_root = tmp_path / "run"
    expected = payload(expected_root)
    expected_evaluations = assert_evaluations(expected_root)
    root = tmp_path / "interrupted"
    argv = ["--config", str(path), "--output-dir", str(root)]
    save = sd3.save_checkpoint

    def stop_after_boundary(*args, **kwargs):
        receipt = save(*args, **kwargs)
        if kwargs["progress"].optimizer_steps == 2:
            raise RuntimeError("injected shutdown after full FLUX checkpoint")
        return receipt

    with monkeypatch.context() as patch:
        patch.setattr(sd3, "save_checkpoint", stop_after_boundary)
        with pytest.raises(RuntimeError, match="injected shutdown after full FLUX checkpoint"):
            main(argv)
    assert not (root / "status.json").exists()
    config = load_config(path, {"runtime": {"output_dir": str(root)}})
    assert_checkpoint(root, config, 2)
    assert_evaluations(root, steps=(0, 2))
    prefix = (root / "metrics.jsonl").read_bytes()
    assert [row["train/optimizer_step"] for row in records(root)] == [1, 2]
    assert main([*argv, "--resume", str(root / "checkpoints/latest.json")]) == 0
    actual = assert_checkpoint(root, config, 4)
    for key in ("model", "optimizer", "stateful", "generators", "rng"):
        assert_tree_equal(actual[key], expected[key])
    for key in ("optimizer_steps", "completed_rollouts", "attempts"):
        assert actual["progress"][key] == expected["progress"][key]
    assert (root / "metrics.jsonl").read_bytes().startswith(prefix)
    actual_logs, expected_logs = records(root), records(expected_root)
    assert [row["train/optimizer_step"] for row in actual_logs] == [1, 2, 3, 4]
    for a, b in zip(actual_logs, expected_logs):
        assert {k: v for k, v in a.items() if k.startswith("train/")} == {
            k: v for k, v in b.items() if k.startswith("train/")}
    assert assert_evaluations(root) == expected_evaluations
