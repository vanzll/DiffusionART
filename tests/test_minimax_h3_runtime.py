"""Execution fixtures, not pretrained H3/scorer or GPU acceptance evidence."""
import hashlib
import json
import os
import shutil
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp

from visual_rl.backends import minimax_h3 as backend
from visual_rl.backends import minimax_h3_runtime as runtime
from visual_rl.config import ConfigError, load_config
from visual_rl.models.minimax_h3 import MiniMaxH3Conditioning
from visual_rl.rewards.audio_video import AVScores
from visual_rl.training.audio_video_types import AVMedia, AVState


class TinyPolicy(torch.nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.video = torch.nn.Linear(2, 2, bias=False)
        self.audio = torch.nn.Linear(3, 3, bias=False)
        self.base = torch.nn.Linear(2, 2, bias=False)
        self.base.requires_grad_(mode == "full")

    def forward(self, state):
        return AVState(self.video(state.video) + self.base(state.video), self.audio(state.audio))


class TinyAdapter:
    def __init__(self, mode):
        self.predictions = 0
        self.transformer = TinyPolicy(mode)
        self.vae = torch.nn.Linear(1, 1)
        self.audio_vae = torch.nn.Linear(1, 1)

    @property
    def trainable_module(self):
        return self.transformer

    def set_trainable_module(self, module):
        self.transformer = module

    def latent_shapes(self, count, height, width, frames, fps):
        self.frames = frames
        return (count, 2), (count, 3)

    def predict_velocity(self, state, video_sigma, audio_sigma, conditioning):
        self.predictions += 1
        values = self.transformer(AVState(state.video.float(), state.audio.float()))
        prompt = conditioning.prompt_embeds[:, 0, 0].reshape(-1, 1) * .001
        return AVState(values.video + prompt, values.audio + prompt)

    def decode(self, state):
        frames = getattr(self, "frames", 5)
        samples = round(frames / 24 * 32000)
        video = state.video.float().mean(1).sigmoid().reshape(-1, 1, 1, 1, 1).expand(-1, 3, frames, 4, 4).clone()
        audio = state.audio.float().mean(1).tanh().reshape(-1, 1, 1).expand(-1, 2, samples).clone()
        return AVMedia(video, audio, 24., 32000)


class TinyConditioner:
    def __init__(self):
        self.text_encoder = torch.nn.Linear(1, 1)

    def encode(self, prompts):
        return MiniMaxH3Conditioning(torch.tensor([len(p) for p in prompts]).float().reshape(-1, 1, 1),
                                    torch.ones(len(prompts), 1, dtype=torch.bool))


class TinyReward:
    closed = False

    def score(self, media, video_prompts, audio_prompts):
        assert len(video_prompts) == len(audio_prompts) == len(media)
        assert all(text.startswith("visual") for text in video_prompts)
        assert all(text.startswith("sound") for text in audio_prompts)
        v = media.video.flatten(1).mean(1)
        a = media.audio.flatten(1).mean(1)
        return AVScores({"video_align": v, "hpsv3": v * 2, "audiobox": a,
                         "clap": a * .2, "desync": v + a * .1}, {"fixture_raw": v})

    def close(self):
        self.closed = True


def fixture_api(created):
    def load(c, device):
        parts = SimpleNamespace(adapter=TinyAdapter(c.model.options["finetune"]["mode"]), conditioner=TinyConditioner(),
                                reward=TinyReward(), manifest={"kind": "explicit_cpu_fixture"})
        created.append(parts)
        return parts
    return SimpleNamespace(validate_config=backend.validate_config, load_datasets=backend.load_datasets,
                           load_components=load, updates_per_rollout=backend.updates_per_rollout)


@pytest.fixture
def config_factory(tmp_path, monkeypatch):
    source = tmp_path / "prompts.jsonl"
    source.write_text("".join(json.dumps({"sample_id": f"p{i}", "prompt": "generate " + "word " * (i + 1),
        "metadata": {"reward_text": {"video": f"visual {i}", "audio": f"sound {i}"}}}) + "\n" for i in range(3)))
    for key, value in {"H3_MODEL": str(tmp_path), "H3_TRAIN_MANIFEST": str(source),
        "H3_TRAIN_SHA256": hashlib.sha256(source.read_bytes()).hexdigest(), "H3_REWARD_MANIFEST": str(tmp_path / "scorers.json")}.items():
        monkeypatch.setenv(key, value)
    def config(algo="nft", mode="lora", *, name="run", evaluation=False):
        root = Path(__file__).resolve().parents[1]
        updates = 4
        overrides = {"sampling": {"steps": 5, "prompts_per_rollout": 2, "samples_per_prompt": 4,
            "microbatch": 3, "height": 32, "width": 32, "frames": 5},
            "training": {"max_updates": updates, "microbatch": 3},
            "runtime": {"output_dir": str(tmp_path / name), "device": "cpu", "distributed": "none", "checkpoint_every": 2}}
        if evaluation:
            overrides["evaluation"] = {"every_updates": updates, "manifest": str(source),
                "manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "sampling_steps": 5}
        return load_config(root / f"configs/examples/minimax_h3_{algo}_{mode}.yaml", overrides)
    return config


@pytest.mark.parametrize("algorithm", ["art", "nft", "fast"])
@pytest.mark.parametrize("mode", ["lora", "full"])
def test_six_execution_fixtures_update_actual_owners_and_publish_complete_checkpoint(config_factory, algorithm, mode):
    c = config_factory(algorithm, mode)
    created = []
    result = runtime.run(c, backend_api=fixture_api(created))
    root = Path(c.runtime.output_dir)
    assert result["optimizer_steps"] == 4
    assert result["completed_rollouts"] == (4 if algorithm == "nft" else 2)
    assert created[0].reward.closed
    run = json.loads((root / "run.json").read_text())
    assert run["pretrained_reference"] is False
    assert run["actual_output_dir"] == str(root.resolve())
    metrics = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
    assert [m["train/optimizer_step"] for m in metrics] == ([1, 2, 3, 4] if algorithm == "nft" else [2, 4])
    assert all(m["train/optimizer_steps_per_rollout"] == (1 if algorithm == "nft" else 2) for m in metrics)
    assert all(m["av/transitions"] == 4 for m in metrics)
    assert all(m["av/training_source_position_min"] >= 0 for m in metrics)
    assert all(m["av/training_source_position_max"] < 4 for m in metrics)
    assert all(m["av/nft_fresh_time_floor_quantized"] == int(algorithm == "nft") for m in metrics)
    for row in metrics:
        assert 0 < row["av/training_video_sigma_min"] <= row["av/training_video_sigma_max"] <= 1
        assert 0 < row["av/training_audio_sigma_min"] <= row["av/training_audio_sigma_max"] <= 1
        if algorithm == "nft":
            assert row["av/training_video_sigma_min"] == row["av/training_audio_sigma_min"]
            assert row["av/training_source_position_max"] == 3
        elif algorithm == "art":
            assert row["av/training_source_position_max"] == 2
            assert row["av/training_video_sigma_min"] > row["av/training_audio_sigma_min"]
    assert all(m["av/training_positions"] == (4 if algorithm == "nft" else 3) for m in metrics)
    latest = json.loads((root / "checkpoints/latest.json").read_text())
    receipt = json.loads((root / "checkpoints" / latest["receipt"]).read_text())
    payload = torch.load(root / "checkpoints" / receipt["file"], weights_only=True)
    assert payload["progress"]["optimizer_steps"] == 4
    assert set(payload["generators"]) == {"av_sampling"}
    assert set(payload["stateful"]) == ({"h3_asset_identity", "prompts", "nft_old_policy", "av_nft_counters"} if algorithm == "nft" else {"h3_asset_identity", "prompts"})
    if algorithm == "nft":
        assert payload["stateful"]["av_nft_counters"]["successful_updates"] == 4


def test_native_grid_has_n_boundaries_and_independent_shifts():
    video, audio = runtime.sigma_schedules(10)
    assert len(video) == len(audio) == 10
    assert video[0] == audio[0] == 1
    assert video[-1] == audio[-1] == 0
    assert (video[1:-1] > audio[1:-1]).all()
    with pytest.raises(ConfigError):
        runtime.sigma_schedules(1)


def _checkpoint_payload(root):
    directory = Path(root) / "checkpoints"
    latest = json.loads((directory / "latest.json").read_text())
    receipt = json.loads((directory / latest["receipt"]).read_text())
    return torch.load(directory / receipt["file"], weights_only=True)


@pytest.mark.parametrize("algorithm", ["art", "nft", "fast"])
def test_real_runtime_save_load_continuation_matches_uninterrupted(config_factory, monkeypatch, algorithm):
    original = config_factory(algorithm, name="original")
    runtime.run(original, backend_api=fixture_api([]))
    expected = _checkpoint_payload(original.runtime.output_dir)
    interrupted = config_factory(algorithm, name="interrupted")
    save = runtime.save_checkpoint
    receipts = []
    class StopAfterPublication(Exception):
        pass
    def stop(*args, **kwargs):
        receipt = save(*args, **kwargs)
        receipts.append(receipt)
        raise StopAfterPublication
    monkeypatch.setattr(runtime, "save_checkpoint", stop)
    with pytest.raises(StopAfterPublication):
        runtime.run(interrupted, backend_api=fixture_api([]))
    monkeypatch.setattr(runtime, "save_checkpoint", save)
    runtime.run(interrupted, resume=receipts[0], backend_api=fixture_api([]))
    actual = _checkpoint_payload(interrupted.runtime.output_dir)
    for name, tensor in expected["model"].items():
        torch.testing.assert_close(actual["model"][name], tensor, atol=0, rtol=0)
    assert actual["stateful"]["prompts"] == expected["stateful"]["prompts"]
    torch.testing.assert_close(actual["generators"]["av_sampling"], expected["generators"]["av_sampling"], atol=0, rtol=0)
    if algorithm == "nft":
        assert actual["stateful"]["av_nft_counters"] == expected["stateful"]["av_nft_counters"]
        for name, tensor in expected["stateful"]["nft_old_policy"].items():
            torch.testing.assert_close(actual["stateful"]["nft_old_policy"][name], tensor, atol=0, rtol=0)


def test_existing_output_is_protected(config_factory):
    c = config_factory()
    Path(c.runtime.output_dir).mkdir()
    protected = Path(c.runtime.output_dir) / "existing"
    protected.write_text("keep")
    with pytest.raises(FileExistsError):
        runtime.run(c, backend_api=fixture_api([]))
    assert protected.read_text() == "keep"


def test_media_real_ffmpeg_preserves_float_wav_and_native_rates(tmp_path):
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("FFmpeg not installed")
    media = AVMedia(torch.rand(1, 3, 1, 4, 4), torch.linspace(-1.2, 1.2, 2666).reshape(1, 2, 1333), 24., 32000)
    path = runtime.save_media(media, tmp_path / "sample.mp4")
    assert path.is_file() and path.with_suffix(".wav").is_file()
    metadata = json.loads(path.with_suffix(".media.json").read_text())
    assert metadata["audio_master_codec"] == "pcm_f32le"
    assert metadata["audio_samples"] == 1333 and metadata["channels"] == 2
    assert metadata["resampled"] is metadata["trimmed"] is False


def test_evaluate_only_restores_checkpoint_and_publishes_av_manifest(config_factory):
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("FFmpeg not installed")
    c = config_factory(evaluation=True)
    api = fixture_api([])
    runtime.run(c, backend_api=api)
    root = Path(c.runtime.output_dir)
    latest = json.loads((root / "checkpoints/latest.json").read_text())
    receipt = root / "checkpoints" / latest["receipt"]
    result = runtime.run(c, resume=receipt, evaluate_only=True, backend_api=api)
    assert result["eval/optimizer_step"] == 4
    assert result["eval/samples"] == 3
    manifest = json.loads(Path(result["manifest"]).read_text())
    assert len(manifest["samples"]) == 3
    for sample in manifest["samples"]:
        path = root / sample["path"]
        assert path.is_file() and path.with_suffix(".wav").is_file()


def _fsdp_worker(rank, config, directory):
    torch.set_num_threads(1)
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE="2")
    session_factory = runtime.distributed_session
    attempt = [0]
    def make_session(*args, **kwargs):
        path = Path(directory) / f"rendezvous-{attempt[0]}"
        attempt[0] += 1
        return session_factory(*args, **kwargs, init_method=f"file://{path}")
    runtime.distributed_session = make_session
    policy_wrapper = runtime.wrap_policy
    runtime.wrap_policy = lambda model, c, device: policy_wrapper(model, c, device, block_types=(torch.nn.Linear,))
    original = replace(config, runtime=replace(config.runtime, output_dir=str(Path(directory) / "original"), distributed="fsdp"))
    created = []
    runtime.run(original, backend_api=fixture_api(created))
    expected = {name: parameter.detach().cpu().clone() for name, parameter in created[-1].adapter.trainable_module.named_parameters()}
    interrupted = replace(config, runtime=replace(config.runtime, output_dir=str(Path(directory) / "resumed"), distributed="fsdp"))
    save = runtime.save_fsdp_checkpoint
    receipts = []
    class StopAfterPublication(Exception):
        pass
    def stop(*args, **kwargs):
        receipt = save(*args, **kwargs)
        receipts.append(receipt)
        raise StopAfterPublication
    runtime.save_fsdp_checkpoint = stop
    try:
        runtime.run(interrupted, backend_api=fixture_api([]))
    except StopAfterPublication:
        pass
    else:
        raise AssertionError("checkpoint interrupt did not occur")
    runtime.save_fsdp_checkpoint = save
    continued = []
    result = runtime.run(interrupted, resume=receipts[0], backend_api=fixture_api(continued))
    actual = dict(continued[-1].adapter.trainable_module.named_parameters())
    for name, expected_value in expected.items():
        torch.testing.assert_close(actual[name].detach().cpu(), expected_value, atol=0, rtol=0)
    assert result["optimizer_steps"] == 4
    (Path(directory) / f"rank-{rank}.ok").write_text("FSDP exact continuation passed")


@pytest.mark.parametrize("algorithm", ["art", "nft", "fast"])
@pytest.mark.parametrize("mode", ["lora", "full"])
def test_real_two_rank_fsdp_six_fixture_paths_save_resume(config_factory, tmp_path, algorithm, mode):
    config = config_factory(algorithm, mode)
    ctx = mp.spawn(_fsdp_worker, args=(config, str(tmp_path)), nprocs=2, join=False)
    deadline = time.monotonic() + 60
    try:
        while not ctx.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("H3 FSDP fixture exceeded 60 second supervisor deadline")
    finally:
        for process in ctx.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    assert all((tmp_path / f"rank-{rank}.ok").is_file() for rank in range(2))


@pytest.mark.parametrize("changed", ["model", "reward"])
@pytest.mark.parametrize("evaluate_only", [False, True])
def test_same_config_changed_legal_asset_identity_refused_before_forward(config_factory, changed, evaluate_only):
    c = config_factory()
    runtime.run(c, backend_api=fixture_api([]))
    directory = Path(c.runtime.output_dir) / "checkpoints"
    latest = json.loads((directory / "latest.json").read_text())
    receipt = directory / latest["receipt"]
    created = []
    api = fixture_api(created)
    original_loader = api.load_components
    def changed_loader(*args):
        parts = original_loader(*args)
        parts.manifest[changed] = {"revision": "another-valid-pinned-identity"}
        return parts
    api.load_components = changed_loader
    with pytest.raises(ConfigError, match="asset identity"):
        runtime.run(c, resume=receipt, evaluate_only=evaluate_only, backend_api=api)
    assert created[0].adapter.predictions == 0


def _identity_worker(rank, config, directory):
    torch.set_num_threads(1)
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE="2")
    session_factory = runtime.distributed_session
    runtime.distributed_session = lambda *args, **kwargs: session_factory(*args, **kwargs,
        init_method=f"file://{directory}/identity-rendezvous")
    api = fixture_api([])
    loader = api.load_components
    def different(*args):
        parts = loader(*args)
        parts.manifest["reward_inventory"] = "a" if rank == 0 else "b"
        return parts
    api.load_components = different
    config = replace(config, runtime=replace(config.runtime, distributed="fsdp"))
    try:
        runtime.run(config, backend_api=api)
    except ConfigError as error:
        assert "identity" in str(error)
    else:
        raise AssertionError("rank-different asset identities were accepted")
    (Path(directory) / f"identity-{rank}.ok").write_text("rejected before any model forward")


def test_rank_different_asset_identity_rejected_before_model_collectives(config_factory, tmp_path):
    config = config_factory()
    ctx = mp.spawn(_identity_worker, args=(config, str(tmp_path)), nprocs=2, join=False)
    deadline = time.monotonic() + 45
    try:
        while not ctx.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("asset identity consensus exceeded 45 seconds")
    finally:
        for process in ctx.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    assert all((tmp_path / f"identity-{rank}.ok").is_file() for rank in range(2))


def test_checkpoint_asset_component_is_immutable():
    identity = runtime.H3AssetIdentity({"model": {"revision": "fixed"}, "reward": {"manifest_sha256": "a"}})
    payload = identity.state_dict()
    payload["assets"]["reward"]["manifest_sha256"] = "b"
    assert identity.state_dict()["assets"]["reward"]["manifest_sha256"] == "a"
    with pytest.raises(ConfigError, match="asset identity"):
        identity.load_state_dict(payload)
