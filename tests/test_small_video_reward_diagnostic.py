"""CPU fixtures for opt-in existing-video scoring; no real reward/GPU evidence."""

import json
import subprocess
from types import SimpleNamespace

import pytest
import torch

from visual_rl import small_video_parity as P
from visual_rl.rewards import small_video_process as R


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    paths = [tmp_path / f"matched-{index}.mp4" for index in range(2)]
    for path in paths:
        path.write_bytes(b"CPU video metadata fixture")
    return dict(
        asset_root=tmp_path / "assets",
        receipt=tmp_path / "receipt.json",
        owned_root=tmp_path,
        output=tmp_path / "diagnostic",
        videos=paths,
        prompts=["private prompt one", "private prompt two"],
        device="cuda",
    )


@pytest.fixture
def runtime(inputs, monkeypatch):
    calls = []
    monkeypatch.setenv("SMALL_VIDEO_REWARD_PYTHON", "/usr/bin/python3")
    monkeypatch.setattr(
        P,
        "_reward_video_profile",
        lambda path, **kwargs: {
            "width": 1216,
            "height": 704,
            "nb_read_frames": "153",
            "avg_frame_rate": "30/1",
        },
    )
    monkeypatch.setattr(
        P,
        "load_manifest",
        lambda *args: {
            "snapshots": [
                {"id": "reward", "revision": R.REWARD_REVISION},
                {"id": "reward_backbone", "revision": "895c3a49bc3fa70a340399125c650a463535e71c"},
            ],
            "sources": [{"id": "reward", "revision": R.REWARD_SOURCE_REVISION}],
        },
    )
    monkeypatch.setattr(
        P,
        "verify_receipt",
        lambda *args, **kwargs: {
            "checksum": "c" * 64,
            "assets": {"reward": "fake/reward", "reward_backbone": "fake/backbone"},
            "sources": [
                {
                    "id": "reward",
                    "path": "fake/canonical-source",
                    "revision": R.REWARD_SOURCE_REVISION,
                    "package": "fastvideo",
                }
            ],
        },
    )
    monkeypatch.setattr(
        P,
        "_reward_device_identity",
        lambda: (torch.device("cuda:0"), {"gpu_uuid": "GPU-fake", "visible_device": "GPU-fake"}),
    )

    class Scorer:
        raw = {
            "VQ": torch.tensor([1.0, 2.0]),
            "MQ": torch.tensor([3.0, 4.0]),
            "TA": torch.tensor([5.0, 6.0]),
        }
        failure = None

        def __init__(self, **kwargs):
            calls.append(("init", kwargs))
            self.last_phase = kwargs["scratch"] / "phase-fake"
            self.last_metrics = {"reward/scoring_seconds": 0.1}

        def to(self, device):
            calls.append(("to", str(device)))
            return self

        def eval(self):
            return self

        def score_paths(self, paths, prompts):
            calls.append(("score", paths, prompts))
            if self.failure is not None:
                raise self.failure
            return SimpleNamespace(raw=self.raw, space="raw")

    monkeypatch.setattr(R, "PhasedVideoReward", Scorer)
    return calls, Scorer


def test_reward_only_default_gate_and_plan_never_query_cuda(inputs, monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CUDA without opt-in"))
    with pytest.raises(ValueError, match="explicit --device cuda"):
        P.run_reward_only(**dict(inputs, device="cpu"))
    assert P.main(["--asset-root", str(inputs["asset_root"]), "--reward-only", "--plan"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["diagnostic_only"] and not result["gpu_execution"]


@pytest.mark.parametrize(
    "change", ["outside", "symlink", "missing", "duplicate", "empty_prompt", "wrong_count"]
)
def test_reward_paths_and_arguments_fail_before_cuda(inputs, monkeypatch, change):
    monkeypatch.setattr(
        P, "_reward_device_identity", lambda: pytest.fail("CUDA before path preflight")
    )
    if change == "outside":
        inputs["output"] = inputs["owned_root"].parent / "outside"
    elif change == "symlink":
        target = inputs["owned_root"] / "alias.mp4"
        target.symlink_to(inputs["videos"][0])
        inputs["videos"][0] = target
    elif change == "missing":
        inputs["videos"][0].unlink()
    elif change == "duplicate":
        inputs["videos"][1] = inputs["videos"][0]
    elif change == "empty_prompt":
        inputs["prompts"][0] = " "
    else:
        inputs["videos"] = inputs["videos"][:1]
    with pytest.raises(ValueError):
        P.run_reward_only(**inputs)


def test_reward_video_profile_checks_actual_ffprobe_response_and_bound_timeout(inputs, monkeypatch):
    commands = []
    stream = {"width": 1216, "height": 704, "nb_read_frames": "153", "avg_frame_rate": "30/1"}

    def probe(command, **kwargs):
        commands.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps({"streams": [stream]}))

    monkeypatch.setattr(P.subprocess, "run", probe)
    assert P._reward_video_profile(inputs["videos"][0], ffprobe="ffprobe") == stream
    assert "-count_frames" in commands[0][0] and commands[0][1]["timeout"] == 120
    stream["nb_read_frames"] = "152"
    with pytest.raises(ValueError, match="153frames"):
        P._reward_video_profile(inputs["videos"][0], ffprobe="ffprobe")


def test_reward_only_raw_components_pins_and_cpu_reset_without_prompt_stdout(
    inputs, runtime, capsys
):
    calls, _ = runtime
    result = P.run_reward_only(**inputs)
    assert result["status"] == "passed" and result["diagnostic_only"]
    assert not result["training_acceptance"] and result["optimizer_calls"] == 0
    assert result["rows"][0]["raw_VQ_MQ_mean"] == 2.0
    assert result["hardware"]["gpu_uuid"] == "GPU-fake"
    assert result["reward_fps"] == 4 and result["use_norm"] is False
    assert calls[-1] == ("to", "cpu")
    assert "private prompt" not in json.dumps(result) and not capsys.readouterr().out
    json.dumps(json.loads((inputs["output"] / "report.json").read_text()), allow_nan=False)


def test_reward_uses_real_default_manifest_and_actual_receipt_source_shape(
    inputs, runtime, monkeypatch
):
    from visual_rl.small_video_assets import load_manifest

    # Actual published default manifest; no asset reads/downloads or GPU calls.
    # Receipt metadata retains the real list-of-records public schema.
    monkeypatch.setattr(P, "load_manifest", load_manifest)
    spec = load_manifest()
    source = next(row for row in spec["sources"] if row["id"] == "reward")
    assert source["revision"] == R.REWARD_SOURCE_REVISION
    assert source["package"] == "fastvideo"
    result = P.run_reward_only(**inputs)
    calls, _ = runtime
    init = next(call[1] for call in calls if call[0] == "init")
    assert init["native_root"] == "fake/canonical-source"
    assert result["status"] == "passed"
    assert result["reward_source_revision"] == source["revision"]


def test_reward_wrong_source_pin_fails_before_cuda_or_worker(inputs, runtime, monkeypatch):
    calls, _ = runtime
    spec = P.load_manifest()
    spec["sources"][0]["revision"] = "0" * 40
    monkeypatch.setattr(P, "load_manifest", lambda *args: spec)
    monkeypatch.setattr(
        P, "_reward_device_identity", lambda: pytest.fail("CUDA before canonical source check")
    )
    with pytest.raises(ValueError, match="pinned canonical"):
        P.run_reward_only(**inputs)
    assert not calls


@pytest.mark.parametrize("failure", ["nonfinite", "missing", "worker"])
def test_reward_error_report_and_final_cpu_reset(inputs, runtime, failure):
    calls, scorer = runtime
    if failure == "worker":
        scorer.failure = R.RewardProcessError("owned worker already drained")
    elif failure == "nonfinite":
        scorer.raw = {**scorer.raw, "TA": torch.tensor([float("nan"), 1.0])}
    else:
        scorer.raw = {"VQ": torch.ones(2), "MQ": torch.ones(2)}
    with pytest.raises((ValueError, R.RewardProcessError)):
        P.run_reward_only(**inputs)
    report = json.loads((inputs["output"] / "report.json").read_text())
    assert report["status"] == "error" and not report["training_acceptance"]
    assert calls[-1] == ("to", "cpu")


def test_reward_cli_dispatch_is_separate_from_default_parity(inputs, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(
        P, "run_reward_only", lambda **kwargs: calls.append(kwargs) or {"status": "passed"}
    )
    monkeypatch.setattr(
        P, "run_ltx_parity", lambda **kwargs: pytest.fail("loaded generator for reward-only")
    )
    argv = [
        "--asset-root",
        str(inputs["asset_root"]),
        "--reward-only",
        "--device",
        "cuda",
        "--owned-root",
        str(inputs["owned_root"]),
        "--output",
        str(inputs["output"]),
        "--reward-videos",
        *map(str, inputs["videos"]),
        "--reward-prompts",
        *inputs["prompts"],
    ]
    assert P.main(argv) == 0 and len(calls) == 1
    assert "private prompt" not in capsys.readouterr().out
    with pytest.raises(SystemExit):
        P.main(
            ["--asset-root", str(inputs["asset_root"]), "--owned-root", str(inputs["owned_root"])]
        )
