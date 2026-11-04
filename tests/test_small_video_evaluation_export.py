"""CPU artifact/RNG/identity fixtures, not pretrained decode or reward evidence."""

import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from visual_rl.backends import small_video as backend
from visual_rl.config import load_config
from visual_rl.data import Prompt, PromptDataset
from visual_rl.small_video_export import load_final_into_adapter, verify_final
from visual_rl.training.distributed import DistributedSession, RankInfo
from visual_rl.training.small_video_checkpoint import SmallVideoIdentity, _hash

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config(monkeypatch, tmp_path):
    monkeypatch.setenv("SMALL_VIDEO_ASSETS", str(tmp_path / "assets"))
    monkeypatch.setenv("SMALL_VIDEO_OUTPUT", str(tmp_path / "run"))
    monkeypatch.setenv("SMALL_VIDEO_REWARD_PYTHON", "/unused/fake-python")
    return load_config(ROOT / "configs/small_video/ltx_video_2b_distilled_full.yaml")


def test_paired_eval_preserves_all_rng_and_validates_complete_files(config, monkeypatch, tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    session = DistributedSession(RankInfo(), torch.device("cpu"), False)
    policy = torch.nn.Linear(1, 1)
    parts = SimpleNamespace(
        adapter=SimpleNamespace(transformer=policy), scheduler_factory=None, offset=0.0
    )
    data = PromptDataset(
        tuple(Prompt(f"heldout-{i}", f"heldout prompt {i}") for i in range(8)), "a" * 64
    )
    monkeypatch.setattr(backend, "latent_shape", lambda c, n: (n, 2, 2, 1, 2))
    monkeypatch.setattr(
        backend, "make_predict", lambda *args: lambda x, sigma, rows, step: x * 0.05
    )

    def score(c, parts, endpoints, prompts, directory, device):
        directory.mkdir(parents=True)
        path = directory / "fixture.mp4"
        path.write_bytes(b"not-a-video; artifact-only-fixture")
        random.random()
        np.random.rand()
        torch.rand(1)  # all must be restored after evaluation
        raw = {
            key: torch.tensor([value + parts.offset])
            for key, value in (("VQ", 1.0), ("MQ", 2.0), ("TA", 3.0))
        }
        return [path], SimpleNamespace(raw=raw)

    monkeypatch.setattr(backend, "score_videos", score)
    before = (torch.get_rng_state().clone(), random.getstate(), np.random.get_state())
    base = backend.evaluate(session, config, parts, data, 0, root)
    parts.offset = 2.0
    post = backend.evaluate(session, config, parts, data, 10, root)
    assert torch.equal(before[0], torch.get_rng_state()) and before[1] == random.getstate()
    assert np.array_equal(before[2][1], np.random.get_state()[1])
    delta = backend.paired_evaluation_change(base / "samples.json", post / "samples.json")
    assert delta["VQ_MQ_equal_mean"]["paired_mean_delta"] == 2.0
    assert delta["VQ_MQ_equal_mean"]["paired_bootstrap_95_percent_interval"] == [2.0, 2.0]
    marker = (post / "COMPLETE.json").read_bytes()
    assert backend.evaluate(session, config, parts, data, 10, root) == post
    assert (post / "COMPLETE.json").read_bytes() == marker
    sample = json.loads((post / "samples.json").read_text())["records"][0]
    (post / sample["video"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        backend.evaluate(session, config, parts, data, 10, root)


def test_paired_raw_analysis_rejects_noise_mismatch(tmp_path):
    row = dict(
        index=0,
        sample_id="x",
        prompt="heldout",
        seed=42,
        initial_noise_sha256="a",
        transition_innovations_sha256="b",
        decode_rng_sha256="c",
        raw={"VQ": 1.0, "MQ": 2.0, "TA": 3.0},
    )
    paths = [tmp_path / "before.json", tmp_path / "after.json"]
    paths[0].write_text(json.dumps({"records": [row]}))
    paths[1].write_text(json.dumps({"records": [{**row, "decode_rng_sha256": "other"}]}))
    with pytest.raises(ValueError, match="noise identity"):
        backend.paired_evaluation_change(*paths)


def test_named_full_export_fresh_load_exact_and_corruption_fails(config, tmp_path):
    session = DistributedSession(RankInfo(), torch.device("cpu"), False)
    model = torch.nn.Linear(3, 2)
    adapter = SimpleNamespace(transformer=model, trainable_module=model)
    root = tmp_path / "run"
    root.mkdir()
    protocol = backend.resolved_protocol(config)
    identity = SmallVideoIdentity(
        config.model.name,
        "full",
        "original",
        "50f0aef1297574a4b29acbdb874e3a363a6d5f2e",
        config.model.revision,
        config.reward.revision,
        "9d947fb510d67a8e5334ec1d8c7d8927f7658c08",
        config.data.sha256,
        _hash(protocol),
        config.evaluation.manifest_sha256,
        config.fingerprint(),
    )
    state = SimpleNamespace(identity=identity, budget=SimpleNamespace(completed=10))
    final = backend.export_final(
        session, config, SimpleNamespace(adapter=adapter), state, "fixture-receipt", root
    )
    assert verify_final(final)["format"] == "native_transformer_state_dict"
    fresh = torch.nn.Linear(3, 2)
    load_final_into_adapter(config, final, SimpleNamespace(trainable_module=fresh))
    torch.testing.assert_close(fresh(torch.ones(2, 3)), model(torch.ones(2, 3)), atol=0, rtol=0)
    assert (
        backend.export_final(
            session, config, SimpleNamespace(adapter=adapter), state, "fixture-receipt", root
        )
        == final
    )
    (final / "policy.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity differs"):
        verify_final(final)
