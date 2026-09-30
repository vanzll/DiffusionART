"""Annotation identity survives replication, decode chunks and scorer composition."""

import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from test_sd3_runner import config_file, threads, tiny_parts  # noqa: F401

from visual_rl.backends import sd3
from visual_rl.config import load_config
from visual_rl.rewards.composite import CompositeImageReward
from visual_rl.rewards.image_dispatch import score_image_batch
from visual_rl.rewards.scalar import weighted_score
from visual_rl.training.image import decode_and_score_images


class AnnotatedScore(torch.nn.Module):
    requires_metadata = True

    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, images, prompts, *, microbatch, metadata):
        assert [row["prompt"] for row in metadata] == list(prompts)
        values = torch.tensor([row["value"] for row in metadata], dtype=torch.float32)
        self.seen.extend(row["value"] for row in metadata)
        for row in metadata:
            row["value"] = -100
        return weighted_score({"annotation": values}, {"annotation": 1}, space="raw")


class PlainScore(torch.nn.Module):
    def forward(self, images, prompts, *, microbatch):
        return weighted_score({"plain": torch.ones(len(prompts))}, {"plain": 1}, space="raw")


@pytest.mark.parametrize("decode_microbatch", [1, 3, 8])
def test_chunks_pair_annotations_and_do_not_mutate_caller(decode_microbatch):
    prompts = ["A", "A", "B", "B"]
    metadata = [{"prompt": p, "value": i} for i, p in enumerate(prompts)]
    scorer = AnnotatedScore()
    adapter = SimpleNamespace(vae=torch.nn.Linear(1, 1), decode=lambda x: x)
    result = decode_and_score_images(
        adapter, torch.zeros(4, 3, 2, 2), prompts, scorer,
        decode_microbatch=decode_microbatch, score_microbatch=2, metadata=metadata,
    )
    assert result.scores.weighted.tolist() == list(range(4))
    assert scorer.seen == list(range(4))
    assert [row["value"] for row in metadata] == list(range(4))


@pytest.mark.parametrize("metadata", [None, [], [None], [{}, {}], "wrong"])
def test_missing_or_misaligned_annotations_fail_before_decode(metadata):
    def decode(x):
        raise AssertionError("invalid metadata must be caught before VAE work")

    adapter = SimpleNamespace(vae=torch.nn.Linear(1, 1), decode=decode)
    with pytest.raises(ValueError, match="metadata"):
        decode_and_score_images(adapter, torch.zeros(1, 3, 2, 2), ["A"], AnnotatedScore(),
                                decode_microbatch=1, score_microbatch=1, metadata=metadata)


def test_composite_routes_metadata_only_to_explicit_consumers():
    scorer = CompositeImageReward(
        {"annotation": AnnotatedScore(), "plain": PlainScore()},
        {"annotation": 2, "plain": 3}, space="raw",
    )
    metadata = [{"prompt": "A", "value": 4}]
    result = score_image_batch(scorer, torch.zeros(1, 3, 2, 2), ["A"],
                               microbatch=1, metadata=metadata)
    assert result.weighted.item() == 11
    assert metadata[0]["value"] == 4
    with pytest.raises(ValueError, match="requires metadata"):
        scorer(torch.zeros(1, 3, 2, 2), ["A"], microbatch=1)


def test_native_train_and_eval_keep_annotation_identity(tmp_path, monkeypatch):
    path = config_file(tmp_path)
    data_path = tmp_path / "prompts.jsonl"
    data = [json.loads(line) for line in data_path.read_text().splitlines()]
    for row in data:
        row["metadata"] = {"prompt": row["prompt"], "nested": [row["sample_id"]]}
    data_path.write_text("".join(json.dumps(row) + "\n" for row in data))
    import yaml

    config = yaml.safe_load(path.read_text())
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    config["data"]["sha256"] = config["evaluation"]["manifest_sha256"] = digest
    path.write_text(yaml.safe_dump(config))
    seen = []

    class CheckedScorer(torch.nn.Module):
        requires_metadata = True

        def __init__(self, original):
            super().__init__()
            self.original = original

        def forward(self, images, prompts, *, microbatch, metadata):
            for prompt, row in zip(prompts, metadata, strict=True):
                assert row == {"prompt": prompt, "nested": [str(int(prompt) - 2)]}
                seen.append(prompt)
                row["nested"].clear()
            return self.original(images, prompts, microbatch=microbatch)

    def components(c):
        parts = tiny_parts(c)
        return sd3.Components(parts.adapter, parts.text, CheckedScorer(parts.scorer))

    monkeypatch.setattr(sd3, "load_components", components)
    sd3.run(load_config(path))
    # Three complete evals, plus two rollouts of P2/K2.
    assert len(seen) == 3 * 4 + 2 * 4
    assert seen[:4] == ["2", "3", "4", "5"]
