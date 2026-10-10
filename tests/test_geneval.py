import copy
from io import BytesIO

import numpy as np
import pytest
import torch
from PIL import Image

from visual_rl.rewards.geneval import (
    TAGS,
    GenEvalScore,
    jpeg_bytes,
    summarize_diagnostics,
    validate_metadata,
)
from visual_rl.rewards.image_dispatch import score_image_batch
from visual_rl.rewards.scalar import ScoreResult, weighted_score


def metadata(count=3):
    return [{"prompt": f"p{i}", "tag": TAGS[i % len(TAGS)],
             "include": [{"class": "cat", "count": 1}]} for i in range(count)]


class Engine:
    def __init__(self):
        self.calls = []

    def __call__(self, images, metadata, *, only_strict):
        self.calls.append((images, copy.deepcopy(metadata), only_strict))
        n = len(images)
        ordinary, strict = [0.0 if only_strict else 1.0] * n, [0.0] * n
        grouped = {tag: [r if m["tag"] == tag else -10.0
                         for r, m in zip(ordinary, metadata, strict=True)] for tag in TAGS}
        grouped_strict = {tag: [r if m["tag"] == tag else -10.0
                                for r, m in zip(strict, metadata, strict=True)] for tag in TAGS}
        return [-2.0] * n, ordinary, strict, grouped, grouped_strict


def test_jpeg_exact_historical_bytes_and_no_direct_pixel_bypass():
    image = torch.rand(2, 3, 11, 13, generator=torch.Generator().manual_seed(8))
    result = jpeg_bytes(image)
    raw = (image * 255).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
    for pixels, encoded in zip(raw, result, strict=True):
        buffer = BytesIO()
        Image.fromarray(pixels).save(buffer, format="JPEG")
        assert buffer.getvalue() == encoded
        decoded = np.array(Image.open(BytesIO(encoded)))
        assert not np.array_equal(decoded, pixels)


@pytest.mark.parametrize("evaluation", [False, True])
@pytest.mark.parametrize("microbatch", [1, 2, 8])
def test_dense_not_binary_and_diagnostics_follow_metadata(evaluation, microbatch):
    engine = Engine()
    scorer = GenEvalScore(engine)
    rows = metadata()
    result = score_image_batch(scorer, torch.zeros(3, 3, 8, 8), [m["prompt"] for m in rows],
                               microbatch=microbatch, metadata=rows, evaluation=evaluation)
    assert result.raw["geneval"].tolist() == [-2.0] * 3
    assert result.weighted.tolist() == [-2.0] * 3
    assert all(call[2] is not evaluation for call in engine.calls)
    assert [row for _, batch, _ in engine.calls for row in batch] == rows
    assert result.diagnostics["geneval/strict_accuracy"].tolist() == [0.0] * 3
    reduced = summarize_diagnostics(result.diagnostics)
    assert reduced["geneval/strict_accuracy_macro"] == 0
    assert reduced["geneval/strict_accuracy_tags_present"] == 3
    assert ("geneval/accuracy" in reduced) is evaluation
    assert "geneval/tag/color_attr/strict_accuracy" not in reduced


@pytest.mark.parametrize("change", [
    lambda rows: rows[0].update(prompt="wrong"),
    lambda rows: rows[0].update(tag="unknown"),
    lambda rows: rows[0].update(include=[]),
    lambda rows: rows[0]["include"][0].update(count=True),
    lambda rows: rows[0]["include"][0].update(color="unknown"),
    lambda rows: rows[0]["include"][0].update(position=["above", 0]),
    lambda rows: rows[0]["include"][0].update(position=["above", -1]),
])
def test_bad_metadata_rejected_before_engine(change):
    engine = Engine()
    rows = metadata(1)
    change(rows)
    with pytest.raises(ValueError):
        GenEvalScore(engine)(torch.zeros(1, 3, 8, 8), ["p0"], microbatch=1, metadata=rows)
    assert not engine.calls


def test_position_reference_to_earlier_clause_is_valid():
    rows = metadata(1)
    rows[0]["include"].append({"class": "dog", "count": 1, "position": ["above", 0]})
    validate_metadata(["p0"], rows)


@pytest.mark.parametrize("corrupt", [
    lambda reply: reply[0].clear(),
    lambda reply: reply[0].__setitem__(0, float("nan")),
    lambda reply: reply[2].__setitem__(0, .5),
    lambda reply: reply[4]["single_object"].__setitem__(0, -10),
])
def test_bad_results_are_fatal_not_zero(corrupt):
    class Bad(Engine):
        def __call__(self, *args, **kwargs):
            reply = super().__call__(*args, **kwargs)
            corrupt(reply)
            return reply

    with pytest.raises(ValueError):
        GenEvalScore(Bad())(torch.zeros(1, 3, 8, 8), ["p0"], microbatch=1, metadata=metadata(1))


def test_diagnostics_serialize_but_never_change_total():
    result = weighted_score({"geneval": -.5}, {"geneval": 1}, space="raw",
                            diagnostics={"geneval/strict_accuracy": 0.0})
    assert result.weighted == -.5
    assert ScoreResult.from_dict(result.to_dict()).to_dict() == result.to_dict()
    old = weighted_score({"other": 2}, {"other": 3}, space="raw")
    assert "diagnostics" not in old.to_dict()
    assert ScoreResult.from_dict(old.to_dict()).weighted == 6
    with pytest.raises(ValueError):
        weighted_score({"x": torch.ones(2)}, {"x": 1}, space="raw",
                       diagnostics={"wrong": torch.ones(1)})


def test_composite_preserves_diagnostics_without_weighting_them():
    from visual_rl.rewards.composite import CompositeImageReward

    class Plain(torch.nn.Module):
        def forward(self, images, prompts, *, microbatch):
            return weighted_score({"plain": torch.ones(len(prompts))}, {"plain": 1}, space="raw")

    scorer = CompositeImageReward({"geneval": GenEvalScore(Engine()), "plain": Plain()},
                                  {"geneval": 2, "plain": 3}, space="raw")
    rows = metadata(1)
    result = score_image_batch(scorer, torch.zeros(1, 3, 8, 8), ["p0"], microbatch=1,
                               metadata=rows, evaluation=True)
    assert result.weighted.item() == -1
    assert result.diagnostics["geneval/accuracy"].item() == 1
    assert result.diagnostics["geneval/strict_accuracy"].item() == 0
