import hashlib
import json

import pytest

from visual_rl.data import PromptDataset
from visual_rl.prepare_data import main, prepare


def test_text_conversion_preserves_repeats_order_and_source(tmp_path):
    source, output = tmp_path / "source.txt", tmp_path / "native.jsonl"
    source.write_bytes(b"  forest  \r\nforest\nriver\n")
    before = source.read_bytes()
    receipt = prepare(source, output, format="text", prefix="train")
    dataset = PromptDataset.load(output, expected_sha256=receipt["sha256"], task="t2i")
    assert [row.text for row in dataset.rows] == ["forest", "forest", "river"]
    assert [row.sample_id for row in dataset.rows] == ["train-000000", "train-000001", "train-000002"]
    assert source.read_bytes() == before
    assert receipt["source_sha256"] == hashlib.sha256(before).hexdigest()
    assert receipt["samples"] == 3
    second = prepare(source, tmp_path / "second.jsonl", format="text", prefix="train")
    assert receipt["sha256"] == second["sha256"]
    with pytest.raises(FileExistsError):
        prepare(source, output, format="text")


def test_geneval_conversion_preserves_entire_annotation_tree(tmp_path, capsys):
    source, output = tmp_path / "geneval.jsonl", tmp_path / "native.jsonl"
    annotation = {"tag": "counting", "include": [{"class": "apple", "count": 2}],
                  "exclude": [{"class": "apple", "count": 3}], "prompt": "two apples"}
    source.write_text(json.dumps(annotation) + "\n")
    assert main(["--source", str(source), "--output", str(output), "--format", "geneval"]) == 0
    receipt = json.loads(capsys.readouterr().out)
    dataset = PromptDataset.load(output, expected_sha256=receipt["sha256"], task="t2v")
    assert dataset.rows[0].metadata == annotation


@pytest.mark.parametrize("content,format", [
    ("", "text"), ("one\n\n", "text"), ("{}\n", "geneval"),
    ('{"tag":"counting","include":[{}],"prompt":"x","score":NaN}\n', "geneval"),
])
def test_failure_never_publishes_partial_manifest(tmp_path, content, format):
    source, output = tmp_path / "source", tmp_path / "output"
    source.write_text(content)
    with pytest.raises(ValueError):
        prepare(source, output, format=format)
    assert not output.exists()
    assert not list(tmp_path.glob(".output.*"))


def test_existing_or_dangling_target_is_not_overwritten(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    source.write_text("prompt\n")
    output.symlink_to("missing")
    with pytest.raises(FileExistsError):
        prepare(source, output, format="text")
    assert output.is_symlink() and not output.exists()
