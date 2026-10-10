import hashlib
import json

import pytest

from visual_rl.data import PromptDataset, PromptStream


def dataset(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text(
        "\n".join(json.dumps({"sample_id": str(i), "prompt": f"Example {i}"}) for i in range(7))
        + "\n"
    )
    return PromptDataset.load(
        path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), task="t2v"
    )


def test_rank_sharding_and_resume(tmp_path):
    data = dataset(tmp_path)
    stream = PromptStream(data, seed=42, shuffle=True)
    a, b = PromptStream(data, seed=42, shuffle=True), PromptStream(data, seed=42, shuffle=True)
    for _ in range(3):
        global_batch = stream.take(4)
        assert a.take(4, rank=0, world_size=2) + b.take(4, rank=1, world_size=2) == global_batch
    restored = PromptStream(data, seed=42, shuffle=True)
    restored.load_state_dict(stream.state_dict())
    assert restored.take(12) == stream.take(12)


def test_invalid_dataset_hash_and_duplicate_identity(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text('{"sample_id":"1","prompt":"A"}\n')
    with pytest.raises(ValueError, match="hash mismatch"):
        PromptDataset.load(path, expected_sha256="a" * 64, task="t2i")
    path.write_text(path.read_text() * 2)
    with pytest.raises(ValueError, match="unique"):
        PromptDataset.load(
            path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), task="t2i"
        )


def test_i2v_reference_hash_required(tmp_path):
    image = tmp_path / "reference.png"
    image.write_bytes(b"identity test; decoding belongs to image adapter")
    path = tmp_path / "prompts.jsonl"
    path.write_text(
        json.dumps(
            {
                "sample_id": "one",
                "prompt": "Move",
                "image": image.name,
                "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            }
        )
        + "\n"
    )
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert PromptDataset.load(path, expected_sha256=sha, task="i2v").rows[0].image == image
    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="image hash"):
        PromptDataset.load(path, expected_sha256=sha, task="i2v")


def test_metadata_is_hashed_and_cannot_mutate_prompt_stream(tmp_path):
    path = tmp_path / "prompts.jsonl"
    annotation = {"tag": "counting", "include": [{"class": "apple", "count": 2}]}
    row = {"sample_id": "one", "prompt": "Two apples", "metadata": annotation}
    path.write_text(json.dumps(row) + "\n")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    data = PromptDataset.load(path, expected_sha256=sha, task="t2i")
    stream = PromptStream(data, seed=42, shuffle=False)
    a, b = stream.take(2)
    assert a.metadata == b.metadata == annotation
    a.metadata["include"][0]["count"] = 200
    assert b.metadata == annotation
    row["metadata"]["include"][0]["count"] = 3
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        PromptDataset.load(path, expected_sha256=sha, task="t2i")


@pytest.mark.parametrize("metadata", [None, [], "bad", {"score": float("nan")},
                                       {"score": float("inf")}])
def test_metadata_rejects_nonobjects_and_nonfinite_values(tmp_path, metadata):
    path = tmp_path / "prompts.jsonl"
    path.write_text(json.dumps({"sample_id": "one", "prompt": "Two", "metadata": metadata}))
    with pytest.raises(ValueError):
        PromptDataset.load(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           task="t2i")
