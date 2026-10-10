"""Identity-preserving prompt inputs and resumable deterministic prompt order."""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import torch


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Prompt:
    sample_id: str
    text: str
    image: Path | None = None
    metadata_json: str | None = field(default=None, repr=False)
    image_sha256: str | None = field(default=None, repr=False)

    @property
    def metadata(self):
        # Return a fresh tree so a scorer cannot mutate the hashed dataset.
        return None if self.metadata_json is None else json.loads(self.metadata_json)


@dataclass(frozen=True)
class PromptDataset:
    rows: tuple[Prompt, ...]
    sha256: str

    @classmethod
    def load(cls, path, *, expected_sha256: str | None = None, task: str):
        path = Path(path)
        if task not in ("t2i", "t2v", "i2v"):
            raise ValueError("unknown prompt task")
        if expected_sha256 is not None and (
            not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or any(c not in "0123456789abcdef" for c in expected_sha256)
        ):
            raise ValueError("dataset SHA256 must be a digest or null")
        digest, references, rows, seen = hashlib.sha256(), [], [], set()
        with path.open("rb") as stream:
            for index in range(1_000_001):
                line = stream.readline(1024 * 1024 + 1)
                if not line:
                    break
                if index == 1_000_000 or len(line) > 1024 * 1024:
                    raise ValueError("prompt dataset exceeds row/line limit")
                digest.update(line)
                if not line.strip():
                    raise ValueError(f"blank dataset row at line {index + 1}")
                row = json.loads(line)
                if not isinstance(row, dict) or set(row) - {
                    "sample_id",
                    "prompt",
                    "image",
                    "image_sha256",
                    "metadata",
                }:
                    raise ValueError(f"invalid prompt schema at line {index + 1}")
                sample_id, text = row.get("sample_id"), row.get("prompt")
                if not isinstance(sample_id, str) or not sample_id.strip() or sample_id in seen:
                    raise ValueError("sample IDs must be unique nonempty strings")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("prompt must be a nonempty string")
                seen.add(sample_id)
                image = None
                if task == "i2v":
                    if not isinstance(row.get("image"), str) or not row["image"]:
                        raise ValueError("I2V rows require an image path")
                    if "image_sha256" in row and (
                        not isinstance(row["image_sha256"], str)
                        or len(row["image_sha256"]) != 64
                        or any(c not in "0123456789abcdef" for c in row["image_sha256"])
                    ):
                        raise ValueError("reference image SHA256 must be a digest")
                    image = (path.parent / row["image"]).resolve()
                    h = hashlib.sha256()
                    with image.open("rb") as media:
                        for block in iter(lambda: media.read(1024 * 1024), b""):
                            h.update(block)
                    if "image_sha256" in row and h.hexdigest() != row["image_sha256"]:
                        raise ValueError(f"reference image hash differs: {sample_id}")
                    # Include unpinned image content in dataset identity so resume
                    # still detects changes without asking users to supply hashes.
                    if "image_sha256" not in row:
                        references.append(h.digest())
                elif "image" in row or "image_sha256" in row:
                    raise ValueError("text-only tasks must not silently ignore reference images")
                metadata_json = None
                if "metadata" in row:
                    if not isinstance(row["metadata"], dict):
                        raise ValueError("prompt metadata must be a JSON object")
                    metadata_json = json.dumps(row["metadata"], allow_nan=False, sort_keys=True)
                rows.append(Prompt(sample_id, text, image, metadata_json,
                                   h.hexdigest() if image is not None else None))
        identity = digest.hexdigest()
        if references:
            identity = hashlib.sha256(digest.digest() + b"".join(references)).hexdigest()
        if not rows or (expected_sha256 is not None and expected_sha256 not in {digest.hexdigest(), identity}):
            raise ValueError("empty dataset or dataset hash mismatch")
        return cls(tuple(rows), identity)


class PromptStream:
    """Global prompt-slot cursor; sharding happens after one shared selection.

    Each rank receives its slice of one shared prompt order.
    """

    def __init__(self, dataset: PromptDataset, *, seed: int, shuffle: bool):
        if type(seed) is not int or not 0 <= seed < 2**63 or type(shuffle) is not bool:
            raise ValueError("invalid prompt stream seed/shuffle")
        if not dataset.rows:
            raise ValueError("empty prompt stream")
        self.dataset, self.seed, self.shuffle = dataset, seed, shuffle
        self.epoch = self.cursor = 0

    def _order(self):
        if not self.shuffle:
            return list(range(len(self.dataset.rows)))
        generator = torch.Generator().manual_seed((self.seed + self.epoch) % (2**63))
        return torch.randperm(len(self.dataset.rows), generator=generator).tolist()

    def take(self, prompt_count: int, *, rank: int = 0, world_size: int = 1):
        if (
            any(type(v) is not int for v in (prompt_count, rank, world_size))
            or world_size < 1
            or not 0 <= rank < world_size
            or prompt_count < world_size
            or prompt_count % world_size
        ):
            raise ValueError("global prompt slots must divide evenly over ranks")
        selected = []
        while len(selected) < prompt_count:
            order = self._order()
            count = min(prompt_count - len(selected), len(order) - self.cursor)
            selected.extend(self.dataset.rows[i] for i in order[self.cursor : self.cursor + count])
            self.cursor += count
            if self.cursor == len(order):
                self.epoch += 1
                self.cursor = 0
        per_rank = prompt_count // world_size
        return tuple(selected[rank * per_rank : (rank + 1) * per_rank])

    def state_dict(self):
        return {
            "schema_version": 1,
            "dataset_sha256": self.dataset.sha256,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "epoch": self.epoch,
            "cursor": self.cursor,
        }

    def load_state_dict(self, state):
        expected = self.state_dict()
        if set(state) != set(expected):
            raise ValueError("prompt cursor schema differs")
        for key in ("schema_version", "dataset_sha256", "seed", "shuffle"):
            if state[key] != expected[key]:
                raise ValueError(f"prompt cursor {key} differs")
        if (
            type(state["epoch"]) is not int
            or state["epoch"] < 0
            or type(state["cursor"]) is not int
            or not 0 <= state["cursor"] < len(self.dataset.rows)
        ):
            raise ValueError("invalid prompt cursor")
        self.epoch, self.cursor = state["epoch"], state["cursor"]
