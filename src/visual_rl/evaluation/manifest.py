"""Content-addressed evaluation records with explicit expected sample identities."""

import hashlib
import json
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from visual_rl.rewards import ScoreResult
from visual_rl.rewards.scalar import finite_number


def relative_path(value, *, allow_empty=False):
    """Validate canonical, decoded POSIX paths. URLs and traversal are not paths."""
    if value == "" and allow_empty:
        return value
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or ":" in value
        or any(ord(c) < 32 or ord(c) == 127 for c in value)
        or any(p in ("", ".", "..") for p in value.split("/"))
    ):
        raise ValueError("expected canonical relative POSIX path without traversal")
    return value


def _integer(value, name, minimum=0):
    # Keep JSON identities exact in browsers, including seeds.
    if type(value) is not int or not minimum <= value <= 2**53 - 1:
        raise ValueError(f"{name} must be a JavaScript-safe integer >= {minimum}")


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")


def canonical_json(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":")
    )


def _digest(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def resolve_media(root, path):
    relative_path(path)
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("media root must be a directory")
    candidate = root.joinpath(*path.split("/"))
    # Refuse links, including dangling links; publish only files under this root.
    current = root
    for part in path.split("/"):
        current = current / part
        if current.is_symlink():
            raise ValueError("symlink media paths are not allowed")
    candidate.resolve().relative_to(root)
    return candidate


def file_sha256(path, *, max_bytes=256 * 1024 * 1024):
    """Bound bytes read; callers must isolate I/O for hung remote filesystems."""
    _integer(max_bytes, "max_bytes", 1)
    info = Path(path).stat()
    if not stat.S_ISREG(info.st_mode) or info.st_size <= 0 or info.st_size > max_bytes:
        raise ValueError("media must be a nonempty regular file within max_bytes")
    digest, size = hashlib.sha256(), 0
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(min(1024 * 1024, max_bytes + 1 - size))
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes:
                raise ValueError("media exceeds max_bytes")
            digest.update(chunk)
    after = Path(path).stat()
    if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or size != info.st_size:
        raise ValueError("media changed during hashing")
    return digest.hexdigest()


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    prompt: str
    seed: int
    optimizer_step: int
    framecount: int
    fps: float
    path: str
    content_sha256: Optional[str]
    scores: Optional[ScoreResult] = None

    def __post_init__(self):
        _text(self.sample_id, "sample_id")
        _text(self.prompt, "prompt")
        for key in ("seed", "optimizer_step", "framecount"):
            _integer(getattr(self, key), key, 1 if key == "framecount" else 0)
        object.__setattr__(self, "fps", finite_number(self.fps, "fps"))
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        relative_path(self.path)
        if self.content_sha256 is not None and (
            not isinstance(self.content_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", self.content_sha256)
        ):
            raise ValueError("content_sha256 must be a lowercase SHA-256 or None")
        if self.scores is not None:
            # Snapshot scalar provenance; tensors and inconsistent hand-built results fail closed.
            object.__setattr__(self, "scores", ScoreResult.from_dict(self.scores.to_dict()))

    @property
    def prompt_sha256(self):
        return hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()

    def _payload(self):
        return {
            k: getattr(self, k)
            for k in (
                "sample_id",
                "prompt",
                "seed",
                "optimizer_step",
                "framecount",
                "fps",
                "path",
                "content_sha256",
            )
        }

    def to_dict(self):
        data = dict(
            self._payload(),
            prompt_sha256=self.prompt_sha256,
            scores=None if self.scores is None else self.scores.to_dict(),
        )
        return dict(data, record_sha256=_digest(data))

    @classmethod
    def from_file(cls, *, media_root, max_bytes=256 * 1024 * 1024, **fields):
        path = resolve_media(media_root, fields["path"])
        return cls(content_sha256=file_sha256(path, max_bytes=max_bytes), **fields)

    @classmethod
    def from_dict(cls, data):
        fields = dict(data)
        fields.pop("prompt_sha256", None)
        fields.pop("record_sha256", None)
        if fields.get("scores") is not None:
            fields["scores"] = ScoreResult.from_dict(fields["scores"])
        record = cls(**fields)
        if record.to_dict() != data:
            raise ValueError("sample content/identity/provenance hash mismatch")
        return record


@dataclass(frozen=True)
class EvaluationManifest:
    optimizer_step: int
    expected_sample_ids: Tuple[str, ...]
    samples: Tuple[SampleRecord, ...]

    def __post_init__(self):
        _integer(self.optimizer_step, "optimizer_step")
        if isinstance(self.expected_sample_ids, str):
            raise ValueError("expected_sample_ids must be a sequence, not text")
        object.__setattr__(self, "expected_sample_ids", tuple(self.expected_sample_ids))
        object.__setattr__(self, "samples", tuple(self.samples))
        for key in self.expected_sample_ids:
            _text(key, "expected sample_id")
        if not self.expected_sample_ids or len(set(self.expected_sample_ids)) != len(
            self.expected_sample_ids
        ):
            raise ValueError("expected sample IDs must be nonempty and unique")
        ids = [s.sample_id for s in self.samples]
        if len(ids) != len(set(ids)) or not set(ids) <= set(self.expected_sample_ids):
            raise ValueError("duplicate or unexpected sample identity")
        if any(s.optimizer_step != self.optimizer_step for s in self.samples):
            raise ValueError("sample optimizer_step differs from manifest")

    def to_dict(self):
        # Order is presentation only, never identity or digest input.
        return {
            "schema_version": 1,
            "optimizer_step": self.optimizer_step,
            "expected_sample_ids": sorted(self.expected_sample_ids),
            "samples": [s.to_dict() for s in sorted(self.samples, key=lambda s: s.sample_id)],
        }

    @property
    def sha256(self):
        return _digest(self.to_dict())

    def to_json(self):
        return canonical_json(dict(self.to_dict(), manifest_sha256=self.sha256)) + "\n"

    @classmethod
    def from_json(cls, text):
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate JSON key")
                result[key] = value
            return result

        data = json.loads(text, object_pairs_hook=unique)
        if set(data) != {
            "schema_version",
            "optimizer_step",
            "expected_sample_ids",
            "samples",
            "manifest_sha256",
        }:
            raise ValueError("invalid manifest schema")
        if type(data["schema_version"]) is not int or data["schema_version"] != 1:
            raise ValueError("unsupported manifest schema version")
        if not isinstance(data["expected_sample_ids"], list) or not isinstance(
            data["samples"], list
        ):
            raise ValueError("manifest identities and samples must be arrays")
        result = cls(
            data["optimizer_step"],
            tuple(data["expected_sample_ids"]),
            tuple(SampleRecord.from_dict(s) for s in data["samples"]),
        )
        if result.sha256 != data["manifest_sha256"]:
            raise ValueError("manifest hash mismatch")
        return result


@dataclass(frozen=True)
class SampleAudit:
    sample_id: str
    media_status: str
    scores_present: bool
    detail: str = ""

    @property
    def complete(self):
        return self.media_status == "verified" and self.scores_present


@dataclass(frozen=True)
class EvaluationAudit:
    manifest_sha256: str
    samples: Tuple[SampleAudit, ...]

    @property
    def complete(self):
        return bool(self.samples) and all(s.complete for s in self.samples)

    @property
    def completed_count(self):
        return sum(s.complete for s in self.samples)


def audit_manifest(manifest, *, media_root, max_bytes=256 * 1024 * 1024):
    """Audit all expected samples, including absent records and failed file checks.

    No decoder dependency: framecount/fps are producer assertions, not probed
    codec metadata. This is a local snapshot, not proof files stay unchanged.
    """
    by_id = {s.sample_id: s for s in manifest.samples}
    audits = []
    for sample_id in manifest.expected_sample_ids:
        s = by_id.get(sample_id)
        if s is None:
            audits.append(SampleAudit(sample_id, "missing_record", False))
            continue
        status, detail = "verified", ""
        try:
            actual = file_sha256(resolve_media(media_root, s.path), max_bytes=max_bytes)
            if s.content_sha256 is None:
                status = "missing_hash"
            elif actual != s.content_sha256:
                status = "hash_mismatch"
        except FileNotFoundError:
            status = "missing_file"
        except (OSError, ValueError) as exc:
            status, detail = "unreadable_file", str(exc)
        audits.append(SampleAudit(sample_id, status, s.scores is not None, detail))
    return EvaluationAudit(manifest.sha256, tuple(audits))
