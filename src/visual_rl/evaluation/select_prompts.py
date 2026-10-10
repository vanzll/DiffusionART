"""Freeze a score-blind VidProM subset without changing the existing prefix."""

import argparse
import csv
import hashlib
import os
import random
import re
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from ..config import ConfigError, _read
from ..prepare_assets import _hash, _vidprom_manifest, validate_assets


def _normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _tokens(text):
    return frozenset(re.findall(r"[a-z0-9]+", _normalize(text)))


class _Exclusions:
    def __init__(self, texts):
        self.exact, self.rows, self.postings = set(), [], defaultdict(list)
        for text in texts:
            self.add(text)

    def add(self, text):
        normalized = _normalize(text)
        if normalized in self.exact:
            return
        self.exact.add(normalized)
        row, index = _tokens(text), len(self.rows)
        self.rows.append(row)
        for token in row:
            self.postings[token].append(index)

    def matches(self, text):
        if _normalize(text) in self.exact:
            return True
        row = _tokens(text)
        overlap = Counter(index for token in row for index in self.postings.get(token, ()))
        return any(common / (len(row) + len(self.rows[index]) - common) >= .8
                   for index, common in overlap.items())


def _usable(text):
    return (4 <= len(text.split()) <= 120 and len(_tokens(text)) >= 3
            and not re.search(r"https?://|www\.|[<>]", text, re.I)
            and not any(unicodedata.category(c) == "Cc" and not c.isspace() for c in text)
            and sum(c.isascii() for c in text) / max(1, len(text)) >= .98)


def select_ids(source, training, *, count, seed=20260908):
    if type(count) is not int or not 1 <= count <= 10000:
        raise ConfigError("evaluation count must be between 1 and 10000")
    with Path(source).open(newline="", encoding="utf-8") as stream:
        candidates = [row for row in csv.DictReader(stream) if _usable(row["prompt"])]
    random.Random(seed).shuffle(candidates)
    exclusions = _Exclusions(Path(training).read_text(encoding="utf-8").splitlines())
    selected = []
    for row in candidates:
        text = " ".join(row["prompt"].split())
        if not exclusions.matches(text):
            selected.append(row["uuid"])
            exclusions.add(text)
            if len(selected) == count:
                return selected
    raise ConfigError("not enough disjoint VidProM prompts; never supplement from training")


def freeze_selection(base_config, source, training, output, *, count=1000):
    base_config, output = Path(base_config).resolve(), Path(output).resolve()
    source, training = Path(source), Path(training)
    if output.exists():
        raise ConfigError("refusing to overwrite a frozen evaluation selection")
    spec = _read(base_config)["assets"]
    validate_assets(spec)
    train = next(e for e in spec["prompts"] if e["env"] == "HUNYUAN_TRAIN_RAW_MANIFEST")
    original = next(e for e in spec["prompts"] if e["env"] == "HUNYUAN_EVAL_RAW_MANIFEST")
    if _hash(source) != original["source_sha256"] or _hash(training) != train["source_sha256"]:
        raise ConfigError("source CSV or training text differs from the pinned public assets")
    ids = select_ids(source, training, count=count)
    if ids[:len(original["ids"])] != original["ids"]:
        raise ConfigError("new selection does not preserve the frozen evaluation prefix")
    with tempfile.TemporaryDirectory() as temporary:
        existing = Path(temporary) / "existing.jsonl"
        _vidprom_manifest(source, existing, original["ids"])
        if _hash(existing) != original["sha256"]:
            raise ConfigError("original evaluation manifest identity differs")
        manifest = Path(temporary) / "expanded.jsonl"
        _vidprom_manifest(source, manifest, ids)
        expanded = {**original, "ids": ids, "sha256": _hash(manifest)}
    document = {"extends": os.path.relpath(base_config, output.parent),
                "assets": {"prompts": [train, expanded]}}
    validate_assets({**spec, "prompts": [train, expanded]})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write("# Frozen score-blind selection; seed 20260908; original eval100 prefix preserved.\n")
        yaml.safe_dump(document, stream, sort_keys=False)
    return {"count": len(ids), "manifest_sha256": expanded["sha256"],
            "ordered_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--training", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=1000)
    args = parser.parse_args()
    print(freeze_selection(args.base_config, args.source, args.training, args.output, count=args.count))


if __name__ == "__main__":
    main()
