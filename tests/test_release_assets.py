import hashlib
import json
from pathlib import Path


def test_readme_examples_match_preserved_asset_identities():
    root = Path(__file__).resolve().parents[1] / "assets/readme"
    provenance = json.loads((root / "provenance.json").read_text())
    assert {row["task"] for row in provenance["samples"]} == {"t2i", "t2v"}
    for row in provenance["samples"]:
        for side in ("before", "after"):
            assert hashlib.sha256((root / row[side]).read_bytes()).hexdigest() == row[f"{side}_sha256"]
