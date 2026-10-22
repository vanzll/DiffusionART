"""No downloads or GPU operations in environment-probe tests."""

import hashlib
import json
import subprocess
from types import SimpleNamespace

import pytest

from visual_rl.models import check_minimax_h3_environment as probe


def test_native_check_verifies_actual_bytes_instead_of_version(tmp_path, monkeypatch):
    path = tmp_path / "models/transformer.py"
    path.parent.mkdir()
    path.write_text("fixed-native-source")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(probe, "NATIVE_SOURCE_SHA256", {"models/transformer.py": digest})
    actual, errors = probe.check_native_sources(tmp_path)
    assert actual == {"models/transformer.py": digest} and not errors
    path.write_text("changed-native-src!")
    assert probe.check_native_sources(tmp_path)[1]


def test_missing_import_is_failure_without_silent_fallback():
    packages, sources, errors = probe.probe_imports((("nonexistent_h3_test_dependency", "fixture"),))
    assert not packages and not sources and len(errors) == 1
    assert "ModuleNotFoundError" in errors[0]


def test_source_checkout_import_still_records_actual_source_hash(tmp_path, monkeypatch):
    file = tmp_path / "native_fixture.py"
    file.write_text("# source fixture\n")
    monkeypatch.setattr(probe.importlib, "import_module", lambda _: SimpleNamespace(__file__=str(file)))
    packages, sources, errors = probe.probe_imports((("fixture", "nonexistent_h3_distribution"),))
    assert not errors
    assert sources["fixture"]["sha256"] == hashlib.sha256(file.read_bytes()).hexdigest()
    assert "metadata absent" in packages["nonexistent_h3_distribution"]


def test_cuda_requires_explicit_authorized_allocation_before_worker():
    with pytest.raises(SystemExit) as exc:
        probe.main(["--cuda"])
    assert exc.value.code == 2


def test_bounded_worker_timeout_is_machine_readable_failure(monkeypatch, capsys):
    def timeout(*args, **kwargs):
        assert kwargs["timeout"] == 1
        raise subprocess.TimeoutExpired(args[0], 1)
    monkeypatch.setattr(probe.subprocess, "run", timeout)
    assert probe.main(["--timeout", "1"]) == 1
    output = json.loads(capsys.readouterr().out)
    assert not output["ok"] and "timed out" in output["errors"][0]


def test_missing_ffmpeg_and_ffprobe_fail(monkeypatch):
    monkeypatch.setattr(probe.shutil, "which", lambda _: None)
    tools, errors = probe.probe_tools()
    assert not tools and len(errors) == 2


def test_all_scorers_requires_actual_source_manifest():
    with pytest.raises(ValueError, match="--reward-manifest"):
        probe.probe_synchformer(None)
