import subprocess
import sys
from types import ModuleType

import pytest

from visual_rl.models.source_pin import import_pinned_source, pinned_source


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    root = tmp_path / "native"
    (root / "native_fixture").mkdir(parents=True)
    (root / "native_fixture/__init__.py").write_text("VALUE=73\n")

    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    git("init", "-q")
    git("add", "native_fixture/__init__.py")
    git(
        "-c",
        "user.name=fixture",
        "-c",
        "user.email=fixture@localhost",
        "commit",
        "-q",
        "-m",
        "native source fixture",
    )
    yield root, git("rev-parse", "HEAD")
    for name in list(sys.modules):
        if name == "native_fixture" or name.startswith("native_fixture."):
            sys.modules.pop(name, None)


def test_native_import_requires_exact_source_identity(source, monkeypatch):
    root, revision = source
    module = import_pinned_source(root, package="native_fixture", revision=revision)
    assert module.VALUE == 73
    assert module.__file__ == str(root / "native_fixture/__init__.py")


@pytest.mark.parametrize("problem", ["wrong_sha", "dirty", "nested_root", "wrong_submodule"])
def test_mismatched_or_shadowed_sources_fail_before_asset_loading(source, monkeypatch, problem):
    root, revision = source
    if problem == "wrong_sha":
        revision = "0" * 40
    elif problem == "dirty":
        (root / "native_fixture/__init__.py").write_text("VALUE=99\n")
    elif problem == "nested_root":
        root = root / "subdirectory"
        (root / "native_fixture").mkdir(parents=True)
        (root / "native_fixture/__init__.py").write_text("VALUE=42\n")
    else:
        module = ModuleType("native_fixture.hidden")
        module.__file__ = "/another/source/native_fixture/hidden.py"
        monkeypatch.setitem(sys.modules, "native_fixture.hidden", module)
    with pytest.raises(ValueError):
        pinned_source(root, package="native_fixture", revision=revision)
