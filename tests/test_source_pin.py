import importlib
import subprocess
import sys
from importlib.machinery import ModuleSpec
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


@pytest.fixture
def namespace_source(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    root = tmp_path / "namespace_source"
    anchor = root / "namespace_fixture/models/score.py"
    anchor.parent.mkdir(parents=True)
    anchor.write_text("VALUE=91\n")

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True, timeout=10
        ).strip()

    git("init", "-q")
    git("add", "namespace_fixture/models/score.py")
    git(
        "-c",
        "user.name=fixture",
        "-c",
        "user.email=fixture@localhost",
        "commit",
        "-qm",
        "namespace fixture",
    )
    yield root, git("rev-parse", "HEAD")
    for name in list(sys.modules):
        if name == "namespace_fixture" or name.startswith("namespace_fixture."):
            sys.modules.pop(name, None)


def namespace_pin(root, revision):
    return import_pinned_source(
        root,
        package="namespace_fixture",
        revision=revision,
        allow_namespace=True,
        tracked_anchor="namespace_fixture/models/score.py",
    )


def test_namespace_is_explicit_tracked_and_fixed_to_one_checkout(namespace_source):
    root, revision = namespace_source
    with pytest.raises(ValueError):
        import_pinned_source(root, package="namespace_fixture", revision=revision)
    with pytest.raises(ValueError):
        import_pinned_source(
            root, package="namespace_fixture", revision=revision, allow_namespace=True
        )
    package = namespace_pin(root, revision)
    assert package.__file__ is None
    assert package.__path__ == [str(root / "namespace_fixture")]
    score = importlib.import_module("namespace_fixture.models.score")
    assert score.VALUE == 91 and score.__file__ == str(root / "namespace_fixture/models/score.py")
    assert namespace_pin(root, revision) is package


@pytest.mark.parametrize(
    "problem", ["wrong_revision", "dirty", "untracked_anchor", "outside_anchor", "new_regular_init"]
)
def test_namespace_source_still_requires_immutable_tracked_identity(namespace_source, problem):
    root, revision = namespace_source
    anchor = "namespace_fixture/models/score.py"
    if problem == "wrong_revision":
        revision = "0" * 40
    elif problem == "dirty":
        (root / anchor).write_text("VALUE=22\n")
    elif problem == "untracked_anchor":
        anchor = "namespace_fixture/models/untracked.py"
        (root / anchor).write_text("VALUE=3\n")
    elif problem == "outside_anchor":
        anchor = "../outside.py"
    else:
        (root / "namespace_fixture/__init__.py").write_text("raise RuntimeError('must not run')")
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        import_pinned_source(
            root,
            package="namespace_fixture",
            revision=revision,
            allow_namespace=True,
            tracked_anchor=anchor,
        )


@pytest.mark.parametrize("regular", [True, False])
def test_foreign_unimported_regular_or_namespace_is_rejected_before_execution(
    namespace_source, tmp_path, monkeypatch, regular
):
    root, revision = namespace_source
    foreign = tmp_path / "foreign"
    (foreign / "namespace_fixture").mkdir(parents=True)
    if regular:
        (foreign / "namespace_fixture/__init__.py").write_text(
            "raise RuntimeError('FOREIGN EXECUTED')"
        )
    monkeypatch.syspath_prepend(str(foreign))
    with pytest.raises(ValueError, match="foreign regular/namespace"):
        namespace_pin(root, revision)
    assert "namespace_fixture" not in sys.modules


@pytest.mark.parametrize(
    "problem", ["regular_root", "namespace_root", "mixed_child_namespace", "foreign_submodule"]
)
def test_preimported_foreign_namespace_regular_and_submodules_fail_closed(
    namespace_source, tmp_path, monkeypatch, problem
):
    root, revision = namespace_source
    name = (
        "namespace_fixture"
        if problem.endswith("root")
        else (
            "namespace_fixture.models"
            if problem == "mixed_child_namespace"
            else "namespace_fixture.models.score"
        )
    )
    module = ModuleType(name)
    if problem in ("regular_root", "foreign_submodule"):
        module.__file__ = str(tmp_path / "foreign.py")
    else:
        module.__path__ = [str(tmp_path / "foreign_namespace")]
        if problem == "mixed_child_namespace":
            module.__path__.insert(0, str(root / "namespace_fixture/models"))
        module.__spec__ = ModuleSpec(name, loader=None, is_package=True)
        module.__spec__.submodule_search_locations = list(module.__path__)
    monkeypatch.setitem(sys.modules, name, module)
    with pytest.raises(ValueError, match="origin|namespace|regular"):
        namespace_pin(root, revision)


def test_namespace_revalidation_rejects_foreign_score_submodule(namespace_source, monkeypatch):
    root, revision = namespace_source
    namespace_pin(root, revision)
    importlib.import_module("namespace_fixture.models.score")
    foreign = ModuleType("namespace_fixture.models.score")
    foreign.__file__ = "/foreign/score.py"
    monkeypatch.setitem(sys.modules, foreign.__name__, foreign)
    with pytest.raises(ValueError, match="origin"):
        namespace_pin(root, revision)
