"""Opt-in native package loading from an immutable, clean source checkout."""

import importlib
import re
import subprocess
import sys
from pathlib import Path


def pinned_source(root, *, package, revision):
    if root is None or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("native code requires an explicit source root and immutable Git SHA")
    root = Path(root).resolve(strict=True)
    if not (root / package / "__init__.py").is_file():
        raise ValueError("native package is absent from its declared source root")

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(root), *args], check=True, capture_output=True, text=True, timeout=15
        ).stdout.strip()

    if (
        Path(git("rev-parse", "--show-toplevel")).resolve() != root
        or git("rev-parse", "HEAD") != revision
        or git("status", "--porcelain", "--untracked-files=no")
    ):
        raise ValueError("native source must be clean at its pinned revision")
    git("ls-files", "--error-unmatch", f"{package}/__init__.py")
    _check_origins(root, package)
    # Never execute a foreign package merely because the correct root occurred
    # later in sys.path; provenance validation precedes optional imports.
    sys.path[:] = [str(root), *(p for p in sys.path if p != str(root))]
    return root


def _check_origins(root, package):
    for name, module in list(sys.modules.items()):
        if name != package and not name.startswith(package + "."):
            continue
        origin = getattr(module, "__file__", None)
        if origin is None or not Path(origin).resolve().is_relative_to(root / package):
            raise ValueError("another native package or submodule origin is already imported")


def import_pinned_source(root, *, package, revision):
    root = pinned_source(root, package=package, revision=revision)
    module = importlib.import_module(package)
    if Path(module.__file__).resolve() != root / package / "__init__.py":
        raise ValueError("imported native package differs from the pinned source")
    _check_origins(root, package)
    return module
