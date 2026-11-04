"""Opt-in native package loading from an immutable, clean source checkout."""

import importlib
import re
import subprocess
import sys
from importlib.machinery import PathFinder
from pathlib import Path


def pinned_source(root, *, package, revision, allow_namespace=False, tracked_anchor=None):
    if root is None or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("native code requires an explicit source root and immutable Git SHA")
    root = Path(root).resolve(strict=True)
    if type(allow_namespace) is not bool:
        raise ValueError("namespace loading requires an explicit boolean opt-in")
    if allow_namespace:
        directory = root / package
        anchor = Path(tracked_anchor) if isinstance(tracked_anchor, str) else None
        if (
            not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", package)
            or not directory.is_dir()
            or directory.resolve() != directory
            or (directory / "__init__.py").exists()
            or anchor is None
            or anchor.is_absolute()
            or ".." in anchor.parts
            or not anchor.parts
            or anchor.parts[0] != package
            or not (root / anchor).is_file()
            or (root / anchor).resolve() != root / anchor
        ):
            raise ValueError(
                "namespace source requires its exact directory and explicit tracked package anchor"
            )
    elif tracked_anchor is not None or not (root / package / "__init__.py").is_file():
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
    git(
        "ls-files",
        "--error-unmatch",
        tracked_anchor if allow_namespace else f"{package}/__init__.py",
    )
    _check_origins(root, package, allow_namespace=allow_namespace)
    # Never execute a foreign package merely because the correct root occurred
    # later in sys.path; provenance validation precedes optional imports.
    sys.path[:] = [str(root), *(p for p in sys.path if p != str(root))]
    if allow_namespace:
        # A later regular package wins over an earlier PEP420 portion. Inspect
        # the finder metadata BEFORE import so foreign code cannot execute.
        spec = PathFinder.find_spec(package, sys.path)
        if (
            spec is None
            or spec.origin is not None
            or not _exact_package_paths(spec.submodule_search_locations, root / package)
        ):
            raise ValueError(
                "foreign regular/namespace package would override or merge pinned source"
            )
    return root


def _exact_package_paths(paths, expected):
    return paths is not None and list(map(lambda path: Path(path).resolve(), paths)) == [expected]


def _check_origins(root, package, *, allow_namespace=False):
    for name, module in list(sys.modules.items()):
        if name != package and not name.startswith(package + "."):
            continue
        origin = getattr(module, "__file__", None)
        if allow_namespace:
            expected = root.joinpath(*name.split("."))
            paths = getattr(module, "__path__", None)
            spec = getattr(module, "__spec__", None)
            if origin is None:
                if (
                    not _exact_package_paths(paths, expected)
                    or spec is None
                    or spec.origin is not None
                    or not _exact_package_paths(spec.submodule_search_locations, expected)
                ):
                    raise ValueError(
                        "foreign native namespace or submodule search path is already imported"
                    )
                continue
            if name == package or (
                paths is not None
                and (
                    not _exact_package_paths(paths, expected)
                    or spec is None
                    or not _exact_package_paths(spec.submodule_search_locations, expected)
                )
            ):
                raise ValueError(
                    "foreign regular package or submodule search path is already imported"
                )
        if origin is None or not Path(origin).resolve().is_relative_to(root / package):
            raise ValueError("another native package or submodule origin is already imported")


def import_pinned_source(root, *, package, revision, allow_namespace=False, tracked_anchor=None):
    root = pinned_source(
        root,
        package=package,
        revision=revision,
        allow_namespace=allow_namespace,
        tracked_anchor=tracked_anchor,
    )
    module = importlib.import_module(package)
    if allow_namespace:
        _check_origins(root, package, allow_namespace=True)
        # Freeze the root namespace to its single verified checkout instead of
        # allowing _NamespacePath to merge future foreign sys.path additions.
        module.__path__ = [str(root / package)]
        module.__spec__.submodule_search_locations = module.__path__
        return module
    if Path(module.__file__).resolve() != root / package / "__init__.py":
        raise ValueError("imported native package differs from the pinned source")
    _check_origins(root, package)
    return module
