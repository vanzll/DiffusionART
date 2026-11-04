"""CPU mocks for early native-Wan dependency failures; no CUDA/FA execution."""

import importlib.metadata
import json
import sys
from types import SimpleNamespace

import pytest
import torch
from test_wan_t2v_provenance import native_assets as _native_assets_fixture
from test_wan_t2v_provenance import native_source as _native_source_fixture

from visual_rl.models import wan_t2v
from visual_rl.models import wan_t2v_dependencies as dependencies
from visual_rl.models.wan_t2v import WanT2VAdapter
from visual_rl.small_video_wan_parity import run_wan_parity

native_assets = _native_assets_fixture
native_source = _native_source_fixture


class NoCudaQueries:
    def __getattr__(self, name):
        pytest.fail(f"dependency preflight queried CUDA: {name}")


@pytest.fixture
def dependency_runtime(monkeypatch):
    monkeypatch.delitem(sys.modules, "wan.modules.attention", raising=False)
    torch_module = SimpleNamespace(
        __version__="2.6.0+cu124",
        version=SimpleNamespace(cuda="12.4"),
        _C=SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=False),
        cuda=NoCudaQueries(),
    )
    fa = SimpleNamespace(
        __version__="2.7.4.post1",
        __file__="/fake/flash_attn/__init__.py",
        flash_attn_varlen_func=lambda: None,
    )
    binary = SimpleNamespace(
        __file__="/fake/flash_attn_2_cuda.so", varlen_fwd=lambda: None, varlen_bwd=lambda: None
    )
    config = {"version": "2.7.4.post1", "fa3": False, "import_error": None}
    imported = []
    original_import = dependencies.importlib.import_module

    def import_module(name, *args, **kwargs):
        if name in ("torch", "flash_attn", "flash_attn_2_cuda", "flash_attn_interface"):
            imported.append(name)
        if name == "flash_attn_interface":
            pytest.fail("preflight imported FA3 instead of rejecting its default-branch override")
        if name == "torch":
            return torch_module
        if name in ("flash_attn", "flash_attn_2_cuda"):
            if config["import_error"] is not None:
                raise config["import_error"]
            return fa if name == "flash_attn" else binary
        return original_import(name, *args, **kwargs)

    original_version = dependencies.importlib.metadata.version

    def version(name):
        if name != "flash-attn":
            return original_version(name)
        if config["version"] is None:
            raise importlib.metadata.PackageNotFoundError(name)
        return config["version"]

    monkeypatch.setattr(dependencies.importlib, "import_module", import_module)
    monkeypatch.setattr(dependencies.importlib.metadata, "version", version)
    monkeypatch.setattr(
        dependencies,
        "_available",
        lambda name: config["fa3"] if name == "flash_attn_interface" else False,
    )
    return config, imported, torch_module, fa, binary


def test_pinned_binary_preflight_reports_versions_abi_fa2_without_cuda_queries(dependency_runtime):
    _, imported, _, _, _ = dependency_runtime
    report = dependencies.check_native_wan_dependencies()
    assert report["status"] == "passed" and report["native_attention"] == "flash_attention_2"
    assert report["torch_version"] == "2.6.0+cu124" and report["torch_cuda_build"] == "12.4"
    assert report["torch_cxx11_abi"] is False and report["fa3_interface_present"] is False
    assert (
        report["flash_attn_distribution_version"]
        == report["flash_attn_module_version"]
        == "2.7.4.post1"
    )
    assert report["kernel_execution"] is False
    assert imported == ["torch", "flash_attn", "flash_attn_2_cuda"]


@pytest.mark.parametrize("version", [None, "2.7.3", "2.8.0"])
def test_missing_or_wrong_fa_version_fails_before_binary_import(dependency_runtime, version):
    config, imported, _, _, _ = dependency_runtime
    config["version"] = version
    with pytest.raises(dependencies.NativeWanDependencyError, match="flash-attn") as error:
        dependencies.check_native_wan_dependencies()
    assert imported == ["torch"] and error.value.report["status"] == "failed"


def test_fa3_default_override_is_rejected_without_importing_or_patching_it(dependency_runtime):
    config, imported, _, _, _ = dependency_runtime
    config["fa3"] = True
    with pytest.raises(dependencies.NativeWanDependencyError, match="prefer FA3"):
        dependencies.check_native_wan_dependencies()
    assert imported == ["torch"]


def test_native_cached_missing_fa2_requires_restart_without_mutating_flags(
    dependency_runtime, monkeypatch
):
    cached = SimpleNamespace(FLASH_ATTN_2_AVAILABLE=False, FLASH_ATTN_3_AVAILABLE=False)
    monkeypatch.setitem(sys.modules, "wan.modules.attention", cached)
    with pytest.raises(dependencies.NativeWanDependencyError, match="restart the interpreter"):
        dependencies.check_native_wan_dependencies()
    assert cached.FLASH_ATTN_2_AVAILABLE is False and cached.FLASH_ATTN_3_AVAILABLE is False


@pytest.mark.parametrize(
    "problem", ["torch", "cuda", "abi", "binary", "shadowed_module", "interface"]
)
def test_wrong_runtime_or_incompatible_binary_fails_closed(dependency_runtime, problem):
    config, _, torch_module, fa, binary = dependency_runtime
    if problem == "torch":
        torch_module.__version__ = "2.7.0+cu126"
    elif problem == "cuda":
        torch_module.version.cuda = "12.8"
    elif problem == "abi":
        torch_module._C._GLIBCXX_USE_CXX11_ABI = True
    elif problem == "binary":
        config["import_error"] = ImportError("undefined symbol fixture")
    elif problem == "shadowed_module":
        fa.__version__ = "2.7.3"
    else:
        binary.varlen_fwd = None
    with pytest.raises(dependencies.NativeWanDependencyError) as error:
        dependencies.check_native_wan_dependencies()
    assert error.value.report["status"] == "failed"


@pytest.mark.parametrize("version", [None, "2.8.0"])
def test_factory_dependency_failure_precedes_native_import_or_weights(
    dependency_runtime, native_source, native_assets, monkeypatch, version
):
    config, imported, _, _, _ = dependency_runtime
    config["version"] = version
    source, revision = native_source
    assets, _ = native_assets
    monkeypatch.setattr(wan_t2v, "WAN_T2V_SOURCE_REVISION", revision)
    import visual_rl.models.wan_t2v_provenance as provenance

    def heavy_import(*args, **kwargs):
        pytest.fail("native model modules/weights loaded before dependency preflight")

    monkeypatch.setattr(provenance, "import_native_components", heavy_import)
    with pytest.raises(dependencies.NativeWanDependencyError):
        WanT2VAdapter.from_pretrained(assets, native_root=source)
    assert "flash_attn" not in imported and "flash_attn_2_cuda" not in imported


def test_parity_dependency_failure_is_reported_before_cuda_or_official_pipeline(
    dependency_runtime, tmp_path, monkeypatch
):
    config, _, _, _, _ = dependency_runtime
    config["version"] = None
    from visual_rl import small_video_assets
    from visual_rl.models import source_pin

    monkeypatch.setattr(small_video_assets, "load_manifest", lambda: {})
    monkeypatch.setattr(
        small_video_assets, "verify_bound_receipt", lambda *a, **k: {"checksum": "verified-fixture"}
    )

    def forbidden(*args, **kwargs):
        pytest.fail("CUDA queried or official pipeline loaded before dependency failure")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(source_pin, "import_pinned_source", forbidden)
    output = tmp_path / "parity"
    with pytest.raises(dependencies.NativeWanDependencyError):
        run_wan_parity(
            asset_root=tmp_path,
            receipt=tmp_path / "receipt.json",
            output=output,
            owned_root=tmp_path,
            device="cuda",
        )
    report = json.loads((output / "metrics.json").read_text())
    assert report["status"] == "failed" and report["error_type"] == "NativeWanDependencyError"
    assert report["native_dependencies"]["kernel_execution"] is False
    assert not list(output.glob("*.mp4"))
