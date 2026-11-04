"""Lazy, allocation-free dependency checks for the pinned native Wan backend.

Importing this module does not import FlashAttention, Torch or native Wan.
An explicit check imports the FA2 binary but does not query/initialize CUDA or
execute a kernel. Kernel/inference validation is a separate acceptance gate.
"""

import importlib
import importlib.metadata
import importlib.util
import json
import sys

FLASH_ATTN_VERSION = "2.7.4.post1"
TORCH_VERSION = "2.6.0"
TORCH_CUDA_VERSION = "12.4"


class NativeWanDependencyError(RuntimeError):
    def __init__(self, message, report):
        self.report = dict(report)
        super().__init__(
            f"{message}. Native Wan dependency report: {json.dumps(report, sort_keys=True)}"
        )


def _available(name):
    if name in sys.modules:
        return sys.modules[name] is not None
    return importlib.util.find_spec(name) is not None


def check_native_wan_dependencies():
    """Freeze the official default attention branch to the installed FA2 binary.

    Native Wan calls flash_attention(version=None). A discoverable FA3 interface
    would change that upstream default; reject it rather than patching native
    code, flags, version arguments, attention math, or selecting an SDPA fallback.
    """
    report = {
        "schema_version": "wan-native-dependencies/v1",
        "status": "checking",
        "python": sys.version.split()[0],
        "required_flash_attn_version": FLASH_ATTN_VERSION,
        "native_attention": "flash_attention_2",
        "kernel_execution": False,
    }

    def fail(message):
        report["status"] = "failed"
        raise NativeWanDependencyError(message, report)

    try:
        torch = importlib.import_module("torch")
        report.update(
            torch_version=str(torch.__version__),
            torch_cuda_build=torch.version.cuda,
            torch_cxx11_abi=getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", None),
        )
    except Exception as exc:
        report["torch_import_error_type"] = type(exc).__name__
        fail("Cannot import the pinned Torch runtime before native Wan loading")
    if (
        report["torch_version"].split("+", 1)[0] != TORCH_VERSION
        or report["torch_cuda_build"] != TORCH_CUDA_VERSION
        or report["torch_cxx11_abi"] is not False
    ):
        fail("Native Wan requires the pinned Torch2.6.0 CUDA12.4 ABI=False build")
    report["fa3_interface_present"] = _available("flash_attn_interface")
    if report["fa3_interface_present"]:
        fail(
            "flash_attn_interface is present and official Wan would prefer FA3; use a dedicated FA2-only environment"
        )
    try:
        report["flash_attn_distribution_version"] = importlib.metadata.version("flash-attn")
    except importlib.metadata.PackageNotFoundError:
        fail(
            "Native Wan requires optional flash-attn==2.7.4.post1; install its Wan-only requirement after Torch with --no-build-isolation --no-deps"
        )
    if report["flash_attn_distribution_version"] != FLASH_ATTN_VERSION:
        fail("Installed flash-attn version differs from the pinned native Wan FA2 version")
    try:
        module = importlib.import_module("flash_attn")
        binary = importlib.import_module("flash_attn_2_cuda")
    except Exception as exc:
        report["flash_attn_import_error_type"] = type(exc).__name__
        report["binary_import_compatible"] = False
        fail(
            "Pinned FA2 binary cannot import against this Torch/CUDA/ABI; rebuild it after the pinned Torch using --no-build-isolation --no-deps"
        )
    report.update(
        flash_attn_module_version=getattr(module, "__version__", None),
        flash_attn_module_origin=getattr(module, "__file__", None),
        flash_attn_binary_origin=getattr(binary, "__file__", None),
    )
    if (
        report["flash_attn_module_version"] != FLASH_ATTN_VERSION
        or not callable(getattr(module, "flash_attn_varlen_func", None))
        or not callable(getattr(binary, "varlen_fwd", None))
        or not callable(getattr(binary, "varlen_bwd", None))
    ):
        fail("Imported FA2 module version/interface differs from its pinned installed distribution")
    cached = sys.modules.get("wan.modules.attention")
    if cached is not None and getattr(cached, "FLASH_ATTN_3_AVAILABLE", False):
        fail(
            "Cached native Wan attention would select FA3; restart in the dedicated FA2-only environment"
        )
    if cached is not None and getattr(cached, "FLASH_ATTN_2_AVAILABLE", None) is not True:
        fail(
            "Native Wan attention was already imported without FA2; restart the interpreter after installation"
        )
    report["binary_import_compatible"] = True
    report["status"] = "passed"
    return report


def main():
    try:
        report = check_native_wan_dependencies()
    except NativeWanDependencyError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
