"""CPU-only UUID metadata tests. No CUDA properties/NVML calls reach hardware."""

import uuid
from types import SimpleNamespace

import pytest
import torch

from visual_rl import small_video_parity as P
from visual_rl.rewards import small_video_reward_worker as W
from visual_rl.rewards.gpu_uuid import normalize_cuda_uuid

BARE = "01234567-89ab-cdef-0123-456789abcdef"


class CUuuidLike:
    """Mirror the official2.6 __str__/vector<uint8_t> interface without a GPU."""

    def __init__(self, text=BARE):
        self.text = text
        self.bytes = list(uuid.UUID(BARE).bytes)

    def __str__(self):
        return self.text


@pytest.mark.parametrize(
    "value,expected",
    [
        (BARE, "GPU-" + BARE),
        (BARE.upper(), "GPU-" + BARE),
        ("GPU-" + BARE, "GPU-" + BARE),
        ("MIG-" + BARE, "MIG-" + BARE),
        (CUuuidLike(), "GPU-" + BARE),
    ],
)
def test_cuda_uuid_normalizes_only_valid128bit_identifiers(value, expected):
    normalized, raw = normalize_cuda_uuid(value)
    assert normalized == expected and raw == str(value)


@pytest.mark.parametrize(
    "value",
    [
        None,
        "GPU-fake",
        "",
        "0",
        "GPU-0",
        "{component}",
        BARE.replace("-", ""),
        " " + BARE,
        "00000000-0000-0000-0000-000000000000",
        "<torch._C._CUuuid at 0x1234>",
    ],
)
def test_cuda_uuid_rejects_guesses_repr_nil_and_noncanonical_values(value):
    with pytest.raises(ValueError):
        normalize_cuda_uuid(value)


@pytest.mark.parametrize("data", [[0] * 15, [0] * 16, [-1] * 16, [256] * 16, [True] * 16])
def test_cuuuid_bytes_must_be16_uint8_matching_string(data):
    value = CUuuidLike()
    value.bytes = data
    with pytest.raises(ValueError, match="actual bytes must match"):
        normalize_cuda_uuid(value)


def test_reward_identity_uses_cuuuid_bytes_and_preserves_actual_visible_binding(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda device: SimpleNamespace(uuid=CUuuidLike())
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6")
    device, metadata = P._reward_device_identity()
    assert str(device) == "cuda:0" and metadata == {
        "gpu_uuid": "GPU-" + BARE,
        "torch_uuid_raw": BARE,
        "visible_device": "6",
    }


def test_installed_cuuuid_type_if_available_has_no_cpu_constructor():
    native = getattr(torch._C, "_CUuuid", None)
    if native is None:
        pytest.skip("this Torch build has no native CUDA UUID Python binding")
    assert hasattr(native, "bytes")
    # Official2.6 binding has no py::init. Do not create unsafe uninitialized
    # native memory via __new__ or fetch real device properties just for a test.
    with pytest.raises(TypeError):
        native()


def test_worker_nvml_metadata_uses_same_normalized_identifier(monkeypatch):
    seen = []

    class Library:
        def nvmlInit_v2(self):
            return 0

        def nvmlDeviceGetHandleByUUID(self, identifier, handle):
            seen.append(identifier)
            return 0

        def nvmlDeviceGetMemoryInfo(self, handle, memory):
            memory._obj.used = 123456
            return 0

        def nvmlShutdown(self):
            seen.append("shutdown")

    monkeypatch.setattr(W.ctypes, "CDLL", lambda name: Library())
    fake = SimpleNamespace(
        cuda=SimpleNamespace(get_device_properties=lambda index: SimpleNamespace(uuid=CUuuidLike()))
    )
    assert W._nvml_used(fake, SimpleNamespace(type="cuda")) == 123456
    assert seen == [("GPU-" + BARE).encode(), "shutdown"]
    fake.cuda.get_device_properties = lambda index: SimpleNamespace(uuid="GPU-invalid")
    assert W._nvml_used(fake, SimpleNamespace(type="cuda")) is None
