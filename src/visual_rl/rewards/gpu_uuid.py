"""Strict CUDA UUID presentation for diagnostic metadata; no device queries."""

import uuid


def normalize_cuda_uuid(value):
    """Return (NVML-style identifier, original Torch string).

    Torch2.6 binds CUuuid.__str__ as bare8-4-4-4-12 and CUuuid.bytes as
    sixteen uint8 values. Its stub incorrectly annotates the property as str.
    Accept canonical strings too; never accept arbitrary reprs or invent an ID.
    """
    raw = str(value)
    prefix = "MIG-" if raw.startswith("MIG-") else "GPU-"
    candidate = raw[len(prefix) :] if raw.startswith(prefix) else raw
    try:
        parsed = uuid.UUID(candidate)
    except (ValueError, TypeError, AttributeError) as error:
        raise ValueError("CUDA UUID must contain a canonical128-bit identifier") from error
    if candidate.lower() != str(parsed) or parsed.int == 0:
        raise ValueError("CUDA UUID must be canonical8-4-4-4-12 and nonzero")
    if not isinstance(value, str):
        data = getattr(value, "bytes", None)
        if (
            not isinstance(data, (list, tuple, bytes, bytearray))
            or len(data) != 16
            or any(type(part) is not int or not 0 <= part <= 255 for part in data)
            or uuid.UUID(bytes=bytes(data)) != parsed
        ):
            raise ValueError("CUDA UUID string and16 actual bytes must match")
    return prefix + str(parsed), raw
