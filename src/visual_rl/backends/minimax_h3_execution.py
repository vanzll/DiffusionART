"""Scoped deterministic execution for native H3 training and checkpoint replay."""

import os
from contextlib import contextmanager

import torch

from ..config import ConfigError


@contextmanager
def execution_policy():
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace not in (None, ":4096:8"):
        raise ConfigError("H3 requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA initialization")
    if workspace is None and torch.cuda.is_initialized():
        raise ConfigError("H3 deterministic execution must be configured before CUDA initialization")
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        yield
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        if workspace is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace
