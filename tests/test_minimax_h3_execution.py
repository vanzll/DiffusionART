import os

import pytest
import torch

from visual_rl.backends.minimax_h3_execution import execution_policy
from visual_rl.config import ConfigError


@pytest.mark.parametrize("deterministic,warn_only", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("fail", [False, True])
def test_policy_restores_caller_flags_and_environment(monkeypatch, deterministic, warn_only, fail):
    original = (torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled())
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    try:
        def execute():
            with execution_policy():
                assert torch.are_deterministic_algorithms_enabled()
                assert not torch.is_deterministic_algorithms_warn_only_enabled()
                assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
                if fail:
                    raise ValueError("failed H3 operation")
        if fail:
            with pytest.raises(ValueError, match="failed H3"):
                execute()
        else:
            execute()
        assert torch.are_deterministic_algorithms_enabled() == deterministic
        assert torch.is_deterministic_algorithms_warn_only_enabled() == warn_only
        assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ
    finally:
        torch.use_deterministic_algorithms(original[0], warn_only=original[1])


def test_policy_rejects_late_cuda_configuration(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(ConfigError, match="before CUDA initialization"):
        with execution_policy():
            pytest.fail("late configuration accepted")


def test_policy_rejects_conflicting_workspace(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    with pytest.raises(ConfigError, match="CUBLAS_WORKSPACE_CONFIG"):
        with execution_policy():
            pytest.fail("conflicting configuration accepted")
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"


def test_policy_preserves_preconfigured_cuda_workspace(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    with execution_policy():
        with execution_policy():
            assert torch.are_deterministic_algorithms_enabled()
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
