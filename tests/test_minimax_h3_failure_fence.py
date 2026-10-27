import subprocess

import pytest

from visual_rl.backends import minimax_h3_failure_fence as fence


def test_fence_reports_before_outer_teardown_and_preserves_exception(monkeypatch):
    order = []
    monkeypatch.setattr(fence, "report_failure", lambda name: order.append(name))
    error = ValueError("private data must not be logged")
    with pytest.raises(ValueError) as raised:
        try:
            with fence.native_failure_fence():
                raise error
        finally:
            order.append("teardown")
    assert raised.value is error
    assert order == ["ValueError", "teardown"]


def test_reporting_failure_cannot_replace_original_exception(monkeypatch, capsys):
    def fail(name):
        raise OSError("private payload")
    monkeypatch.setattr(fence, "report_failure", fail)
    with pytest.raises(ArithmeticError):
        with fence.native_failure_fence():
            raise ArithmeticError("private payload")
    assert "private payload" not in capsys.readouterr().out


def test_report_child_wait_is_bounded(monkeypatch):
    actions = []
    class Child:
        def wait(self, timeout):
            actions.append(timeout)
            raise subprocess.TimeoutExpired("child", timeout)
        def kill(self):
            actions.append("kill")
    monkeypatch.setenv("H3_ATTEMPT_NONCE", "test")
    monkeypatch.setattr(fence.subprocess, "Popen", lambda *args, **kwargs: Child())
    fence.report_failure("RuntimeError", timeout=.05)
    assert actions == [.05, "kill"]


def test_foreign_attempt_rejected_before_write(monkeypatch, tmp_path):
    attempt = tmp_path / "attempts" / "old"
    attempt.mkdir(parents=True)
    monkeypatch.setenv("H3_ACCEPTANCE_ROOT", str(tmp_path))
    monkeypatch.setenv("H3_ATTEMPT_DIR", str(attempt))
    monkeypatch.setenv("H3_ATTEMPT_NONCE", "new")
    with pytest.raises(RuntimeError, match="identity changed"):
        fence.publish_owned_failure(1, "0", "RuntimeError")
    assert not list(attempt.glob("failure.*"))


def test_cp2_logical_prompt_ownership_and_reject_physical_dp64():
    from types import SimpleNamespace

    from visual_rl.backends.minimax_h3_parallel import H3Parallel
    from visual_rl.config import ConfigError

    config = SimpleNamespace(model=SimpleNamespace(options={"context_parallel_size": 2, "finetune": {"mode": "full"}}),
                             runtime=SimpleNamespace(distributed="fsdp"), algorithm=SimpleNamespace(name="diffusionart"))
    identities = []
    for rank in range(64):
        parallel = H3Parallel.from_config(config, SimpleNamespace(rank=rank, world_size=64))
        assert parallel.local_prompt_count(32) == 1
        identities.append((parallel.dp_rank, parallel.cp_rank))
    assert identities == [(dp, cp) for dp in range(32) for cp in range(2)]
    with pytest.raises(ConfigError):
        H3Parallel(1, 0, 64).local_prompt_count(32)
