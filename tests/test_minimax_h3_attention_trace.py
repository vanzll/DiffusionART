import json

import pytest
import torch
import torch.nn.functional as functional

from visual_rl.backends.minimax_h3_attention_trace import OneShotAttentionTrace


def test_actual_operator_types_and_values_gradients_rng_preserved(tmp_path, monkeypatch):
    monkeypatch.setenv("H3_ATTEMPT_NONCE", "cpu-attention-audit")
    monkeypatch.setenv("RANK", "0")
    original = functional.scaled_dot_product_attention
    values = torch.arange(256, dtype=torch.float32).reshape(1, 2, 16, 8)/256
    baseline = values.clone().requires_grad_()
    actual = values.clone().requires_grad_()
    before = torch.get_rng_state()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        expected = original(baseline, baseline, baseline)
    expected.sum().backward()
    expected_rng = torch.get_rng_state()
    torch.set_rng_state(before)
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(functional, "scaled_dot_product_attention", counted)
    observer = OneShotAttentionTrace(tmp_path, enabled=True, device_type="cpu")
    observer.install()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = functional.scaled_dot_product_attention(actual, actual, actual)
    result.sum().backward()
    assert len(calls) == 1
    assert functional.scaled_dot_product_attention is counted
    assert torch.equal(result, expected)
    assert torch.equal(actual.grad, baseline.grad)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    receipt = json.loads((tmp_path/"attention-traces/cpu-attention-audit/rank-00000-first-sdpa.json").read_text())
    assert receipt["cuda_activity_enabled"] is False
    assert receipt["operation_replayed"] is False
    assert receipt["output_dtype"] == "torch.bfloat16"
    assert any(record["input_types"][:3] == ["c10::BFloat16"]*3
               for record in receipt["attention_operators"])
    assert any("_scaled_dot_product" in record["name"] for record in receipt["attention_operators"])
    assert not torch.cuda.is_initialized()


def test_disabled_and_other_device_leave_original_execution(tmp_path, monkeypatch):
    original = functional.scaled_dot_product_attention
    disabled = OneShotAttentionTrace(tmp_path)
    disabled.install()
    assert functional.scaled_dot_product_attention is original
    observer = OneShotAttentionTrace(tmp_path, enabled=True)
    observer.install()
    inputs = torch.ones(1, 1, 2, 4)
    actual = functional.scaled_dot_product_attention(query=inputs, key=inputs, value=inputs)
    assert torch.equal(actual, original(inputs, inputs, inputs))
    assert not observer.used
    observer.close()
    assert functional.scaled_dot_product_attention is original
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("failure", ["start", "export", "operator", "nested", "unsafe_nonce"])
def test_diagnostic_failure_never_replays_or_masks_operator(tmp_path, monkeypatch, failure):
    monkeypatch.setenv("H3_ATTEMPT_NONCE", "../unsafe" if failure == "unsafe_nonce" else "audit")
    monkeypatch.setattr(torch.autograd, "_profiler_enabled", lambda: failure == "nested")
    calls = []
    sentinel = ValueError("original attention failure")
    def original(*args, **kwargs):
        calls.append(1)
        if failure == "operator":
            raise sentinel
        return args[0]
    class FakeProfiler:
        def __enter__(self):
            if failure == "start":
                raise RuntimeError("profiler start failed")
            return self
        def __exit__(self, *args):
            if failure == "operator":
                raise RuntimeError("cleanup must not mask operator")
        def export_chrome_trace(self, path):
            raise RuntimeError("export failed")
    monkeypatch.setattr(functional, "scaled_dot_product_attention", original)
    monkeypatch.setattr(torch.profiler, "profile", lambda **kwargs: FakeProfiler())
    observer = OneShotAttentionTrace(tmp_path, enabled=True, device_type="cpu")
    observer.install()
    inputs = torch.ones(1, 1, 2, 4)
    if failure == "operator":
        with pytest.raises(ValueError) as caught:
            functional.scaled_dot_product_attention(inputs, inputs, inputs)
        assert caught.value is sentinel
    else:
        assert functional.scaled_dot_product_attention(inputs, inputs, inputs) is inputs
    assert calls == [1]
    assert functional.scaled_dot_product_attention is original
