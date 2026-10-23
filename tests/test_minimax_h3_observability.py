import json

import torch

from visual_rl.backends.minimax_h3_observability import BatchTrace


def test_batch_trace_preserves_values_and_gradients_and_removes_hooks(capsys):
    model = torch.nn.Linear(3, 2)
    baseline = torch.nn.Linear(3, 2)
    baseline.load_state_dict(model.state_dict())
    inputs = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    trace = BatchTrace(enabled=True)
    trace.install(model)
    with torch.no_grad():
        torch.testing.assert_close(model(inputs), baseline(inputs))
    actual = model(inputs)
    expected = baseline(inputs)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(model.weight.grad, baseline.weight.grad)
    trace.close()
    assert not model._forward_pre_hooks
    assert not trace.handles
    records = [json.loads(line.removeprefix("H3_PROGRESS ")) for line in capsys.readouterr().out.splitlines()]
    assert {record["mode"] for record in records if record["stage"] == "physical_batch_shape"} == {"inference", "update"}
    assert all(record["shapes"]["arg_0"] == [4, 3] for record in records if record["stage"] == "physical_batch_shape")
    assert records[-1]["hooks_removed"] is True
    assert records[-1]["observed_module_calls"] == 2


def test_disabled_batch_trace_does_not_install_hooks(capsys):
    model = torch.nn.Linear(3, 2)
    trace = BatchTrace()
    trace.install(model)
    trace.close()
    assert not model._forward_pre_hooks
    assert not capsys.readouterr().out


def test_batch_trace_records_nested_padding_mask_without_copy(capsys):
    class Model(torch.nn.Module):
        def forward(self, inputs, *, attention_kwargs):
            return inputs

    model = Model()
    trace = BatchTrace(enabled=True)
    trace.install(model)
    inputs = torch.ones(4, 3)
    mask = torch.ones(4, 7, dtype=torch.bool)
    assert model(inputs, attention_kwargs={"h3_text_mask": mask}) is inputs
    trace.close()
    record = json.loads(capsys.readouterr().out.splitlines()[0].removeprefix("H3_PROGRESS "))
    assert record["shapes"]["attention_kwargs/h3_text_mask"] == [4, 7]
    assert record["dtypes"]["attention_kwargs/h3_text_mask"] == "torch.bool"
