import pytest
import torch

from visual_rl.models.finetune import FineTuneConfig, parameter_inventory, prepare_policy


@pytest.mark.parametrize("kwargs", [dict(mode="unknown"), dict(mode="full", rank=4),
                                    dict(mode="lora", rank=0, alpha=4, target_modules=("0",)),
                                    dict(mode="lora", rank=2, alpha=float("nan"), target_modules=("0",)),
                                    dict(mode="lora", rank=2, alpha=4, target_modules="0")])
def test_bad_finetune_config(kwargs):
    with pytest.raises(ValueError):
        FineTuneConfig(**kwargs)


def test_full_precision_and_gradient_reset():
    model = torch.nn.Linear(3, 2).bfloat16()
    model.weight.grad = torch.ones_like(model.weight)
    result = prepare_policy(model, FineTuneConfig(mode="full"))
    assert result is model
    assert parameter_inventory(result) == {"trainable": {"float32": 8}, "frozen": {}}
    assert all(p.grad is None for p in result.parameters())


def test_lora_init_output_parity_freezes_base_and_updates_only_adapters():
    pytest.importorskip("peft")
    with torch.random.fork_rng():
        torch.manual_seed(31)
        model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 4))
        inputs = torch.randn(3, 4)
        before = model(inputs).detach()
        result = prepare_policy(model, FineTuneConfig(
            mode="lora", rank=2, alpha=4, target_modules=("0", "2")
        ))
        torch.testing.assert_close(result(inputs), before, rtol=0, atol=0)
        frozen = {name: p.detach().clone() for name, p in result.named_parameters()
                  if not p.requires_grad}
        trainable = {name: p for name, p in result.named_parameters() if p.requires_grad}
        assert trainable and all("lora_" in name for name in trainable)
        assert all(p.dtype == torch.float32 for p in trainable.values())
        optimizer = torch.optim.AdamW(trainable.values(), lr=1e-3)
        result(inputs).square().mean().backward()
        optimizer.step()
        assert not torch.equal(result(inputs), before)
        for name, p in result.named_parameters():
            if name in frozen:
                torch.testing.assert_close(p, frozen[name], rtol=0, atol=0)
                assert p.grad is None
        with pytest.raises(ValueError, match="fresh"):
            prepare_policy(result, FineTuneConfig(mode="full"))


def test_bad_target_fails_before_mutation():
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.ReLU())
    with pytest.raises(ValueError, match="linear"):
        prepare_policy(model, FineTuneConfig(mode="lora", rank=2, alpha=2, target_modules=("1",)))
    assert all(p.requires_grad for p in model.parameters())
