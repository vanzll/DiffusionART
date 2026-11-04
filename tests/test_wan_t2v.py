"""Native batch, masking, time and VAE contracts without pretrained assets."""

import importlib.util
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from visual_rl.models.finetune import FineTuneConfig, parameter_inventory, prepare_policy
from visual_rl.models.wan_t2v import (
    WAN_T2V_NATIVE_LORA_TARGETS,
    WanT2VAdapter,
    WanT2VConditioning,
    WanT2VConfig,
    masked_wan_context,
)


class NativeField(nn.Module):
    """Real batched differentiable projections using the official forward API."""

    def __init__(self):
        super().__init__()
        self.model_type, self.in_dim, self.out_dim, self.text_len = "t2v", 2, 2, 5
        self.q, self.k, self.v, self.o = [nn.Linear(2, 2) for _ in range(4)]
        self.context = nn.Linear(3, 2)
        self.calls = []

    def forward(self, x, t, context, seq_len):
        self.calls.append((len(x), t.detach().clone(), context, seq_len))
        state = torch.stack(x)
        # Project all video rows together, making accidental per-row forward visible.
        hidden = state.movedim(1, -1)
        hidden = self.o(self.v(self.k(self.q(hidden))))
        prompt = self.context(torch.stack(context)).mean(1)
        output = hidden.movedim(-1, 1) + prompt[:, :, None, None, None]
        output = output + t[:, None, None, None, None] / 1000
        return list(output.unbind())


class NativeVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.2))
        self.calls = []

    def decode(self, x, scale):
        self.calls.append((x.detach().clone(), scale))
        return x * self.weight


def setup_adapter(cfg=6.0):
    config = WanT2VConfig((0.1, -0.2), (2.0, 3.0), cfg_scale=cfg, text_len=5)
    return WanT2VAdapter(NativeField(), NativeVAE(), config)


def conditioning():
    # Positive and negative prompts may have different padded lengths.
    return WanT2VConditioning(
        torch.randn(2, 4, 3),
        torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]]),
        torch.randn(2, 2, 3),
        torch.tensor([[1, 0], [1, 1]]),
        timesteps=torch.tensor([873, 123], dtype=torch.int64),
    )


def test_native_cfg_is_one_batched_call_and_matches_separate_branch_gradients():
    adapter, c = setup_adapter(), conditioning()
    z = torch.randn(2, 2, 2, 4, 6, requires_grad=True)
    sigma = torch.tensor([0.87399, 0.12391])
    actual = adapter(z, sigma, c)
    call = adapter.transformer.calls[0]
    assert call[0] == 4 and call[3] == 12
    assert len(adapter.transformer.calls) == 1
    assert call[1].tolist() == [873, 123, 873, 123]
    assert all(row.shape == (5, 3) for row in call[2])
    positive = masked_wan_context(c.prompt_embeds, c.prompt_attention_mask, text_len=5)
    negative = masked_wan_context(
        c.negative_prompt_embeds, c.negative_prompt_attention_mask, text_len=5
    )
    model = adapter.transformer
    vp = torch.stack(model(list(z.unbind()), c.timesteps, list(positive.unbind()), 12))
    vn = torch.stack(model(list(z.unbind()), c.timesteps, list(negative.unbind()), 12))
    expected = vn + 6 * (vp - vn)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.square().sum(), z, retain_graph=True)[0],
        torch.autograd.grad(expected.square().sum(), z)[0],
    )


def test_padding_values_cannot_affect_native_velocity_and_valid_tokens_keep_gradients():
    adapter, c = setup_adapter(None), conditioning()
    prompt = c.prompt_embeds.detach().requires_grad_()
    c = replace(c, prompt_embeds=prompt)
    z = torch.randn(2, 2, 1, 2, 2)
    first = adapter(z, 0.4, c)
    corrupted = prompt.detach().clone()
    corrupted[~c.prompt_attention_mask.bool()] = float("nan")
    second = adapter(z, 0.4, replace(c, prompt_embeds=corrupted))
    torch.testing.assert_close(first, second)
    first.sum().backward()
    assert prompt.grad[c.prompt_attention_mask.bool()].abs().sum() > 0
    assert prompt.grad[~c.prompt_attention_mask.bool()].abs().sum() == 0


@pytest.mark.parametrize(
    "mask", [torch.tensor([[0, 1]]), torch.tensor([[0, 0]]), torch.tensor([[1, 2]])]
)
def test_invalid_text_masks_fail_closed(mask):
    with pytest.raises(ValueError, match="mask"):
        masked_wan_context(torch.zeros(1, 2, 3), mask, text_len=5)


def test_native_time_conditions_are_required_and_never_rebuilt_from_sigma():
    adapter, c = setup_adapter(None), conditioning()
    with pytest.raises(ValueError, match="timesteps"):
        adapter(torch.zeros(2, 2, 1, 2, 2), 0.3, replace(c, timesteps=None))
    with pytest.raises(ValueError, match="INT64"):
        adapter(torch.zeros(2, 2, 1, 2, 2), 0.3, replace(c, timesteps=c.timesteps.float()))


def test_native_decode_affine_once_freezes_vae_and_retains_input_derivative():
    adapter = setup_adapter(None)
    z = (torch.randn(2, 2, 2, 2, 2) * 0.01).requires_grad_()
    pixels = adapter.decode(z)
    mean = torch.tensor([0.1, -0.2]).reshape(1, 2, 1, 1, 1)
    std = torch.tensor([2.0, 3.0]).reshape(1, 2, 1, 1, 1)
    expected = (z * std + mean) * adapter.vae.weight
    torch.testing.assert_close(pixels, expected)
    torch.testing.assert_close(torch.autograd.grad(pixels.sum(), z)[0], (std * 0.2).expand_as(z))
    assert adapter.vae.calls[0][1] == (0.0, 1.0)
    assert not adapter.vae.weight.requires_grad and adapter.vae.weight.grad is None
    adapter.train()
    assert not adapter.vae.training


def test_recommended_latent_shape_and_native_incompatible_components():
    config = WanT2VConfig()
    assert config.latent_shape(height=480, width=832, frames=81) == (16, 21, 60, 104)
    with pytest.raises(ValueError, match="align"):
        config.latent_shape(height=480, width=832, frames=80)
    model = NativeField()
    model.model_type = "i2v"
    with pytest.raises(ValueError, match="model_type"):
        WanT2VAdapter(model, NativeVAE(), setup_adapter().config)


def test_native_full_fp32_optimizer_contract_and_backward():
    adapter, c = setup_adapter(None), conditioning()
    adapter.transformer = prepare_policy(
        adapter.transformer.to(torch.bfloat16), FineTuneConfig("full")
    )
    inventory = parameter_inventory(adapter.transformer)
    assert not inventory["frozen"] and set(inventory["trainable"]) == {"float32"}
    adapter(torch.randn(2, 2, 1, 2, 2), 0.5, c).square().mean().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in adapter.transformer.parameters()
    )


def test_native_lora_targets_save_load_disable_and_fp32_optimizer(tmp_path):
    peft = pytest.importorskip("peft")
    adapter, c = setup_adapter(None), conditioning()
    base = adapter.transformer
    base_state = {key: value.clone() for key, value in base.state_dict().items()}
    z = torch.randn(2, 2, 1, 2, 2)
    before = adapter(z, 0.5, c).detach()
    adapter.transformer = prepare_policy(
        base, FineTuneConfig("lora", rank=32, alpha=32, target_modules=WAN_T2V_NATIVE_LORA_TARGETS)
    )
    torch.testing.assert_close(adapter(z, 0.5, c), before)
    trainable = [p for p in adapter.transformer.parameters() if p.requires_grad]
    assert trainable and all(p.dtype == torch.float32 for p in trainable)
    assert all(
        layer.lora_dropout["default"].p == 0
        for layer in adapter.transformer.modules()
        if hasattr(layer, "lora_dropout") and hasattr(layer.lora_dropout["default"], "p")
    )
    optimizer = torch.optim.AdamW(trainable, lr=0.01)
    adapter(z, 0.5, c).square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainable)
    optimizer.step()
    changed = adapter(z, 0.5, c).detach()
    assert not torch.allclose(changed, before)
    with adapter.transformer.disable_adapter():
        torch.testing.assert_close(adapter(z, 0.5, c), before)
    adapter.transformer.save_pretrained(tmp_path)
    fresh = NativeField()
    fresh.load_state_dict(base_state)
    adapter.transformer = peft.PeftModel.from_pretrained(fresh, tmp_path, is_trainable=True)
    torch.testing.assert_close(adapter(z, 0.5, c), changed)


def test_diffusers_injection_preserves_native_zero_padding_and_batched_time():
    class Field(nn.Module):
        config = SimpleNamespace(in_channels=2, out_channels=2, image_dim=None)

        def forward(self, *, hidden_states, timestep, encoder_hidden_states, return_dict):
            assert len(hidden_states) == 4 and timestep.dtype == torch.int64
            assert encoder_hidden_states.shape == (4, 5, 3)
            return (hidden_states,)

    config = replace(setup_adapter().config, transformer_api="diffusers")
    adapter = WanT2VAdapter(Field(), NativeVAE(), config)
    z = torch.randn(2, 2, 1, 2, 2)
    torch.testing.assert_close(adapter(z, 0.5, conditioning()), z)


def test_real_official_native_transformer_batch_with_cpu_sdpa(monkeypatch):
    """Real pinned DiT blocks/RoPE/patch/time/text interfaces, CPU attention only.

    SDPA replaces the GPU-only flash kernel for this test; this is explicitly
    not a FlashAttention numerical parity test or pretrained weight validation.
    """
    source = os.environ.get("WAN_T2V_NATIVE_SOURCE")
    if not source:
        pytest.skip("WAN_T2V_NATIVE_SOURCE not set")
    root = Path(source)
    path = root / "wan/modules/model.py"
    if not path.is_file():
        path = root / "wan__modules__model.py"
    if not path.is_file():
        pytest.fail("official native model source is missing")
    pytest.importorskip("diffusers")

    def cpu_attention(q, k, v, *, k_lens=None, window_size=(-1, -1), **kwargs):
        assert window_size == (-1, -1)
        mask = None
        if k_lens is not None:
            mask = torch.arange(k.shape[1])[None, :] < k_lens[:, None]
            mask = mask[:, None, None, :]
        result = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask
        )
        return result.transpose(1, 2)

    package_name = "_wan_t2v_cpu_source"
    package = ModuleType(package_name)
    package.__path__ = []
    attention = ModuleType(package_name + ".attention")
    attention.flash_attention = cpu_attention
    monkeypatch.setitem(sys.modules, package_name, package)
    monkeypatch.setitem(sys.modules, package_name + ".attention", attention)
    spec = importlib.util.spec_from_file_location(package_name + ".model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = (
        module.WanModel(
            model_type="t2v",
            text_len=5,
            in_dim=2,
            out_dim=2,
            dim=24,
            ffn_dim=48,
            freq_dim=8,
            text_dim=3,
            num_heads=2,
            num_layers=1,
        )
        .float()
        .eval()
    )
    # Native initialization zeros the final projection; make input gradients observable.
    with torch.no_grad():
        model.head.head.weight.normal_(std=0.05)
    adapter = WanT2VAdapter(model, NativeVAE(), setup_adapter().config)
    c = conditioning()
    z = torch.randn(2, 2, 1, 2, 4, requires_grad=True)
    physical_batches = []
    hook = model.blocks[0].register_forward_pre_hook(
        lambda block, inputs: physical_batches.append(len(inputs[0]))
    )
    actual = adapter(z, torch.tensor([0.87399, 0.12391]), c)
    hook.remove()
    assert physical_batches == [4]
    pos = [row[mask.bool()] for row, mask in zip(c.prompt_embeds, c.prompt_attention_mask)]
    neg = [
        row[mask.bool()]
        for row, mask in zip(c.negative_prompt_embeds, c.negative_prompt_attention_mask)
    ]
    vp = torch.stack(model(list(z.unbind()), c.timesteps, pos, seq_len=2))
    vn = torch.stack(model(list(z.unbind()), c.timesteps, neg, seq_len=2))
    expected = vn + 6 * (vp - vn)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    actual_gradient = torch.autograd.grad(actual.square().sum(), z, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected.square().sum(), z)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=2e-5, rtol=2e-4)
    assert torch.isfinite(actual_gradient).all() and actual_gradient.abs().sum() > 0
