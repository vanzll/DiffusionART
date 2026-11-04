"""Observation/comparison contract tests only; these are NOT model parity evidence."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from visual_rl import small_video_parity as parity
from visual_rl.training.small_video_replay import collect_native_video


class ToyTransformer(nn.Module):
    def forward(self, state, **kwargs):
        return (torch.ones_like(state) * 0.2,)


class ToyVAE:
    def decode(self, value, **kwargs):
        return (value.mean(1, keepdim=True).expand(-1, 3, -1, -1, -1),)


class ObservedScheduler:
    """Small independently written reference for observation wiring only."""

    def __init__(self):
        self.timesteps = torch.tensor(
            [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875]
        )

    def add_noise(self, clean, noise, sigma):
        return (1 - sigma) * clean + sigma * noise

    def step(self, prediction, sigma, state, return_dict=False, stochastic_sampling=False):
        assert stochastic_sampling
        boundaries = torch.cat((self.timesteps, torch.zeros(1)))
        following = boundaries[boundaries < sigma.flatten()[0] - 1e-6][0]
        clean = state - sigma[..., None] * prediction
        return (self.add_noise(clean, torch.randn_like(state), following.reshape(1, 1, 1)),)


class ObservedPipeline:
    def __init__(self, module):
        self.module, self.transformer = module, ToyTransformer()
        self.scheduler, self.vae = ObservedScheduler(), ToyVAE()

    def prepare_latents(self, generator):
        packed = torch.randn(2, 2, 2, generator=generator, dtype=torch.bfloat16)
        return packed.reshape(2, 1, 1, 2, 2).permute(0, 4, 1, 2, 3)

    def encode_prompt(self):
        return (torch.ones(2, 2, 3), torch.tensor([[1, 0], [1, 1]]), None, None)

    def __call__(self, generator):
        self.encode_prompt()
        initial = self.prepare_latents(generator)
        state = initial.permute(0, 2, 3, 4, 1).reshape(2, 2, 2)
        for sigma in self.scheduler.timesteps:
            prediction = self.transformer(
                state.to(torch.bfloat16), timestep=sigma.expand(2, 1), return_dict=False
            )[0]
            state = self.scheduler.step(
                prediction, sigma.reshape(1, 1), state, return_dict=False, stochastic_sampling=True
            )[0]
        endpoint = state.reshape(2, 1, 1, 2, 2).permute(0, 4, 1, 2, 3)
        noise = self.module.torch.randn_like(endpoint)
        return self.vae.decode(
            0.975 * endpoint + 0.025 * noise, timestep=torch.tensor([0.05, 0.05])
        )[0]


def test_observation_preserves_actual_native_calls_outputs_rng_and_descriptors():
    module = SimpleNamespace(torch=torch)
    pipe = ObservedPipeline(module)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        baseline = pipe(torch.Generator().manual_seed(42))
        baseline_rng = torch.random.get_rng_state().clone()
        torch.manual_seed(11)
        with parity.observe_official_ltx(pipe, module, device=torch.device("cpu")) as trace:
            actual = pipe(torch.Generator().manual_seed(42))
        torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
        assert torch.equal(torch.random.get_rng_state(), baseline_rng)
    assert module.torch is torch
    assert "step" not in vars(pipe.scheduler) and "decode" not in vars(pipe.vae)
    assert "prepare_latents" not in vars(pipe)
    parity.validate_official_trace(trace, shape=(2, 2, 1, 1, 2))
    assert len(trace.steps) == len(trace.forwards) == 8
    assert len(trace.decode_noise) == 1
    assert trace.steps[-1]["next_sigma"].item() == 0
    assert trace.steps[-1]["innovation"].shape == (2, 2, 2)
    assert trace.initial.dtype == torch.bfloat16 and trace.steps[-1]["state"].dtype == torch.float32


def test_captured_stream_drives_independent_native_collector_without_new_noise_plan():
    module = SimpleNamespace(torch=torch)
    pipe = ObservedPipeline(module)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        with parity.observe_official_ltx(pipe, module, device=torch.device("cpu")) as trace:
            pipe(torch.Generator().manual_seed(42))
        parity._set_rng(trace.rng_after_initial, torch.device("cpu"))

        def predict(state, sigma, rows, index):
            return torch.ones_like(state, dtype=torch.bfloat16) * 0.2

        cache = collect_native_video(
            predict,
            trace.initial,
            model="ltx_video_2b_distilled",
            selected=tuple(range(8)),
            microbatch=2,
        )
    metrics = parity.compare_native_cache(trace, cache, (2, 2, 1, 1, 2))
    assert len(metrics) == 42 and all(item["passed"] for item in metrics)
    # A single bad entry must fail even when the global MSE would be tiny.
    cache.next_states[0, 4, 0, 0, 0, 0] += 0.1
    assert not all(
        item["passed"] for item in parity.compare_native_cache(trace, cache, (2, 2, 1, 1, 2))
    )


def test_observers_restore_after_reference_failure():
    module = SimpleNamespace(torch=torch)
    pipe = ObservedPipeline(module)
    with pytest.raises(RuntimeError, match="injected"):
        with parity.observe_official_ltx(pipe, module, device=torch.device("cpu")):
            raise RuntimeError("injected")
    assert module.torch is torch and not pipe.transformer._forward_hooks
    assert "step" not in vars(pipe.scheduler) and "prepare_latents" not in vars(pipe)


def test_native_draw_observer_records_actual_storage_mapping_without_changing_rng():
    value = torch.zeros(2, 3, 2).permute(0, 2, 1)
    draws = []
    observer = parity._TorchDrawObserver(
        torch, lambda x, y, before: draws.append((x.stride(), y.clone(), before))
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(97)
        expected = torch.randn_like(value)
        after = torch.random.get_rng_state().clone()
        torch.manual_seed(97)
        actual = observer.randn_like(value)
        assert torch.equal(torch.random.get_rng_state(), after)
    assert draws[0][0] == value.stride()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(draws[0][1], expected, rtol=0, atol=0)


def test_comparator_exposes_dtype_tolerance_and_fails_nonfinite_or_shape():
    ref = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
    metric = parity.compare_tensor("v", ref, ref.float() + 0.001, tolerance="bf16")
    assert metric["passed"] and metric["reference_dtype"] == "torch.bfloat16"
    assert metric["atol"] == 1 / 128
    assert not parity.compare_tensor("x", ref, ref.float() + 0.1, tolerance="fp32")["passed"]
    with pytest.raises(ValueError, match="nonfinite"):
        parity.compare_tensor("x", ref, torch.full_like(ref, float("nan")), tolerance="bf16")
    with pytest.raises(ValueError, match="shape"):
        parity.compare_tensor("x", ref, ref[:, :1], tolerance="exact")


def test_unsupported_official_kwargs_are_not_swallowed_by_variadic_signature():
    class Signature:
        def __call__(self, prompt=None, **kwargs):
            pass

    parity.validate_official_kwargs(Signature, {"prompt": ["a", "b"], "is_video": True})
    with pytest.raises(ValueError, match="unsupported"):
        parity.validate_official_kwargs(Signature, {"silently_ignored_new_solver": "euler"})
    kwargs = parity.official_pipeline_kwargs(["short", "a much longer prompt"], object())
    assert kwargs["stochastic_sampling"] and kwargs["num_inference_steps"] == 8
    assert kwargs["guidance_scale"] == 1 and kwargs["stg_scale"] == 0
    assert kwargs["decode_timestep"] == 0.05 and kwargs["decode_noise_scale"] == 0.025
    assert not kwargs["enhance_prompt"] and "sampler" not in kwargs


def test_default_execution_gate_and_plan_never_touch_cuda_or_assets(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        torch.cuda, "is_available", lambda: pytest.fail("CUDA queried without opt-in")
    )
    monkeypatch.setattr(
        parity, "verify_receipt", lambda *a, **k: pytest.fail("loaded assets without opt-in")
    )
    with pytest.raises(ValueError, match="explicit --device cuda"):
        parity.run_ltx_parity(
            asset_root=tmp_path, receipt=tmp_path / "receipt", output=tmp_path / "out"
        )
    assert parity.main(["--asset-root", str(tmp_path), "--plan"]) == 0
    assert '"pretrained_inference": false' in capsys.readouterr().out
    assert not (tmp_path / "out").exists()


class ProjectionAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q, self.to_k, self.to_v = (nn.Linear(4, 4) for _ in range(3))
        self.to_out = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.0))

    def forward(self, value):
        return self.to_out((self.to_q(value) + self.to_k(value) + self.to_v(value)) / 3)


class ProjectionPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn1, self.attn2 = ProjectionAttention(), ProjectionAttention()

    def forward(self, value):
        return self.attn2(self.attn1(value))


def candidate_fixture():
    policy = ProjectionPolicy().bfloat16().requires_grad_(False).eval()
    vae = nn.Linear(2, 2).requires_grad_(False).eval()
    encoder = nn.Linear(2, 2).requires_grad_(False).eval()
    return SimpleNamespace(transformer=policy, vae=vae), encoder


def test_candidate_cannot_mutate_native_reference_before_capture():
    adapter, encoder = candidate_fixture()
    values = {name: parameter.clone() for name, parameter in adapter.transformer.named_parameters()}
    with pytest.raises(ValueError, match="official reference before"):
        parity.prepare_parity_candidate(
            adapter, encoder, "full", reference_captured=False, expected_attention_layers=1
        )
    for name, parameter in adapter.transformer.named_parameters():
        assert parameter.dtype == torch.bfloat16 and not parameter.requires_grad
        torch.testing.assert_close(parameter, values[name], atol=0, rtol=0)


def test_full_candidate_has_all_original_fp32_parameters_and_frozen_services():
    adapter, encoder = candidate_fixture()
    names = {name for name, _ in adapter.transformer.named_parameters()}
    total = sum(p.numel() for p in adapter.transformer.parameters())
    report = parity.prepare_parity_candidate(
        adapter, encoder, "full", reference_captured=True, expected_attention_layers=1
    )
    assert set(report["trainable_names"]) == names
    assert report["after"] == {"trainable": {"float32": total}, "frozen": {}}
    assert report["attention_target_counts"] == dict.fromkeys(
        ("to_q", "to_k", "to_v", "to_out.0"), 2
    )
    assert report["optimizer_calls"] == 0 and report["candidate_autocast"] == "bf16"
    assert not report["vae"]["trainable"] and not report["text_encoder"]["trainable"]
    assert all(
        p.dtype == torch.float32 and p.requires_grad and p.grad is None
        for p in adapter.transformer.parameters()
    )


def test_real_lora_zero_enabled_disable_base_parity_and_exact_attention_coverage():
    pytest.importorskip("peft")
    adapter, encoder = candidate_fixture()
    inputs = torch.arange(8.0).reshape(2, 4).bfloat16()
    with torch.no_grad():
        base = adapter.transformer(inputs).clone()
        report = parity.prepare_parity_candidate(
            adapter, encoder, "lora", reference_captured=True, expected_attention_layers=1
        )
        enabled = adapter.transformer(inputs)
        with adapter.transformer.disable_adapter():
            disabled = adapter.transformer(inputs)
    torch.testing.assert_close(enabled, base, rtol=0, atol=0)
    torch.testing.assert_close(disabled, base, rtol=0, atol=0)
    assert report["lora"]["adapter_layers"] == 8 and report["lora"]["zero_B_verified"]
    assert report["lora"]["rank"] == report["lora"]["alpha"] == 32
    assert report["lora"]["dropout"] == 0.0
    assert all("lora_" in name for name in report["trainable_names"])
    assert set(report["after"]["trainable"]) == {"float32"}
    assert all(parameter.grad is None for parameter in adapter.transformer.parameters())
    assert all(
        not p.requires_grad for module in (adapter.vae, encoder) for p in module.parameters()
    )


def test_target_and_frozen_service_mismatch_fail_before_candidate_preparation():
    adapter, encoder = candidate_fixture()
    adapter.transformer.attn2.to_v = nn.Identity()
    with pytest.raises(ValueError, match="target count/type"):
        parity.prepare_parity_candidate(
            adapter, encoder, "lora", reference_captured=True, expected_attention_layers=1
        )
    assert not getattr(adapter.transformer, "peft_config", None)
    adapter, encoder = candidate_fixture()
    encoder.requires_grad_(True)
    with pytest.raises(ValueError, match="T5 must already be frozen"):
        parity.prepare_parity_candidate(
            adapter, encoder, "full", reference_captured=True, expected_attention_layers=1
        )


def test_finetune_and_optional_tensor_cli_dispatch_without_gpu(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(
        parity, "run_ltx_parity", lambda **kwargs: calls.append(kwargs) or {"status": "passed"}
    )
    assert (
        parity.main(
            [
                "--asset-root",
                str(tmp_path),
                "--output",
                str(tmp_path / "out"),
                "--device",
                "cuda",
                "--finetune",
                "full",
            ]
        )
        == 0
    )
    assert calls[-1]["finetune"] == "full" and calls[-1]["save_tensors"] is False
    assert (
        parity.main(
            [
                "--asset-root",
                str(tmp_path),
                "--output",
                str(tmp_path / "out-lora"),
                "--device",
                "cuda",
                "--finetune",
                "lora",
                "--save-tensors",
            ]
        )
        == 0
    )
    assert calls[-1]["finetune"] == "lora" and calls[-1]["save_tensors"] is True
    assert parity.main(["--asset-root", str(tmp_path), "--plan", "--finetune", "lora"]) == 0
    assert '"finetune": "lora"' in capsys.readouterr().out
