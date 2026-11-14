"""Profile guardrails, true batched conditioning and native noise layout."""

import sys
import types
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from visual_rl.backends import small_video as backend
from visual_rl.config import ConfigError, load_config
from visual_rl.models.ltx_video_sampling import unpack_ltx_latents

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(
    params=[
        "wan_t2v_1_3b_lora",
        "wan_t2v_1_3b_full",
        "ltx_video_2b_distilled_lora",
        "ltx_video_2b_distilled_full",
    ]
)
def config(request, monkeypatch, tmp_path):
    monkeypatch.setenv("SMALL_VIDEO_REWARD_PYTHON", sys.executable)
    monkeypatch.setenv("SMALL_VIDEO_ASSETS", str(tmp_path / "assets"))
    monkeypatch.setenv("SMALL_VIDEO_OUTPUT", str(tmp_path / "output"))
    return load_config(ROOT / "configs/small_video" / (request.param + ".yaml"))


def test_native_original_flash_only_change_selection_and_identity(config, monkeypatch):
    backend.validate_config(config)
    flash = load_config(
        ROOT / "configs/small_video" / (config.name.removesuffix("_original") + "_flash.yaml")
    )
    backend.validate_config(flash)
    one, two = config.to_dict(), flash.to_dict()
    one.pop("name")
    two.pop("name")
    one["training"].pop("timestep_selection", None)
    two["training"].pop("timestep_selection", None)
    assert one == two
    protocol = backend.resolved_protocol(config)
    assert len(protocol["eligible_positions"]) == config.sampling.steps
    assert protocol["sigmas"][-1] == 0 and protocol["cumulative_actual_optimizer_step_limit"] == 10
    assert protocol["flash_pool_sigmas"][0] > 0.99


@pytest.mark.parametrize(
    "mutation", ["steps", "frames", "K", "learning_rate", "reward", "budget", "evaluation"]
)
def test_scientific_contract_changes_cannot_silently_pass(config, mutation):
    if mutation in ("steps", "frames"):
        c = replace(
            config,
            sampling=replace(config.sampling, **{mutation: getattr(config.sampling, mutation) - 1}),
        )
    elif mutation == "K":
        c = replace(config, sampling=replace(config.sampling, samples_per_prompt=4))
    elif mutation == "learning_rate":
        c = replace(config, training=replace(config.training, learning_rate=0.01))
    elif mutation == "reward":
        c = replace(
            config, reward=replace(config.reward, weights={"VQ": 1.0, "MQ": 0.0, "TA": 0.0})
        )
    elif mutation == "budget":
        c = replace(config, training=replace(config.training, max_updates=12))
    else:
        c = replace(config, evaluation=replace(config.evaluation, every_updates=2))
    with pytest.raises(ConfigError):
        backend.validate_config(c)


def test_pending_science_fails_before_gpu_or_optional_loader(config, monkeypatch):
    monkeypatch.setattr(
        torch.cuda, "is_available", lambda: pytest.fail("CUDA touched before science gate")
    )
    with pytest.raises(ConfigError, match="scientific decision"):
        backend.run(config)


def test_ltx_initial_noise_is_native_packed_draw_not_raw_layout(monkeypatch, tmp_path):
    monkeypatch.setenv("SMALL_VIDEO_REWARD_PYTHON", sys.executable)
    monkeypatch.setenv("SMALL_VIDEO_ASSETS", str(tmp_path))
    monkeypatch.setenv("SMALL_VIDEO_OUTPUT", str(tmp_path / "output"))
    c = load_config(ROOT / "configs/small_video/ltx_video_2b_distilled_lora.yaml")
    monkeypatch.setattr(backend, "latent_shape", lambda c, n: (n, 2, 3, 2, 1))
    before = torch.get_rng_state()
    actual = backend.initial_noise(c, 2, torch.device("cpu"), 42)
    expected = unpack_ltx_latents(
        torch.randn((2, 6, 2), dtype=torch.bfloat16, generator=torch.Generator().manual_seed(42)),
        (2, 2, 3, 2, 1),
    )
    assert torch.equal(actual, expected) and torch.equal(before, torch.get_rng_state())
    raw = torch.randn(
        (2, 2, 3, 2, 1), dtype=torch.bfloat16, generator=torch.Generator().manual_seed(42)
    )
    assert not torch.equal(actual, raw)


def test_wan_conditioning_preserves_lengths_negative_branch_and_integer_native_times(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("SMALL_VIDEO_REWARD_PYTHON", sys.executable)
    monkeypatch.setenv("SMALL_VIDEO_ASSETS", str(tmp_path))
    monkeypatch.setenv("SMALL_VIDEO_OUTPUT", str(tmp_path / "output"))
    c = load_config(ROOT / "configs/small_video/wan_t2v_1_3b_lora.yaml")
    cfg = types.ModuleType("wan.configs.wan_t2v_1_3B")
    cfg.t2v_1_3B = SimpleNamespace(sample_neg_prompt="native negative")
    monkeypatch.setitem(sys.modules, "wan.configs.wan_t2v_1_3B", cfg)

    class Encoder:
        model = torch.nn.Linear(1, 1)

        def __call__(self, prompts, device):
            assert prompts == ["short", "a longer prompt", "native negative"]
            return [torch.full((n, 4), float(n)) for n in (2, 5, 3)]

    class Adapter:
        def predict_velocity(self, state, sigma, condition):
            assert condition.prompt_embeds.shape == (4, 512, 4)
            assert condition.prompt_attention_mask.sum(1).tolist() == [2, 2, 5, 5]
            assert condition.negative_prompt_attention_mask.sum(1).tolist() == [3] * 4
            assert (
                condition.timesteps.dtype == torch.int64
                and condition.timesteps.tolist() == [999] * 4
            )
            return state * 0

    parts = SimpleNamespace(text_encoder=Encoder(), adapter=Adapter())
    predict = backend.make_predict(c, parts, ["short", "a longer prompt"], 2, torch.device("cpu"))
    result = predict(torch.ones(4, 1, 1, 1, 1), torch.ones(4), torch.arange(4), 0)
    assert result.shape[0] == 4
