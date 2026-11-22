import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from visual_rl.backends import hunyuan
from visual_rl.config import load_config
from visual_rl.prepare_assets import plan_environment, validate_bindings
from visual_rl.registry import get_backend
from visual_rl.sampling import shifted_schedule


def recipe(name, root):
    path = Path(__file__).resolve().parents[1] / "configs/examples" / f"hunyuan_{name}_lora_8gpu.yaml"
    with patch.dict(os.environ, {"DANCEGRPO_ROOT": str(root), "VISUAL_RL_ASSET_ROOT": str(root)}):
        plan = plan_environment(path)
        config = load_config(path)
        validate_bindings(config, plan)
        get_backend(config)
    return config


@pytest.mark.parametrize("name,updates,pool,count", [
    ("art", 2, range(10), 10),
    ("nft", 1, range(10), 10),
    ("fast", 2, range(6), 2),
])
def test_published_recipe(name, updates, pool, count, tmp_path):
    c = recipe(name, tmp_path)
    assert c.runtime.distributed == "fsdp"
    assert (c.sampling.prompts_per_rollout, c.sampling.samples_per_prompt) == (8, 8)
    assert (c.sampling.microbatch, c.training.microbatch) == (1, 1)
    assert (c.sampling.width, c.sampling.height, c.sampling.frames) == (1280, 720, 5)
    assert c.sampling.steps == 10 and c.sampling.options["shift"] == 5
    assert c.sampling.options["train_timestep_count"] == count
    assert c.sampling.options["train_timestep_pool"] == list(pool)
    assert c.model.options["finetune"]["rank"] == 64
    assert c.model.options["finetune"]["alpha"] == 256
    assert c.model.options["continuous_timesteps"] and c.model.options["embedded_guidance_raw"] == 1000
    assert c.reward.name == "pickscore" and c.reward.normalization == "fixed"
    assert hunyuan.updates_per_rollout(c) == updates
    assert c.training.max_updates == 100 * updates
    assert c.runtime.checkpoint_every == 25 * updates
    assert not c.evaluation.enabled


def test_art_native_trajectory_and_disjoint_updates(tmp_path):
    c = recipe("art", tmp_path)
    model = torch.nn.Linear(1, 1)
    parts = SimpleNamespace(adapter=SimpleNamespace(transformer=model))
    sigmas = shifted_schedule(10, 5)
    noise = torch.arange(8, dtype=torch.bfloat16)[:, None] + 1
    selected = torch.arange(10).repeat(8, 1)
    def predict(state, times, rows):
        return state * 0.375

    cache = hunyuan.collect_training(c, parts, predict, (noise, selected, None), sigmas, "cpu")
    state = noise
    for step in range(10):
        velocity = state * 0.375
        torch.testing.assert_close(cache.old_velocities[:, step], velocity.float(), rtol=0, atol=0)
        state = (state.float() + (sigmas[step + 1] - sigmas[step]) * velocity.float()).bfloat16()
    torch.testing.assert_close(cache.endpoints, state.float(), rtol=0, atol=0)
    scores = SimpleNamespace(weighted=torch.arange(8).float())

    def prepare(attempt):
        return hunyuan.prepare_objective(c, parts, cache, predict, scores, scores, "cpu", attempt=attempt)

    objective, windows = prepare(1)
    assert objective.has_guidance and not objective.reconstruct_old_from_mean
    assert windows == prepare(1)[1] and windows != prepare(2)[1]
    assert set(windows[0]).isdisjoint(windows[1])
    assert all(len({row for row, slot in window}) == 4 for window in windows)
    assert sorted(windows[0] + windows[1]) == [(row, slot) for row in range(8) for slot in range(10)]
    assert torch.isfinite(objective([(0, 9)])[0]).all()
