"""Flash retains full rollouts and normalizes actual selected training items."""
import copy
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from test_config import write_config
from test_hunyuan_runner import config_file as hunyuan_config
from test_wan_runner import config_file as wan_config

from visual_rl.backends import hunyuan, wan
from visual_rl.config import (
    ConfigError,
    Training,
    config_fingerprint,
    load_config,
    resume_signature,
)
from visual_rl.config import _read as read_yaml
from visual_rl.training.objectives import ARTObjective, prompt_disjoint_windows
from visual_rl.training.rollout import collect_rollout
from visual_rl.training.timesteps import art_training_positions
from visual_rl.training.update import accumulated_step


@pytest.mark.parametrize("recipe,parent", [
    ("sd3_art_flash", "sd3_pickscore"),
    ("flux_art_flash", "flux_pickscore"),
    ("hunyuan_art_flash", "hunyuan_art"),
    ("wan_art_flash", "wan_art"),
    ("minimax_h3_art_flash_full_8gpu", "minimax_h3_art_full_8gpu"),
    ("minimax_h3_art_flash_full_32gpu", "minimax_h3_art_full_32gpu"),
])
def test_flash_recipes_change_only_selection_and_output_identity(recipe, parent):
    root = Path(__file__).resolve().parents[1] / "configs/examples"
    original = read_yaml(root / f"{parent}.yaml")
    flash = read_yaml(root / f"{recipe}.yaml")
    assert flash["training"].pop("timestep_selection") == "first"
    flash["name"] = original["name"]
    flash["runtime"]["output_dir"] = original["runtime"]["output_dir"]
    assert flash == original


def test_flash_switch_preserves_default_checkpoint_identity(tmp_path):
    base = load_config(write_config(tmp_path))
    explicit = replace(base, training=replace(base.training, timestep_selection="configured"))
    assert explicit.to_dict() == base.to_dict()
    assert "timestep_selection" not in base.to_dict()["training"]
    assert config_fingerprint(base.to_dict()) == explicit.fingerprint()
    flash = replace(base, training=replace(base.training, timestep_selection="first"))
    assert flash.fingerprint() != base.fingerprint()
    assert resume_signature(flash.to_dict()) != resume_signature(base.to_dict())
    assert art_training_positions(flash.training, (3, 4, 5)) == (0,)
    assert art_training_positions(base.training, (3, 4, 5)) == (3, 4, 5)


@pytest.mark.parametrize("bad", ["random", "first2", True, None, 1])
def test_unknown_selection_is_rejected(bad):
    with pytest.raises(ConfigError, match="timestep_selection"):
        Training(timestep_selection=bad)


def test_flash_cannot_be_silently_ignored_by_another_algorithm(tmp_path):
    config = load_config(write_config(tmp_path))
    with pytest.raises(ConfigError, match="only by DiffusionART"):
        replace(config, algorithm=replace(config.algorithm, name="diffusionnft"),
                training=replace(config.training, timestep_selection="first"))


@pytest.mark.parametrize("backend,config_file", [(hunyuan, hunyuan_config), (wan, wan_config)])
def test_video_flash_preserves_noise_and_selects_initial_state(tmp_path, backend, config_file):
    base = load_config(config_file(tmp_path))
    flash = replace(base, training=replace(base.training, timestep_selection="first"))
    backend.validate_config(flash)
    original = backend.training_inputs(base, attempt=3, rank=1)
    actual = backend.training_inputs(flash, attempt=3, rank=1)
    assert torch.equal(original[0], actual[0])
    assert actual[1].shape == (len(actual[0]), 1)
    assert actual[1].eq(0).all()
    assert flash.sampling.steps == base.sampling.steps
    assert hunyuan.updates_per_rollout(flash) == hunyuan.updates_per_rollout(base) == 2


@pytest.mark.parametrize("microbatch", [1, 3, 8])
@pytest.mark.parametrize("masked", [False, True])
def test_flash_two_adam_updates_match_direct_selected_sample_mean(microbatch, masked):
    torch.manual_seed(47)
    model = torch.nn.Linear(2, 2)
    reference = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    control = torch.optim.AdamW(reference.parameters(), lr=.001)
    def predict(module):
        return lambda state, sigma, rows: module(state)
    sigmas = torch.tensor([1., .8, .5, .2, 0.])
    visited = []
    def rollout_predict(state, sigma, rows):
        visited.extend(sigma.tolist())
        return model(state)
    cache = collect_rollout(rollout_predict, torch.randn(8, 2), sigmas,
                            selected=art_training_positions(Training(timestep_selection="first"), (0, 1, 2)),
                            microbatch=8)
    assert cache.selected == (0,)
    assert set(round(t, 2) for t in visited) == {1., .8, .5, .2}
    args = (cache, torch.ones(8, 2), torch.ones(8), torch.zeros(8, dtype=torch.bool), torch.device("cpu"))
    def objective(module):
        return ARTObjective(args[0], predict(module), *args[1:], options={"corrective_mask_enabled": False})
    live, direct = objective(model), objective(reference)
    windows = prompt_disjoint_windows(torch.zeros(8, dtype=torch.long), len(cache.selected))
    assert [len(window) for window in windows] == [4, 4]
    def per_item(fn, items):
        losses = fn(items)
        if masked:
            losses = losses * losses.new_tensor([float(row % 3 != 0) for row, _ in items])
        return losses
    for items in windows:
        control.zero_grad(set_to_none=True)
        expected = per_item(direct, items).mean()
        expected.backward()
        norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 100.)
        control.step()
        result = accumulated_step(model, optimizer, items, lambda chunk: per_item(live, chunk),
            microbatch=microbatch, denominator=len(items), max_grad_norm=100.)
        assert result["train/loss"] == pytest.approx(expected.item(), abs=1e-6)
        assert result["train/grad_norm/before_clip"] == pytest.approx(float(norm), abs=2e-6)
        for actual, wanted in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, wanted, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("model", ["sd3", "flux"])
def test_native_image_flash_caches_first_position_and_keeps_full_rollout(tmp_path, monkeypatch, model):
    from test_image_ddp_runner import configured, parts_for_model

    from visual_rl.backends import sd3

    config = load_config(configured(tmp_path, "diffusionart", model=model))
    config = replace(config, training=replace(config.training, max_updates=2, timestep_selection="first"),
                     runtime=replace(config.runtime, distributed="none"),
                     evaluation=replace(config.evaluation, every_updates=0))
    monkeypatch.setattr(sd3, "load_components", parts_for_model)
    collect = sd3.collect_rollout
    observed = []
    def capture(predict, noise, sigmas, **kwargs):
        cache = collect(predict, noise, sigmas, **kwargs)
        observed.append((cache.selected, len(sigmas) - 1))
        return cache
    monkeypatch.setattr(sd3, "collect_rollout", capture)
    sd3.run(config)
    assert observed and all(selected == (0,) for selected, _ in observed)
    assert {count for _, count in observed} == {config.sampling.steps}
