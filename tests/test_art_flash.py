"""Flash retains full rollouts and normalizes actual selected training items."""
import copy
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
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
    assert flash["training"].pop("timestep_selection") == ("first" if recipe == "wan_art_flash" else "high_noise")
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
    high_noise = replace(base, training=replace(base.training, timestep_selection="high_noise"))
    assert high_noise.fingerprint() not in (base.fingerprint(), flash.fingerprint())
    assert high_noise.to_dict()["training"]["timestep_selection"] == "high_noise"
    assert resume_signature(high_noise.to_dict()) != resume_signature(base.to_dict())


@pytest.mark.parametrize("bad", ["random", "first2", True, None, 1])
def test_unknown_selection_is_rejected(bad):
    with pytest.raises(ConfigError, match="timestep_selection"):
        Training(timestep_selection=bad)


@pytest.mark.parametrize("selection", ["first", "high_noise"])
def test_flash_cannot_be_silently_ignored_by_another_algorithm(tmp_path, selection):
    config = load_config(write_config(tmp_path))
    with pytest.raises(ConfigError, match="only by DiffusionART"):
        replace(config, algorithm=replace(config.algorithm, name="diffusionnft"),
                training=replace(config.training, timestep_selection=selection))


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
@pytest.mark.parametrize("selection,positions", [("first", (0,)), ("high_noise", (2, 3))])
def test_flash_two_adam_updates_match_direct_selected_sample_mean(microbatch, masked, selection, positions):
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
                            selected=art_training_positions(Training(timestep_selection=selection), (0, 1, 2),
                                                            seed=42, attempt=1),
                            microbatch=8)
    assert cache.selected == positions
    assert set(round(t, 2) for t in visited) == {1., .8, .5, .2}
    args = (cache, torch.ones(8, 2), torch.ones(8), torch.zeros(8, dtype=torch.bool), torch.device("cpu"))
    def objective(module):
        return ARTObjective(args[0], predict(module), *args[1:], options={"corrective_mask_enabled": False})
    live, direct = objective(model), objective(reference)
    windows = prompt_disjoint_windows(torch.zeros(8, dtype=torch.long), len(cache.selected))
    assert [len(window) for window in windows] == [4 * len(positions)] * 2
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
@pytest.mark.parametrize("selection", ["first", "high_noise"])
def test_native_image_flash_caches_selected_positions_and_keeps_full_rollout(tmp_path, monkeypatch, model, selection):
    from test_image_ddp_runner import configured, parts_for_model

    from visual_rl.backends import sd3

    config = load_config(configured(tmp_path, "diffusionart", model=model))
    config = replace(config, sampling=replace(config.sampling, steps=4),
                     training=replace(config.training, max_updates=2, timestep_selection=selection),
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
    expected = art_training_positions(config.training, (), seed=config.seed, attempt=1)
    assert observed and all(selected == expected for selected, _ in observed)
    assert {count for _, count in observed} == {config.sampling.steps}


def test_high_noise_pair_is_global_stateless_and_does_not_consume_rng():
    training = Training(timestep_selection="high_noise")
    random.seed(193)
    np.random.seed(194)
    torch.manual_seed(195)
    rollout_rng = torch.Generator().manual_seed(196)
    states = (random.getstate(), np.random.get_state(), torch.random.get_rng_state(), rollout_rng.get_state())
    pairs = [art_training_positions(training, (8, 9), seed=42, attempt=attempt)
             for attempt in range(1, 101)]
    assert pairs[0] == (2, 3)  # Stable domain/version and counter convention.
    assert set(pairs) == {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}
    assert all(len(pair) == 2 and tuple(sorted(set(pair))) == pair for pair in pairs)
    # A retry or reconstructed process uses only persisted global seed/attempt.
    assert pairs == [art_training_positions(training, (), seed=42, attempt=attempt)
                     for attempt in range(1, 101)]
    assert pairs != [art_training_positions(training, (), seed=43, attempt=attempt)
                     for attempt in range(1, 101)]
    assert states[0] == random.getstate()
    np_state = np.random.get_state()
    assert states[1][0] == np_state[0] and np.array_equal(states[1][1], np_state[1])
    assert states[1][2:] == np_state[2:]
    assert torch.equal(states[2], torch.random.get_rng_state())
    assert torch.equal(states[3], rollout_rng.get_state())


@pytest.mark.parametrize("seed,attempt", [(None, 1), (42, None), (True, 1), (42, 0), (42, -1)])
def test_high_noise_requires_explicit_global_counter(seed, attempt):
    with pytest.raises(ConfigError, match="one-based rollout attempt"):
        art_training_positions(Training(timestep_selection="high_noise"), (), seed=seed, attempt=attempt)


@pytest.mark.parametrize("model,minimum", [("sd3", 4), ("flux", 4), ("hunyuan_public", 5), ("minimax_h3", 6)])
def test_high_noise_rejects_short_schedules_before_loading(tmp_path, model, minimum):
    base = load_config(write_config(tmp_path))
    training = replace(base.training, timestep_selection="high_noise")
    with pytest.raises(ConfigError, match="high_noise requires"):
        replace(base, model=replace(base.model, name=model), training=training,
                sampling=replace(base.sampling, steps=minimum - 1))
    valid = replace(base, model=replace(base.model, name=model), training=training,
                    sampling=replace(base.sampling, steps=minimum))
    assert valid.sampling.steps == minimum


def test_video_high_noise_global_pair_and_native_cache_preserve_complete_generation(tmp_path):
    from types import SimpleNamespace

    from visual_rl.sampling import shifted_schedule

    backend = hunyuan
    base = load_config(hunyuan_config(tmp_path))
    base = replace(base, sampling=replace(base.sampling, steps=5))
    flash = replace(base, training=replace(base.training, timestep_selection="high_noise"))
    backend.validate_config(flash)
    parts = SimpleNamespace(adapter=SimpleNamespace(transformer=torch.nn.Linear(1, 1)))
    sigmas = shifted_schedule(flash.sampling.steps, flash.sampling.options["shift"])
    for attempt in (1, 3, 7):
        pair = art_training_positions(flash.training, (), seed=flash.seed, attempt=attempt)
        for rank in (0, 1):
            original = backend.training_inputs(base, attempt=attempt, rank=rank)
            actual = backend.training_inputs(flash, attempt=attempt, rank=rank)
            assert torch.equal(original[0], actual[0])
            assert actual[1].tolist() == [list(pair)] * len(actual[0])
            visits = []
            def predict(state, sigma, rows):
                visits.extend(sigma.tolist())
                return state.float() * .1
            reference = backend.collect_training(base, parts, predict, original, sigmas, torch.device("cpu"))
            visits.clear()
            cache = backend.collect_training(flash, parts, predict, actual, sigmas, torch.device("cpu"))
            assert cache.selected == pair
            assert cache.states.shape[1] == 2
            assert torch.equal(cache.endpoints, reference.endpoints)
            assert set(visits) == set(sigmas[:-1].tolist())
            assert len(prompt_disjoint_windows(torch.zeros(len(cache.endpoints), dtype=torch.long),
                                              len(cache.selected))) == 2


def test_tensor_fixture_high_noise_uses_attempt_pair_and_two_windows(tmp_path, monkeypatch):
    from visual_rl.backends import tensor_fixture

    config = load_config("configs/examples/cpu_smoke.yaml", {
        "training": {"timestep_selection": "high_noise", "max_updates": 4},
        "runtime": {"output_dir": str(tmp_path / "run")},
        "sampling": {"steps": 5}})
    original = tensor_fixture.collect_rollout
    observed = []
    def collect(*args, **kwargs):
        cache = original(*args, **kwargs)
        observed.append(cache.selected)
        return cache
    monkeypatch.setattr(tensor_fixture, "collect_rollout", collect)
    tensor_fixture.run(config)
    assert observed == [art_training_positions(config.training, (), seed=config.seed, attempt=attempt)
                        for attempt in (1, 2)]


def test_wan_rejects_high_noise_before_component_load(tmp_path, monkeypatch):
    base = load_config(wan_config(tmp_path))
    flash = replace(base, training=replace(base.training, timestep_selection="high_noise"))
    def forbidden(*args, **kwargs):
        pytest.fail("unsupported Wan selection reached component loading")
    monkeypatch.setattr(wan, "load_components", forbidden)
    with pytest.raises(ConfigError, match="only three nonfinal ART positions"):
        wan.run(flash)


def test_high_noise_two_rank_hunyuan_fsdp_and_exact_resume(tmp_path, monkeypatch):
    import test_hunyuan_fsdp
    import yaml

    original = test_hunyuan_fsdp.config_file
    def high_noise(*args, **kwargs):
        path = original(*args, **kwargs)
        content = read_yaml(path)
        content["sampling"]["steps"] = 5
        content["training"]["timestep_selection"] = "high_noise"
        path.write_text(yaml.safe_dump(content))
        return path
    monkeypatch.setattr(test_hunyuan_fsdp, "config_file", high_noise)
    test_hunyuan_fsdp.test_native_hunyuan_fsdp_eval_and_resume(tmp_path, "diffusionart")
