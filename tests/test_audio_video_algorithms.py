"""AV science contracts: independent schedules/routes, real steps and replay."""
import copy

import pytest
import torch

from visual_rl.algorithms import clipped_grpo_loss, nft_loss
from visual_rl.training.audio_video import AVNFTState, prompt_disjoint_windows, run_av_updates
from visual_rl.training.audio_video_objectives import (
    prepare_art_objective,
    prepare_fast_objective,
    prepare_nft_objective,
    route_rewards,
    select_nft_fresh_times,
)
from visual_rl.training.audio_video_rollout import (
    collect_av_rollout,
    h3_cps_transition,
    select_fast_window,
)
from visual_rl.training.audio_video_types import AVState
from visual_rl.training.update import TerminalStepError


class TinyAV(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.video = torch.nn.Parameter(torch.tensor(.1))
        self.audio = torch.nn.Parameter(torch.tensor(.2))
        self.calls = []

    def forward(self, state, sv, sa, rows):
        self.calls.append((sv.detach().clone(), sa.detach().clone(), rows.detach().clone()))
        return AVState(self.video * state.video, self.audio * state.audio)


class CountingSGD(torch.optim.SGD):
    def __init__(self, model):
        super().__init__(model.parameters(), lr=.001)
        self.steps = 0

    def step(self, closure=None):
        self.steps += 1
        return super().step(closure)


def inputs(n=8):
    g = torch.Generator().manual_seed(35)
    noise = AVState(torch.randn(n, 2, 3, generator=g), torch.randn(n, 3, generator=g))
    return noise, torch.tensor([1., .8, .4, .1, 0.]), torch.tensor([1., .6, .2, .05, 0.]), torch.arange(n) // 4


def art(model, *, microbatch=3):
    noise, sv, sa, groups = inputs()
    cache = collect_av_rollout(model, noise, sv, sa, microbatch=microbatch)
    rewards = torch.tensor([0., 1., 2., 4., 1., 0., 3., 6.])
    objective = prepare_art_objective(cache, model, rewards, rewards.flip(0), groups, "cpu",
        loss_options={"corrective_mask_enabled": False})
    return cache, groups, objective


def nft(model, *, microbatch=3):
    endpoint, sv, _, groups = inputs()
    state = AVNFTState(model)
    g = torch.Generator().manual_seed(19)
    times = select_nft_fresh_times(sv, len(endpoint), positions=3, generator=g)
    noise = AVState(torch.randn(8, 3, 2, 3, generator=g), torch.randn(8, 3, 3, generator=g))
    def old_predict(*args):
        with state.use_old():
            return model(*args)
    rewards = torch.tensor([0., 1., 2., 4., 1., 0., 3., 6.])
    objective = prepare_nft_objective(endpoint, model, old_predict, rewards, rewards.flip(0), groups,
        times, noise, "cpu", microbatch=microbatch)
    return state, groups, objective


def fast(model):
    noise, sv, sa, groups = inputs()
    g = torch.Generator().manual_seed(4)
    selected = (0, 1, 3)  # Includes terminal to catch Gaussian variance guards.
    innovations = {i: AVState(torch.randn(noise.video.shape, generator=g), torch.randn(noise.audio.shape, generator=g)) for i in selected}
    cache = collect_av_rollout(model, noise, sv, sa, selected=selected, stochastic=selected,
        innovations=innovations, microbatch=3, unselected_kind="cps_zero_noise")
    rewards = torch.tensor([0., 1., 2., 4., 1., 0., 3., 6.])
    return cache, groups, prepare_fast_objective(cache, model, rewards, groups, "cpu")


def test_dual_schedules_identity_and_art_final_exclusion():
    model = TinyAV()
    noise, sv, sa, _ = inputs()
    cache = collect_av_rollout(model, noise, sv, sa, microbatch=3)
    expected_v, expected_a = noise.video.clone(), noise.audio.clone()
    for step in range(4):
        expected_v *= 1 + (sv[step + 1] - sv[step]) * model.video.detach()
        expected_a *= 1 + (sa[step + 1] - sa[step]) * model.audio.detach()
    torch.testing.assert_close(cache.endpoints.video, expected_v)
    torch.testing.assert_close(cache.endpoints.audio, expected_a)
    assert cache.selected == (0, 1, 2)
    assert any(not torch.equal(video, audio) for video, audio, _ in model.calls)
    assert [r.tolist() for _, _, r in model.calls[::4]] == [[0, 1, 2], [3, 4, 5], [6, 7]]
    assert not cache.states[0].video.requires_grad


def test_rollout_microbatch_invariance():
    m1, m2 = TinyAV(), TinyAV()
    one = art(m1, microbatch=1)[0]
    many = art(m2, microbatch=5)[0]
    torch.testing.assert_close(one.endpoints.video, many.endpoints.video)
    torch.testing.assert_close(one.endpoints.audio, many.endpoints.audio)
    for first, second in zip(one.old_velocities, many.old_velocities):
        torch.testing.assert_close(first.video, second.video)


def test_prompt_half_k_windows_even_when_rows_interleaved():
    groups = torch.tensor([9, 3, 9, 3, 9, 3, 9, 3])
    windows = prompt_disjoint_windows(groups, 3)
    assert set(windows[0]).isdisjoint(windows[1])
    assert len(set(windows[0] + windows[1])) == 24
    for window in windows:
        rows = sorted({row for row, _ in window})
        assert torch.bincount((groups[rows] == 9).long()).tolist() == [2, 2]
    with pytest.raises(ValueError, match="even K"):
        prompt_disjoint_windows(torch.tensor([0, 0, 0, 1, 1, 1]), 2)


def test_rewards_routed_and_desync_shared_only_once():
    components = {key: torch.zeros(2) for key in ("video_align", "hpsv3", "audiobox", "clap", "desync")}
    components["desync"][:] = 2
    routed = route_rewards(components)
    assert all(torch.equal(value, torch.full((2,), 2.)) for value in routed.values())
    components["hpsv3"][:] = 4
    routed = route_rewards(components)
    assert torch.equal(routed["audio"], torch.full((2,), 2.))
    assert torch.equal(routed["video"], torch.full((2,), 8.))


def test_art_routing_complete_group_frozen_across_u2():
    model = TinyAV()
    cache, groups, obj = art(model)
    base_v = obj.directions.video.clone()
    rewards = torch.arange(8).float()
    changed = prepare_art_objective(cache, model, rewards, rewards.square(), groups, "cpu")
    changed_audio = prepare_art_objective(cache, model, rewards, -rewards.square(), groups, "cpu")
    torch.testing.assert_close(changed.directions.video, changed_audio.directions.video)
    torch.testing.assert_close(changed.directions.audio, -changed_audio.directions.audio)
    optimizer = CountingSGD(model)
    metrics = run_av_updates(model, optimizer, obj, groups, algorithm="art", microbatch=5, max_grad_norm=10)
    assert optimizer.steps == metrics["train/optimizer_steps_per_rollout"] == 2
    torch.testing.assert_close(base_v, obj.directions.video)
    assert not obj.directions.video.requires_grad


def test_art_modality_losses_are_equal_weight_despite_feature_counts():
    model = TinyAV()
    _, _, obj = art(model)
    values = obj(obj.items[:5])
    assert values.shape == (5,)
    assert values.mean().item() == pytest.approx(.5 * (obj.diagnostics["av/video_loss"] + obj.diagnostics["av/audio_loss"]), rel=1e-6)


def test_nft_global_raw_population_std_not_group_or_local_std():
    model = TinyAV()
    endpoint, sv, _, groups = inputs()
    rewards = torch.tensor([0., 1., 2., 4., 1., 0., 3., 6.])
    population = torch.cat([rewards, rewards + 100])
    noise = AVState(torch.ones(8, 1, 2, 3), torch.ones(8, 1, 3))
    obj = prepare_nft_objective(endpoint, model, model, rewards, rewards, groups, sv[:1], noise,
        "cpu", global_video_rewards=population, global_audio_rewards=population)
    expected = (rewards.reshape(2, 4) - rewards.reshape(2, 4).mean(1, keepdim=True)).flatten() / (population.std(unbiased=False) + 1e-4)
    torch.testing.assert_close(obj.video_advantages, expected)
    torch.testing.assert_close(obj.audio_advantages, expected)


def test_nft_is_existing_normalized_x0_kernel():
    model = TinyAV()
    _, _, obj = nft(model)
    indices = torch.tensor([0, 5, 17])
    x, old, endpoint = [value.select(indices) for value in (obj.states, obj.old_predictions, obj.endpoints)]
    sigma = obj.video_sigmas[indices]
    pred = model(x, sigma, sigma, obj.rows[indices])
    expected = []
    for name, advantages in (("video", obj.video_advantages), ("audio", obj.audio_advantages)):
        expected.append(nft_loss(getattr(pred, name), getattr(old, name), getattr(endpoint, name), getattr(x, name),
            sigma, advantages[indices], nft_beta=1., adv_clip_max=5., eps=1e-5)["per_item_loss"])
    torch.testing.assert_close(obj(indices.tolist()), .5 * expected[0] + .5 * expected[1])


def test_nft_real_u1_and_one_slow_ema_refresh_checkpoint():
    model = TinyAV()
    state, groups, obj = nft(model)
    optimizer = CountingSGD(model)
    frozen = obj.old_predictions.video.clone()
    metrics = run_av_updates(model, optimizer, obj, groups, algorithm="nft", microbatch=5,
        max_grad_norm=10, state=state)
    assert optimizer.steps == metrics["train/optimizer_steps_per_rollout"] == 1
    assert state.successful_updates == 1
    assert metrics["nft/old_policy_decay"] == 0
    torch.testing.assert_close(obj.old_predictions.video, frozen)
    assert set(state.stateful()) == {"nft_old_policy", "av_nft_counters"}
    snapshot = {key: copy.deepcopy(value.state_dict()) for key, value in state.stateful().items()}
    model2 = TinyAV()
    model2.load_state_dict(model.state_dict())
    state2 = AVNFTState(model2)
    for key, value in state2.stateful().items():
        value.load_state_dict(snapshot[key])
    assert state2.successful_updates == 1
    assert state2.after_update()["nft/old_policy_decay"] == .00125


def test_nft_microbatch_denominator_invariant():
    models = [TinyAV(), TinyAV()]
    for model, microbatch in zip(models, [1, 7]):
        state, groups, obj = nft(model)
        run_av_updates(model, CountingSGD(model), obj, groups, algorithm="nft", microbatch=microbatch,
            max_grad_norm=10, state=state)
    for a, b in zip(models[0].parameters(), models[1].parameters()):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)


def test_nft_frozen_shared_fresh_times_and_noise():
    model = TinyAV()
    _, _, obj = nft(model)
    torch.testing.assert_close(obj.video_sigmas, obj.audio_sigmas)
    assert all(len(torch.unique(row)) == 3 for row in obj.video_sigmas.reshape(8, 3))
    states = obj.states.video.clone()
    noise_reconstructed = (obj.states.video - (1 - obj.video_sigmas[:, None, None]) * obj.endpoints.video) / obj.video_sigmas[:, None, None]
    assert not torch.equal(noise_reconstructed[0], noise_reconstructed[1])
    obj([0, 1, 2])
    obj([0, 1, 2])
    torch.testing.assert_close(states, obj.states.video)


def test_nft_nonfinite_failure_does_not_step_or_refresh():
    model = TinyAV()
    state, groups, obj = nft(model)
    def broken(state, sv, sa, rows):
        return AVState(state.video * model.video * float("nan"), state.audio * model.audio)
    obj.predict = broken
    optimizer = CountingSGD(model)
    with pytest.raises(TerminalStepError):
        run_av_updates(model, optimizer, obj, groups, algorithm="nft", microbatch=4,
            max_grad_norm=10, state=state)
    assert optimizer.steps == state.successful_updates == 0


@pytest.mark.parametrize("algorithm", ["art", "nft", "flow_grpo_fast"])
def test_zero_reward_only_skips_art_covariance_rule(algorithm):
    model = TinyAV()
    if algorithm == "nft":
        state, groups, obj = nft(model)
        obj.video_advantages.zero_()
        obj.audio_advantages.zero_()
    elif algorithm == "art":
        _, groups, obj = art(model)
        obj.video_magnitudes.zero_()
        obj.audio_magnitudes.zero_()
        state = None
    else:
        _, groups, obj = fast(model)
        obj.advantages.zero_()
        state = None
    optimizer = CountingSGD(model)
    result = run_av_updates(model, optimizer, obj, groups, algorithm=algorithm, microbatch=3,
        max_grad_norm=10, state=state)
    expected = {"art": 0, "nft": 1, "flow_grpo_fast": 2}[algorithm]
    assert result["train/optimizer_steps_per_rollout"] == optimizer.steps == expected
    assert state is None or state.successful_updates == expected


def test_fast_replay_initial_ratio_one_including_terminal():
    model = TinyAV()
    cache, _, obj = fast(model)
    obj(obj.items)
    assert obj.diagnostics["grpo/ratio"] == pytest.approx(1.)
    assert obj.diagnostics["grpo/clip_fraction"] == 0
    assert cache.selected[-1] == len(cache.video_sigmas) - 2


def test_h3_cps_is_negative_mse_surrogate_with_valid_feature_mean():
    x, v = torch.tensor([[1., 2., 100.]]), torch.tensor([[.2, .7, 99.]])
    innovation = torch.ones_like(x)
    mask = torch.tensor([[True, True, False]])
    sampled = h3_cps_transition(x, v, .8, .2, innovation=innovation, mask=mask)
    replay = h3_cps_transition(x, v + .1, .8, .2, observed=sampled.sample, mask=mask)
    expected = -(sampled.sample[:, :2] - replay.mean[:, :2]).square().mean(1)
    torch.testing.assert_close(replay.log_prob, expected)
    terminal = h3_cps_transition(x, v, .1, 0., innovation=innovation)
    assert terminal.std.item() == 0
    torch.testing.assert_close(terminal.log_prob, torch.zeros(1))
    assert torch.isfinite(terminal.log_prob).all()


def test_fast_one_combined_ratio_and_real_u2():
    model = TinyAV()
    cache, groups, obj = fast(model)
    optimizer = CountingSGD(model)
    result = run_av_updates(model, optimizer, obj, groups, algorithm="flow_grpo_fast", microbatch=5,
        max_grad_norm=10)
    assert optimizer.steps == result["train/optimizer_steps_per_rollout"] == 2
    assert "grpo/ratio" in result


def test_fast_window_boundaries_seed_and_terminal():
    for index in range(40):
        selected = select_fast_window(8, rollout_index=index)
        assert len(selected) == 3
        assert tuple(range(selected[0], selected[0] + 3)) == selected
        assert min(selected) >= 0 and max(selected) < 8
    assert any(7 in select_fast_window(8, rollout_index=i) for i in range(40))
    assert select_fast_window(8, rollout_index=2) == select_fast_window(8, rollout_index=2)
    with pytest.raises(ValueError):
        select_fast_window(2)


def test_art_valid_feature_mask_preserves_loss_and_gradient():
    model = TinyAV()
    _, groups, obj = art(model)
    original = obj(obj.items)
    noise, sv, sa, _ = inputs()
    padded_noise = AVState(torch.cat([noise.video, torch.full((8, 2, 4), 999.)], dim=-1),
        torch.cat([noise.audio, torch.full((8, 10), 999.)], dim=-1),
        torch.tensor([[[True] * 3 + [False] * 4] * 2] * 8),
        torch.tensor([[True] * 3 + [False] * 10] * 8))
    cache = collect_av_rollout(model, padded_noise, sv, sa, microbatch=3)
    rewards = torch.tensor([0., 1., 2., 4., 1., 0., 3., 6.])
    padded = prepare_art_objective(cache, model, rewards, rewards.flip(0), groups, "cpu",
        loss_options={"corrective_mask_enabled": False})
    torch.testing.assert_close(padded(padded.items), original)
    original.sum().backward()
    gradient = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    padded(padded.items).sum().backward()
    for p, expected in zip(model.parameters(), gradient):
        torch.testing.assert_close(p.grad, expected)


def test_fast_combines_logp_before_single_clipping_surrogate():
    model = TinyAV()
    cache, _, obj = fast(model)
    with torch.no_grad():
        model.video.add_(1.7)
        model.audio.sub_(.9)
    from visual_rl.training.audio_video_rollout import gather_slots
    pairs = torch.tensor(obj.items)
    rows, slots = pairs.unbind(1)
    steps = torch.tensor(cache.selected)[slots]
    state = gather_slots(cache.states, rows, slots)
    observed = gather_slots(cache.next_states, rows, slots)
    pred = model(state, cache.video_sigmas[steps], cache.audio_sigmas[steps], rows)
    current, old = [], []
    old_velocity = gather_slots(cache.old_velocities, rows, slots)
    for name, schedule in (("video", cache.video_sigmas), ("audio", cache.audio_sigmas)):
        current.append(h3_cps_transition(getattr(state, name), getattr(pred, name), schedule[steps], schedule[steps + 1],
            observed=getattr(observed, name)).log_prob)
        old.append(h3_cps_transition(getattr(state, name), getattr(old_velocity, name), schedule[steps], schedule[steps + 1],
            observed=getattr(observed, name)).log_prob)
    advantage = obj.advantages[rows]
    expected = clipped_grpo_loss(.5 * current[0] + .5 * current[1], .5 * old[0] + .5 * old[1],
        advantage, ratio_clip=1e-4)["per_item_loss"]
    independent = .5 * sum(clipped_grpo_loss(c, o, advantage, ratio_clip=1e-4)["per_item_loss"]
                          for c, o in zip(current, old))
    torch.testing.assert_close(obj(obj.items), expected)
    assert not torch.allclose(expected, independent, atol=1e-6, rtol=1e-6)


def test_cps_unselected_step_uses_expanded_zero_noise_arithmetic(monkeypatch):
    import visual_rl.training.audio_video_rollout as rollout_module
    original = rollout_module.flow_transition
    kinds = []
    def traced(*args, **kwargs):
        kinds.append(kwargs.get("kind"))
        return original(*args, **kwargs)
    monkeypatch.setattr(rollout_module, "flow_transition", traced)
    model = TinyAV()
    cache, _, _ = fast(model)
    # Position2 is unselected and must never call Euler ODE. Selected position3
    # is terminal and follows H3 CPS's explicitly deterministic MSE surrogate.
    assert "ode" not in kinds
    assert kinds.count("cps_zero_noise") == 6  # 3 microbatches x 2 modalities.
    x = cache.next_states[1]
    sv, sa = cache.video_sigmas, cache.audio_sigmas
    for name, schedule in (("video", sv), ("audio", sa)):
        state = getattr(x, name)
        velocity = state * getattr(model, name).detach()
        sigma, nxt = schedule[2], schedule[3]
        expected = (state - sigma * velocity) * (1 - nxt) + (state + (1 - sigma) * velocity) * nxt
        torch.testing.assert_close(getattr(cache.states[2], name), expected)


def test_nft_fresh_times_floor_source_millisecond_quantization():
    sigmas = torch.tensor([1., .9189189, .5714286, .20123456, 0.])
    selected = select_nft_fresh_times(sigmas, 3, generator=torch.Generator().manual_seed(42))
    expected = torch.tensor([1., .918, .571, .201]).sort().values
    for row in selected:
        torch.testing.assert_close(row.sort().values, expected)
    with pytest.raises(ValueError, match="pool"):
        select_nft_fresh_times(torch.tensor([1., .0001, 0.]), 2, generator=torch.Generator())


def test_diagnostic_means_keep_original_item_weight_and_ignore_replays():
    model = TinyAV()
    _, _, obj = art(model)
    # Different microbatch sizes and repeated items must not change telemetry.
    loss = obj(obj.items)
    expected = dict(obj.diagnostics)
    obj._diagnostic_items = {}
    obj.diagnostics.clear()
    for start in range(0, len(obj.items), 5):
        obj(obj.items[start:start + 5])
    obj(obj.items[:1])
    assert obj.diagnostics == pytest.approx(expected)
    assert loss.mean().item() == pytest.approx(.5 * (obj.diagnostics["av/video_loss"] + obj.diagnostics["av/audio_loss"]), rel=1e-6)


def test_nft_padding_is_excluded_from_normalized_residual_and_feature_mean():
    model = TinyAV()
    _, _, obj = nft(model)
    reference = obj(obj.items)
    reference.sum().backward()
    expected_grad = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    def pad(value):
        vm = torch.tensor([[[True] * 3 + [False] * 4] * 2] * len(value))
        am = torch.tensor([[True] * 3 + [False] * 5] * len(value))
        return AVState(torch.cat([value.video, torch.full((len(value), 2, 4), 37.)], -1),
                       torch.cat([value.audio, torch.full((len(value), 5), -45.)], -1), vm, am)
    obj.endpoints, obj.states, obj.old_predictions = [pad(value) for value in (obj.endpoints, obj.states, obj.old_predictions)]
    actual = obj(obj.items)
    torch.testing.assert_close(actual, reference)
    actual.sum().backward()
    for p, expected in zip(model.parameters(), expected_grad):
        torch.testing.assert_close(p.grad, expected)


@pytest.mark.parametrize("algorithm", ["art", "nft", "flow_grpo_fast"])
def test_owner_telemetry_invariant_to_ragged_microbatch(algorithm):
    results = []
    for microbatch in (1, 7):
        model = TinyAV()
        if algorithm == "nft":
            state, groups, obj = nft(model)
        elif algorithm == "art":
            _, groups, obj = art(model)
            state = None
        else:
            _, groups, obj = fast(model)
            state = None
        results.append(run_av_updates(model, CountingSGD(model), obj, groups, algorithm=algorithm,
            microbatch=microbatch, max_grad_norm=10, state=state))
    names = [key for key in results[0] if key.startswith(("av/video_", "av/audio_", "grpo/"))
             or (key.startswith("av/window_") and any(word in key for word in ("video_", "audio_", "grpo/")))]
    for name in names:
        assert results[0][name] == pytest.approx(results[1][name], rel=1e-5, abs=1e-7)
    if algorithm != "flow_grpo_fast":
        for result in results:
            assert result["train/loss"] == pytest.approx(.5 * (result["av/video_loss"] + result["av/audio_loss"]), rel=1e-6)


def test_nft_checkpoint_resume_preserves_optimizer_old_counter_and_next_update(tmp_path):
    from visual_rl.training.checkpoint import load_checkpoint, save_checkpoint
    from visual_rl.training.state import Progress

    def make_objective(model, state, generator):
        endpoint, sv, _, groups = inputs()
        times = select_nft_fresh_times(sv, 8, positions=2, generator=generator)
        noise = AVState(torch.randn(8, 2, 2, 3, generator=generator), torch.randn(8, 2, 3, generator=generator))
        rewards = torch.tensor([0., 1., 2., 4., 1., 0., 3., 6.])
        def old_predict(*args):
            with state.use_old():
                return model(*args)
        obj = prepare_nft_objective(endpoint, model, old_predict, rewards, rewards.flip(0), groups,
            times, noise, "cpu", microbatch=3)
        return groups, obj

    model = TinyAV()
    state = AVNFTState(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    generator = torch.Generator().manual_seed(54)
    groups, obj = make_objective(model, state, generator)
    run_av_updates(model, optimizer, obj, groups, algorithm="nft", microbatch=5, max_grad_norm=10, state=state)
    receipt = save_checkpoint(tmp_path, model=model, optimizer=optimizer,
        progress=Progress(attempts=1, completed_rollouts=1, optimizer_steps=1), config_sha256="a" * 64,
        stateful=state.stateful(), generators={"av_sampling": generator})
    groups, obj = make_objective(model, state, generator)
    reference = run_av_updates(model, optimizer, obj, groups, algorithm="nft", microbatch=5, max_grad_norm=10, state=state)

    restored = TinyAV()
    restored_state = AVNFTState(restored)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=.001)
    restored_generator = torch.Generator().manual_seed(1)
    progress = load_checkpoint(receipt, model=restored, optimizer=restored_optimizer, config_sha256="a" * 64,
        stateful=restored_state.stateful(), generators={"av_sampling": restored_generator})
    assert progress.optimizer_steps == restored_state.successful_updates == 1
    groups, obj = make_objective(restored, restored_state, restored_generator)
    result = run_av_updates(restored, restored_optimizer, obj, groups, algorithm="nft", microbatch=5,
        max_grad_norm=10, state=restored_state)
    assert result["nft/old_policy_decay"] == reference["nft/old_policy_decay"] == .00125
    assert result["train/loss"] == pytest.approx(reference["train/loss"], abs=0, rel=0)
    assert restored_state.successful_updates == 2
    for actual, expected in zip(restored.parameters(), model.parameters()):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for key, expected in state.old.shadow.items():
        torch.testing.assert_close(restored_state.old.shadow[key], expected, atol=0, rtol=0)


def test_advantage_zero_nft_keeps_ema_restoration_gradient_and_updates():
    model = TinyAV()
    state, groups, objective = nft(model)
    objective.video_advantages.zero_()
    objective.audio_advantages.zero_()
    with torch.no_grad():
        model.video.add_(.7)
        model.audio.sub_(.4)
    loss = objective(objective.items).mean()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().item() > 0 for p in model.parameters())
    model.zero_grad(set_to_none=True)
    optimizer = CountingSGD(model)
    metrics = run_av_updates(model, optimizer, objective, groups, algorithm="nft", microbatch=5,
        max_grad_norm=10, state=state)
    assert optimizer.steps == state.successful_updates == metrics["train/optimizer_steps_per_rollout"] == 1


def test_art_window0_cache_drift_snapshot_is_preserved_separately():
    from visual_rl.training.audio_video_rollout import feature_mean, gather_slots
    model = TinyAV()
    cache, groups, objective = art(model)
    with torch.no_grad():
        model.video.add_(.03)
        model.audio.sub_(.02)
    first = prompt_disjoint_windows(groups, len(cache.selected))[0]
    pair = torch.tensor(first)
    rows, slots = pair.unbind(1)
    states = gather_slots(cache.states, rows, slots)
    old = gather_slots(cache.old_velocities, rows, slots)
    current = model(states, torch.ones(len(first)), torch.ones(len(first)), rows)
    expected = {name: float(feature_mean((getattr(current, name).detach() - getattr(old, name)).square()).sqrt().mean())
                for name in ("video", "audio")}
    result = run_av_updates(model, CountingSGD(model), objective, groups, algorithm="art", microbatch=5,
                            max_grad_norm=10)
    for name in ("video", "audio"):
        first_rms = result[f"av/window_0/{name}_current_old_rms"]
        second_rms = result[f"av/window_1/{name}_current_old_rms"]
        assert first_rms == pytest.approx(expected[name], rel=1e-6)
        assert first_rms > 0
        assert result[f"av/{name}_current_old_rms"] == pytest.approx(.5 * (first_rms + second_rms), rel=1e-6)
        assert f"av/window_0/{name}_mask_fraction" in result
        assert f"av/window_1/{name}_mask_fraction" in result


def test_nft_drift_and_advantage_population_std_come_from_frozen_items():
    from visual_rl.training.audio_video_rollout import feature_mean
    model = TinyAV()
    state, groups, objective = nft(model)
    with torch.no_grad():
        model.video.add_(.03)
        model.audio.sub_(.02)
    prediction = model(objective.states, objective.video_sigmas, objective.audio_sigmas, objective.rows)
    expected_rms = {name: float(feature_mean((getattr(prediction, name).detach() - getattr(objective.old_predictions, name)).square()).sqrt().mean())
                    for name in ("video", "audio")}
    result = run_av_updates(model, CountingSGD(model), objective, groups, algorithm="nft", microbatch=7,
                            max_grad_norm=10, state=state)
    for name, advantage in (("video", objective.video_advantages), ("audio", objective.audio_advantages)):
        assert result[f"av/{name}_current_old_rms"] == pytest.approx(expected_rms[name], rel=1e-6)
        assert result[f"av/{name}_advantage_std"] == pytest.approx(float(advantage.std(unbiased=False)), rel=1e-6)
        assert result[f"av/{name}_advantage_mean"] == pytest.approx(float(advantage.mean()), abs=1e-7)
        assert result[f"av/window_0/{name}_advantage_std"] == result[f"av/{name}_advantage_std"]
