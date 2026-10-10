import math
from dataclasses import fields

import pytest
import torch

from visual_rl.training import grpo_sampling, rollout
from visual_rl.training.grpo_sampling import collect_grpo_rollout
from visual_rl.training.rollout import replay_transition


def _predict(state, sigma, rows):
    shape = (-1,) + (1,) * (state.ndim - 1)
    return state.sin() * 0.13 + sigma.reshape(shape) * 0.21 + rows.reshape(shape) * 0.037


def _arguments(**overrides):
    arguments = dict(
        predict=_predict,
        noise=torch.linspace(-1, 1, 72).reshape(12, 2, 3),
        sigmas=torch.tensor([1, 0.9, 0.7, 0.5, 0.3, 0.1, 0]),
        pool=(0, 1, 2, 3, 4),
        count=2,
        selection="random_subset",
        logical_batch_size=4,
        microbatch=3,
        transition_kind="sde",
        likelihood="gaussian",
        noise_level=0.7,
        unselected_noise="ode",
        generator=torch.Generator().manual_seed(17),
    )
    arguments.update(overrides)
    return arguments


@pytest.mark.parametrize("selection", ["fixed", "contiguous", "random_subset"])
@pytest.mark.parametrize("kind", ["sde", "cps"])
@pytest.mark.parametrize("likelihood", ["gaussian", "mse"])
@pytest.mark.parametrize("unselected", ["ode", "all"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_physical_microbatch_invariance_and_replay(selection, kind, likelihood, unselected, dtype):
    arguments = _arguments(
        selection=selection,
        count=5 if selection == "fixed" else 2,
        transition_kind=kind,
        likelihood=likelihood,
        unselected_noise=unselected,
    )
    arguments["noise"] = arguments["noise"].to(dtype).requires_grad_()
    arguments["sigmas"].requires_grad_()
    noise_before = arguments["noise"].detach().clone()
    sigmas_before = arguments["sigmas"].detach().clone()
    global_rng = torch.get_rng_state().clone()
    baseline = None
    for microbatch in (1, 3, 4, 99):
        generator = torch.Generator().manual_seed(17)
        calls = []

        def predict(state, sigma, rows):
            assert not torch.is_grad_enabled()
            assert not state.requires_grad and not sigma.requires_grad
            assert sigma.dtype == torch.float32
            assert state.dtype == (
                dtype if bool((sigma == arguments["sigmas"][0]).all()) else torch.float32
            )
            assert len(state) <= min(microbatch, arguments["logical_batch_size"])
            assert rows.dtype == torch.long and rows.device == state.device
            calls.append(rows.clone())
            return _predict(state, sigma, rows)

        cache = collect_grpo_rollout(
            **dict(arguments, microbatch=microbatch, generator=generator, predict=predict)
        )
        assert len(calls) == 3 * math.ceil(4 / microbatch) * 6
        assert set(torch.cat(calls).tolist()) == set(range(12))
        for field in fields(cache):
            value = getattr(cache, field.name)
            if isinstance(value, torch.Tensor):
                assert value.device.type == "cpu" and not value.requires_grad
                assert value.dtype == (
                    torch.long if field.name == "step_indices" else torch.float32
                )
        if baseline is None:
            baseline, rng = cache, generator.get_state().clone()
        else:
            assert torch.equal(generator.get_state(), rng)
            for field in fields(cache):
                actual, expected = getattr(cache, field.name), getattr(baseline, field.name)
                if isinstance(actual, torch.Tensor):
                    assert torch.equal(actual, expected), field.name
                else:
                    assert actual == expected
        rows = torch.arange(12).repeat_interleave(arguments["count"])
        slots = torch.arange(arguments["count"]).repeat(12)
        velocity = cache.old_velocities[rows, slots].clone().requires_grad_()
        result = replay_transition(cache, velocity, rows, slots)
        torch.testing.assert_close(
            result.log_prob, cache.old_log_probs[rows, slots], rtol=0, atol=0
        )
        torch.testing.assert_close(result.mean, cache.old_means[rows, slots], rtol=0, atol=0)
        result.log_prob.sum().backward()
        assert velocity.grad is not None and torch.isfinite(velocity.grad).all()
    assert torch.equal(global_rng, torch.get_rng_state())
    assert torch.equal(arguments["noise"], noise_before)
    assert torch.equal(arguments["sigmas"], sigmas_before)
    assert arguments["noise"].grad is arguments["sigmas"].grad is None
    for start in range(0, 12, 4):
        assert torch.equal(
            baseline.step_indices[start : start + 4],
            baseline.step_indices[start].expand(4, -1),
        )
    assert baseline.selected == tuple(baseline.step_indices[0].tolist())


@pytest.mark.parametrize("kind", ["sde", "cps"])
@pytest.mark.parametrize("likelihood", ["gaussian", "mse"])
@pytest.mark.parametrize("unselected", ["ode", "all"])
@pytest.mark.parametrize("selection", ["contiguous", "random_subset"])
def test_independent_reference_trajectory(kind, likelihood, unselected, selection):
    """Canonical constant-noise SDE/CPS formulas, without production helpers."""
    args = _arguments(
        transition_kind=kind,
        likelihood=likelihood,
        unselected_noise=unselected,
        selection=selection,
    )
    reference_rng = torch.Generator().manual_seed(17)
    actual = collect_grpo_rollout(**args)
    expected_indices = []
    union = set()
    for start in range(0, 12, 4):
        if selection == "contiguous":
            offset = int(torch.randint(len(args["pool"]) - 1, (), generator=reference_rng))
            selected = args["pool"][offset : offset + 2]
        else:
            selected = tuple(
                sorted(
                    args["pool"][i]
                    for i in torch.randperm(len(args["pool"]), generator=reference_rng)[:2].tolist()
                )
            )
        expected_indices.extend([selected] * 4)
        stochastic = set(range(5 if kind == "cps" else 6)) if unselected == "all" else set(selected)
        union.update(stochastic)
        innovations = {
            i: torch.randn((4, 2, 3), generator=reference_rng, dtype=torch.float32)
            for i in sorted(stochastic)
        }
        state = args["noise"][start : start + 4].clone()
        for i in range(6):
            s, t = args["sigmas"][i : i + 2].tolist()
            dt = t - s
            velocity = _predict(state, torch.full((4,), s), torch.arange(start, start + 4))
            if i not in stochastic:
                following = state + dt * velocity
            else:
                if kind == "sde":
                    guard = float(args["sigmas"][1]) if s == 1 else s
                    diffusion = math.sqrt(s / (1 - guard)) * args["noise_level"]
                    mean = state * (1 + diffusion**2 / (2 * s) * dt)
                    mean += velocity * (1 + diffusion**2 * (1 - s) / (2 * s)) * dt
                    std = diffusion * math.sqrt(-dt)
                else:
                    std = t * math.sin(args["noise_level"] * math.pi / 2)
                    clean = state - s * velocity
                    noise_estimate = state + (1 - s) * velocity
                    mean = clean * (1 - t) + noise_estimate * math.sqrt(max(t * t - std * std, 0))
                following = mean + std * innovations[i]
                if i in selected:
                    slot = selected.index(i)
                    rows = slice(start, start + 4)
                    torch.testing.assert_close(
                        actual.states[rows, slot], state, atol=5e-7, rtol=2e-6
                    )
                    torch.testing.assert_close(actual.next_states[rows, slot], following)
                    torch.testing.assert_close(actual.old_means[rows, slot], mean)
                    torch.testing.assert_close(
                        actual.transition_stds[rows, slot], torch.full((4,), std)
                    )
                    logp = -(following - mean).square()
                    if likelihood == "gaussian":
                        logp = logp / (2 * std * std) - math.log(std) - math.log(2 * math.pi) / 2
                    torch.testing.assert_close(
                        actual.old_log_probs[rows, slot],
                        logp.flatten(1).mean(1),
                        atol=3e-6,
                        rtol=2e-5,
                    )
            state = following
        torch.testing.assert_close(actual.endpoints[start : start + 4], state)
    assert torch.equal(actual.step_indices, torch.tensor(expected_indices))
    assert actual.stochastic == tuple(sorted(union))
    assert torch.equal(args["generator"].get_state(), reference_rng.get_state())


def test_contiguous_draws_include_every_valid_window(monkeypatch):
    real_randint = torch.randint
    bounds = []

    def randint(high, *args, **kwargs):
        bounds.append(high)
        return real_randint(high, *args, **kwargs)

    monkeypatch.setattr(torch, "randint", randint)
    cache = collect_grpo_rollout(
        **_arguments(
            noise=torch.zeros(64, 2),
            logical_batch_size=1,
            selection="contiguous",
            pool=(1, 2, 3, 4),
            count=2,
        )
    )
    assert bounds == [3] * 64
    assert set(map(tuple, cache.step_indices.tolist())) == {(1, 2), (2, 3), (3, 4)}


def test_each_logical_batch_draws_its_own_window(monkeypatch):
    offsets = iter((0, 2, 1))
    monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: torch.tensor(next(offsets)))
    cache = collect_grpo_rollout(**_arguments(selection="contiguous", pool=(1, 2, 3, 4)))
    expected = torch.tensor([[1, 2]] * 4 + [[3, 4]] * 4 + [[2, 3]] * 4)
    assert torch.equal(cache.step_indices, expected)
    rows = torch.tensor([0, 4, 8, 5, 10])
    slots = torch.tensor([0, 0, 0, 1, 1])
    assert torch.equal(cache.indices(rows, slots), expected[rows, slots])
    velocity = _predict(cache.states[rows, slots], cache.sigmas[expected[rows, slots]], rows)
    assert torch.equal(velocity, cache.old_velocities[rows, slots])
    replay = replay_transition(cache, velocity, rows, slots)
    assert torch.equal(replay.log_prob, cache.old_log_probs[rows, slots])


def test_fixed_order_single_window_and_full_subset():
    fixed = collect_grpo_rollout(**_arguments(selection="fixed", pool=(4, 1, 0), count=3))
    assert fixed.selected == (4, 1, 0)
    assert torch.equal(fixed.step_indices, torch.tensor([4, 1, 0]).repeat(12, 1))
    contiguous = collect_grpo_rollout(**_arguments(selection="contiguous", count=5))
    subset = collect_grpo_rollout(**_arguments(selection="random_subset", count=5))
    assert torch.equal(contiguous.step_indices, subset.step_indices)
    with pytest.raises(ValueError, match="row-specific"):
        fixed.slot(4)


@pytest.mark.parametrize("kind", ["sde", "cps"])
@pytest.mark.parametrize("unselected", ["ode", "all"])
@pytest.mark.parametrize("terminal", [0.0, 0.05])
def test_exact_stochastic_steps_and_endpoint_noise(monkeypatch, kind, unselected, terminal):
    real_transition = rollout.flow_transition
    seen = []

    def transition(*args, **kwargs):
        result = real_transition(*args, **kwargs)
        seen.append(
            (kwargs["kind"], result.sample.clone(), result.mean.clone(), result.std.clone())
        )
        return result

    real_collect = grpo_sampling.collect_rollout

    def collect(*args, **kwargs):
        return real_collect(*args, **kwargs, transition_fn=transition)

    monkeypatch.setattr(grpo_sampling, "collect_rollout", collect)
    cache = collect_grpo_rollout(
        **_arguments(
            noise=torch.ones(4, 2),
            microbatch=4,
            selection="fixed",
            pool=(1,),
            count=1,
            sigmas=torch.tensor([1, 0.8, 0.5, terminal]),
            transition_kind=kind,
            unselected_noise=unselected,
        )
    )
    expected = ["ode", kind, "ode"] if unselected == "ode" else [kind] * 3
    if kind == "cps" and terminal == 0:
        expected[-1] = "ode"
    assert [item[0] for item in seen] == expected
    assert cache.stochastic == tuple(i for i, mode in enumerate(expected) if mode != "ode")
    for mode, sample, mean, std in seen:
        if mode == "ode":
            assert torch.equal(sample, mean) and (std == 0).all()
        else:
            assert not torch.equal(sample, mean) and (std > 0).all()
    assert torch.equal(cache.endpoints, seen[-1][1])


def test_sde_terminal_selected_and_cps_positive_terminal_selected():
    for kind, terminal in (("sde", 0.0), ("cps", 0.05)):
        cache = collect_grpo_rollout(
            **_arguments(
                sigmas=torch.tensor([1, 0.8, 0.5, terminal]),
                selection="fixed",
                pool=(2,),
                count=1,
                transition_kind=kind,
            )
        )
        assert (cache.transition_stds > 0).all()
        assert torch.equal(cache.endpoints, cache.next_states[:, 0])
        result = replay_transition(
            cache, cache.old_velocities[:, 0], torch.arange(12), torch.zeros(12).long()
        )
        assert torch.equal(result.log_prob, cache.old_log_probs[:, 0])


def test_innovations_predrawn_fp32_and_cache_storage_independent(monkeypatch):
    real_collect = grpo_sampling.collect_rollout
    args = _arguments(selection="fixed", pool=(1, 3), count=2)
    noise_before, sigmas_before = args["noise"].clone(), args["sigmas"].clone()
    expected_rng = torch.Generator().manual_seed(17)
    captured = []

    def collect(predict, noise, sigmas, **kwargs):
        for step in (1, 3):
            innovation = kwargs["innovations"][step]
            expected = torch.randn((4, 2, 3), generator=expected_rng, dtype=torch.float32)
            assert innovation.dtype == torch.float32
            assert torch.equal(innovation, expected)
        assert torch.equal(args["generator"].get_state(), expected_rng.get_state())
        cache = real_collect(predict, noise, sigmas, **kwargs)
        captured.append((cache, kwargs["innovations"]))
        return cache

    monkeypatch.setattr(grpo_sampling, "collect_rollout", collect)
    result = collect_grpo_rollout(**args)
    snapshots = {
        field.name: getattr(result, field.name).clone()
        for field in fields(result)
        if isinstance(getattr(result, field.name), torch.Tensor)
    }
    assert torch.equal(args["noise"], noise_before) and torch.equal(args["sigmas"], sigmas_before)
    args["noise"].fill_(100)
    args["sigmas"].fill_(100)
    for cache, innovations in captured:
        for name in snapshots:
            value = getattr(cache, name)
            if isinstance(value, torch.Tensor) and name != "sigmas":
                value.fill_(100)
        for innovation in innovations.values():
            innovation.fill_(100)
    for name, snapshot in snapshots.items():
        assert torch.equal(getattr(result, name), snapshot)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"predict": None}, "predict"),
        ({"noise": None}, "noise"),
        ({"noise": torch.ones(12)}, "noise"),
        ({"noise": torch.empty(0, 2)}, "noise"),
        ({"noise": torch.empty(12, 0)}, "noise"),
        ({"noise": torch.ones(12, 2, dtype=torch.long)}, "noise"),
        ({"noise": torch.ones(12, 2, dtype=torch.complex64)}, "noise"),
        ({"noise": torch.full((12, 2), float("nan"))}, "noise"),
        ({"noise": torch.full((12, 2), 1e100, dtype=torch.float64)}, "noise"),
        ({"logical_batch_size": 5}, "divisible"),
        ({"logical_batch_size": 0}, "logical_batch_size"),
        ({"logical_batch_size": True}, "logical_batch_size"),
        ({"logical_batch_size": 4.0}, "logical_batch_size"),
        ({"microbatch": 0}, "microbatch"),
        ({"microbatch": True}, "microbatch"),
        ({"microbatch": 1.5}, "microbatch"),
        ({"sigmas": [1, 0]}, "sigmas"),
        ({"sigmas": torch.tensor([1, 0])}, "sigmas"),
        ({"sigmas": torch.tensor([[1.0, 0.0]])}, "sigmas"),
        ({"sigmas": torch.tensor([1.0, 0.5, 0.5, 0.0])}, "sigmas"),
        ({"sigmas": torch.tensor([1.0, float("nan"), 0.0])}, "sigmas"),
        ({"sigmas": torch.tensor([1.0, 1.0 - 1e-10, 0], dtype=torch.float64)}, "sigmas"),
        ({"selection": "prefix"}, "selection"),
        ({"transition_kind": "ode"}, "transition_kind"),
        ({"likelihood": "ode_mse"}, "likelihood"),
        ({"unselected_noise": "after_window"}, "unselected_noise"),
        ({"noise_level": 0}, "noise_level"),
        ({"noise_level": -0.1}, "noise_level"),
        ({"noise_level": 1.1}, "noise_level"),
        ({"noise_level": float("nan")}, "noise_level"),
        ({"noise_level": float("inf")}, "noise_level"),
        ({"noise_level": True}, "noise_level"),
        ({"noise_level": "0.7"}, "noise_level"),
        ({"generator": None}, "generator"),
        ({"pool": [0, 1]}, "pool"),
        ({"pool": ()}, "pool"),
        ({"pool": (1, 1)}, "pool"),
        ({"pool": (True,)}, "pool"),
        ({"pool": (1.0,)}, "pool"),
        ({"pool": (-1,)}, "pool"),
        ({"pool": (6,)}, "pool"),
        ({"pool": ([],)}, "pool"),
        ({"count": 0}, "count"),
        ({"count": True}, "count"),
        ({"count": 1.5}, "count"),
        ({"count": 6}, "count"),
        ({"selection": "fixed", "count": 1}, "count"),
        ({"selection": "contiguous", "pool": (0, 2, 3)}, "consecutive"),
        ({"selection": "contiguous", "pool": (2, 1, 0)}, "consecutive"),
        ({"transition_kind": "cps", "pool": (1, 5)}, "zero-variance CPS"),
        ({"sigmas": torch.tensor([1.0, 0.0]), "pool": (0,), "count": 1}, "sigma_max"),
    ],
)
def test_invalid_inputs_fail_before_prediction_or_rng(overrides, match):
    def forbidden(*args):
        pytest.fail("prediction called before validation")

    args = _arguments(predict=forbidden)
    generator = args["generator"]
    rng = generator.get_state().clone()
    args.update(overrides)
    with pytest.raises(ValueError, match=match):
        collect_grpo_rollout(**args)
    assert torch.equal(generator.get_state(), rng)


@pytest.mark.parametrize(
    "bad",
    [
        None,
        torch.ones(3),
        torch.ones(3, 2, 3, dtype=torch.long),
        torch.ones(3, 2, 3, dtype=torch.complex64),
        torch.ones(3, 2, 3, device="meta"),
        torch.full((3, 2, 3), float("nan")),
    ],
)
def test_invalid_predict_output(bad):
    with pytest.raises(ValueError, match="predict"):
        collect_grpo_rollout(**_arguments(predict=lambda *args: bad))


def test_underflowed_stochastic_transition_never_falls_back_to_ode():
    with pytest.raises(ValueError, match="zero-variance"):
        collect_grpo_rollout(**_arguments(noise_level=1e-100))
