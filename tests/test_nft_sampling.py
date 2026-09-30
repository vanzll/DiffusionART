import math

import pytest
import torch

from visual_rl.training.nft_sampling import collect_nft_endpoints


def _velocity(state, sigma, rows):
    shape = (-1,) + (1,) * (state.ndim - 1)
    return (
        state.float().sin() * 0.17 + sigma.reshape(shape) * 0.31 + rows.reshape(shape) * 0.027
    ).to(state.dtype)


def _reference(noise, sigmas, solver, quantize):
    """Independent scalar coefficient recurrence; no production solver helpers."""
    state = noise.clone()
    grid = sigmas.float().tolist()
    old_clean = None
    rows = torch.arange(len(noise))
    for i, (s, t) in enumerate(zip(grid[:-1], grid[1:])):
        time = torch.tensor(s)
        if quantize:
            time = (time * 1000).long().float() / 1000
        v = _velocity(state, time.expand(len(noise)), rows).float()
        x = state.float()
        if solver == "euler":
            state = x + (torch.tensor(t) - torch.tensor(s)) * v
        else:
            clean = x - s * v
            if i == 0 or i == len(grid) - 2:
                eps = (x - (1 - s) * clean) / s
                state = (1 - t) * clean + t * eps
            else:

                def logsnr(u):
                    return math.log1p(-u) - math.log(u)

                h = logsnr(t) - logsnr(s)
                previous_h = math.inf if grid[i - 1] == 1 else logsnr(s) - logsnr(grid[i - 1])
                correction = (h / previous_h) * (clean - old_clean)
                coefficient = (1 - t) * math.expm1(-h)
                state = (t / s) * x - coefficient * clean - 0.5 * coefficient * correction
            old_clean = clean
        state = state.to(noise.dtype)
    return state.float()


@pytest.mark.parametrize("solver", ["dpm2", "euler"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("quantize", [False, True])
@pytest.mark.parametrize(
    "grid",
    [
        [1, 0],
        [0.83, 0.41, 0],
        [1, 0.8237, 0.5319, 0.2113, 0],
        [0.9712, 0.8034, 0.4009, 0.1802, 0.0317],
    ],
)
def test_independent_reference_recurrence(solver, dtype, quantize, grid):
    noise = torch.linspace(-1.3, 1.7, 42).reshape(7, 2, 3).to(dtype)
    sigmas = torch.tensor(grid, dtype=torch.float32)
    expected = _reference(noise, sigmas, solver, quantize)
    actual = collect_nft_endpoints(
        _velocity, noise, sigmas, solver=solver, microbatch=3, quantize_model_time=quantize
    )
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)


@pytest.mark.parametrize("solver", ["dpm2", "euler"])
def test_constant_velocity_analytical_endpoint(solver):
    noise = torch.tensor([[0.5, -0.25], [0.75, 0.125]])
    velocity = torch.tensor([[0.125, 0.5], [0.25, -0.125]])
    sigmas = torch.tensor([1, 0.79, 0.55, 0.24, 0])
    result = collect_nft_endpoints(lambda x, t, rows: velocity[rows], noise, sigmas, solver=solver)
    torch.testing.assert_close(result, noise - velocity, atol=2e-7, rtol=0)


@pytest.mark.parametrize("solver", ["dpm2", "euler"])
@pytest.mark.parametrize("microbatch", [1, 3, 7, 20])
def test_physical_partition_calls_detachment_and_no_mutation(solver, microbatch):
    noise = torch.linspace(-1, 1, 42).reshape(7, 2, 3).requires_grad_()
    sigmas = torch.tensor([1, 0.8291, 0.5017, 0.1239, 0], requires_grad=True)
    original_noise, original_sigmas = noise.detach().clone(), sigmas.detach().clone()
    weight = torch.tensor(0.3, requires_grad=True)
    calls = []

    def predict(state, time, rows):
        assert not torch.is_grad_enabled()
        assert not state.requires_grad and not time.requires_grad
        assert state.dtype == noise.dtype and time.dtype == torch.float32
        assert rows.dtype == torch.long and rows.device == state.device
        assert time.shape == rows.shape == (len(state),)
        assert len(state) <= microbatch
        calls.append((rows.clone(), time.clone()))
        return _velocity(state, time, rows) * weight

    rng = torch.get_rng_state().clone()
    result = collect_nft_endpoints(predict, noise, sigmas, solver=solver, microbatch=microbatch)
    assert torch.equal(rng, torch.get_rng_state())
    assert result.dtype == torch.float32 and result.device.type == "cpu"
    assert not result.requires_grad and result.grad_fn is None
    assert weight.grad is noise.grad is sigmas.grad is None
    assert torch.equal(noise, original_noise) and torch.equal(sigmas, original_sigmas)
    assert len(calls) == math.ceil(len(noise) / microbatch) * (len(sigmas) - 1)
    pos = 0
    for start in range(0, len(noise), microbatch):
        rows = torch.arange(start, min(start + microbatch, len(noise)))
        for sigma in sigmas.detach()[:-1]:
            assert torch.equal(calls[pos][0], rows)
            assert torch.equal(calls[pos][1], ((sigma * 1000).floor() / 1000).expand(len(rows)))
            pos += 1
    whole = collect_nft_endpoints(
        lambda x, t, rows: _velocity(x, t, rows) * weight,
        noise,
        sigmas,
        solver=solver,
        microbatch=len(noise),
    )
    assert torch.equal(result, whole)


@pytest.mark.parametrize("solver", ["dpm2", "euler"])
def test_quantization_changes_model_time_not_integration_grid(solver):
    sigmas = torch.tensor([0.82379, 0.40129, 0.00079, 0])
    noise = torch.ones(2, 3)
    for quantize in (False, True):
        seen = []

        def predict(x, t, rows):
            seen.append(t[0].clone())
            return torch.ones_like(x)

        result = collect_nft_endpoints(
            predict, noise, sigmas, solver=solver, quantize_model_time=quantize
        )
        expected_times = (sigmas[:-1] * 1000).floor() / 1000 if quantize else sigmas[:-1]
        assert torch.equal(torch.stack(seen), expected_times)
        torch.testing.assert_close(result, noise - sigmas[0], atol=2e-7, rtol=0)


def test_default_quantization_and_fp32_schedule_conversion():
    noise = torch.ones(2, 3, dtype=torch.float32)
    sigmas = torch.tensor([1, 0.78991, 0.33239, 0], dtype=torch.float64)
    default = collect_nft_endpoints(_velocity, noise, sigmas)
    explicit = collect_nft_endpoints(
        _velocity, noise, sigmas.float(), solver="dpm2", quantize_model_time=True
    )
    assert torch.equal(default, explicit)
    unquantized = collect_nft_endpoints(_velocity, noise, sigmas, quantize_model_time=False)
    assert not torch.equal(default, unquantized)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"predict": None}, "callable"),
        ({"solver": "rk2"}, "solver"),
        ({"microbatch": 0}, "microbatch"),
        ({"microbatch": -2}, "microbatch"),
        ({"microbatch": 1.5}, "microbatch"),
        ({"microbatch": True}, "microbatch"),
        ({"quantize_model_time": 1}, "quantize_model_time"),
        ({"noise": None}, "noise"),
        ({"noise": torch.ones(3)}, "noise"),
        ({"noise": torch.empty(0, 3)}, "noise"),
        ({"noise": torch.empty(2, 0)}, "noise"),
        ({"noise": torch.ones(2, 3, dtype=torch.long)}, "noise"),
        ({"noise": torch.ones(2, 3, dtype=torch.complex64)}, "noise"),
        ({"noise": torch.full((2, 3), float("nan"))}, "noise"),
        ({"noise": torch.full((2, 3), float("inf"))}, "noise"),
        ({"sigmas": [1, 0]}, "sigmas"),
        ({"sigmas": torch.tensor([1, 0])}, "sigmas"),
        ({"sigmas": torch.tensor([1 + 0j, 0])}, "sigmas"),
        ({"sigmas": torch.tensor(1.0)}, "sigmas"),
        ({"sigmas": torch.tensor([])}, "sigmas"),
        ({"sigmas": torch.tensor([1.0])}, "sigmas"),
        ({"sigmas": torch.tensor([[1.0, 0.0]])}, "sigmas"),
        ({"sigmas": torch.tensor([1.1, 0])}, "sigmas"),
        ({"sigmas": torch.tensor([1.0, -0.1])}, "sigmas"),
        ({"sigmas": torch.tensor([0.0, 1.0])}, "sigmas"),
        ({"sigmas": torch.tensor([1.0, 0.5, 0.5, 0])}, "sigmas"),
        ({"sigmas": torch.tensor([1.0, float("nan"), 0])}, "sigmas"),
        ({"sigmas": torch.tensor([float("inf"), 0])}, "sigmas"),
        ({"sigmas": torch.tensor([1.0, 1.0 - 1e-10, 0], dtype=torch.float64)}, "sigmas"),
    ],
)
def test_invalid_inputs_fail_before_prediction(overrides, match):
    def forbidden(*args):
        pytest.fail("prediction called before validation")

    arguments = dict(predict=forbidden, noise=torch.ones(2, 3), sigmas=torch.tensor([1.0, 0]))
    arguments.update(overrides)
    with pytest.raises(ValueError, match=match):
        collect_nft_endpoints(**arguments)


@pytest.mark.parametrize(
    "bad",
    [
        None,
        torch.ones(2),
        torch.ones(2, 3, dtype=torch.long),
        torch.ones(2, 3, dtype=torch.complex64),
        torch.ones(2, 3, device="meta"),
        torch.full((2, 3), float("nan")),
        torch.full((2, 3), float("inf")),
    ],
)
def test_invalid_prediction(bad):
    with pytest.raises(ValueError, match="predict"):
        collect_nft_endpoints(lambda *args: bad, torch.ones(2, 3), torch.tensor([1.0, 0]))


@pytest.mark.parametrize("solver", ["dpm2", "euler"])
def test_nonfinite_update_rejected(solver):
    noise = torch.full((2, 3), 65504.0, dtype=torch.float16)
    with pytest.raises(ValueError, match="nonfinite NFT state"):
        collect_nft_endpoints(lambda x, t, rows: -x, noise, torch.tensor([1.0, 0]), solver=solver)
