import copy
import time
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP

from visual_rl.training.checkpoint import load_checkpoint, save_checkpoint
from visual_rl.training.state import PolicyShadow, Progress
from visual_rl.training.update import TerminalStepError, accumulated_step


class CountingAdamW(torch.optim.AdamW):
    def __init__(self, parameters):
        super().__init__(parameters, lr=0.01)
        self.steps = 0
        self.last_gradients = []

    def step(self, closure=None):
        self.steps += 1
        self.last_gradients = [
            p.grad.clone() for group in self.param_groups for p in group["params"]
            if p.grad is not None
        ]
        return super().step(closure)


class RecordingScaler(torch.amp.GradScaler):
    def __init__(self, **kwargs):
        super().__init__("cpu", init_scale=128.0, growth_interval=2, **kwargs)
        self.steps = 0
        self.updates = []

    def step(self, optimizer, *args, **kwargs):
        self.steps += 1
        return super().step(optimizer, *args, **kwargs)

    def update(self, new_scale=None):
        self.updates.append(new_scale)
        return super().update(new_scale)


def case(dropout=0.0):
    torch.manual_seed(73)
    model = torch.nn.Sequential(torch.nn.Dropout(dropout), torch.nn.Linear(3, 2))
    optimizer = CountingAdamW(model.parameters())
    x, y = torch.randn(7, 3), torch.randn(7, 2)
    mask = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])

    def loss(rows):
        return (model(x[rows]) - y[rows]).square().mean(1) * mask[rows]

    return model, optimizer, loss


def run(model, optimizer, loss, **kwargs):
    return accumulated_step(
        model, optimizer, list(range(7)), loss, microbatch=2,
        denominator=7, max_grad_norm=0.1, **kwargs,
    )


def assert_state_equal(actual, expected):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_state_equal(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_state_equal(left, right)
    else:
        assert actual == expected


@pytest.mark.parametrize("microbatch", [1, 2, 7, 8])
def test_scaled_clipping_matches_unscaled_optimizer_exactly(microbatch):
    plain, control, plain_loss = case()
    model, optimizer, loss = case()
    scaler = RecordingScaler()
    for step in range(3):
        options = dict(microbatch=microbatch, denominator=7, max_grad_norm=0.1)
        expected = accumulated_step(plain, control, list(range(7)), plain_loss, **options)
        metrics = accumulated_step(
            model, optimizer, list(range(7)), loss, scaler=scaler, **options,
        )
        assert {key: metrics[key] for key in expected} == expected
        assert set(expected) == {
            "train/loss", "train/grad_norm/before_clip", "train/grad_clip_triggered",
        }
        assert metrics["train/grad_clip_triggered"] == 1
        assert metrics["train/overflow_retries"] == 0
        assert metrics["train/loss_scale"] == (128 if step < 2 else 256)
        assert metrics["train/next_loss_scale"] == (128 if step == 0 else 256)
        assert_state_equal(model.state_dict(), plain.state_dict())
        assert_state_equal(optimizer.state_dict(), control.state_dict())
        assert_state_equal(optimizer.last_gradients, control.last_gradients)
        assert all(p.grad is None for p in model.parameters())
    assert scaler.steps == optimizer.steps == 3
    assert scaler.updates == [None] * 3


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_default_matches_original_unscaled_arithmetic(dtype):
    torch.manual_seed(17)
    model = torch.nn.Linear(3, 2)
    reference = copy.deepcopy(model)
    optimizer, control = CountingAdamW(model.parameters()), CountingAdamW(reference.parameters())
    x, y = torch.randn(7, 3), torch.randn(7, 2)
    mask = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])

    def loss(network, rows):
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
            output = network(x[rows])
        return (output.float() - y[rows]).square().mean(1) * mask[rows]

    for _ in range(2):
        loss_sum = 0.0
        for start in range(0, 7, 2):
            losses = loss(reference, list(range(start, min(start + 2, 7))))
            loss_sum += float(losses.detach().double().sum())
            (losses.sum() * (1 / 7)).backward()
        norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.1, error_if_nonfinite=True)
        control.step()
        control.zero_grad(set_to_none=True)
        metrics = run(model, optimizer, lambda rows: loss(model, rows))
        assert metrics == {
            "train/loss": loss_sum / 7,
            "train/grad_norm/before_clip": float(norm),
            "train/grad_clip_triggered": float(norm > 0.1),
        }
        assert_state_equal(model.state_dict(), reference.state_dict())
        assert_state_equal(optimizer.state_dict(), control.state_dict())


@pytest.mark.parametrize("bad_gradient", [float("inf"), float("nan")])
def test_overflow_replays_whole_window_and_restores_dropout_rng(bad_gradient):
    model, optimizer, loss = case(dropout=0.5)
    reference, control, reference_loss = case(dropout=0.5)
    scaler = RecordingScaler()
    state = scaler.state_dict()
    state["_growth_tracker"] = 1
    scaler.load_state_dict(state)
    shadow = PolicyShadow(model)
    original = copy.deepcopy(model.state_dict())
    original_shadow = copy.deepcopy(shadow.state_dict())
    rng = torch.get_rng_state()
    seen, losses_seen, scales = [], [], []

    def overflow_once(rows):
        assert optimizer.steps == scaler.steps == 0
        assert not optimizer.state
        assert_state_equal(model.state_dict(), original)
        assert_state_equal(shadow.state_dict(), original_shadow)
        seen.append(list(rows))
        scales.append(scaler.get_scale())
        result = loss(rows)
        losses_seen.append(result.detach().clone())
        if len(seen) == 3:
            result.register_hook(lambda grad: torch.full_like(grad, bad_gradient))
        return result

    actual = run(model, optimizer, overflow_once, scaler=scaler)
    actual_rng = torch.get_rng_state()
    assert seen == [[0, 1], [2, 3], [4, 5], [6]] * 2
    assert scales == [128] * 4 + [64] * 4
    assert_state_equal(losses_seen[:4], losses_seen[4:])
    assert actual["train/overflow_retries"] == 1
    assert actual["train/loss_scale"] == actual["train/next_loss_scale"] == 64
    assert scaler.state_dict()["_growth_tracker"] == 1
    assert scaler.updates == [64.0, None]
    assert optimizer.steps == scaler.steps == 1
    assert all(p.grad is None for p in model.parameters())
    torch.set_rng_state(rng)
    expected = run(reference, control, reference_loss)
    assert torch.equal(actual_rng, torch.get_rng_state())
    assert {key: actual[key] for key in expected} == expected
    assert_state_equal(model.state_dict(), reference.state_dict())
    assert_state_equal(optimizer.state_dict(), control.state_dict())
    assert_state_equal(shadow.state_dict(), original_shadow)


@pytest.mark.parametrize("cuda_available", [False, True])
def test_cpu_scaler_never_touches_cuda_rng_on_retry(monkeypatch, cuda_available):
    model, optimizer, loss = case()
    scaler = RecordingScaler()

    def forbidden(*args, **kwargs):
        pytest.fail("CPU scaling must not access CUDA RNG or initialize CUDA contexts")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    # Adam's unrelated graph-capture probe also sees the fake CUDA availability.
    # Isolate it so this test exercises the runner's CPU retry/RNG handling.
    for name in ("_cuda_graph_capture_health_check", "_accelerator_graph_capture_health_check"):
        if hasattr(optimizer, name):
            monkeypatch.setattr(optimizer, name, lambda: None)
    for name in ("get_rng_state_all", "set_rng_state_all", "get_rng_state", "set_rng_state",
                 "init", "current_device"):
        monkeypatch.setattr(torch.cuda, name, forbidden)

    def overflow_once(rows):
        result = loss(rows)
        if scaler.get_scale() == 128:
            result.register_hook(lambda grad: torch.full_like(grad, float("inf")))
        return result

    metrics = run(model, optimizer, overflow_once, scaler=scaler)
    assert metrics["train/overflow_retries"] == 1


@pytest.mark.parametrize("retry_limit", [0, 2, None])
def test_overflow_exhaustion_is_bounded_and_scaler_can_start_a_new_window(retry_limit):
    model, optimizer, loss = case()
    scaler = RecordingScaler()
    original = copy.deepcopy(model.state_dict())
    calls = []

    def always_overflow(rows):
        calls.append(list(rows))
        result = loss(rows)
        result.register_hook(lambda grad: torch.full_like(grad, float("inf")))
        return result

    options = {} if retry_limit is None else {"max_scale_retries": retry_limit}
    limit = 16 if retry_limit is None else retry_limit
    with pytest.raises(RuntimeError, match=f"overflow exhausted {limit} retries"):
        run(model, optimizer, always_overflow, scaler=scaler, **options)
    assert len(calls) == 4 * (limit + 1)
    assert scaler.steps == optimizer.steps == 0
    assert not optimizer.state
    assert scaler.updates == [128.0 * 0.5**i for i in range(1, limit + 2)]
    assert scaler.state_dict()["_growth_tracker"] == 0
    assert all(p.grad is None for p in model.parameters())
    assert_state_equal(model.state_dict(), original)
    metrics = run(model, optimizer, loss, scaler=scaler)
    assert metrics["train/overflow_retries"] == 0
    assert scaler.steps == optimizer.steps == 1


@pytest.mark.parametrize("failure", ["loss", "forward", "backward", "preparation", "zero"])
def test_terminal_errors_never_retry_or_advance_scaler(failure):
    model, optimizer, loss = case()
    scaler = RecordingScaler()
    original, original_scaler = copy.deepcopy(model.state_dict()), scaler.state_dict()
    calls = []

    def fail_backward(grad):
        raise ValueError("injected backward error")

    def failing(rows):
        calls.append(list(rows))
        if failure == "forward" and len(calls) == 2:
            raise ValueError("injected model error")
        result = loss(rows)
        if failure == "zero":
            return result * 0
        if len(calls) == 2:
            if failure == "loss":
                result = result * float("nan")
            elif failure == "backward":
                result.register_hook(fail_backward)
        return result

    def prepare(rows):
        if failure == "preparation" and rows[0] == 2:
            raise ValueError("injected preparation error")

    error = TerminalStepError if failure in ("forward", "backward") else RuntimeError
    messages = {
        "loss": "loss validation", "forward": "model/loss execution failed",
        "backward": "backward failed", "preparation": "input preparation",
        "zero": "nonzero policy gradient",
    }
    with pytest.raises(error, match=messages[failure]):
        run(model, optimizer, failing, scaler=scaler, validate_batch=prepare,
            require_nonzero_grad=True)
    assert len(calls) == {"zero": 4, "preparation": 1}.get(failure, 2)
    assert scaler.steps == optimizer.steps == 0
    assert scaler.updates == []
    assert scaler.state_dict() == original_scaler
    assert_state_equal(model.state_dict(), original)
    assert not optimizer.state
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("retry_limit", [-1, 17, 1.5, True])
def test_invalid_retry_budget_rejected_before_forward(retry_limit):
    model, optimizer, _ = case()
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    with pytest.raises(RuntimeError, match="max_scale_retries validation"):
        run(model, optimizer, lambda rows: pytest.fail("unexpected forward"),
            scaler=RecordingScaler(), max_scale_retries=retry_limit)
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("scaler", [torch.amp.GradScaler("cpu", enabled=False), object()])
def test_only_enabled_grad_scaler_is_supported(scaler):
    model, optimizer, _ = case()
    with pytest.raises(RuntimeError, match="enabled GradScaler validation"):
        run(model, optimizer, lambda rows: pytest.fail("unexpected forward"), scaler=scaler)
    assert optimizer.steps == 0
    assert all(p.grad is None for p in model.parameters())


def test_cuda_scaler_is_not_treated_as_enabled_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.warns(UserWarning, match="CUDA is not available.*Disabling"):
        scaler = torch.amp.GradScaler("cuda", enabled=True)
    assert not scaler.is_enabled()
    assert scaler.state_dict() == {}
    model, optimizer, _ = case()
    with pytest.raises(RuntimeError, match="enabled GradScaler validation"):
        run(model, optimizer, lambda rows: pytest.fail("unexpected forward"), scaler=scaler)
    assert optimizer.steps == 0
    assert all(p.grad is None for p in model.parameters())


def test_scaler_checkpoint_roundtrip_reproduces_continuation(tmp_path):
    model, optimizer, loss = case(dropout=0.5)
    scaler = RecordingScaler()

    def overflow_once(rows):
        result = loss(rows)
        if scaler.get_scale() == 128:
            result.register_hook(lambda grad: torch.full_like(grad, float("inf")))
        return result

    run(model, optimizer, overflow_once, scaler=scaler)
    receipt = save_checkpoint(
        tmp_path, model=model, optimizer=optimizer, progress=Progress(optimizer_steps=1),
        config_sha256="c" * 64, stateful={"grad_scaler": scaler},
    )
    saved_scaler = scaler.state_dict()
    expected = [run(model, optimizer, loss, scaler=scaler) for _ in range(3)]
    expected_rng = torch.get_rng_state()
    resumed, resumed_optimizer, resumed_loss = case(dropout=0.5)
    resumed_scaler = RecordingScaler()
    progress = load_checkpoint(
        receipt, model=resumed, optimizer=resumed_optimizer, config_sha256="c" * 64,
        stateful={"grad_scaler": resumed_scaler},
    )
    assert progress.optimizer_steps == 1
    assert resumed_scaler.state_dict() == saved_scaler
    actual = [run(resumed, resumed_optimizer, resumed_loss, scaler=resumed_scaler) for _ in range(3)]
    assert actual == expected
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert_state_equal(resumed.state_dict(), model.state_dict())
    assert_state_equal(resumed_optimizer.state_dict(), optimizer.state_dict())
    assert resumed_scaler.state_dict() == scaler.state_dict()


class LocalOverflowScaler(RecordingScaler):
    """Inject after DDP reduction to exercise rank-local overflow consensus."""

    def __init__(self, rank):
        super().__init__()
        self.rank = rank
        self.unscales = 0

    def unscale_(self, optimizer):
        if self.rank == 1 and self.unscales == 0:
            optimizer.param_groups[0]["params"][0].grad.fill_(float("inf"))
        self.unscales += 1
        return super().unscale_(optimizer)


def distributed_worker(rank, init_file, mode):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2,
        timeout=timedelta(seconds=15),
    )
    try:
        if mode == "mismatch":
            for key, different in [
                ("scale", 64.0), ("_growth_tracker", 1), ("growth_factor", 4.0),
                ("backoff_factor", 0.25), ("growth_interval", 3), ("retry_limit", 1),
            ]:
                model = DDP(torch.nn.Linear(3, 2))
                optimizer = CountingAdamW(model.parameters())
                scaler = RecordingScaler()
                if rank == 1 and key != "retry_limit":
                    state = scaler.state_dict()
                    state[key] = different
                    scaler.load_state_dict(state)
                limit = different if key == "retry_limit" and rank == 1 else 16
                with pytest.raises(RuntimeError, match="identical scaler state"):
                    run(model, optimizer, lambda rows: pytest.fail("unexpected forward"),
                        scaler=scaler, max_scale_retries=limit)
                assert optimizer.steps == scaler.steps == 0
                assert all(p.grad is None for p in model.parameters())
            return

        if mode == "fsdp":
            model = FSDP(torch.nn.Linear(3, 2), device_id=torch.device("cpu"))
            optimizer = CountingAdamW(model.parameters())
            with pytest.raises(ValueError, match="FSDP with a GradScaler"):
                run(model, optimizer, lambda rows: pytest.fail("unexpected forward"),
                    scaler=RecordingScaler())
            assert optimizer.steps == 0
            assert all(p.grad is None for p in model.parameters())
            return

        torch.manual_seed(41)
        module = torch.nn.Linear(3, 2)
        reference = copy.deepcopy(module)
        model = DDP(module)
        optimizer, control = CountingAdamW(model.parameters()), CountingAdamW(reference.parameters())
        x, y = torch.randn(7, 3), torch.randn(7, 2)
        indices = [0, 1, 2, 3] if rank == 0 else [4, 5, 6, 0]
        scaler = LocalOverflowScaler(rank) if mode == "local_overflow" else RecordingScaler()
        state = scaler.state_dict()
        state["_growth_tracker"] = 1
        scaler.load_state_dict(state)
        calls = []

        def loss(rows):
            assert optimizer.steps == scaler.steps == 0
            assert not optimizer.state
            calls.append(list(rows))
            actual = [indices[i] for i in rows]
            result = (model(x[actual]) - y[actual]).square().mean(1)
            result = result * result.new_tensor([0 if rank == 1 and i == 3 else 1 for i in rows])
            if mode == "backward_overflow" and rank == 1 and len(calls) == 1:
                result.register_hook(lambda grad: torch.full_like(grad, float("inf")))
            return result

        metrics = accumulated_step(
            model, optimizer, list(range(4)), loss, microbatch=2,
            denominator=4 if rank == 0 else 3, max_grad_norm=0.1, scaler=scaler,
        )
        assert calls == [[0, 1], [2, 3]] * 2
        assert metrics["train/overflow_retries"] == 1
        assert metrics["train/loss_scale"] == metrics["train/next_loss_scale"] == 64
        assert scaler.state_dict()["_growth_tracker"] == 1
        assert scaler.updates == [64.0, None]
        assert optimizer.steps == scaler.steps == 1
        assert all(p.grad is None for p in model.parameters())
        full_loss = (reference(x) - y).square().mean(1).mean()
        full_loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.1)
        control.step()
        assert metrics["train/loss"] == pytest.approx(full_loss.item(), abs=2e-7)
        assert metrics["train/grad_norm/before_clip"] == pytest.approx(norm.item(), abs=2e-7)
        for expected, actual in zip(reference.parameters(), module.parameters()):
            torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
        states = [None, None]
        dist.all_gather_object(states, scaler.state_dict())
        assert states[0] == states[1]
        weights = [torch.empty_like(module.weight) for _ in range(2)]
        dist.all_gather(weights, module.weight.detach())
        assert torch.equal(weights[0], weights[1])
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="requires Gloo")
@pytest.mark.parametrize("mode", ["backward_overflow", "local_overflow", "mismatch", "fsdp"])
def test_two_rank_scaled_update(tmp_path, mode):
    context = mp.spawn(
        distributed_worker, args=(str(tmp_path / "rendezvous"), mode), nprocs=2, join=False,
    )
    deadline = time.monotonic() + 60
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("scaled DDP regression exceeded 60 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
