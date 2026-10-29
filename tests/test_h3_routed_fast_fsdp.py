"""Real CPU FSDP heads, inactive-modal backward and exact routed Fast resume.

Uses only tiny random weights; this does not establish native H3 GPU capacity.
"""

import copy
import random
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from visual_rl.backends.minimax_h3_reference import ReferenceState, prepare_reference_predictions
from visual_rl.training.audio_video import run_av_updates
from visual_rl.training.audio_video_objectives import prepare_routed_fast_objective
from visual_rl.training.audio_video_rollout import collect_av_rollout
from visual_rl.training.audio_video_types import AVState
from visual_rl.training.checkpoint import rng_state
from visual_rl.training.fsdp_checkpoint import load_fsdp_checkpoint, save_fsdp_checkpoint
from visual_rl.training.state import Progress


class SeparateHeads(torch.nn.Module):
    """Each output head owns separate FSDP backward hooks and collectives."""

    def __init__(self):
        super().__init__()
        self.video = FSDP(torch.nn.Linear(4, 4, bias=False), device_id=torch.device("cpu"), use_orig_params=False)
        self.audio = FSDP(torch.nn.Linear(3, 3, bias=False), device_id=torch.device("cpu"), use_orig_params=False)

    def forward(self, video, audio, sv, sa):
        # Tuple output keeps the test independent of dataclass traversal hooks.
        return self.video(video) * (1 + sv[:, None]), self.audio(audio) * (1 + sa[:, None])


def _same(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            _same(a, b)
    else:
        assert left == right


def _worker(rank, root, kind):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{root}/rendezvous", rank=rank,
                            world_size=2, timeout=timedelta(seconds=30))
    try:
        torch.manual_seed(37)
        model = FSDP(SeparateHeads(), device_id=torch.device("cpu"), use_orig_params=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
        reference = ReferenceState(model)
        initial_reference = copy.deepcopy(reference.state_dict())
        generator = torch.Generator().manual_seed(700 + rank)
        torch.manual_seed(100 + rank)
        random.seed(200 + rank)
        np.random.seed(300 + rank)
        groups = torch.zeros(8, dtype=torch.long)
        selections = {"video": (0, 2), "audio": (1, 2)}
        union = (0, 1, 2)
        video_sigmas = torch.tensor([1., .9, .7, .4, 0.])
        audio_sigmas = torch.tensor([1., .85, .6, .3, 0.])

        def predict(state, sv, sa, rows):
            values = model(state.video, state.audio, sv, sa)
            return AVState(*values)

        def cycle():
            # Exercise every restored RNG family, not only the sampling generator.
            scale = random.random() + float(np.random.random()) + float(torch.rand(()))
            noise = AVState(torch.randn(8, 4, generator=generator) * scale,
                            torch.randn(8, 3, generator=generator) * scale)
            innovations = {step: AVState(torch.randn(8, 4, generator=generator),
                                         torch.randn(8, 3, generator=generator)) for step in union}
            cache = collect_av_rollout(predict, noise, video_sigmas, audio_sigmas, selected=union,
                stochastic=union, innovations=innovations, modality_selected=selections,
                modality_stochastic=selections, unselected_kind="ode" if kind == "sde" else "cps_zero_noise",
                transition_kind=kind, microbatch=1)
            with reference.use_reference():
                targets = prepare_reference_predictions(cache, predict, "cpu", 1)
            objective = prepare_routed_fast_objective(cache, predict, torch.arange(8).float() - 3.5,
                torch.arange(8).float().flip(0) - 3.5, groups, "cpu", reference_mse_beta=.001,
                reference_predictions=targets)
            metrics = run_av_updates(model, optimizer, objective, groups, algorithm="flow_grpo_fast",
                microbatch=1, max_grad_norm=100.)
            assert metrics["train/optimizer_steps_per_rollout"] == 2
            assert metrics["train/grad_norm/before_clip"] > 0
            assert all(np.isfinite(value) for value in metrics.values())
            assert all(parameter.grad is None for parameter in model.parameters())
            _same(reference.state_dict(), initial_reference)
            return metrics

        cycle()
        kwargs = dict(model=model, optimizer=optimizer, config_sha256="c" * 64,
                      stateful=reference.stateful(), generators={"sampling": generator})
        receipt = save_fsdp_checkpoint(Path(root) / "checkpoints",
            progress=Progress(attempts=1, completed_rollouts=1, optimizer_steps=2), **kwargs)
        expected_metrics = cycle()
        expected_model = [parameter.detach().clone() for parameter in model.parameters()]
        expected_optimizer = copy.deepcopy(optimizer.state_dict())
        expected_rng, expected_sampling = rng_state(), generator.get_state().clone()
        expected_reference = copy.deepcopy(reference.state_dict())
        restored = load_fsdp_checkpoint(receipt, **kwargs)
        assert restored.attempts == restored.completed_rollouts == 1 and restored.optimizer_steps == 2
        _same(reference.state_dict(), initial_reference)
        actual_metrics = cycle()
        _same(actual_metrics, expected_metrics)
        for parameter, expected in zip(model.parameters(), expected_model):
            _same(parameter, expected)
        _same(optimizer.state_dict(), expected_optimizer)
        _same(rng_state(), expected_rng)
        _same(generator.get_state(), expected_sampling)
        _same(reference.state_dict(), expected_reference)
        (Path(root) / f"rank-{rank}.ok").write_text("routed CPS U2 and reference exact resume passed\n")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("kind", ["cps", "sde"])
def test_two_rank_routed_fast_inactive_heads_reference_and_exact_resume(tmp_path, kind):
    context = mp.spawn(_worker, args=(str(tmp_path), kind), nprocs=2, join=False)
    deadline = time.monotonic() + 75
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("routed Fast CPU FSDP test exceeded75s")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    assert all((tmp_path / f"rank-{rank}.ok").is_file() for rank in range(2))
