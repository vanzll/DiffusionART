"""Real two-rank native tiny H3: sparse modality replay and exact checkpoint resume."""

import copy
import time
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from visual_rl.backends.minimax_h3_posttraining import selected_positions
from visual_rl.backends.minimax_h3_runtime import sigma_schedules
from visual_rl.training.audio_video import run_av_updates
from visual_rl.training.audio_video_objectives import prepare_art_objective, route_rewards
from visual_rl.training.audio_video_rollout import collect_av_rollout
from visual_rl.training.audio_video_types import AVState
from visual_rl.training.fsdp_checkpoint import load_fsdp_checkpoint, save_fsdp_checkpoint
from visual_rl.training.state import Progress


def _same(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _same(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _same(x, y)
    else:
        assert a == b


def _worker(rank, root):
    from test_audio_video_native_fsdp import _native_policy

    from visual_rl.models.minimax_h3 import MiniMaxH3Conditioning

    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{root}/rendezvous', rank=rank,
                            world_size=2, timeout=timedelta(seconds=45))
    try:
        torch.manual_seed(23)
        adapter, model = _native_policy('full')
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        generator = torch.Generator().manual_seed(100 + rank)
        groups = torch.zeros(8, dtype=torch.long)
        sigmas = sigma_schedules(6)
        weights = dict(video_align=1., hpsv3=1.5, audiobox=.5, clap=1., desync=1., imagebind=.5)

        def cycle(attempt):
            conditioning = MiniMaxH3Conditioning(torch.randn(8, 4, 8, generator=generator),
                                                 torch.ones(8, 4, dtype=torch.bool))
            def predict(state, sv, sa, rows):
                return adapter.predict_velocity(state, sv, sa, conditioning.select(rows))
            selected = selected_positions(42, attempt)
            identity = torch.tensor([*selected['video'], *selected['audio']])
            received = [torch.empty_like(identity) for _ in range(2)]
            dist.all_gather(received, identity)
            assert torch.equal(*received)
            union = tuple(sorted(set(selected['video']) | set(selected['audio'])))
            noise = AVState(torch.randn(8, 2, 2, 4, 4, generator=generator),
                            torch.randn(8, 2, 3, 3, generator=generator))
            cache = collect_av_rollout(predict, noise, *sigmas, selected=union,
                                       modality_selected=selected, microbatch=2)
            scores = {name: (torch.arange(8).float() + rank).pow(1 + i % 2)
                      for i, name in enumerate(weights)}
            routes = route_rewards(scores, groups=groups, weights=weights, prompt_group_std=True)
            objective = prepare_art_objective(cache, predict, routes['video'], routes['audio'], groups,
                'cpu', loss_options={'corrective_mask_enabled': False})
            result = run_av_updates(model, optimizer, objective, groups, algorithm='art', microbatch=2,
                                    max_grad_norm=1.)
            assert result['train/optimizer_steps_per_rollout'] == 2
            return selected, result

        cycle(1)
        kwargs = dict(model=model, optimizer=optimizer, config_sha256='a' * 64,
                      generators={'sampling': generator})
        receipt = save_fsdp_checkpoint(Path(root) / 'checkpoints',
            progress=Progress(attempts=1, completed_rollouts=1, optimizer_steps=2), **kwargs)
        selected, expected_metrics = cycle(2)
        expected = [p.detach().clone() for p in model.parameters()]
        expected_adam = copy.deepcopy(optimizer.state_dict())
        expected_rng = generator.get_state().clone()
        restored = load_fsdp_checkpoint(receipt, **kwargs)
        assert restored.attempts == restored.completed_rollouts == 1 and restored.optimizer_steps == 2
        actual_selected, actual_metrics = cycle(restored.attempts + 1)
        assert selected == actual_selected
        _same(expected_metrics, actual_metrics)
        for p, value in zip(model.parameters(), expected):
            _same(p, value)
        _same(optimizer.state_dict(), expected_adam)
        _same(generator.get_state(), expected_rng)
        (Path(root) / f'{rank}.ok').write_text('native sparse U2 and exact resume passed')
    finally:
        dist.destroy_process_group()


def test_native_h3_sparse_modality_selection_fsdp_exact_resume(tmp_path):
    pytest.importorskip('diffusers.models.transformers.transformer_minimax_h3')
    pytest.importorskip('peft')
    context = mp.spawn(_worker, args=(str(tmp_path),), nprocs=2, join=False)
    deadline = time.monotonic() + 90
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail('native H3 post-training test exceeded 90s')
    finally:
        for child in context.processes:
            if child.is_alive():
                child.terminate()
            child.join(timeout=5)
    assert all((tmp_path / f'{rank}.ok').is_file() for rank in range(2))
