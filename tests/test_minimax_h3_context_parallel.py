"""Real CPU FSDP4/DP2xCP2 against replicated full-sequence DDP4 Adam."""

import copy
import os
import random
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank, directory):
    from diffusers import MiniMaxH3Transformer3DModel
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.nn.parallel import DistributedDataParallel as DDP

    from visual_rl.backends.minimax_h3_checkpoint_audit import audit_restored_checkpoint
    from visual_rl.backends.minimax_h3_parallel import H3Parallel, enable_context_parallel
    from visual_rl.backends.minimax_h3_reference import ReferenceState
    from visual_rl.backends.minimax_h3_runtime import global_scores, wrap_policy
    from visual_rl.data import Prompt, PromptDataset, PromptStream
    from visual_rl.models.minimax_h3 import MiniMaxH3Adapter, MiniMaxH3Conditioning
    from visual_rl.rewards.audio_video import AVScores
    from visual_rl.training.audio_video import prompt_disjoint_windows
    from visual_rl.training.audio_video_types import AVState
    from visual_rl.training.distributed import DistributedSession, RankInfo
    from visual_rl.training.fsdp_checkpoint import load_fsdp_checkpoint, save_fsdp_checkpoint
    from visual_rl.training.state import Progress
    from visual_rl.training.update import accumulated_step

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{directory}/rendezvous",
                            rank=rank, world_size=4, timeout=timedelta(seconds=90))
    try:
        parallel = H3Parallel(2, rank, 4).initialize()
        torch.manual_seed(23)
        native = MiniMaxH3Transformer3DModel(
            num_attention_heads=56, attention_head_dim=12, hidden_size=24, num_layers=2,
            num_refiner_layers=1, ffn_dim=32, in_channels=2, audio_in_channels=3,
            patch_size=(1, 2, 2), text_dim=8, freq_dim=8, time_embed_hidden_dim=16,
            time_embed_dim=12, rope_freq_dim=2)
        reference = copy.deepcopy(native)
        adapter, oracle = MiniMaxH3Adapter(native), MiniMaxH3Adapter(reference)
        enable_context_parallel(native, parallel)
        config = SimpleNamespace(model=SimpleNamespace(options={
            "finetune": {"mode": "full"}, "gradient_checkpointing": True}))
        model = wrap_policy(native, config, torch.device("cpu"))
        frozen_reference = ReferenceState(model)
        baseline = DDP(reference)
        adapter.set_trainable_module(model)
        oracle.set_trainable_module(baseline)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
        expected_optimizer = torch.optim.AdamW(baseline.parameters(), lr=1e-5, weight_decay=1e-4)
        generator = torch.Generator().manual_seed(90 + parallel.dp_rank)
        lengths = torch.tensor([1, 4, 2, 3, 4, 2, 1, 3]) + parallel.dp_rank * 3
        mask = torch.arange(int(lengths.max()))[None] < lengths[:, None]
        embeds = torch.randn(8, mask.shape[1], 8, generator=generator)
        embeds[~mask] = 10000
        conditioning = MiniMaxH3Conditioning(embeds, mask)
        state = AVState(torch.randn(8, 2, 2, 4, 4, generator=generator),
                        torch.randn(8, 2, 3, 3, generator=generator))
        sv, sa = torch.tensor([1., .99, .98]), torch.tensor([1., .96, .92])
        frozen = []
        for slot in range(3):
            with torch.no_grad():
                actual = adapter.predict_velocity(state, sv[slot], sa[slot], conditioning)
                expected = oracle.predict_velocity(state, sv[slot], sa[slot], conditioning)
            torch.testing.assert_close(actual.video, expected.video, rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(actual.audio, expected.audio, rtol=2e-5, atol=2e-6)
            frozen.append(expected)
        # Leader-only frozen media/reward work may consume process RNG. Policy
        # dropout is zero; its sampling generator remains independently owned.
        torch.manual_seed(700 + rank)
        random.seed(800 + rank)
        np.random.seed(900 + rank)
        physical = []
        hook = native.register_forward_pre_hook(
            lambda module, args, kwargs: physical.append(kwargs["hidden_states"].shape[0]), with_kwargs=True)

        def objective(policy):
            def loss(items):
                rows = torch.tensor([row for row, _ in items])
                slots = torch.tensor([slot for _, slot in items])
                prediction = policy.predict_velocity(state.select(rows), sv[slots], sa[slots], conditioning.select(rows))
                video_ref = torch.cat([frozen[slot].video[row:row + 1] for row, slot in items])
                audio_ref = torch.cat([frozen[slot].audio[row:row + 1] for row, slot in items])
                video = (prediction.video - video_ref - .07).square().flatten(1).mean(1)
                audio = (prediction.audio - audio_ref + .03).square().flatten(1).mean(1)
                # Disjoint modality positions and corrective inactivity keep the
                # original item denominator. The independent reference is unmasked.
                mask_v = ((slots != 1) & (rows % 3 != 0)).float()
                mask_a = ((slots != 0) & (rows % 2 != 0)).float()
                ref = .5 * ((prediction.video - video_ref).square().flatten(1).mean(1)
                            + (prediction.audio - audio_ref).square().flatten(1).mean(1))
                return .5 * 1.5 * (video * mask_v + audio * mask_a) + 1e-3 * ref
            return loss

        for items in prompt_disjoint_windows(torch.zeros(8, dtype=torch.long), 3):
            actual = accumulated_step(model, optimizer, items, objective(adapter), microbatch=8,
                                      denominator=12, max_grad_norm=.01)
            expected = accumulated_step(baseline, expected_optimizer, items, objective(oracle), microbatch=8,
                                        denominator=12, max_grad_norm=.01)
            for key in ("train/loss", "train/grad_norm/before_clip", "train/grad_clip_triggered"):
                assert actual[key] == pytest.approx(expected[key], rel=3e-4, abs=2e-7)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                                      FullStateDictConfig(offload_to_cpu=False, rank0_only=False)):
                actual_state = model.state_dict()
                actual_adam = FSDP.optim_state_dict(model, optimizer)["state"]
            expected_state = reference.state_dict()
            for name in expected_state:
                torch.testing.assert_close(actual_state[name], expected_state[name], rtol=3e-5, atol=3e-6)
            named = dict(reference.named_parameters())
            for name, values in actual_adam.items():
                for key, value in values.items():
                    torch.testing.assert_close(value, expected_optimizer.state[named[name]][key], rtol=3e-4, atol=3e-7)
        assert physical == [8, 4, 8, 4]
        hook.remove()
        # CP replicas must not double the complete prompt-K population.
        identity = RankInfo(rank, 4, rank, 4)
        session = DistributedSession(identity, torch.device("cpu"), True)
        values = torch.arange(8).float() + 10 * parallel.dp_rank
        scores = global_scores(session, AVScores({"test": values}, {}), parallel)
        torch.testing.assert_close(scores.components["test"], torch.cat((torch.arange(8), torch.arange(8) + 10)).float())
        # Exercise the production local-shard save/load and exact restore auditor.
        # This uses actual CP2 groups and the immutable initial-policy shadow.
        stream = PromptStream(PromptDataset(tuple(Prompt(str(i), f"prompt {i}") for i in range(12)),
                                            "a" * 64), seed=42, shuffle=True)
        stream.take(2, rank=parallel.dp_rank, world_size=parallel.dp_world)
        stateful = {"prompts": stream, **frozen_reference.stateful()}
        generators = {"av_sampling": generator}
        progress = Progress(attempts=1, completed_rollouts=1, optimizer_steps=2,
                            elapsed_seconds=10., training_seconds=8.)
        arguments = dict(model=model, optimizer=optimizer, config_sha256="c" * 64,
                         stateful=stateful, generators=generators)
        receipt = save_fsdp_checkpoint(Path(directory) / "checkpoints", progress=progress, **arguments)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            for values in optimizer.state.values():
                for value in values.values():
                    if isinstance(value, torch.Tensor):
                        value.fill_(-7)
        changed_reference = frozen_reference.state_dict()
        for tensor in changed_reference["parameters"]["parameters"].values():
            tensor.zero_()
        frozen_reference.load_state_dict(changed_reference)
        stream.take(2, rank=parallel.dp_rank, world_size=parallel.dp_world)
        generator.manual_seed(999)
        torch.manual_seed(888)
        restored = load_fsdp_checkpoint(receipt, **arguments)
        assert restored == progress
        os.environ["H3_ATTEMPT_NONCE"] = "cpu-cp2-recovery-test"
        audit_restored_checkpoint(session, receipt, model, optimizer, stateful, generators,
                                  restored, Path(directory))
        with frozen_reference.use_reference(session):
            prediction = adapter.predict_velocity(state, sv[0], sa[0], conditioning)
        torch.testing.assert_close(prediction.video, frozen[0].video, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(prediction.audio, frozen[0].audio, rtol=2e-5, atol=2e-6)
        # Continue one real B8+4 Adam update after restore against uninterrupted DDP.
        items = next(iter(prompt_disjoint_windows(torch.zeros(8, dtype=torch.long), 3)))
        actual = accumulated_step(model, optimizer, items, objective(adapter), microbatch=8,
                                  denominator=12, max_grad_norm=.01)
        expected = accumulated_step(baseline, expected_optimizer, items, objective(oracle), microbatch=8,
                                    denominator=12, max_grad_norm=.01)
        assert actual["train/loss"] == pytest.approx(expected["train/loss"], rel=3e-4, abs=2e-7)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                                  FullStateDictConfig(offload_to_cpu=False, rank0_only=False)):
            after = model.state_dict()
            adam = FSDP.optim_state_dict(model, optimizer)["state"]
        for name, expected_tensor in reference.state_dict().items():
            torch.testing.assert_close(after[name], expected_tensor, rtol=3e-5, atol=3e-6)
        for name, values in adam.items():
            for key, value in values.items():
                torch.testing.assert_close(value, expected_optimizer.state[named[name]][key], rtol=3e-4, atol=3e-7)
        (Path(directory) / f"rank-{rank}.ok").write_text("B8 DP2xCP2 output/gradient/clip/Adam/U2/population oracle passed")
    finally:
        dist.destroy_process_group()


def test_cp2_production_checkpoint_restore_reference_rng_cursor_adam_and_continue(tmp_path):
    pytest.importorskip("diffusers.models.transformers.transformer_minimax_h3")
    context = mp.spawn(_worker, args=(str(tmp_path),), nprocs=4, join=False)
    deadline = time.monotonic() + 210
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("H3 CP2 CPU oracle exceeded210s")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    assert all((tmp_path / f"rank-{rank}.ok").is_file() for rank in range(4))
