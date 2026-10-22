"""Actual pinned tiny H3 + real PEFT/native-refiner FSDP integration.

No pretrained weights, real scorer assets or GPUs are used. Deterministic
fixture rewards isolate optimizer/old/checkpoint semantics from decoder assets.
"""
import math
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from visual_rl.backends.minimax_h3_runtime import sigma_schedules, wrap_policy
from visual_rl.training.audio_video import AVNFTState, run_av_updates
from visual_rl.training.audio_video_objectives import (
    prepare_art_objective,
    prepare_fast_objective,
    prepare_nft_objective,
    select_nft_fresh_times,
)
from visual_rl.training.audio_video_rollout import collect_av_rollout, select_fast_window
from visual_rl.training.audio_video_types import AVState
from visual_rl.training.fsdp_checkpoint import load_fsdp_checkpoint, save_fsdp_checkpoint
from visual_rl.training.state import Progress

TARGETS = ("to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2")


def _native_policy(mode):
    from diffusers import MiniMaxH3Transformer3DModel

    from visual_rl.models.finetune import FineTuneConfig, prepare_policy
    from visual_rl.models.minimax_h3 import MiniMaxH3Adapter

    native = MiniMaxH3Transformer3DModel(
        num_attention_heads=2, attention_head_dim=12, hidden_size=24, num_layers=1,
        num_refiner_layers=1, ffn_dim=32, in_channels=2, audio_in_channels=3,
        patch_size=(1, 2, 2), text_dim=8, freq_dim=8, time_embed_hidden_dim=16,
        time_embed_dim=12, rope_freq_dim=2)
    adapter = MiniMaxH3Adapter(native)
    finetune = (FineTuneConfig("full") if mode == "full" else
                FineTuneConfig("lora", rank=2, alpha=2, target_modules=TARGETS))
    native = prepare_policy(native, finetune)
    c = SimpleNamespace(model=SimpleNamespace(options={"finetune": {"mode": mode}, "gradient_checkpointing": True}))
    model = wrap_policy(native, c, torch.device("cpu"))
    adapter.set_trainable_module(model)
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    assert any("refiner_blocks.0" in name and isinstance(module, FSDP) for name, module in model.named_modules())
    # These native modules are queried for dtype before forward. Their FSDP
    # Parameters must stay visible instead of being flattened into their parent.
    assert any("proj_in" in name and isinstance(module, FSDP) for name, module in model.named_modules())
    return adapter, model


def _worker(rank, root):
    from visual_rl.models.minimax_h3 import MiniMaxH3Conditioning

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{root}/rendezvous", rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        for mode in ("full", "lora"):
            for algorithm in ("art", "nft", "flow_grpo_fast"):
                torch.manual_seed(23)
                adapter, model = _native_policy(mode)
                optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
                old = AVNFTState(model) if algorithm == "nft" else None
                generator = torch.Generator().manual_seed(101 + rank)
                conditioning = MiniMaxH3Conditioning(torch.randn(2, 4, 8, generator=generator),
                                                    torch.ones(2, 4, dtype=torch.bool))
                def predict(state, video_sigma, audio_sigma, rows):
                    return adapter.predict_velocity(state, video_sigma, audio_sigma, conditioning.select(rows))
                sigmas = sigma_schedules(5)
                groups = torch.zeros(2, dtype=torch.long)
                reward = torch.tensor([-1., 1.]) + rank * 3
                audio_reward = -reward
                population = []
                for values in (reward, audio_reward):
                    gathered = [torch.empty_like(values) for _ in range(2)]
                    dist.all_gather(gathered, values)
                    population.append(torch.cat(gathered))
                def cycle():
                    noise = AVState(torch.randn(2, 2, 2, 4, 4, generator=generator),
                                    torch.randn(2, 2, 3, 3, generator=generator))
                    options = {}
                    if algorithm == "flow_grpo_fast":
                        selected = select_fast_window(4)
                        innovations = {step: AVState(torch.randn(noise.video.shape, generator=generator),
                            torch.randn(noise.audio.shape, generator=generator)) for step in selected}
                        options.update(selected=selected, stochastic=selected, innovations=innovations,
                                       unselected_kind="cps_zero_noise")
                    elif old is not None:
                        options["selected"] = ()
                    if old is not None:
                        with old.use_old():
                            cache = collect_av_rollout(predict, noise, *sigmas, microbatch=1, **options)
                        times = select_nft_fresh_times(sigmas[0], 2, positions=2, generator=generator)
                        fresh = AVState(torch.randn(2, 2, 2, 2, 4, 4, generator=generator),
                                        torch.randn(2, 2, 2, 3, 3, generator=generator))
                        with old.use_old():
                            objective = prepare_nft_objective(cache.endpoints, predict, predict, reward, audio_reward,
                                groups, times, fresh, "cpu", microbatch=1,
                                global_video_rewards=population[0], global_audio_rewards=population[1])
                    else:
                        cache = collect_av_rollout(predict, noise, *sigmas, microbatch=1, **options)
                        objective = (prepare_art_objective(cache, predict, reward, audio_reward, groups, "cpu")
                            if algorithm == "art" else prepare_fast_objective(cache, predict, reward, groups, "cpu",
                                global_shared_rewards=population[0]))
                    result = run_av_updates(model, optimizer, objective, groups, algorithm=algorithm, microbatch=1,
                                            max_grad_norm=10, state=old)
                    assert all(math.isfinite(value) for value in result.values())
                    assert result["train/optimizer_steps_per_rollout"] == (1 if old is not None else 2)
                    assert result["train/grad_norm/before_clip"] > 0
                    return result
                initial = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
                result = cycle()
                assert any(not torch.equal(initial[name], p) for name, p in model.named_parameters() if p.requires_grad)
                stateful = old.stateful() if old is not None else {}
                kwargs = dict(model=model, optimizer=optimizer, config_sha256="a" * 64,
                              stateful=stateful, generators={"sampling": generator})
                checkpoint = save_fsdp_checkpoint(Path(root) / mode / algorithm,
                    progress=Progress(attempts=1, completed_rollouts=1,
                        optimizer_steps=int(result["train/optimizer_steps_per_rollout"])), **kwargs)
                if old is not None:
                    assert old.successful_updates == 1
                    reference = cycle()
                    expected = {name: p.detach().clone() for name, p in model.named_parameters()}
                    expected_old = {key: value.clone() for key, value in old.old.shadow.items()}
                    restored = load_fsdp_checkpoint(checkpoint, **kwargs)
                    assert restored.optimizer_steps == old.successful_updates == 1
                    actual = cycle()
                    assert actual["nft/old_policy_decay"] == reference["nft/old_policy_decay"] == .00125
                    assert actual["train/loss"] == reference["train/loss"]
                    for name, value in model.named_parameters():
                        torch.testing.assert_close(value, expected[name], atol=0, rtol=0)
                    for name, value in old.old.shadow.items():
                        torch.testing.assert_close(value, expected_old[name], atol=0, rtol=0)
                else:
                    restored = load_fsdp_checkpoint(checkpoint, **kwargs)
                    assert restored.optimizer_steps == 2
                (Path(root) / f"{mode}-{algorithm}-{rank}.ok").write_text("actual native H3 passed")
    finally:
        dist.destroy_process_group()


def test_actual_native_h3_full_and_six_target_lora_fsdp_three_algorithms_checkpoint(tmp_path):
    pytest.importorskip("diffusers.models.transformers.transformer_minimax_h3")
    pytest.importorskip("peft")
    ctx = mp.spawn(_worker, args=(str(tmp_path),), nprocs=2, join=False)
    deadline = time.monotonic() + 90
    try:
        while not ctx.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("actual native H3 six-path FSDP test exceeded 90 seconds")
    finally:
        for process in ctx.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    for mode in ("full", "lora"):
        for algorithm in ("art", "nft", "flow_grpo_fast"):
            for rank in range(2):
                assert (tmp_path / f"{mode}-{algorithm}-{rank}.ok").is_file()
