"""True B8 native H3 against singleton oracles across heterogeneous FSDP ranks."""
import copy
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _singleton(model, state, conditioning, sv, sa):
    from visual_rl.models.minimax_h3 import native_h3_helpers, unpack_video_rows
    from visual_rl.training.audio_video_types import AVState

    build, patchify = native_h3_helpers()
    videos, audios = [], []
    for row in range(len(state)):
        length = int(conditioning.attention_mask[row].sum())
        pos, tags, vi, ai, ti, _, _ = build(
            torch.ones(length).long(), 2, 4, 4, 3, (1, 2, 2), 2, 2, 0)
        times = torch.full((len(tags),), 1 - float(sv[row]))
        times[ai] = 1 - sa[row]
        times, indices = times.unique(sorted=True, return_inverse=True)
        video, audio = model(
            hidden_states=patchify(state.video[row:row + 1], (1, 2, 2))[None],
            audio_hidden_states=state.audio[row:row + 1].permute(0, 1, 3, 2).reshape(1, 6, 3),
            encoder_hidden_states=conditioning.prompt_embeds[row:row + 1, :length],
            timestep=times, timestep_indices=indices, token_tags=tags, position_ids=pos,
            video_indices=vi, audio_indices=ai, text_indices=ti, return_dict=False)
        videos.append(-unpack_video_rows(video, (1, 2, 2, 4, 4), (1, 2, 2)))
        audios.append(-audio.reshape(1, 2, 3, 3).permute(0, 1, 3, 2))
    return AVState(torch.cat(videos), torch.cat(audios))


def _worker(rank, directory):
    from diffusers import MiniMaxH3Transformer3DModel
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    from visual_rl.backends.minimax_h3_runtime import wrap_policy
    from visual_rl.models.finetune import FineTuneConfig, prepare_policy
    from visual_rl.models.minimax_h3 import MiniMaxH3Adapter, MiniMaxH3Conditioning
    from visual_rl.training.audio_video_types import AVState

    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{directory}/rendezvous',
                            rank=rank, world_size=2, timeout=timedelta(seconds=40))
    try:
        for mode in ('full', 'lora'):
            torch.manual_seed(23)
            native = MiniMaxH3Transformer3DModel(
                num_attention_heads=2, attention_head_dim=12, hidden_size=24, num_layers=1,
                num_refiner_layers=1, ffn_dim=32, in_channels=2, audio_in_channels=3,
                patch_size=(1, 2, 2), text_dim=8, freq_dim=8, time_embed_hidden_dim=16,
                time_embed_dim=12, rope_freq_dim=2)
            reference = copy.deepcopy(native)
            adapter = MiniMaxH3Adapter(native)
            fine = (FineTuneConfig('full') if mode == 'full' else FineTuneConfig(
                'lora', rank=2, alpha=2,
                target_modules=('to_q', 'to_k', 'to_v', 'to_out.0', 'ff.net.0.proj', 'ff.net.2')))
            torch.manual_seed(71)
            native = prepare_policy(native, fine)
            torch.manual_seed(71)
            reference = prepare_policy(reference, fine)
            c = SimpleNamespace(model=SimpleNamespace(options={
                'finetune': {'mode': mode}, 'gradient_checkpointing': True}))
            model = wrap_policy(native, c, torch.device('cpu'))
            adapter.set_trainable_module(model)
            physical = []
            hook = native.register_forward_pre_hook(
                lambda module, args, kwargs: physical.append(kwargs['hidden_states'].shape[0]),
                with_kwargs=True)
            generator = torch.Generator().manual_seed(90 + rank)
            # Different max text lengths across ranks must not alter collective order.
            lengths = torch.tensor([1, 4, 2, 3, 4, 2, 1, 3]) + rank * 3
            mask = torch.arange(int(lengths.max()))[None] < lengths[:, None]
            embeds = torch.randn(8, mask.shape[1], 8, generator=generator)
            embeds[~mask] = 10000
            conditioning = MiniMaxH3Conditioning(embeds, mask)
            state = AVState(torch.randn(8, 2, 2, 4, 4, generator=generator),
                            torch.randn(8, 2, 3, 3, generator=generator))
            sv, sa = torch.linspace(.2, .9, 8), torch.linspace(.8, .1, 8)
            expected = _singleton(reference, state, conditioning, sv, sa)
            actual = adapter.predict_velocity(state, sv, sa, conditioning)
            assert physical == [8]
            torch.testing.assert_close(actual.video, expected.video, rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(actual.audio, expected.audio, rtol=2e-5, atol=2e-6)
            for value in (actual, expected):
                (.5 * value.video.square().mean() + .5 * value.audio.square().mean()).backward()
            for parameter in reference.parameters():
                if parameter.grad is not None:
                    dist.all_reduce(parameter.grad)
                    parameter.grad /= 2
            torch.optim.SGD(model.parameters(), lr=.01).step()
            torch.optim.SGD(reference.parameters(), lr=.01).step()
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                                      FullStateDictConfig(offload_to_cpu=True, rank0_only=False)):
                actual_state = model.state_dict()
            expected_state = reference.state_dict()
            assert actual_state.keys() == expected_state.keys()
            for name in actual_state:
                torch.testing.assert_close(actual_state[name], expected_state[name],
                                           rtol=2e-5, atol=2e-6, msg=lambda msg: f'{name}: {msg}')
            hook.remove()
            (Path(directory) / f'{mode}-{rank}.ok').write_text('B8 forward/backward/update passed')
    finally:
        dist.destroy_process_group()


def test_b8_full_and_lora_two_rank_fsdp_match_singleton_native_updates(tmp_path):
    pytest.importorskip('diffusers.models.transformers.transformer_minimax_h3')
    pytest.importorskip('peft')
    context = mp.spawn(_worker, args=(str(tmp_path),), nprocs=2, join=False)
    deadline = time.monotonic() + 90
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail('H3 B8 FSDP test exceeded 90 seconds')
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    for mode in ('full', 'lora'):
        for rank in range(2):
            assert (tmp_path / f'{mode}-{rank}.ok').is_file()
