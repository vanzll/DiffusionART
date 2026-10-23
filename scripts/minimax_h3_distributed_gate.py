import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import socket
import statistics
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist


def emit(stage, **values):
    print(json.dumps(dict(stage=stage, rank=int(os.environ['RANK']),
                          unix=time.time(), **values)), flush=True)


def file_sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def library_identity():
    paths = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                    if '/' in line and any(name in line for name in
                       ('libnccl', 'libcuda.so', 'libcudart', 'libtorch_cuda'))})
    result = []
    for path in paths:
        resolved = Path(path).resolve(strict=True)
        metadata = resolved.stat()
        result.append(dict(path=path, realpath=str(resolved), size=metadata.st_size,
                           sha256=file_sha(resolved) if 'libnccl' in path else None))
    core = [item for item in result if Path(item['realpath']).name.startswith('libnccl.so')]
    plugin = [item for item in result if Path(item['realpath']).name.startswith('libnccl-net.so')]
    if len(core) != 1 or len(plugin) != 1:
        raise RuntimeError(f'Expected one actual NCCL core/plugin, got {result}')
    if not Path(core[0]['realpath']).is_relative_to(Path(os.environ['H3_PYTHON']).parents[1]):
        raise RuntimeError('Actual NCCL core is outside this clean environment')
    return result


def collective_case(elements, device):
    rank, world = dist.get_rank(), dist.get_world_size()
    local = (elements + world - 1) // world
    source = torch.full((local,), rank + 1, dtype=torch.float32, device=device)
    gathered = torch.empty(local * world, dtype=torch.float32, device=device)
    reduced = torch.empty_like(source)
    timings = []
    for iteration in range(5):
        dist.barrier()
        torch.cuda.synchronize()
        begin = time.perf_counter()
        dist.all_gather_into_tensor(gathered, source)
        torch.cuda.synchronize()
        ag_seconds = time.perf_counter() - begin
        expected = torch.arange(1, world + 1, device=device, dtype=torch.float32)[:, None]
        if not bool(torch.all(gathered.view(world, local) == expected)):
            raise RuntimeError('AllGather numeric correctness failed')
        gathered.fill_(rank + 1)
        torch.cuda.synchronize()
        begin = time.perf_counter()
        dist.reduce_scatter_tensor(reduced, gathered)
        torch.cuda.synchronize()
        rs_seconds = time.perf_counter() - begin
        if not bool(torch.all(reduced == world * (world + 1) / 2)):
            raise RuntimeError('ReduceScatter numeric correctness failed')
        if iteration:
            timings.append(dict(ag_seconds=ag_seconds, rs_seconds=rs_seconds))
    return dict(full_tensor_bytes=local * world * 4, local_shard_bytes=local * 4,
                dtype='float32', timings=timings,
                median_ag_rs_seconds=statistics.median(item['ag_seconds'] + item['rs_seconds'] for item in timings))


def native_fsdp_case(device):
    from diffusers import MiniMaxH3Transformer3DModel
    from visual_rl.backends.minimax_h3_runtime import wrap_policy
    from visual_rl.models.minimax_h3 import MiniMaxH3Adapter, MiniMaxH3Conditioning
    from visual_rl.training.audio_video_types import AVState

    torch.manual_seed(23)
    native = MiniMaxH3Transformer3DModel(
        num_attention_heads=2, attention_head_dim=128, hidden_size=256, num_layers=1,
        num_refiner_layers=1, ffn_dim=512, in_channels=2, audio_in_channels=3,
        patch_size=(1, 2, 2), text_dim=8, freq_dim=32, time_embed_hidden_dim=256,
        time_embed_dim=128, rope_freq_dim=16)
    adapter = MiniMaxH3Adapter(native)
    physical_batches = []
    hook = native.register_forward_pre_hook(
        lambda module, args, kwargs: physical_batches.append(kwargs['hidden_states'].shape[0]),
        with_kwargs=True)
    config = SimpleNamespace(model=SimpleNamespace(options={
        'finetune': {'mode': 'full'}, 'gradient_checkpointing': True}))
    model = wrap_policy(native, config, device)
    adapter.set_trainable_module(model)
    lengths = torch.tensor([1, 4, 2, 3, 4, 2, 1, 3], device=device) + dist.get_rank() % 4
    mask = torch.arange(int(lengths.max()), device=device)[None] < lengths[:, None]
    conditioning = MiniMaxH3Conditioning(torch.randn(8, mask.shape[1], 8, device=device), mask)
    state = AVState(torch.randn(8, 2, 2, 4, 4, device=device), torch.randn(8, 2, 3, 3, device=device))
    video_sigma = torch.linspace(.2, .9, 8, device=device)
    audio_sigma = torch.linspace(.8, .1, 8, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    with torch.autocast('cuda', dtype=torch.bfloat16):
        predicted = adapter.predict_velocity(state, video_sigma, audio_sigma, conditioning)
        loss = .5 * predicted.video.float().square().mean() + .5 * predicted.audio.float().square().mean()
    loss.backward()
    grad_norm = model.clip_grad_norm_(1)
    if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(grad_norm)) or float(grad_norm) <= 0:
        raise RuntimeError('Native FSDP loss/gradient invalid')
    optimizer.step()
    changed = sum(int(torch.count_nonzero(parameter.detach() != previous))
                  for parameter, previous in zip(model.parameters(), before))
    if changed == 0 or not optimizer.state:
        raise RuntimeError('Native FSDP AdamW did not change the policy')
    hook.remove()
    if not physical_batches or set(physical_batches) != {8}:
        raise RuntimeError(f'Expected physical B8 transformer calls, got {physical_batches}')
    return dict(physical_batches=physical_batches, attention_head_dim=128, loss=float(loss.detach()), grad_norm=float(grad_norm),
                changed_local_elements=changed, optimizer_states=len(optimizer.state))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout', required=True)
    args = parser.parse_args()
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    emit('cuda_bound', torch=torch.__version__, cuda=torch.version.cuda, nccl=torch.cuda.nccl.version())
    dist.init_process_group('nccl', device_id=device, timeout=timedelta(seconds=180))
    try:
        emit('communicator_ready')
        layout = json.loads(Path(args.layout).read_text())
        results = []
        for elements in sorted({512 * 1024 * 1024 // 4, layout['largest_block_parameters']}):
            emit('collective_begin', full_elements=elements)
            results.append(collective_case(elements, device))
            emit('collective_complete', **results[-1])
        emit('native_fsdp_begin')
        native = native_fsdp_case(device)
        torch.cuda.synchronize()
        properties = torch.cuda.get_device_properties(device)
        report = dict(rank=dist.get_rank(), world_size=dist.get_world_size(), host=socket.gethostname(),
                      local_rank=local_rank, gpu_uuid=str(properties.uuid), gpu_total_bytes=properties.total_memory,
                      torch=torch.__version__, torch_source=torch.__file__, cuda=torch.version.cuda,
                      nccl=torch.cuda.nccl.version(), libraries=library_identity(), collectives=results,
                      native_fsdp=native, peak_allocated=torch.cuda.max_memory_allocated(),
                      peak_reserved=torch.cuda.max_memory_reserved(),
                      environment={key: os.environ.get(key) for key in (
                          'NCCL_NET', 'NCCL_NET_PLUGIN', 'NCCL_ALGO', 'NCCL_PROTO', 'NCCL_IB_HCA',
                          'NCCL_IB_PCI_RELAXED_ORDERING', 'NCCL_P2P_DISABLE', 'TORCH_NCCL_BLOCKING_WAIT')})
        destination = Path(os.environ['H3_ATTEMPT_DIR']) / f'gate-rank-{dist.get_rank():02d}.json'
        temporary = destination.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(destination)
        dist.barrier()
        emit('gate_passed', report=str(destination))
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
