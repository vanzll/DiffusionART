import argparse
import json
import os
import shlex
import signal
import subprocess
import time
from pathlib import Path


def identity(pid):
    try:
        fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(') ', 1)[1].split()
        return dict(pid=int(pid), state=fields[0], parent_pid=int(fields[1]), start_ticks=fields[19])
    except FileNotFoundError:
        return None


def validate_lease(path, expected):
    actual = {}
    for line in Path(path).read_text().splitlines():
        name, value = line.split('=', 1)
        values = shlex.split(value)
        actual[name] = values[0] if values else ''
    if not all(actual.get(key) == value for key, value in expected.items()):
        raise RuntimeError('Lease identity/generation changed')
    if actual.get('GPU_LEASE_STATUS') not in {'active', 'quarantined'}:
        raise RuntimeError('Lease is not active or quarantined')
    return actual


def supervisor_identities(expected):
    """Protect the live lease owner and both supervision chains from cleanup.

    Linux /proc/PID/environ may retain an inherited nonce after shell unset.
    Nonce matching alone therefore cannot distinguish a trainer from its MPI,
    timeout, or lease supervisor. Include start ticks so PID reuse is harmless.
    """
    owner_pid = int(expected['GPU_LEASE_PID'])
    owner = identity(owner_pid)
    if (owner is None or owner['state'] == 'Z'
            or owner['start_ticks'] != expected['GPU_LEASE_START_TICKS']):
        raise RuntimeError('Lease owner process identity changed')
    protected = set()
    for pid in (owner_pid, os.getpid()):
        for _ in range(128):
            if pid <= 0:
                break
            process = identity(pid)
            if process is None:
                break
            key = process['pid'], process['start_ticks']
            if key in protected:
                break
            protected.add(key)
            pid = process['parent_pid']
        else:
            raise RuntimeError('Cannot bound H3 supervisor ancestry')
    return protected


def owned_processes(nonce, root, *, protected=()):
    result = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit() or int(path.name) == os.getpid():
            continue
        try:
            environment = dict(item.split('=', 1) for item in
                               (path / 'environ').read_bytes().decode().split('\0') if '=' in item)
            if environment.get('H3_ATTEMPT_NONCE') != nonce or environment.get('H3_ACCEPTANCE_ROOT') != root:
                continue
            process = identity(path.name)
            if (process and process['state'] != 'Z'
                    and (process['pid'], process['start_ticks']) not in protected):
                result.append(process)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return result


def terminate_owned(lease_path, expected, nonce, root, *, grace=60):
    sent = set()
    for requested, seconds in [(signal.SIGTERM, grace), (signal.SIGKILL, 15)]:
        deadline = time.monotonic() + seconds
        while True:
            validate_lease(lease_path, expected)
            processes = owned_processes(nonce, root, protected=supervisor_identities(expected))
            if not processes:
                return dict(drained=True, signaled=[list(item) for item in sorted(sent)])
            for process in processes:
                key = process['pid'], process['start_ticks'], int(requested)
                if key in sent:
                    continue
                validate_lease(lease_path, expected)
                if (process['pid'], process['start_ticks']) in supervisor_identities(expected):
                    continue
                current = identity(process['pid'])
                if current is None or current['start_ticks'] != process['start_ticks']:
                    continue
                try:
                    os.kill(process['pid'], requested)
                    sent.add(key)
                except ProcessLookupError:
                    continue
            if time.monotonic() >= deadline:
                break
            time.sleep(.2)
    remaining = owned_processes(nonce, root, protected=supervisor_identities(expected))
    return dict(drained=not remaining, remaining=remaining,
                signaled=[list(item) for item in sorted(sent)])


def gpu_processes():
    output = subprocess.check_output([
        'timeout', '-k', '2', '8', '/usr/bin/nvidia-smi',
        '--query-compute-apps=pid,gpu_uuid,used_memory', '--format=csv,noheader,nounits',
    ], text=True, timeout=11)
    result = []
    for line in output.splitlines():
        if not line.strip():
            continue
        pid, uuid, memory = [item.strip() for item in line.split(',')]
        process = identity(pid)
        argv = (Path('/proc') / pid / 'cmdline').read_bytes().decode().split('\0')
        result.append(dict(pid=int(pid), gpu_uuid=uuid, memory_mib=memory,
                           process=process, argv=argv))
    return result


def local_hcas():
    ports = []
    for port in sorted(Path('/sys/class/infiniband').glob('*/ports/*')):
        if not (port / 'state').read_text().startswith('4:'):
            continue
        if not (port / 'rate').read_text().startswith('400 '):
            continue
        if (port / 'link_layer').read_text().strip() != 'Ethernet':
            raise RuntimeError('Expected Ethernet RDMA')
        if (port / 'gid_attrs/types/3').read_text().strip() != 'RoCE v2':
            raise RuntimeError('Expected RoCE v2 GID index3')
        if (port / 'gids/3').read_text().strip() in {'', '::', '0:0:0:0:0:0:0:0'}:
            raise RuntimeError('Empty RoCE GID')
        ports.append(port.parents[1].name + ':' + port.name)
    if len(ports) != 8:
        raise RuntimeError(f'Expected eight active local400G HCA ports, got {ports}')
    return '=' + ','.join(ports)


def observe_owned(nonce, root, *, protected=()):
    processes = owned_processes(nonce, root, protected=protected)
    for process in processes:
        directory = Path('/proc') / str(process['pid'])
        try:
            process['argv'] = (directory / 'cmdline').read_bytes().decode().strip('\0').split('\0')
            process['memory'] = [line for line in (directory / 'status').read_text().splitlines()
                                 if line.startswith(('VmRSS:', 'VmHWM:', 'VmPTE:', 'Threads:'))]
            process['pss'] = [line for line in (directory / 'smaps_rollup').read_text().splitlines()
                              if line.startswith('Pss:')]
            process['io'] = (directory / 'io').read_text().splitlines()
        except (FileNotFoundError, ProcessLookupError):
            process['exited_during_observation'] = True
    gpu = subprocess.check_output([
        'timeout', '-k', '2', '8', '/usr/bin/nvidia-smi',
        '--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits',
    ], text=True, timeout=11)
    return dict(unix=time.time(), processes=processes, gpu_rows=gpu.splitlines(),
                host_memory=[line for line in Path('/proc/meminfo').read_text().splitlines()
                             if line.startswith(('MemTotal:', 'MemAvailable:'))])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lease')
    parser.add_argument('--owner-pid')
    parser.add_argument('--owner-start-ticks')
    parser.add_argument('--generation')
    parser.add_argument('--nonce')
    parser.add_argument('--root')
    parser.add_argument('--terminate', action='store_true')
    parser.add_argument('--observe', action='store_true')
    parser.add_argument('--grace', type=int, default=60)
    parser.add_argument('--admit', action='store_true')
    parser.add_argument('--empty', action='store_true')
    parser.add_argument('--hcas', action='store_true')
    args = parser.parse_args()
    if args.hcas:
        print(local_hcas(), flush=True)
        return 0
    if args.admit or args.empty:
        processes = gpu_processes()
        admitted = all('/m2v_intern2/zhenglin/gpu_occupy.py' in item['argv']
                       and item['process'] and item['process']['state'] != 'Z'
                       for item in processes)
        print(json.dumps(dict(admitted=admitted, empty=not processes, processes=processes)), flush=True)
        return 0 if (not processes if args.empty else admitted) else 2
    required = [args.lease, args.owner_pid, args.owner_start_ticks, args.generation, args.nonce, args.root]
    if not all(required):
        parser.error('lease, owner identity, generation, nonce and root are required')
    expected = dict(GPU_LEASE_PID=args.owner_pid, GPU_LEASE_START_TICKS=args.owner_start_ticks,
                    GPU_LEASE_GENERATION=args.generation, GPU_LEASE_NONCE=args.nonce)
    validate_lease(args.lease, expected)
    if args.observe:
        result = observe_owned(args.nonce, args.root, protected=supervisor_identities(expected))
        print(json.dumps(result), flush=True)
        return 0
    if args.terminate:
        result = terminate_owned(args.lease, expected, args.nonce, args.root, grace=args.grace)
    else:
        processes = owned_processes(args.nonce, args.root, protected=supervisor_identities(expected))
        result = dict(drained=not processes, remaining=processes)
    print(json.dumps(result), flush=True)
    return 0 if result['drained'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
