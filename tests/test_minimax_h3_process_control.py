import importlib.util
import os
from pathlib import Path
import shlex
import subprocess
import sys
import uuid

import pytest


specification = importlib.util.spec_from_file_location(
    'h3_control', Path(__file__).resolve().parents[1] / 'scripts/minimax_h3_process_control.py')
control = importlib.util.module_from_spec(specification)
specification.loader.exec_module(control)


def test_generation_change_fails_closed(tmp_path):
    lease = tmp_path / 'owner.env'
    lease.write_text('GPU_LEASE_GENERATION=3\nGPU_LEASE_STATUS=active\n')
    with pytest.raises(RuntimeError, match='generation changed'):
        control.validate_lease(lease, dict(GPU_LEASE_GENERATION='4'))


def test_only_exact_nonce_and_root_child_is_stopped(tmp_path):
    nonce = 'cpu-smoke-' + uuid.uuid4().hex
    environment = dict(os.environ, H3_ATTEMPT_NONCE=nonce, H3_ACCEPTANCE_ROOT=str(tmp_path))
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],
                             env=environment, start_new_session=True)
    foreign = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],
                               env=dict(environment, H3_ATTEMPT_NONCE=nonce + '-other'),
                               start_new_session=True)
    try:
        owner = control.identity(os.getpid())
        expected = dict(GPU_LEASE_PID=str(os.getpid()), GPU_LEASE_START_TICKS=owner['start_ticks'],
                        GPU_LEASE_GENERATION='1', GPU_LEASE_NONCE=nonce)
        lease = tmp_path / 'owner.env'
        lease.write_text(''.join(f'{key}={value}\n' for key, value in
                                 dict(expected, GPU_LEASE_STATUS='active').items()))
        result = control.terminate_owned(lease, expected, nonce, str(tmp_path), grace=2)
        assert result['drained']
        assert child.wait(timeout=3) == -15
        assert foreign.poll() is None
    finally:
        for process in (child, foreign):
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=3)


@pytest.mark.parametrize('node_exits,failure,expected', [
    ([0, 0, 0, 0], False, 0),
    ([0, 1, 0, 0], False, 1),
    ([0, None, 0, 0], False, 1),
    ([0, 0, 0, 0], True, 1),
])
def test_mpi_transport_zero_requires_every_node_receipt(tmp_path, node_exits, failure, expected):
    attempt = tmp_path / 'attempt'
    for node_rank, status in enumerate(node_exits):
        directory = attempt / f'nodes/{node_rank}'
        directory.mkdir(parents=True)
        if status is not None:
            (directory / 'exitcode').write_text(f'{status}\n')
    if failure:
        (attempt / 'failure.2').write_text('failed\n')
    mpi = tmp_path / 'transport.py'
    mpi.write_text('raise SystemExit(0)\n')
    original = Path(__file__).resolve().parents[1] / 'scripts/minimax_h3_mpi_attempt.sh'
    script = tmp_path / 'coordinator.sh'
    script.write_text(original.read_text().replace(
        'source /m2v_intern2/zhenglin/my_bashrc.sh', ':').replace(
        '/usr/local/mpi/bin/mpirun', f'{shlex.quote(sys.executable)} {shlex.quote(str(mpi))}'))
    environment = tmp_path / 'launch.env'
    environment.write_text('\n'.join([
        f'H3_ATTEMPT_DIR={shlex.quote(str(attempt))}', 'H3_ATTEMPT_NONCE=cpu-test',
        'H3_SSH_KNOWN_HOSTS=unused', 'H3_CHECKOUT=unused', 'H3_NNODES=4',
    ]) + '\n')
    result = subprocess.run(['bash', str(script), str(environment)], capture_output=True,
                            text=True, timeout=10)
    assert result.returncode == expected, result.stderr
    assert (attempt / 'mpi.exitcode').read_text().strip() == str(expected)


@pytest.mark.parametrize('relative', ['cache', 'fresh-reproduction/kernel-cache'])
def test_kernel_caches_stay_inside_task_root(tmp_path, relative):
    source = Path(__file__).resolve().parents[1] / 'scripts/minimax_h3_runtime_env.sh'
    fragment = source.read_text().split('export NCCL_IB_HCA', 1)[0].replace(
        'source /m2v_intern2/zhenglin/my_bashrc.sh', ':')
    root = tmp_path / 'task'
    root.mkdir()
    cache = root / relative
    environment = dict(os.environ, H3_ACCEPTANCE_ROOT=str(root), H3_CHECKOUT=str(root),
                       H3_ATTEMPT_DIR=str(root / 'attempt'), H3_PYTHON=sys.executable,
                       H3_NODE_RANK='0', H3_KERNEL_CACHE_ROOT=str(cache), BASH_ENV='/dev/null')
    result = subprocess.run(['bash', '-c', fragment + '\nprintf "%s\\n" "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR"'],
                            env=environment, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [str(cache / 'cuda'), str(cache / 'triton')]
    for directory in ['cuda', 'triton', 'torchinductor', 'torch-extensions', 'numba']:
        assert (cache / directory).is_dir()


def test_kernel_cache_rejects_path_outside_task_before_creation(tmp_path):
    source = Path(__file__).resolve().parents[1] / 'scripts/minimax_h3_runtime_env.sh'
    fragment = source.read_text().split('export NCCL_IB_HCA', 1)[0].replace(
        'source /m2v_intern2/zhenglin/my_bashrc.sh', ':')
    root = tmp_path / 'task'
    root.mkdir()
    outside = tmp_path / 'outside'
    environment = dict(os.environ, H3_ACCEPTANCE_ROOT=str(root), H3_CHECKOUT=str(root),
                       H3_ATTEMPT_DIR=str(root / 'attempt'), H3_PYTHON=sys.executable,
                       H3_NODE_RANK='0', H3_KERNEL_CACHE_ROOT=str(root / '..' / 'outside'),
                       BASH_ENV='/dev/null')
    result = subprocess.run(['bash', '-c', fragment], env=environment, capture_output=True,
                            text=True, timeout=10)
    assert result.returncode != 0
    assert not outside.exists()
