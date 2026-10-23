import importlib.util
import os
from pathlib import Path
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
