"""Owned one-shot CPU workers; no model assets, CUDA or reward imports in child."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

from visual_rl.rewards import small_video_process as process

FAKE_WORKER = r"""
import argparse,json,os,sys,time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--input');p.add_argument('--output');p.add_argument('--parent-pid');p.add_argument('--bashrc')
a=p.parse_args(); request=json.loads(Path(a.input).read_text()); mode=os.environ.get('FAKE_REWARD_MODE','valid')
if mode=='sleep': time.sleep(10)
if mode=='exit': sys.exit(3)
if mode=='verbose': sys.stderr.write('x'*(8*1024*1024+1024))
sys.stderr.write('native worker stderr marker\n')
fields=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
identity={'pid':os.getpid(),'ppid':os.getppid(),'pgid':os.getpgrp(),'start_ticks':int(fields[19]),'visible_device':os.environ.get('CUDA_VISIBLE_DEVICES')}
rows=[{'index':i,'VQ':i+1.,'MQ':2.*(i+1),'TA':-(i+1.)} for i in range(len(request['paths']))]
if mode=='nonfinite': rows[0]['MQ']=float('nan')
if mode=='missing': rows[0].pop('TA')
if mode=='reordered': rows=list(reversed(rows))
if mode=='short': rows=rows[:-1]
if mode=='wrong_device': identity['visible_device']='GPU-foreign'
norm={k:{'mean':.5,'std':2.,'source':'VideoReward@'+request['revision']} for k in ['VQ','MQ','TA']}
if mode=='badnorm': norm['VQ']['std']=0
runtime={'native_source_revision':request['source_revision'],'reward_revision':request['revision'],'transformers':'4.46.1','python':sys.version.split()[0],'torch':'fake-cpu','flash-attn':'fake-cpu'}
if mode=='badenv': runtime['transformers']='4.51.3'
prefixes=('TORCHELASTIC_','ACCELERATE_','TORCH_DISTRIBUTED_','PMI_','PMIX_','OMPI_COMM_WORLD_','MV2_','MPI_')
bootstrap={'RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT','GROUP_RANK','GROUP_WORLD_SIZE','ROLE_RANK','ROLE_WORLD_SIZE','ROLE_NAME','NODE_RANK'}
output={'schema_version':1,'request_id':request['request_id'],'rows':rows,'normalization':norm,'runtime':runtime,'process':identity,
'metrics':{'reward/load_seconds':0.01,'reward/scoring_seconds':0.02,'reward/worker_total_seconds':0.03,'memory/scorer_peak_allocated_bytes':0,'memory/scorer_peak_reserved_bytes':0},
'observed':{'distributed_keys':[k for k in os.environ if k in bootstrap or k.startswith(prefixes)],'hf_offline':os.environ.get('HF_HUB_OFFLINE'),'pythonpath':os.environ.get('PYTHONPATH')}}
Path(a.output).write_text(json.dumps(output))
"""


@pytest.fixture
def worker(tmp_path, monkeypatch):
    if not Path("/proc/self/stat").is_file():
        pytest.skip("Linux process ownership test")
    script = tmp_path / "fake_worker.py"
    script.write_text(FAKE_WORKER)
    monkeypatch.setattr(process, "WORKER_SCRIPT", script)
    directories = [tmp_path / name for name in ("native", "weights", "backbone")]
    for directory in directories:
        directory.mkdir()
    paths = [tmp_path / f"video-{index}.mp4" for index in range(2)]
    for path in paths:
        path.write_bytes(b"fake video; never decoded by the CPU transport fixture")

    def build(**kwargs):
        revision = kwargs.pop("revision", process.REWARD_REVISION)
        return process.PhasedVideoReward(
            python=sys.executable,
            native_root=directories[0],
            checkpoint=directories[1],
            backbone=directories[2],
            revision=revision,
            video_fps=16.0,
            scratch=tmp_path / "phases",
            **kwargs,
        )

    return build, paths


def test_one_shot_ordered_raw_scores_waited_exit_and_safe_process_audit(worker, monkeypatch):
    build, paths = worker
    monkeypatch.setenv("RANK", "7")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_RANK", "7")
    monkeypatch.setenv("MASTER_ADDR", "private-bootstrap-host")
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "owned-test")
    monkeypatch.setenv("ACCELERATE_USE_DEEPSPEED", "true")
    monkeypatch.setenv("PMI_RANK", "7")
    monkeypatch.setenv("MV2_COMM_WORLD_RANK", "7")
    scorer = build().to("cpu").eval()
    result = scorer.score_paths(paths, ["scene one", "scene two"])
    torch.testing.assert_close(result.raw["VQ"], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(result.raw["MQ"], torch.tensor([2.0, 4.0]))
    torch.testing.assert_close(result.raw["TA"], torch.tensor([-1.0, -2.0]))
    torch.testing.assert_close(result.weighted, torch.tensor([1.5, 3.0]))
    assert result.space == "raw" and dict(result.weights) == {"VQ": 0.5, "MQ": 0.5, "TA": 0.0}
    audit = json.loads((scorer.last_phase / "audit.json").read_text())
    reply = json.loads((scorer.last_phase / "response.json").read_text())
    assert audit["status"] == "completed" and audit["exit_code"] == 0
    assert audit["child"]["ppid"] == os.getpid() and audit["child"]["pgid"] == os.getpgrp()
    assert audit["child"]["start_ticks"] == reply["process"]["start_ticks"]
    assert audit["visible_device"] == "" and reply["observed"]["distributed_keys"] == []
    assert reply["observed"]["hf_offline"] == "1"
    assert not Path(f"/proc/{audit['child']['pid']}").exists()
    assert "native worker stderr marker" in (scorer.last_phase / "worker.log").read_text()
    assert scorer.last_metrics["reward/load_seconds"] == 0.01
    assert scorer.last_metrics["memory/scorer_peak_allocated_bytes"] == 0
    first = scorer.last_phase
    scorer.score_paths(paths, ["scene one", "scene two"])
    assert scorer.last_phase != first and (first / "audit.json").exists()


@pytest.mark.parametrize(
    "visible,index,expected",
    [
        ("7,3,5", 1, "3"),
        ("GPU-one,GPU-two", 1, "GPU-two"),
        ("MIG-GPU-abc/1/0,MIG-GPU-def/2/0", 0, "MIG-GPU-abc/1/0"),
    ],
)
def test_physical_gpu_and_uuid_mapping_without_cuda_initialization(
    tmp_path, visible, index, expected
):
    env = {
        "CUDA_VISIBLE_DEVICES": visible,
        "LOCAL_RANK": str(index),
        "WORLD_SIZE": "8",
        "TORCHELASTIC_RUN_ID": "private",
        "MPI_LOCALRANKID": "7",
        "PATH": os.environ["PATH"],
    }
    mapped = process.physical_visible_device(f"cuda:{index}", environ=env)
    assert mapped == expected
    child = process.worker_environment(visible_device=mapped, phase=tmp_path, environ=env)
    assert child["CUDA_VISIBLE_DEVICES"] == expected
    assert (
        "LOCAL_RANK" not in child
        and "WORLD_SIZE" not in child
        and "TORCHELASTIC_RUN_ID" not in child
    )
    assert "MPI_LOCALRANKID" not in child
    assert process.physical_visible_device("cpu", environ=env) == ""
    assert process.physical_visible_device("cuda:2", environ={}) == "2"
    with pytest.raises(ValueError):
        process.physical_visible_device("cuda:4", environ=env)


@pytest.mark.parametrize(
    "mode",
    ["nonfinite", "missing", "reordered", "short", "wrong_device", "badnorm", "badenv", "exit"],
)
def test_missing_nonfinite_reordered_or_wrong_identity_fail_closed(worker, monkeypatch, mode):
    build, paths = worker
    monkeypatch.setenv("FAKE_REWARD_MODE", mode)
    scorer = build()
    with pytest.raises(process.RewardProcessError, match="inspect its owned phase audit"):
        scorer.score_paths(paths, ["first", "second"])
    audit = json.loads((scorer.last_phase / "audit.json").read_text())
    assert audit["status"] == "failed" and not Path(f"/proc/{audit['child']['pid']}").exists()


def test_timeout_kills_only_owned_child_and_reaps_before_return(worker, monkeypatch):
    build, paths = worker
    monkeypatch.setenv("FAKE_REWARD_MODE", "sleep")
    scorer = build(timeout=0.1)
    parent_pid, parent_pgid = os.getpid(), os.getpgrp()
    start = time.monotonic()
    with pytest.raises(process.RewardProcessError):
        scorer.score_paths(paths, ["first", "second"])
    assert time.monotonic() - start < 8
    audit = json.loads((scorer.last_phase / "audit.json").read_text())
    assert audit["status"] == "failed" and audit["error_type"] == "TimeoutExpired"
    assert audit["exit_code"] is not None and not Path(f"/proc/{audit['child']['pid']}").exists()
    assert os.getpid() == parent_pid and os.getpgrp() == parent_pgid


def test_worker_log_is_bounded_while_pipe_is_fully_drained(worker, monkeypatch):
    build, paths = worker
    monkeypatch.setenv("FAKE_REWARD_MODE", "verbose")
    scorer = build()
    scorer.score_paths(paths, ["first", "second"])
    audit = json.loads((scorer.last_phase / "audit.json").read_text())
    assert audit["status"] == "completed" and audit["log_truncated"] is True
    assert (scorer.last_phase / "worker.log").stat().st_size <= process.LOG_LIMIT


def test_invalid_counts_revision_fps_and_json_size_rejected_before_spawn(worker, monkeypatch):
    build, paths = worker
    scorer = build()
    with pytest.raises(ValueError):
        scorer.score_paths(paths, ["one"])
    with pytest.raises(ValueError):
        scorer.score_paths(paths, ["", "two"])
    with pytest.raises(ValueError):
        build(reward_fps=8.0)
    with pytest.raises(ValueError):
        build(revision="main")
    with pytest.raises(ValueError):
        scorer.score_paths(paths, ["a" * process.JSON_LIMIT, "two"])
    assert not list(scorer.scratch.glob("phase-*/audit.json"))


def test_real_worker_bootstrap_rejects_bad_contract_before_model_import(tmp_path):
    if not sys.platform.startswith("linux"):
        pytest.skip("Linux parent-death bootstrap")
    request, response = tmp_path / "request.json", tmp_path / "response.json"
    request.write_text(json.dumps({"schema_version": 1, "revision": "unapproved"}))
    script = Path(process.__file__).with_name("small_video_reward_worker.py")
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--input",
            str(request),
            "--output",
            str(response),
            "--parent-pid",
            str(os.getpid()),
        ],
        capture_output=True,
        text=True,
        timeout=10,
        env=process.worker_environment(visible_device="", phase=tmp_path),
    )
    assert result.returncode == 1 and not response.exists()
    assert "canonical worker request contract differs" in result.stderr
