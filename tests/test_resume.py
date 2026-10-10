import json
from pathlib import Path

import pytest

from visual_rl.config import ConfigError, load_config
from visual_rl.training.resume import prepare_resume_history, validate_resume_request
from visual_rl.training.state import Progress

EXAMPLE = Path(__file__).resolve().parents[1] / "configs/examples/cpu_smoke.yaml"


def setup_history(tmp_path):
    c = load_config(EXAMPLE, {"runtime": {"output_dir": str(tmp_path)}})
    rows = [{"train/optimizer_step": i, "cost/elapsed_seconds": i * 2.,
             "cost/training_seconds": float(i)} for i in range(1, 5)]
    (tmp_path / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    later = Progress(attempts=2, completed_rollouts=2, optimizer_steps=4,
                     elapsed_seconds=9, training_seconds=4)
    (tmp_path / "cost.json").write_text(json.dumps({"config_sha256": c.fingerprint(),
                                                  "progress": later.state_dict()}))
    (tmp_path / "status.json").write_text(json.dumps({"config_sha256": c.fingerprint(),
                                                    **later.state_dict(), "status": "completed"}))
    for step in (0, 2, 4):
        directory = tmp_path / "evaluation" / f"step-{step:09d}" / "identity"
        directory.mkdir(parents=True)
        (directory / "video.mp4").write_bytes(b"original media")
    (tmp_path / "evaluation/latest.json").write_text(json.dumps({"optimizer_step": 4}))
    progress = Progress(attempts=1, completed_rollouts=1, optimizer_steps=2,
                        elapsed_seconds=4, training_seconds=2)
    return c, progress


def test_explicit_rewind_preserves_evidence_and_actual_cost(tmp_path):
    c, progress = setup_history(tmp_path)
    original = (tmp_path / "metrics.jsonl").read_bytes()
    with pytest.raises(ConfigError, match="--resume-rewind"):
        prepare_resume_history(c, tmp_path, progress, receipt="checkpoint.json")
    assert (tmp_path / "metrics.jsonl").read_bytes() == original
    assert not (tmp_path / "resume-history").exists()
    prepare_resume_history(c, tmp_path, progress, receipt="checkpoint.json", rewind=True)
    archive, = (tmp_path / "resume-history").iterdir()
    assert (archive / "metrics.jsonl").read_bytes() == original
    assert [json.loads(row)["train/optimizer_step"] for row in
            (tmp_path / "metrics.jsonl").read_text().splitlines()] == [1, 2]
    assert (archive / "evaluation/step-000000004/identity/video.mp4").read_bytes() == b"original media"
    assert (tmp_path / "evaluation/step-000000002/identity/video.mp4").is_file()
    assert not (tmp_path / "evaluation/latest.json").exists()
    assert not (tmp_path / "status.json").exists()
    assert progress.optimizer_steps == 2 and progress.elapsed_seconds == 9 and progress.training_seconds == 4
    assert json.loads((archive / "resume.json").read_text())["state"] == "completed"
    prepare_resume_history(c, tmp_path, progress, receipt="checkpoint.json", rewind=True)
    assert len(list((tmp_path / "resume-history").iterdir())) == 1


def test_rewind_can_retry_interrupted_publication(tmp_path, monkeypatch):
    from visual_rl.training import resume

    c, progress = setup_history(tmp_path)
    atomic = resume._atomic_json

    def fail_cost(path, record):
        if path == tmp_path / "cost.json":
            raise OSError("injected disk failure")
        atomic(path, record)

    monkeypatch.setattr(resume, "_atomic_json", fail_cost)
    with pytest.raises(OSError, match="injected"):
        prepare_resume_history(c, tmp_path, progress, receipt="checkpoint.json", rewind=True)
    archive, = (tmp_path / "resume-history").iterdir()
    assert json.loads((archive / "resume.json").read_text())["state"] == "prepared"
    assert (archive / "evaluation/step-000000004/identity/video.mp4").exists()
    monkeypatch.setattr(resume, "_atomic_json", atomic)
    prepare_resume_history(c, tmp_path, progress, receipt="checkpoint.json", rewind=True)
    assert progress.elapsed_seconds == 9 and progress.optimizer_steps == 2
    assert len(list((tmp_path / "resume-history").iterdir())) == 1
    assert not (tmp_path / ".resume-rewind.json").exists()


def test_interrupted_rewind_restores_cost_from_durable_journal(tmp_path, monkeypatch):
    from visual_rl.training import resume

    c, progress = setup_history(tmp_path)
    (tmp_path / "cost.json").unlink()
    (tmp_path / "status.json").unlink()
    fresh = Progress.from_state_dict(progress.state_dict())
    atomic = resume._atomic_json

    def fail_cost(path, record):
        if path == tmp_path / "cost.json":
            raise OSError("cost publication failed")
        atomic(path, record)

    monkeypatch.setattr(resume, "_atomic_json", fail_cost)
    with pytest.raises(OSError):
        prepare_resume_history(c, tmp_path, progress, receipt="checkpoint.json", rewind=True)
    monkeypatch.setattr(resume, "_atomic_json", atomic)
    prepare_resume_history(c, tmp_path, fresh, receipt="checkpoint.json", rewind=True)
    assert fresh.elapsed_seconds == 8 and fresh.training_seconds == 4
    assert fresh.optimizer_steps == 2


def test_same_step_skipped_attempt_is_archived(tmp_path):
    c = load_config(EXAMPLE, {"runtime": {"output_dir": str(tmp_path)}})
    progress = Progress(attempts=1, completed_rollouts=1, optimizer_steps=2)
    rows = [{"train/optimizer_step": 2, "system/rollout_attempts": attempt,
             "system/completed_rollouts": 1} for attempt in (1, 2)]
    original = "".join(json.dumps(row) + "\n" for row in rows)
    (tmp_path / "metrics.jsonl").write_text(original)
    prepare_resume_history(c, tmp_path, progress, receipt="checkpoint.json", rewind=True)
    kept = [json.loads(row) for row in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert len(kept) == 1 and kept[0]["system/rollout_attempts"] == 1
    archive, = (tmp_path / "resume-history").iterdir()
    assert (archive / "metrics.jsonl").read_text() == original


def test_rewind_rejects_foreign_metadata_without_changing_history(tmp_path):
    c, progress = setup_history(tmp_path)
    (tmp_path / "cost.json").write_text('{"config_sha256": "foreign"}')
    before = (tmp_path / "metrics.jsonl").read_bytes()
    with pytest.raises(ConfigError, match="different run"):
        prepare_resume_history(c, tmp_path, progress, receipt="checkpoint.json", rewind=True)
    assert before == (tmp_path / "metrics.jsonl").read_bytes()


def test_rewind_archives_partial_last_line_but_rejects_corrupt_complete_rows(tmp_path):
    c, progress = setup_history(tmp_path)
    with (tmp_path / "metrics.jsonl").open("ab") as file:
        file.write(b'{"train/optimizer_step":')
    prepare_resume_history(c, tmp_path, progress, receipt="checkpoint.json", rewind=True)
    with (tmp_path / "metrics.jsonl").open("ab") as file:
        file.write(b"not JSON\n")
    with pytest.raises(ConfigError, match="invalid metrics"):
        prepare_resume_history(c, tmp_path, progress, receipt="checkpoint.json", rewind=True)


def test_resume_flags_are_unambiguous():
    with pytest.raises(ConfigError, match="explicit checkpoint"):
        validate_resume_request(None, True, False)
    with pytest.raises(ConfigError, match="must not rewind"):
        validate_resume_request("receipt.json", True, True)
    validate_resume_request("receipt.json", False, True)


def test_rewind_completes_unterminated_valid_json_row(tmp_path):
    c = load_config(EXAMPLE, {"runtime": {"output_dir": str(tmp_path)}})
    progress = Progress(optimizer_steps=2)
    row = json.dumps({"train/optimizer_step": 2})
    (tmp_path / "metrics.jsonl").write_text(row)
    prepare_resume_history(c, tmp_path, progress, receipt="checkpoint.json", rewind=True)
    assert (tmp_path / "metrics.jsonl").read_text() == row + "\n"
