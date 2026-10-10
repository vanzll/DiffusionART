import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from visual_rl.config import ConfigError, load_config, resume_signature
from visual_rl.train import main
from visual_rl.training.no_signal import NoSignalGuard, NoSignalRolloutLimit


def test_unlimited_skips_warn_at_exponentially_spaced_streaks(caplog):
    guard = NoSignalGuard()
    for _ in range(1024):
        guard.skipped()
    assert guard.consecutive == 1024
    assert [record.args[0] for record in caplog.records] == [2**i for i in range(11)]
    guard.updated()
    assert guard.consecutive == 0
    guard.skipped()
    assert caplog.records[-1].args[0] == 1


@pytest.mark.parametrize("maximum", [1, 2, 5, 9])
def test_exact_limit_and_update_reset(maximum, caplog):
    guard = NoSignalGuard(maximum)
    for _ in range(maximum - 1):
        guard.skipped(warn=False)
    guard.updated()
    for _ in range(maximum - 1):
        guard.skipped(warn=False)
    with pytest.raises(NoSignalRolloutLimit, match=f"{maximum} consecutive no-signal"):
        guard.skipped(warn=False)
    assert guard.consecutive == maximum
    assert not caplog.records


@pytest.mark.parametrize("maximum", [None, 1, 5])
def test_runtime_budget_keeps_legacy_config_and_checkpoint_identity(maximum):
    from test_infra_entrypoint import EXAMPLE

    baseline = load_config(EXAMPLE)
    changed = replace(baseline, runtime=replace(
        baseline.runtime, max_consecutive_no_signal_rollouts=maximum))
    legacy = baseline.to_dict()
    legacy["runtime"].pop("max_consecutive_no_signal_rollouts", None)
    legacy["runtime"].pop("checkpoint_keep", None)
    digest = hashlib.sha256(json.dumps(legacy, sort_keys=True, allow_nan=False,
                                      separators=(",", ":")).encode()).hexdigest()
    assert baseline.runtime.max_consecutive_no_signal_rollouts is None
    assert baseline.fingerprint() == changed.fingerprint() == digest
    assert resume_signature(legacy) == resume_signature(changed.to_dict())
    assert (baseline.execution_fingerprint() == changed.execution_fingerprint()) is (maximum is None)


@pytest.mark.parametrize("maximum", [0, -1, True, False, "2", 2.5])
def test_invalid_runtime_budget_rejected_at_entrypoint(tmp_path, capsys, maximum):
    import yaml
    from test_infra_entrypoint import EXAMPLE

    values = load_config(EXAMPLE).to_dict()
    values["runtime"]["max_consecutive_no_signal_rollouts"] = maximum
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(values))
    with pytest.raises(ConfigError, match="max_consecutive_no_signal_rollouts"):
        load_config(path)
    assert main(["--config", str(path), "--print-config"]) == 2
    assert "max_consecutive_no_signal_rollouts" in capsys.readouterr().err
    assert not (tmp_path / "run").exists()


def test_entrypoint_reports_only_the_expected_liveness_stop(monkeypatch, capsys):
    from test_infra_entrypoint import EXAMPLE

    def stop(*args, **kwargs):
        NoSignalGuard(1).skipped()

    monkeypatch.setattr("visual_rl.train.get_backend", lambda c: SimpleNamespace(load=lambda: stop))
    assert main(["--config", str(EXAMPLE)]) == 1
    assert "training stopped: DiffusionART stopped after 1 consecutive no-signal" in capsys.readouterr().err

    def unexpected(*args, **kwargs):
        raise RuntimeError("unrelated failure")

    monkeypatch.setattr("visual_rl.train.get_backend", lambda c: SimpleNamespace(load=lambda: unexpected))
    with pytest.raises(RuntimeError, match="unrelated failure"):
        main(["--config", str(EXAMPLE)])


@pytest.mark.parametrize("maximum", [None, 1, 7])
def test_legacy_checkpoint_resumes_with_changed_liveness_budget(tmp_path, maximum):
    from test_infra_entrypoint import EXAMPLE

    from visual_rl.backends.tensor_fixture import run
    from visual_rl.training.resume import prepare_resume_config

    c = load_config(EXAMPLE, {"runtime": {"output_dir": str(tmp_path / "run")}})
    run(c)
    original = (tmp_path / "run/resolved-config.json").read_bytes()
    assert "max_consecutive_no_signal_rollouts" not in json.loads(original)["runtime"]
    changed = replace(c, training=replace(c.training, max_updates=6), runtime=replace(
        c.runtime, max_consecutive_no_signal_rollouts=maximum))
    bound, _ = prepare_resume_config(changed, "latest")
    assert bound.fingerprint() == c.fingerprint()
    run(changed, resume="latest")
    assert (tmp_path / "run/resolved-config.json").read_bytes() == original
    assert json.loads((tmp_path / "run/status.json").read_text())["optimizer_steps"] == 6
    active = json.loads((tmp_path / "run/resume-config.json").read_text())
    assert active["runtime"].get("max_consecutive_no_signal_rollouts") == maximum
