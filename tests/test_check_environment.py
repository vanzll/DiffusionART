import subprocess

import pytest

from visual_rl import check_environment as env


@pytest.mark.parametrize("profile", ["hunyuan", "wan"])
def test_exact_versions_allow_cuda_wheel_suffix(profile):
    observed = {**env.CORE, **env.PROFILES[profile]}
    observed["torch"] += "+cu124"
    assert env.check_versions(profile, version=observed.__getitem__) == observed
    observed["transformers"] = "5.17.0"
    with pytest.raises(RuntimeError, match="transformers"):
        env.check_versions(profile, version=observed.__getitem__)


def test_cpu_probe_hides_cuda_and_is_bounded(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(env.subprocess, "run", run)
    assert env.main(["--profile", "wan", "--native-root", "/native", "--timeout-seconds", "3"]) == 124
    command, options = calls[0]
    assert "--worker" in command and "--cuda" not in command
    assert options["timeout"] == 3
    assert options["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert options["env"]["HF_HUB_OFFLINE"] == "1"


def test_cuda_probe_requires_explicit_flag_and_propagates_failure(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")

    def run(command, **kwargs):
        assert "--cuda" in command
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "2"
        return subprocess.CompletedProcess(command, 7)

    monkeypatch.setattr(env.subprocess, "run", run)
    assert env.main(["--profile", "hunyuan", "--native-root", "/native", "--cuda"]) == 7


@pytest.mark.parametrize("timeout", ["0", "-1", "nan", "inf"])
def test_bad_deadline_rejected(timeout):
    with pytest.raises(SystemExit):
        env.main(["--profile", "wan", "--native-root", "/native", "--timeout-seconds", timeout])


def test_requirements_match_version_check():
    from pathlib import Path

    from packaging.requirements import Requirement

    directory = Path(__file__).resolve().parents[1] / "environments"
    for profile in env.PROFILES:
        pins = {}
        for source in (directory / "video-common.txt", directory / f"{profile}.txt"):
            for line in source.read_text().splitlines():
                if line and not line.startswith(("#", "-r")):
                    requirement = Requirement(line)
                    pins[requirement.name.lower()] = str(requirement.specifier)
        for name, version in {**env.CORE, **env.PROFILES[profile]}.items():
            if name != "flash-attn":
                assert pins[name] == f"=={version}"
