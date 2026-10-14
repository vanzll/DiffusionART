import json

import pytest

from visual_rl.config import ConfigError, load_config
from visual_rl.train import main


def test_sd3_fp16_entrypoint_and_flux_precision_guard():
    from visual_rl.registry import get_backend

    c = load_config("configs/examples/sd3_pickscore_fp16.yaml")
    assert c.training.precision == "fp16"
    assert get_backend(c).name == "sd3"
    assert main(["--config", "configs/examples/sd3_pickscore_fp16.yaml", "--validate"]) == 0
    flux = load_config("configs/examples/flux_pickscore.yaml", {"training": {"precision": "fp16"}})
    with pytest.raises(ConfigError, match="precision"):
        get_backend(flux)


def write_config(tmp_path, extra=""):
    p = tmp_path / "experiment.yaml"
    p.write_text(
        """name: test
task: t2i
algorithm: diffusionart
model: sd35
reward:
  name: pickscore
  weights: {pickscore: 1.0}
sampling: {frames: 1}
"""
        + extra
    )
    return p


def test_inheritance_and_override(tmp_path, monkeypatch):
    base = write_config(tmp_path)
    child = tmp_path / "child.yaml"
    child.write_text(
        "extends: experiment.yaml\nname: child\ntraining: {microbatch: 2}\nruntime: {output_dir: '${TEST_OUTPUT}'}\n"
    )
    monkeypatch.setenv("TEST_OUTPUT", str(tmp_path / "outputs"))
    c = load_config(child)
    assert c.training.microbatch == 2
    assert c.training.learning_rate == 1e-5
    assert c.runtime.output_dir == str(tmp_path / "outputs")
    assert load_config(base).name == "test"
    assert load_config(child).fingerprint() == c.fingerprint()
    assert load_config(child, {"algorithm": "diffusionnft"}).algorithm.name == "diffusionnft"


def test_sampler_options_preserve_default_identity_and_separate_noise_seed(tmp_path):
    path = write_config(tmp_path, "data: {manifest: prompts.jsonl}\n")
    original = load_config(path)
    for key in ("sampler_seed", "order_shards", "drop_last"):
        assert key not in original.to_dict()["data"]
    aligned = load_config(path, {"data": {"sampler_seed": 1237, "order_shards": 8,
                                        "drop_last": True}})
    assert aligned.seed == original.seed == 42
    assert aligned.data.sampler_seed == 1237
    assert aligned.to_dict()["data"]["order_shards"] == 8
    assert aligned.fingerprint() != original.fingerprint()
    assert load_config(path, {"data": {"sampler_seed": 0}}).data.sampler_seed == 0


@pytest.mark.parametrize("options", [{"sampler_seed": -1}, {"sampler_seed": True},
                                    {"order_shards": 0}, {"order_shards": True},
                                    {"drop_last": "true"}])
def test_invalid_sampler_options(tmp_path, options):
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, "data: {manifest: prompts.jsonl}\n"), {"data": options})


@pytest.mark.parametrize(
    "extra",
    [
        "training: {learnng_rate: 0.1}\n",
        "training: {microbatch: true}\n",
        "training: {learning_rate: .nan}\n",
        "seed: -1\n",
        "unknown: 1\n",
        "training: {betas: [0.9, 1]}\n",
        "training: {allow_tf32: 1}\n",
        "evaluation: {at_updates: [10, 10]}\n",
        "evaluation: {at_updates: [20, 10]}\n",
        "evaluation: {at_updates: [0]}\n",
        "evaluation: {at_updates: [true]}\n",
        "evaluation: {at_updates: [401]}\n",
        "evaluation: {shift: 0}\n",
        "evaluation: {shift: .nan}\n",
        "runtime: {checkpoint_at_updates: [0]}\n",
        "evaluation: {retain_all: 'false'}\n",
        "runtime: {output_dir: ''}\n",
        "seed: 42\nseed: 43\n",
    ],
)
def test_invalid_config_rejected(tmp_path, extra):
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, extra))


def test_unknown_env_and_recursive_inheritance(tmp_path, monkeypatch):
    monkeypatch.delenv("UNSET_VISUAL_RL_VAR", raising=False)
    with pytest.raises(ConfigError, match="unset"):
        load_config(write_config(tmp_path, "runtime: {output_dir: '${UNSET_VISUAL_RL_VAR}'}\n"))
    p = write_config(tmp_path, "extends: experiment.yaml\n")
    with pytest.raises(ConfigError, match="cyclic"):
        load_config(p)


def test_refl_k1_and_t2i_frames(tmp_path):
    path = write_config(tmp_path)
    with pytest.raises(ConfigError, match="K=1"):
        load_config(path, {"algorithm": "refl"})
    assert load_config(path, {"algorithm": "refl", "sampling": {"samples_per_prompt": 1}})
    with pytest.raises(ConfigError, match="frames=1"):
        load_config(path, {"sampling": {"frames": 53}})


def test_default_checkpoint_cadence():
    from visual_rl.config import Runtime

    runtime = Runtime()
    assert runtime.checkpoint_every == 200
    assert not runtime.checkpoint_due(0, 1000)
    assert not runtime.checkpoint_due(199, 1000)
    assert all(runtime.checkpoint_due(step, 1000) for step in (200, 400, 600, 800, 1000))
    assert runtime.checkpoint_due(350, 350)
    explicit = Runtime(checkpoint_at_updates=(100,))
    assert explicit.checkpoint_due(100, 1000)
    assert explicit.checkpoint_due(200, 1000)


def test_explicit_artifact_cadence_keeps_periodic_and_final_behavior(tmp_path):
    from visual_rl.config import Evaluation, Runtime

    evaluation = Evaluation(every_updates=0, at_updates=[10, 20])
    checkpoint = Runtime(checkpoint_every=0, checkpoint_at_updates=[10, 20])
    for step in (0, 9, 11, 30):
        assert not evaluation.due(step, 40)
        assert not checkpoint.checkpoint_due(step, 40)
    for step in (10, 20, 40):
        assert evaluation.due(step, 40)
        assert checkpoint.checkpoint_due(step, 40)
    assert evaluation.enabled
    assert not Evaluation(every_updates=0).enabled
    assert not Evaluation(every_updates=0).due(40, 40)
    assert not Runtime(checkpoint_every=0).checkpoint_due(40, 40)
    periodic = Evaluation(every_updates=12, at_updates=[20])
    assert all(periodic.due(step, 40) for step in (12, 20, 24, 40))
    c = load_config(write_config(tmp_path), {"evaluation": {"every_updates": 0, "at_updates": [20]},
                    "runtime": {"checkpoint_every": 0, "checkpoint_at_updates": [20]}})
    assert c.evaluation.at_updates == (20,)
    assert c.runtime.checkpoint_at_updates == (20,)


def test_cli_print_does_not_load_or_download_model(tmp_path, capsys):
    path = write_config(tmp_path)
    assert main(["--config", str(path), "--print-config"]) == 0
    assert json.loads(capsys.readouterr().out)["model"]["name"] == "sd35"
    assert main(["--config", str(path), "--validate"]) == 2
    assert "not integrated" in capsys.readouterr().err


def test_component_switch_does_not_inherit_other_algorithm_options(tmp_path):
    parent = write_config(tmp_path)
    text = parent.read_text().replace(
        "algorithm: diffusionart",
        "algorithm:\n  name: diffusionart\n  options: {corrective_mask_enabled: true}",
    )
    parent.write_text(text)
    child = tmp_path / "child.yaml"
    child.write_text("extends: experiment.yaml\nalgorithm: {name: diffusionnft}\n")
    assert load_config(child).algorithm.options == {}
    assert load_config(parent, {"reward": {"weights": {"other": 1.0}}}).reward.weights == {
        "other": 1.0
    }


@pytest.mark.parametrize("keep", [None, 1, 2])
def test_checkpoint_retention_is_operational_and_legacy_compatible(tmp_path, keep):
    import hashlib

    path = write_config(tmp_path)
    c = load_config(path, {"runtime": {"checkpoint_keep": keep}})
    old = c.to_dict()
    old["runtime"].pop("checkpoint_keep")
    digest = hashlib.sha256(json.dumps(old, sort_keys=True, allow_nan=False,
                                       separators=(",", ":")).encode()).hexdigest()
    assert c.fingerprint() == digest == load_config(path).fingerprint()
    assert c.fingerprint() != load_config(path, {"training": {"learning_rate": 2e-5}}).fingerprint()


@pytest.mark.parametrize("keep", [0, -1, True, "2"])
def test_invalid_checkpoint_retention(tmp_path, keep):
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path), {"runtime": {"checkpoint_keep": keep}})


@pytest.mark.parametrize("flag", ["--resume-rewind", "--eval-only"])
def test_resume_modes_require_checkpoint(tmp_path, flag):
    with pytest.raises(SystemExit) as exc:
        main(["--config", str(write_config(tmp_path)), flag])
    assert exc.value.code == 2
