import json
from pathlib import Path

import pytest

from visual_rl.backends.tensor_fixture import run
from visual_rl.config import load_config
from visual_rl.train import main

EXAMPLE = Path(__file__).resolve().parents[1] / "configs/examples/cpu_smoke.yaml"


@pytest.mark.parametrize(
    "algorithm",
    [
        "diffusionart",
        "diffusionnft",
        "flow_grpo",
        "flow_grpo_fast",
        "dancegrpo",
        "flow_dppo",
        "refl",
    ],
)
def test_every_algorithm_runs_via_same_configuration(tmp_path, algorithm):
    overrides = {"algorithm": algorithm, "runtime": {"output_dir": str(tmp_path / "run")}}
    if algorithm == "refl":
        overrides["sampling"] = {"samples_per_prompt": 1}
    config = load_config(EXAMPLE, overrides)
    run(config)
    root = Path(config.runtime.output_dir)
    metrics = [json.loads(row) for row in (root / "metrics.jsonl").read_text().splitlines()]
    assert [row["train/optimizer_step"] for row in metrics] == [1, 2, 3, 4]
    assert all(row["cost/elapsed_gpu_hours"] == 0 for row in metrics)
    assert all(row["cost/training_seconds"] <= row["cost/elapsed_seconds"] for row in metrics)
    assert json.loads((root / "status.json").read_text())["status"] == "completed"
    with pytest.raises(FileExistsError):
        run(config)
    run(config, resume=root / "checkpoints/latest.json")
    assert len((root / "metrics.jsonl").read_text().splitlines()) == 4


def test_readme_commands(tmp_path):
    assert main(["--config", str(EXAMPLE), "--validate"]) == 0
    assert main(["--config", str(EXAMPLE), "--output-dir", str(tmp_path / "readme")]) == 0


def test_preflight_rejects_backend_specific_options_without_writing(tmp_path):
    from visual_rl.config import ConfigError
    from visual_rl.registry import get_backend

    config = load_config(
        EXAMPLE,
        {
            "runtime": {"output_dir": str(tmp_path / "not-created")},
            "algorithm": {"name": "diffusionart", "options": {"typo": 1}},
        },
    )
    with pytest.raises(ConfigError):
        get_backend(config)
    assert not (tmp_path / "not-created").exists()
