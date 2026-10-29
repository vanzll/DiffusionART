"""Documented asset commands must retain configs as well as model weights."""

import importlib
import re
import shlex
from pathlib import Path

import pytest


@pytest.mark.parametrize("guide, expected", [
    ("sd3-clipscore.md", ["*.json", "*.txt", "*.safetensors"]),
    ("flux-runner.md", ["model_index.json", "scheduler/*", "transformer/*", "vae/*",
                        "text_encoder/*", "text_encoder_2/*", "tokenizer/*", "tokenizer_2/*"]),
])
def test_documented_image_download_patterns_reach_hub_parser(monkeypatch, guide, expected):
    pytest.importorskip("huggingface_hub.cli.download")
    typer = pytest.importorskip("typer")
    from typer.testing import CliRunner

    download = importlib.import_module("huggingface_hub.cli.download")
    document = (Path(__file__).resolve().parents[1] / "docs" / guide).read_text()
    commands = [shlex.split(line) for block in re.findall(r"```bash\n(.*?)```", document, re.S)
                for line in block.replace("\\\n", " ").splitlines() if "hf download " in line]
    assert len(commands) == 1
    command = commands[0]
    arguments = command[command.index("download") + 1:]
    captured = []

    def snapshot(**kwargs):
        captured.append(kwargs)
        return "/unused/network-free-download"

    monkeypatch.setattr(download, "snapshot_download", snapshot)
    app = typer.Typer()
    app.command()(download.download)
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 0, result.output
    assert len(captured) == 1
    assert captured[0]["allow_patterns"] == expected
    assert "Ignoring `--include`" not in result.output
