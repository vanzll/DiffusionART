import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from test_sd3_runner import config_file, payload, threads, tiny_parts  # noqa: F401

from visual_rl.backends import sd3
from visual_rl.config import ConfigError, load_config
from visual_rl.training.image_contracts import tf32_policy


def test_explicit_cadence_native_training_eval_and_resume(tmp_path, monkeypatch):
    c = load_config(config_file(tmp_path))
    c = replace(c, training=replace(c.training, max_updates=6, allow_tf32=True),
                evaluation=replace(c.evaluation, every_updates=0, at_updates=(2, 6)),
                runtime=replace(c.runtime, checkpoint_every=0, checkpoint_at_updates=(2, 6)))
    original_flags = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32

    def parts(config):
        assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
        return tiny_parts(config)

    monkeypatch.setattr(sd3, "load_components", parts)
    sd3.run(c)
    assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == original_flags
    root = Path(c.runtime.output_dir)
    assert sorted(path.name for path in (root / "evaluation").glob("step-*")) == [
        "step-000000000", "step-000000002", "step-000000006"]
    expected = payload(root)
    interrupted = replace(c, runtime=replace(c.runtime, output_dir=str(tmp_path / "interrupted")))
    save = sd3.save_checkpoint

    def stop(*args, **kwargs):
        result = save(*args, **kwargs)
        if kwargs["progress"].optimizer_steps == 2:
            raise RuntimeError("interrupt explicit checkpoint")
        return result

    monkeypatch.setattr(sd3, "save_checkpoint", stop)
    with pytest.raises(RuntimeError, match="interrupt explicit"):
        sd3.run(interrupted)
    monkeypatch.setattr(sd3, "save_checkpoint", save)
    sd3.run(interrupted, resume=Path(interrupted.runtime.output_dir) / "checkpoints/latest.json")
    actual = payload(Path(interrupted.runtime.output_dir))
    for key in expected["model"]:
        torch.testing.assert_close(actual["model"][key], expected["model"][key], rtol=0, atol=0)
    assert torch.equal(actual["generators"]["sampling"], expected["generators"]["sampling"])


def test_unaligned_explicit_steps_fail_native_preflight(tmp_path):
    c = load_config(config_file(tmp_path))
    with pytest.raises(ConfigError, match="align"):
        sd3.validate_config(replace(c, evaluation=replace(c.evaluation, every_updates=0, at_updates=(1,))))
    with pytest.raises(ConfigError, match="align"):
        sd3.validate_config(replace(c, runtime=replace(c.runtime, checkpoint_at_updates=(3,))))


def test_tf32_context_restores_flags_on_failure():
    flags = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    with pytest.raises(RuntimeError):
        with tf32_policy(not flags[0]):
            assert torch.backends.cuda.matmul.allow_tf32 == (not flags[0])
            raise RuntimeError("injected setup failure")
    assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == flags


def test_one_window_art_is_one_full_update_per_rollout(tmp_path, monkeypatch):
    c = load_config(config_file(tmp_path))
    c = replace(c, algorithm=replace(c.algorithm, options={**c.algorithm.options, "updates_per_rollout": 1}),
                training=replace(c.training, max_updates=3),
                evaluation=replace(c.evaluation, every_updates=1),
                runtime=replace(c.runtime, checkpoint_every=1))
    monkeypatch.setattr(sd3, "load_components", tiny_parts)
    sd3.validate_config(c)
    sd3.run(c)
    root = Path(c.runtime.output_dir)
    expected = payload(root)
    metrics = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
    assert [row["train/optimizer_step"] for row in metrics] == [1, 2, 3]
    assert [row["system/completed_rollouts"] for row in metrics] == [1, 2, 3]
    interrupted = replace(c, runtime=replace(c.runtime, output_dir=str(tmp_path / "one-interrupted")))
    save = sd3.save_checkpoint

    def stop(*args, **kwargs):
        result = save(*args, **kwargs)
        if kwargs["progress"].optimizer_steps == 1:
            raise RuntimeError("interrupt first rollout")
        return result

    monkeypatch.setattr(sd3, "save_checkpoint", stop)
    with pytest.raises(RuntimeError, match="first rollout"):
        sd3.run(interrupted)
    monkeypatch.setattr(sd3, "save_checkpoint", save)
    sd3.run(interrupted, resume=Path(interrupted.runtime.output_dir) / "checkpoints/latest.json")
    actual = payload(Path(interrupted.runtime.output_dir))
    for key in expected["model"]:
        torch.testing.assert_close(actual["model"][key], expected["model"][key], rtol=0, atol=0)
    assert torch.equal(actual["generators"]["sampling"], expected["generators"]["sampling"])
