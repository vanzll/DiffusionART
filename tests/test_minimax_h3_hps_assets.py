import pytest


def test_hps_bundle_uses_its_own_base_not_videoalign_base(tmp_path, monkeypatch):
    from visual_rl.rewards._native_av import hpsv3_reward

    bundle = tmp_path / "hps"
    base = bundle / "base_model"
    base.mkdir(parents=True)
    (base / "config.json").write_text('{"hidden_size":3584}')
    checkpoint = bundle / "HPSv3.safetensors"
    checkpoint.write_bytes(b"unit-test-only")
    captured = {}

    def inferencer(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(hpsv3_reward, "_HPSv3Inferencer", inferencer)
    hpsv3_reward.HPSv3Model(str(bundle), "cpu", base_model_path="videoalign-2b")
    assert captured["checkpoint_path"] == str(checkpoint)
    assert captured["base_config"] == str(base)
    assert captured["device"] == "cpu"


def test_hps_bundle_rejects_missing_base_before_model_allocation(tmp_path, monkeypatch):
    from visual_rl.rewards._native_av import hpsv3_reward

    bundle = tmp_path / "hps"
    bundle.mkdir()
    (bundle / "HPSv3.safetensors").write_bytes(b"unit-test-only")

    def unexpected(**kwargs):
        raise AssertionError("Invalid assets must fail before constructing a model")

    monkeypatch.setattr(hpsv3_reward, "_HPSv3Inferencer", unexpected)
    with pytest.raises(ValueError, match="base_model/config.json"):
        hpsv3_reward.HPSv3Model(str(bundle), "cpu")


def test_hps_single_checkpoint_preserves_explicit_legacy_base(monkeypatch):
    from visual_rl.rewards._native_av import hpsv3_reward

    captured = {}

    def inferencer(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(hpsv3_reward, "_HPSv3Inferencer", inferencer)
    hpsv3_reward.HPSv3Model("checkpoint.safetensors", "cpu", base_model_path="matching-7b")
    assert captured["checkpoint_path"] == "checkpoint.safetensors"
    assert captured["base_config"] == "matching-7b"
