import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from visual_rl.rewards.inputs import ContinuousRGBInput
from visual_rl.rewards.video_reward import (
    VideoReward,
    pack_video,
    uniform_indices,
    verify_video_fps,
)


class Tokenizer:
    def convert_tokens_to_ids(self, token):
        return 7

    def __call__(self, texts, **kwargs):
        ids = torch.full((1, texts[0].count("<|video_pad|>")), 7)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class Scorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1., 2., 3.]))
        self.config = SimpleNamespace(video_token_id=7)

    def forward(self, pixel_values_videos, **kwargs):
        return {"logits": pixel_values_videos.mean().reshape(1, 1) * self.weight[None]}


def inferencer():
    ip = SimpleNamespace(patch_size=14, temporal_patch_size=2, merge_size=2,
        do_rescale=True, rescale_factor=1 / 255., do_normalize=True,
        image_mean=[.5] * 3, image_std=[.25] * 3, do_resize=False)
    return SimpleNamespace(model=Scorer(), processor=SimpleNamespace(image_processor=ip,
        tokenizer=Tokenizer(), apply_chat_template=lambda *args, **kwargs: "<|video_pad|> text"),
        data_config=SimpleNamespace(sample_type="uniform", num_frames=None, max_frame_pixels=200704,
                                    eval_dim="all", prompt_template_type="default"),
        inference_config={**{f"{key}_mean": 1. for key in ("VQ", "MQ", "TA")},
                          **{f"{key}_std": 2. for key in ("VQ", "MQ", "TA")}})


def scorer(native=None):
    return VideoReward(native or inferencer(), lambda prompt, *args: prompt,
                       weights={"VQ": .5, "MQ": .5, "TA": 0.}, normalization_source="fixture")


def test_video_reward_differentiable_rgb_and_frozen_weights():
    model = scorer().train()
    assert not model.model.training
    value = torch.full((1, 4, 3, 28, 28), .75, requires_grad=True)
    result = model(ContinuousRGBInput(value, ["scene"], 8., "zero_one"))
    torch.testing.assert_close(result.raw["VQ"], torch.tensor([1.]))
    torch.testing.assert_close(result.weighted, torch.tensor([.25]))
    result.weighted.sum().backward()
    assert torch.isfinite(value.grad).all() and value.grad.sum() > 0
    assert model.model.weight.grad is None and not model.model.weight.requires_grad
    other = (value.detach() * 2 - 1).requires_grad_()
    result2 = model(ContinuousRGBInput(other, ["scene"], 8., "minus_one_one"))
    torch.testing.assert_close(result2.weighted, result.weighted)
    with pytest.raises(ValueError, match="FPS"):
        model(ContinuousRGBInput(value, ["scene"], 18., "zero_one"))


def test_video_reward_reporting_raw_then_normalize_once(tmp_path, monkeypatch):
    native = inferencer()
    calls = []

    def reward(paths, prompts, **kwargs):
        calls.append((paths, prompts, kwargs))
        return [{"VQ": 1., "MQ": 2., "TA": 3., "Overall": 6.}]

    native.reward = reward
    model = scorer(native)
    path = tmp_path / "sample.mp4"
    path.touch()
    monkeypatch.setattr("visual_rl.rewards.video_reward.subprocess.run", lambda *args, **kwargs:
                        SimpleNamespace(stdout=json.dumps({"streams": [{"avg_frame_rate": "8/1"}]})))
    result = model.score_paths([path], ["scene"])
    assert calls[0][2] == {"fps": 4., "max_pixels": 200704, "use_norm": False}
    assert result.weighted.item() == .25 and result.raw["MQ"].item() == 2.
    assert not result.weighted.requires_grad
    monkeypatch.setattr("visual_rl.rewards.video_reward.subprocess.run", lambda *args, **kwargs:
                        SimpleNamespace(stdout=json.dumps({"streams": [{"avg_frame_rate": "18/1"}]})))
    with pytest.raises(ValueError, match="differs"):
        model.score_paths([path], ["scene"])
    assert len(calls) == 1


def test_qwen_patch_order_matches_native_processor():
    transformers = pytest.importorskip("transformers")

    value = torch.arange(4 * 3 * 56 * 84, dtype=torch.float32).reshape(4, 3, 56, 84) / 100000
    patches, grid = pack_video(value)
    options = dict(do_resize=False, do_rescale=False, do_normalize=False, temporal_patch_size=2)
    if hasattr(transformers, "Qwen2VLVideoProcessor"):
        processor = transformers.Qwen2VLVideoProcessor(**options, do_sample_frames=False)
        expected = processor(videos=[value], return_tensors="pt")
    else:
        processor = transformers.Qwen2VLImageProcessor(**options)
        expected = processor(images=None, videos=[list(value)], return_tensors="pt")
    torch.testing.assert_close(patches, expected["pixel_values_videos"], rtol=0, atol=0)
    assert expected["video_grid_thw"].tolist() == [list(grid)]


def test_uniform_sampling_and_missing_stream(tmp_path, monkeypatch):
    indices = uniform_indices(53, 8., 4.)
    assert len(indices) == 26 and indices[0] == 0 and indices[-1] == 52
    path = tmp_path / "header.mp4"
    path.touch()
    monkeypatch.setattr("visual_rl.rewards.video_reward.subprocess.run", lambda *args, **kwargs:
                        SimpleNamespace(stdout='{"streams": []}'))
    with pytest.raises(ValueError, match="stream"):
        verify_video_fps(path, 8.)


def test_reject_fixed_frame_count_before_scoring():
    native = inferencer()
    native.data_config.num_frames = 2
    with pytest.raises(ValueError, match="num_frames=None"):
        scorer(native)
