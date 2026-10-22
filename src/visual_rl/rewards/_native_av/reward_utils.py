# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Standalone payload boundary for the pinned native AV scorers.

Audio normalization follows verl-omni ca61e80; engine batch objects and service
media registries are deliberately not part of this package's interface.
"""

import math
from numbers import Real

import torch


class ArtifactContractError(ValueError):
    pass


def get_device_name():
    return "cuda" if torch.cuda.is_available() else "cpu"


def audio_info_from_batch(extra_info, batch, *, scorer):
    if batch is not None:
        raise ValueError(f"{scorer} expects an explicit standalone media payload")
    return dict(extra_info or {})


def get_audio(extra_info, *, audio_layout=None):
    if "media_artifacts" in extra_info:
        raise ArtifactContractError("pass decoded audio with explicit layout/rate")
    audio = torch.as_tensor(extra_info["audio"]).detach().float().cpu()
    layout = extra_info.get("audio_layout")
    if audio_layout is not None and layout is not None and layout != audio_layout:
        raise ValueError("audio layouts disagree")
    layout = audio_layout if audio_layout is not None else layout
    if not audio.numel() or not torch.isfinite(audio).all():
        raise ValueError("audio must be finite and nonempty")
    if audio.ndim == 2 and layout == "CT":
        audio = audio.mean(0)
    elif audio.ndim != 1 or layout not in (None, "T", "CT"):
        raise ValueError("audio requires explicit CT or T layout")
    rate = extra_info["audio_sample_rate"]
    if isinstance(rate, torch.Tensor):
        if rate.numel() != 1:
            raise ValueError("audio sample rate must be scalar")
        rate = rate.item()
    if (isinstance(rate, bool) or not isinstance(rate, Real) or not math.isfinite(rate)
            or rate <= 0 or int(rate) != rate):
        raise ValueError("audio sample rate must be a positive integer")
    return audio, int(rate)


def resample_audio(waveform, source_rate, target_rate):
    if source_rate == target_rate:
        return waveform
    import torchaudio.functional as audio_functional
    return audio_functional.resample(waveform.unsqueeze(0), orig_freq=source_rate,
                                     new_freq=target_rate).squeeze(0)


def load_torch_state_dict(path):
    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path, device="cpu")
    return torch.load(path, map_location="cpu", weights_only=True)


def visual_reward_frames(solution_image, extra_info, frame_interval=1):
    if type(frame_interval) is not int or frame_interval < 1:
        raise ValueError("frame interval must be a positive integer")
    if extra_info.get("media_kind") != "video":
        raise ArtifactContractError("explicit decoded video media_kind is required")
    video = solution_image
    if (not isinstance(video, torch.Tensor) or video.ndim != 4
            or video.shape[1] != 3 or video.dtype != torch.uint8
            or any(d <= 0 for d in video.shape)):
        raise ArtifactContractError("decoded video must be nonempty uint8 [T,3,H,W]")
    return video[::frame_interval]


def video_reward_tensor(solution_image, extra_info, fps):
    if extra_info.get("fps") != fps:
        raise ArtifactContractError("reward FPS differs from decoded media")
    return visual_reward_frames(solution_image, extra_info)
