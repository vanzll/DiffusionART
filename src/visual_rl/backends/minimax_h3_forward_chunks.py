"""Physical no-grad forward chunks; logical trajectories and RNG stay outside."""
import os

import torch

from ..config import ConfigError
from ..training.audio_video_types import AVState


def inference_forward_cap():
    try:
        value = int(os.environ.get("H3_INFERENCE_FORWARD_MICROBATCH", "0"))
    except ValueError as exc:
        raise ConfigError("H3 inference forward cap must be a nonnegative integer") from exc
    if value < 0:
        raise ConfigError("H3 inference forward cap must be a nonnegative integer")
    return value


def predict_in_chunks(predict, state, video_sigma, audio_sigma, original_rows, *, cap):
    """Split model work only; never redraw noise, select positions or change U2."""
    if torch.is_grad_enabled() or not cap or len(state) <= cap:
        return predict(state, video_sigma, audio_sigma, original_rows)

    def sigma_part(value, part):
        return value if value.ndim == 0 or value.numel() == 1 else value[part]

    outputs = []
    for start in range(0, len(state), cap):
        part = slice(start, min(start + cap, len(state)))
        outputs.append(predict(state.select(part), sigma_part(video_sigma, part),
                               sigma_part(audio_sigma, part), original_rows[part]))
    fields = []
    for name in ("video", "audio", "video_mask", "audio_mask"):
        values = [getattr(output, name) for output in outputs]
        if all(value is None for value in values):
            fields.append(None)
        elif any(value is None for value in values):
            raise ValueError("Physical H3 forward chunks disagree on masks")
        else:
            fields.append(torch.cat(values, dim=0))
    return AVState(*fields)
