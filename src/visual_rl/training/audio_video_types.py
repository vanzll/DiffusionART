"""Small tensor/media contracts shared by opt-in audio-video backends."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AVState:
    """Native latent batches; optional masks mark valid features, never prompts.

    Both tensors begin with the same batch dimension. A boolean mask may have
    the full shape or a prefix (for example [B,tokens] for [B,tokens,C]).
    Packing and velocity-sign conversion belong to the model adapter.
    """

    video: torch.Tensor
    audio: torch.Tensor
    video_mask: torch.Tensor | None = None
    audio_mask: torch.Tensor | None = None

    def __len__(self):
        return len(self.video)

    def select(self, rows):
        def selected(value):
            if value is None:
                return None
            return value[rows.to(value.device) if isinstance(rows, torch.Tensor) else rows]
        return AVState(*(selected(v) for v in (self.video, self.audio, self.video_mask, self.audio_mask)))

    def to(self, device, *, dtype=None):
        return AVState(
            self.video.to(device=device, dtype=dtype),
            self.audio.to(device=device, dtype=dtype),
            None if self.video_mask is None else self.video_mask.to(device=device),
            None if self.audio_mask is None else self.audio_mask.to(device=device),
        )

    def detached_cpu(self):
        return AVState(*(None if v is None else v.detach().cpu().clone()
                         for v in (self.video, self.audio, self.video_mask, self.audio_mask)))


@dataclass(frozen=True)
class AVMedia:
    """RGB [B,3,T,H,W] in [0,1], waveform [B,C,S], explicit native rates."""

    video: torch.Tensor
    audio: torch.Tensor
    fps: float
    sample_rate: int

    def __len__(self):
        return len(self.video)
