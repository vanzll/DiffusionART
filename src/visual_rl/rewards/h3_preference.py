"""Opt-in CLAP + ImageBind + eight-frame mean PickScore; no legacy scorers."""

import asyncio
import math

import torch

from .audio_video import AudioVideoReward, AVScores, validate_asset_inventory, validate_media
from .av_profiles import (
    PICKSCORE_PROCESSOR_REVISION,
    PICKSCORE_REVISION,
    THREE_REWARD_ASSETS,
    THREE_REWARD_WEIGHTS,
)


def pickscore_frame_indices(frames):
    """Eight equal-spaced positions, including both endpoints; never top-k."""
    if type(frames) is not int or frames < 8:
        raise ValueError("H3 PickScore requires at least eight decoded video frames")
    return torch.linspace(0, frames - 1, 8, dtype=torch.float64).round().long()


class VideoPickScore:
    def __init__(self, scorer):
        self.model = scorer
        self.device = torch.device("cpu")

    def infer(self, video, prompt):
        # Input is the same quantized RGB used by the native AV scorers.
        frames = video[pickscore_frame_indices(len(video))].float().div(255)
        result = self.model(frames, [prompt] * len(frames), microbatch=8)
        raw = result.raw["pickscore"]
        if raw.shape != (8,) or not torch.isfinite(raw).all():
            raise ValueError("PickScore must return eight finite per-frame logits")
        return {"score": float(raw.float().mean()), "frame_min_raw": float(raw.min()),
                "frame_max_raw": float(raw.max())}

    def close(self):
        self.model.to("cpu")
        self.device = torch.device("cpu")


class H3PreferenceReward(AudioVideoReward):
    def __init__(self, models, *, device="cpu", full_duration=False):
        if set(models) != set(THREE_REWARD_WEIGHTS):
            raise ValueError("H3 preference requires exactly CLAP, ImageBind and PickScore")
        if type(full_duration) is not bool:
            raise ValueError("full_duration must be boolean")
        self.components = tuple(sorted(THREE_REWARD_WEIGHTS))
        self.models = dict(models)
        self.device = torch.device(device)
        self.full_duration = full_duration

    @classmethod
    def from_paths(cls, paths, *, device, expected_sha256=None, full_duration=False):
        if set(paths) != THREE_REWARD_ASSETS:
            raise ValueError("H3 preference manifest requires exactly five asset entries")
        validate_asset_inventory(paths, expected_sha256)
        from ._native_av.clap import CLAPModel
        from ._native_av.imagebind import ImageBindModel
        from .pickscore import PickScore

        models = {}
        try:
            models["clap"] = CLAPModel(paths["clap"], "cpu",
                model_kwargs={"local_files_only": True}, processor_kwargs={"local_files_only": True})
            models["clap"].model.requires_grad_(False)
            models["imagebind"] = ImageBindModel(paths["imagebind"], "cpu", paths["imagebind_source"])
            scorer = PickScore.from_pretrained(model_path=paths["pickscore"],
                processor_path=paths["pickscore_processor"], model_revision=PICKSCORE_REVISION,
                processor_revision=PICKSCORE_PROCESSOR_REVISION, space="raw", device="cpu",
                dtype=torch.float32, local_files_only=True)
            models["pickscore"] = VideoPickScore(scorer)
            return cls(models, device=device, full_duration=full_duration)
        except BaseException:
            for model in models.values():
                model.close()
            raise

    @torch.no_grad()
    def score(self, media, video_prompts, audio_prompts):
        validate_media(media)
        if (len(video_prompts) != len(media) or len(audio_prompts) != len(media)
                or any(not isinstance(p, str) or not p.strip() for p in (*video_prompts, *audio_prompts))):
            raise ValueError("each row needs explicit nonempty video/audio reward text")
        from ._native_av import clap, imagebind

        values = {key: [] for key in self.components}
        diagnostics = {}
        for i in range(len(media)):
            video = media.video[i].detach().cpu().permute(1, 0, 2, 3).mul(255).round().to(torch.uint8)
            extra = {"audio": media.audio[i].detach().cpu(), "audio_layout": "CT",
                     "audio_sample_rate": media.sample_rate, "fps": media.fps, "media_kind": "video"}

            async def compute():
                with self._active("clap") as model:
                    if self.full_duration:
                        from .duration_windows import clap_window_seconds, score_windows
                        cl = await score_windows(video, extra, clap_window_seconds(self.models["clap"]),
                            lambda frames, info: clap.compute_score(None, frames, audio_prompts[i], info,
                                reward_model=model, score_scale=.5, score_offset=.5, score_min=0., score_max=1.))
                    else:
                        cl = await clap.compute_score(None, video, audio_prompts[i], extra,
                            reward_model=model, score_scale=.5, score_offset=.5, score_min=0., score_max=1.)
                with self._active("imagebind") as model:
                    ib = await imagebind.compute_score(solution_image=video, extra_info=extra, reward_model=model)
                with self._active("pickscore") as model:
                    ps = await model.infer(video, video_prompts[i])
                return {"clap": cl, "imagebind": ib, "pickscore": ps}

            results = asyncio.run(compute())
            for name, result in results.items():
                score = float(result["score"])
                if not math.isfinite(score):
                    raise ValueError(f"nonfinite {name} reward")
                values[name].append(score)
                for key, value in result.items():
                    if key.endswith("_raw"):
                        if not math.isfinite(float(value)):
                            raise ValueError(f"nonfinite {name} diagnostic: {key}")
                        diagnostics.setdefault(f"{name}/{key}", []).append(float(value))
        return AVScores({k: torch.tensor(v, dtype=torch.float32) for k, v in values.items()},
                        {k: torch.tensor(v, dtype=torch.float32) for k, v in diagnostics.items()})
