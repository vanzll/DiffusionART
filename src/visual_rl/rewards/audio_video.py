"""Five legacy AV scorers and an opt-in sixth ImageBind scorer with explicit transforms, native rates and text routing.

Native model/scoring implementations are attributed in _native_av/provenance.json.
This module imports optional dependencies only when the requested scorer loads.
No scorer is silently substituted when its dependency or checkpoint is absent.
"""

import asyncio
import math
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch

from ..training.audio_video_types import AVMedia

COMPONENTS = ("video_align", "hpsv3", "audiobox", "clap", "desync")
SIX_COMPONENTS = (*COMPONENTS, "imagebind")


@dataclass(frozen=True)
class AVScores:
    components: dict[str, torch.Tensor]
    diagnostics: dict[str, torch.Tensor]


class _AsyncInference:
    def __init__(self, native):
        self.native = native

    async def infer(self, *args):
        return self.native.infer(*args)


def validate_media(media):
    if not isinstance(media, AVMedia):
        raise ValueError("expected decoded AVMedia")
    v, a = media.video, media.audio
    if (v.ndim != 5 or v.shape[1] != 3 or a.ndim != 3 or len(v) != len(a)
            or not len(v) or min(*v.shape, *a.shape) < 1):
        raise ValueError("media requires RGB [B,3,T,H,W] and audio [B,C,S]")
    if (not v.is_floating_point() or not a.is_floating_point()
            or not torch.isfinite(v).all() or not torch.isfinite(a).all()
            or v.min() < 0 or v.max() > 1):
        raise ValueError("decoded RGB must be in [0,1] and both streams finite")
    if (isinstance(media.fps, bool) or not math.isfinite(media.fps) or media.fps <= 0
            or type(media.sample_rate) is not int or media.sample_rate < 1):
        raise ValueError("media rates must be explicit and positive")
    # One video-frame tolerance permits native temporal-codec endpoint rounding.
    if abs(v.shape[2] / media.fps - a.shape[-1] / media.sample_rate) > 1 / media.fps + 1e-8:
        raise ValueError("audio/video durations differ by more than one video frame")


class AudioVideoReward:
    """Sequential rowwise scoring, preserving prompt and modality identity."""

    def __init__(self, models, *, device="cpu"):
        if set(models) not in (set(COMPONENTS), set(SIX_COMPONENTS)):
            raise ValueError("supply exactly the five legacy AV scorers or all six scorers")
        self.components = SIX_COMPONENTS if "imagebind" in models else COMPONENTS
        self.models = dict(models)
        self.device = torch.device(device)

    @classmethod
    def from_paths(cls, paths, *, device, expected_sha256=None):
        required = {"video_align", "hpsv3", "audiobox", "clap", "desync",
                    "video_align_base", "desync_source"}
        if (set(paths) not in (required, required | {"imagebind", "imagebind_source"})
                or any(not Path(v).exists() for v in paths.values())):
            raise ValueError("AV scorer paths require the seven legacy assets or those plus imagebind/imagebind_source")
        validate_asset_inventory(paths, expected_sha256)
        from ._native_av.audiobox import AudioBoxModel
        from ._native_av.clap import CLAPModel
        from ._native_av.desync import DeSyncModel
        from ._native_av.hpsv3_reward import HPSv3Model
        from ._native_av.videoalign import VideoAlignModel

        # Frozen scorers stay on CPU until their individual inference phase.
        target_device = device
        device = "cpu"
        models = {}
        try:
            models["video_align"] = VideoAlignModel(paths["video_align"], device, paths["video_align_base"])
            models["hpsv3"] = HPSv3Model(paths["hpsv3"], device,
                                        base_model_path=paths["video_align_base"])
            models["audiobox"] = AudioBoxModel(paths["audiobox"], device)
            models["clap"] = CLAPModel(paths["clap"], device,
                                       model_kwargs={"local_files_only": True},
                                       processor_kwargs={"local_files_only": True})
            models["desync"] = DeSyncModel(paths["desync"], device, paths["desync_source"])
            if "imagebind" in paths:
                from ._native_av.imagebind import ImageBindModel
                models["imagebind"] = ImageBindModel(paths["imagebind"], device, paths["imagebind_source"])
            return cls(models, device=target_device)
        except BaseException:
            for model in models.values():
                model.close()
            raise

    def close(self):
        for model in self.models.values():
            model.close()

    @contextmanager
    def _active(self, name):
        native = self.models[name]
        holder = native._inferencer if name == "hpsv3" else native
        holder.device = self.device
        holder.model.to(self.device)
        if name == "desync":
            native.mel.to(self.device)
        try:
            yield _AsyncInference(native)
        finally:
            holder.model.to("cpu")
            holder.device = torch.device("cpu")
            if name == "desync":
                native.mel.to("cpu")

    @torch.no_grad()
    def score(self, media, video_prompts, audio_prompts):
        validate_media(media)
        if (len(video_prompts) != len(media) or len(audio_prompts) != len(media)
                or any(not isinstance(p, str) or not p.strip()
                       for p in (*video_prompts, *audio_prompts))):
            raise ValueError("each row needs explicit nonempty video/audio reward text")
        # Deliberately no engine batch objects; these functions receive one
        # canonical decoded media row, as in pinned native scorer paths.
        from ._native_av import audiobox, clap, desync, hpsv3_reward, videoalign
        values = {key: [] for key in self.components}
        diagnostics = {}
        for i in range(len(media)):
            video = media.video[i].detach().cpu().permute(1, 0, 2, 3).mul(255).round().to(torch.uint8)
            extra = {"audio": media.audio[i].detach().cpu(), "audio_layout": "CT",
                     "audio_sample_rate": media.sample_rate, "fps": media.fps,
                     "media_kind": "video"}
            async def compute():
                with self._active("video_align") as model:
                    va = await videoalign.compute_score(solution_image=video, ground_truth=video_prompts[i],
                        extra_info=extra, reward_model=model, fps=media.fps,
                        score_weights=[.5, 0., .5], score_means=[3.6757, 0., 2.8105],
                        score_stds=[2.2476, 1., 2.5121])
                with self._active("hpsv3") as model:
                    hps = await hpsv3_reward.compute_score_hpsv3(None, video, video_prompts[i], extra,
                        reward_model=model, reward_scale=1., num_frames=5,
                        top_fraction=.3, score_cap=15, max_batch_size=5)
                with self._active("audiobox") as model:
                    ab = await audiobox.compute_score(extra_info=extra, reward_model=model,
                        axis_weights={"CE": 1., "CU": 1., "PC": -1., "PQ": 1.}, score_scale=.025)
                with self._active("clap") as model:
                    cl = await clap.compute_score(None, video, audio_prompts[i], extra,
                        reward_model=model, score_scale=.5, score_offset=.5, score_min=0., score_max=1.)
                with self._active("desync") as model:
                    ds = await desync.compute_score(solution_image=video, extra_info=extra,
                        reward_model=model, fps=media.fps)
                results = (va, hps, ab, cl, ds)
                if "imagebind" in self.models:
                    from ._native_av import imagebind
                    with self._active("imagebind") as model:
                        ib = await imagebind.compute_score(solution_image=video, extra_info=extra, reward_model=model)
                    results += (ib,)
                return results
            results = asyncio.run(compute())
            for key, result in zip(self.components, results):
                score = float(result["score"])
                if not math.isfinite(score):
                    raise ValueError(f"nonfinite {key} reward")
                values[key].append(score)
                for raw_name, raw_value in result.items():
                    if raw_name.endswith("_raw"):
                        metric = raw_name if raw_name == "hpsv3_raw" else f"{key}/{raw_name}"
                        if not math.isfinite(float(raw_value)):
                            raise ValueError(f"nonfinite scorer diagnostic: {metric}")
                        diagnostics.setdefault(metric, []).append(float(raw_value))
        return AVScores({k: torch.tensor(v, dtype=torch.float32) for k, v in values.items()},
                        {k: torch.tensor(v, dtype=torch.float32) for k, v in diagnostics.items()})


def validate_asset_inventory(paths, inventory):
    """Bind every scorer input file, including directory configs and source."""
    from ..prepare_assets import _hash
    if not isinstance(inventory, dict) or set(inventory) != set(paths):
        raise ValueError("all scorer assets require complete SHA256 inventories")
    for name, value in paths.items():
        root = Path(value)
        if any(p.is_symlink() for p in (root.absolute(), *root.absolute().parents)):
            raise ValueError(f"symlink scorer asset is unsupported: {name}")
        entries = {".": root} if root.is_file() else {}
        if root.is_dir():
            for item in root.rglob("*"):
                if item.is_symlink():
                    raise ValueError(f"symlink scorer asset is unsupported: {name}")
                if item.is_file():
                    entries[item.relative_to(root).as_posix()] = item
                elif not item.is_dir():
                    raise ValueError(f"nonregular scorer asset is unsupported: {name}")
        expected = inventory[name]
        if not entries or not isinstance(expected, dict) or set(expected) != set(entries):
            raise ValueError(f"incomplete scorer asset inventory: {name}")
        for relative, item in entries.items():
            digest = expected[relative]
            if (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
                    or _hash(item) != digest):
                raise ValueError(f"AV scorer asset checksum mismatch: {name}/{relative}")
