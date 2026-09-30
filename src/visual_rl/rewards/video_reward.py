"""VideoReward MP4 reporting and continuous-RGB ReFL scoring.

ABI/preprocessing reference: DanceGRPO 15cc71d53cc2e6e18a68ee607d5fb6ba9a99e344,
fastvideo/models/videoalign/{inference,vision_process}.py. The continuous path
ports the research ReFL adapter, not a straight-through codec approximation.
The native inferencer/model is supplied by the backend's pinned asset loader.
"""

import json
import math
import subprocess
from fractions import Fraction
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .inputs import ContinuousRGBInput
from .scalar import FixedNormalization, finite_number, weighted_score

COMPONENTS = ("VQ", "MQ", "TA")


def uniform_indices(frames, video_fps, reward_fps):
    if type(frames) is not int or frames < 4:
        raise ValueError("VideoReward requires at least four frames")
    if min(finite_number(video_fps, "video_fps"), finite_number(reward_fps, "reward_fps")) <= 0:
        raise ValueError("FPS must be positive")
    maximum = math.floor(min(768, frames) / 2) * 2
    count = min(max(frames / video_fps * reward_fps, 4), maximum)
    count = min(round(count / 2) * 2, frames)
    return torch.linspace(0, frames - 1, count).round().long()


def resized_shape(height, width, *, min_pixels=100352, max_pixels=200704, factor=28):
    if min(height, width) < factor or max(height, width) / min(height, width) > 200:
        raise ValueError("unsupported VideoReward spatial geometry")
    if not 0 < min_pixels <= max_pixels:
        raise ValueError("invalid VideoReward pixel bounds")
    h, w = max(factor, round(height / factor) * factor), max(factor, round(width / factor) * factor)
    if h * w > max_pixels:
        scale = math.sqrt(height * width / max_pixels)
        h, w = math.floor(height / scale / factor) * factor, math.floor(width / scale / factor) * factor
    elif h * w < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        h, w = math.ceil(height * scale / factor) * factor, math.ceil(width * scale / factor) * factor
    return h, w


def pack_video(frames):
    """Qwen2-VL's spatial-merge-major patch layout, preserving input gradients."""
    if frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError("video patches require TCHW RGB")
    t, c, h, w = frames.shape
    if t % 2 or h % 28 or w % 28:
        raise ValueError("video geometry must divide the native Qwen2-VL patches")
    grid = (t // 2, h // 14, w // 14)
    value = frames.reshape(t // 2, 2, c, h // 28, 2, 14, w // 28, 2, 14)
    value = value.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    return value.reshape(math.prod(grid), c * 2 * 14**2), grid


def verify_video_fps(path, expected, *, ffprobe="ffprobe", timeout=20):
    """Check stream metadata before reward sampling; this is not a full decode."""
    path = Path(path).resolve(strict=True)
    result = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0",
                             "-show_entries", "stream=avg_frame_rate", "-of", "json", str(path)],
                            capture_output=True, timeout=timeout, check=True)
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    if len(streams) != 1:
        raise ValueError("reporting MP4 has no unique selected video stream")
    actual = float(Fraction(streams[0]["avg_frame_rate"]))
    if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"reporting MP4 FPS {actual} differs from configured {expected}")
    return str(path)


class VideoReward(nn.Module):
    """One frozen scorer, with explicit reporting and differentiable paths.

    Neither path changes module placement or enables CPU activation offload.
    The backend owns model/VAE memory policy. Native MP4 scoring stays batch-one
    to preserve the paper's scorer behavior; continuous scoring is also rowwise.
    """

    def __init__(self, inferencer, build_prompt, *, weights, normalization_source,
                 space="fixed_normalized", video_fps=8., reward_fps=4., ffprobe="ffprobe"):
        super().__init__()
        self.inferencer, self.build_prompt = inferencer, build_prompt
        self.model = inferencer.model.requires_grad_(False).eval()
        for parameter in self.model.parameters():
            parameter.grad = None
        self.processor, self.data = inferencer.processor, inferencer.data_config
        self.video_fps, self.reward_fps, self.ffprobe = video_fps, reward_fps, ffprobe
        uniform_indices(53, video_fps, reward_fps)
        if self.data.sample_type != "uniform" or self.data.num_frames is not None:
            raise ValueError("VideoReward requires FPS-driven uniform sampling, num_frames=None")
        if self.data.max_frame_pixels != 200704:
            raise ValueError("VideoReward requires max_frame_pixels=200704")
        self.weights, self.space = dict(weights), space
        if set(weights) != set(COMPONENTS):
            raise ValueError("VideoReward weights must explicitly name VQ, MQ and TA")
        config = inferencer.inference_config
        self.normalization = {
            key: FixedNormalization(config[f"{key}_mean"], config[f"{key}_std"], normalization_source)
            for key in COMPONENTS
        }
        weighted_score(dict.fromkeys(COMPONENTS, 0.), self.weights,
                       space=space, normalization=self.normalization)
        ip = self.processor.image_processor
        if (ip.patch_size, ip.temporal_patch_size, ip.merge_size) != (14, 2, 2):
            raise ValueError("unsupported Qwen2-VL patch geometry")
        if not ip.do_rescale or not ip.do_normalize or not math.isclose(ip.rescale_factor, 1 / 255., abs_tol=1e-10):
            raise ValueError("unsupported VideoReward RGB preprocessing")
        if len(ip.image_mean) != 3 or len(ip.image_std) != 3 or any(
            not math.isfinite(x) for x in (*ip.image_mean, *ip.image_std)
        ) or min(ip.image_std) <= 0:
            raise ValueError("invalid VideoReward RGB statistics")
        self.video_token = getattr(self.processor, "video_token", "<|video_pad|>")
        self.video_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.video_token)
        if self.video_token_id != self.model.config.video_token_id:
            raise ValueError("VideoReward tokenizer/model token identity differs")

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()
        return self

    def _scores(self, raw):
        return weighted_score(raw, self.weights, space=self.space, normalization=self.normalization)

    @torch.no_grad()
    def score_paths(self, paths, prompts):
        if not paths or len(paths) != len(prompts) or any(not isinstance(p, str) or not p.strip() for p in prompts):
            raise ValueError("MP4 scoring requires matching nonempty paths/prompts")
        records = []
        self.inferencer.device = str(next(self.model.parameters()).device)
        for path, prompt in zip(paths, prompts):
            path = verify_video_fps(path, self.video_fps, ffprobe=self.ffprobe)
            result = self.inferencer.reward([path], [prompt], fps=self.reward_fps,
                                            max_pixels=200704, use_norm=False)
            if len(result) != 1:
                raise ValueError("VideoReward returned the wrong number of rows")
            records.append(result[0])
        return self._scores({key: torch.tensor([finite_number(r[key], key) for r in records],
                                               dtype=torch.float32) for key in COMPONENTS})

    def _continuous_batch(self, pixels, prompt):
        ip = self.processor.image_processor
        target = resized_shape(*pixels.shape[-2:])
        pixels = F.interpolate(pixels.float(), size=target, mode="bicubic", align_corners=False, antialias=True)
        if ip.do_resize:
            target2 = resized_shape(*target, min_pixels=ip.min_pixels, max_pixels=ip.max_pixels)
            if target2 != target:
                pixels = F.interpolate(pixels, size=target2, mode="bicubic", align_corners=False, antialias=True)
        mean, std = (pixels.new_tensor(value).reshape(1, 3, 1, 1) for value in (ip.image_mean, ip.image_std))
        patches, grid = pack_video((pixels - mean) / std)
        chat = [{"role": "user", "content": [
            {"type": "video", "video": "file://continuous-rgb"},
            {"type": "text", "text": self.build_prompt(prompt, self.data.eval_dim, self.data.prompt_template_type)},
        ]}]
        text = self.processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        if text.count(self.video_token) != 1:
            raise ValueError("VideoReward chat requires one video placeholder")
        count = math.prod(grid) // 4
        text = text.replace(self.video_token, self.video_token * count)
        tokens = self.processor.tokenizer([text], padding=True, return_tensors="pt")
        batch = {key: tokens[key].to(pixels.device) for key in ("input_ids", "attention_mask")}
        if int((batch["input_ids"] == self.video_token_id).sum()) != count:
            raise ValueError("VideoReward tokenization truncated or changed video tokens")
        batch.update(pixel_values_videos=patches, video_grid_thw=torch.tensor([grid], device=pixels.device))
        return batch

    def forward(self, inputs: ContinuousRGBInput):
        inputs.validate()
        if inputs.fps != self.video_fps:
            raise ValueError("continuous video FPS differs from the reporting protocol")
        indices = uniform_indices(inputs.frames.shape[1], self.video_fps, self.reward_fps).to(inputs.frames.device)
        pixels = inputs.frames.index_select(1, indices).float()
        if inputs.value_range == "minus_one_one":
            pixels = (pixels + 1) * .5
        logits = []
        for frames, prompt in zip(pixels, inputs.prompts):
            output = self.model(**self._continuous_batch(frames, prompt), return_dict=True, use_cache=False)["logits"]
            if output.shape != (1, 3) or not output.requires_grad:
                raise ValueError("VideoReward must return differentiable [1,3] logits")
            logits.append(output.float())
        raw = torch.cat(logits)
        return self._scores({key: raw[:, i] for i, key in enumerate(COMPONENTS)})
