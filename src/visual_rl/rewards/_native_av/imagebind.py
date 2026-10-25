# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# ImageBind-derived preprocessing in this file is licensed under CC-BY-NC-SA-4.0.
# See licenses/ImageBind-CC-BY-NC-SA-4.0.txt and THIRD_PARTY_NOTICES.md.
# Local changes: decoded tensor input, explicit stereo downmix/rates, offline
# checkpoint loading, private source imports and unscaled audio/video cosine.

"""Offline ImageBind audio/video cosine with a decoded-media preprocessing bridge.

Transform conventions follow facebookresearch/ImageBind ``data.py`` at
53680b02d7e37b19b124fa37bae4b6c98c38f5be and its pinned pytorchvideo
6cdc929315aab1b5674b6dcf73b16ec99147735f. The implementation and weights
are supplied locally under ImageBind's CC-BY-NC-SA-4.0 license; no assets
are downloaded and no globally named ``imagebind`` package is imported.
"""

import ast
import hashlib
import math
import sys
import threading
from fractions import Fraction
from pathlib import Path
from types import ModuleType

import torch
import torch.nn.functional as F

IMAGEBIND_REPOSITORY = "facebookresearch/ImageBind"
IMAGEBIND_REVISION = "53680b02d7e37b19b124fa37bae4b6c98c38f5be"
IMAGEBIND_LICENSE = "CC-BY-NC-SA-4.0"
SOURCE_SHA256 = {
    "imagebind/models/helpers.py": "ee5d71dde0c9c2016272ee461edb7c9f4acb3449df829aa56757963c893a8396",
    "imagebind/models/multimodal_preprocessors.py": "d096e36002bdcf31a0f394f6daf50546700bbf3ac2927af2820b9ec5cbd6c411",
    "imagebind/models/transformer.py": "0f21a02f6ed207debfa2bbabaaec3080a9b48df742bde9351bdd9838b68b81fe",
    "imagebind/models/imagebind_model.py": "de713318171f94bc6479f6df10153916f19869d175c1360654a00d70e4c0cd66",
}
_IMPORT_LOCK = threading.Lock()
_MEAN = (0.48145466, 0.4578275, 0.40821073)
_STD = (0.26862954, 0.26130258, 0.27577711)


def validate_source(root):
    """Verify immutable implementation bytes, independently of manifest labels."""
    root = Path(root).resolve(strict=True)
    for relative, expected in SOURCE_SHA256.items():
        path = root / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"ImageBind source differs from {IMAGEBIND_REVISION}: {relative}")
    return root


def import_model_source(root):
    """Load four verified modules under a private namespace, without sys.path edits.

    Only absolute imports of ImageBind's own modules are rewritten. Existing
    third-party ``imagebind`` modules are never replaced, even temporarily.
    """
    root = validate_source(root)
    prefix = "_visual_rl_imagebind_" + hashlib.sha256(str(root).encode()).hexdigest()[:20]
    with _IMPORT_LOCK:
        if prefix + ".models.imagebind_model" in sys.modules:
            return sys.modules[prefix + ".models.imagebind_model"]
        created = []
        try:
            for name, folder in ((prefix, root / "imagebind"), (prefix + ".models", root / "imagebind/models")):
                package = ModuleType(name)
                package.__path__ = [str(folder)]
                package.__package__ = name
                sys.modules[name] = package
                created.append(name)
            for relative in SOURCE_SHA256:
                path = root / relative
                name = prefix + ".models." + path.stem
                tree = ast.parse(path.read_text(), filename=str(path))
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("imagebind."):
                        node.module = prefix + node.module[len("imagebind"):]
                module = ModuleType(name)
                module.__file__ = str(path)
                module.__package__ = prefix + ".models"
                sys.modules[name] = module
                created.append(name)
                exec(compile(tree, str(path), "exec"), module.__dict__)
            return sys.modules[prefix + ".models.imagebind_model"]
        except BaseException:
            for name in reversed(created):
                sys.modules.pop(name, None)
            raise


def _clip_times(duration, count):
    # ConstantClipsPerVideoSampler uses Fraction to avoid cumulative drift.
    spacing = Fraction(max(duration - 2, 0)) / max(count - 1, 1)
    return [(spacing * i, spacing * i + 2) for i in range(count)]


def preprocess_audio(audio, sample_rate):
    """Canonical CT audio -> [1,3,1,128,204], using real Kaldi filterbanks.

    H3 stereo is explicitly downmixed to mono before the upstream transform.
    Short clips are mel-padded, not waveform-padded to an invented duration.
    """
    import torchaudio

    waveform = torch.as_tensor(audio).detach().float().cpu()
    if waveform.ndim != 2 or not waveform.shape[0] or not waveform.shape[1]:
        raise ValueError("ImageBind requires nonempty CT audio")
    if type(sample_rate) is not int or sample_rate < 1 or not torch.isfinite(waveform).all():
        raise ValueError("ImageBind audio requires a positive explicit rate and finite samples")
    waveform = waveform.mean(0, keepdim=True)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    clips = []
    for start, end in _clip_times(waveform.shape[-1] / 16000, 3):
        clip = waveform[:, int(start * 16000):int(end * 16000)].clone()
        if clip.shape[-1] < 400:
            raise ValueError("ImageBind audio clip must contain at least 25 ms")
        clip -= clip.mean()
        mel = torchaudio.compliance.kaldi.fbank(
            clip, htk_compat=True, sample_frequency=16000, use_energy=False,
            window_type="hanning", num_mel_bins=128, dither=0.,
            frame_length=25, frame_shift=10,
        ).transpose(0, 1)
        mel = F.pad(mel, (0, max(204 - mel.shape[-1], 0)))[:, :204]
        clips.append((mel.unsqueeze(0) + 4.268) / 9.138)
    return torch.stack(clips).unsqueeze(0)


def preprocess_video(video, fps):
    """RGB uint8 TCHW -> [1,15,3,2,224,224], matching decoded CFR windows."""
    video = torch.as_tensor(video).detach().cpu()
    if (video.ndim != 4 or video.shape[1] != 3 or min(video.shape) < 1
            or video.dtype != torch.uint8 or not math.isfinite(fps) or fps <= 0):
        raise ValueError("ImageBind requires uint8 TCHW RGB video and explicit positive fps")
    count, _, height, width = video.shape
    clips = []
    for start, end in _clip_times(count / fps, 5):
        # Match pinned EncodedVideoDecord.get_clip and UniformTemporalSubsample(2).
        first, stop = math.ceil(fps * start), min(math.ceil(fps * end), count)
        if stop <= first:
            raise ValueError("ImageBind temporal window contains no decoded frames")
        indices = torch.linspace(first, stop - 1, 2).long()
        clip = video[indices].permute(1, 0, 2, 3).float() / 255.
        size = (int(height / width * 224), 224) if width < height else (224, int(width / height * 224))
        clip = F.interpolate(clip, size=size, mode="bilinear", align_corners=False)
        clip = (clip - torch.tensor(_MEAN)[:, None, None, None]) / torch.tensor(_STD)[:, None, None, None]
        for spatial_idx in range(3):
            y = math.ceil((size[0] - 224) / 2)
            x = math.ceil((size[1] - 224) / 2)
            if size[0] > size[1]:
                y = (0, y, size[0] - 224)[spatial_idx]
            else:
                x = (0, x, size[1] - 224)[spatial_idx]
            clips.append(clip[:, :, y:y + 224, x:x + 224])
    return torch.stack(clips).unsqueeze(0)


class ImageBindModel:
    def __init__(self, checkpoint, device="cpu", source_root=None):
        if source_root is None or not Path(checkpoint).is_file():
            raise ValueError("ImageBind requires an explicit local checkpoint file and source directory")
        native = import_model_source(source_root)
        self.device = torch.device(device)
        self.model = native.imagebind_huge(pretrained=False)
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        self.model.eval().requires_grad_(False).to(self.device)

    @torch.no_grad()
    def infer(self, video, audio):
        return self.model({"vision": video.to(self.device), "audio": audio.to(self.device)})

    def close(self):
        self.model.to("cpu")
        self.device = torch.device("cpu")


async def compute_score(*, solution_image, extra_info, reward_model):
    if extra_info.get("audio_layout") != "CT":
        raise ValueError("ImageBind requires explicit CT audio layout")
    video = preprocess_video(solution_image, extra_info["fps"])
    audio = preprocess_audio(extra_info["audio"], extra_info["audio_sample_rate"])
    embeddings = await reward_model.infer(video, audio)
    vision, sound = embeddings["vision"].float(), embeddings["audio"].float()
    if (vision.ndim != 2 or sound.shape != vision.shape or vision.shape[0] != 1
            or not torch.isfinite(vision).all() or not torch.isfinite(sound).all()
            or (vision.norm(dim=-1) == 0).any() or (sound.norm(dim=-1) == 0).any()):
        raise ValueError("ImageBind returned invalid or zero audio/video embeddings")
    score = float(F.cosine_similarity(vision, sound, dim=-1).clamp(-1., 1.).item())
    return {"score": score, "cosine_raw": score}
