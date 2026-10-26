"""Explicit H3 six-reward high-noise recipe; no rollout RNG consumption."""

import hashlib
import random
from dataclasses import replace

from ..config import ConfigError

# Keep legacy defaults for frozen configs; current recipes declare their pools.
POST_TRAINING = {"video_pool": 4, "audio_pool": 3, "count": 2,
                 "reward_normalization": "prompt_group_std", "eval_gallery": True}

DURATION_OPTIONS = {"duration_frames", "full_duration_rewards"}


def duration_frames(c):
    """Frozen configurations remain single-length; only the explicit recipe mixes."""
    post = c.algorithm.options.get("post_training") or {}
    return tuple(post.get("duration_frames", (c.sampling.frames,)))


def duration_config(c, frames):
    """A local workload view; checkpoint and run identity use the original config."""
    return replace(c, sampling=replace(c.sampling, frames=frames))


def rollout_config(c, completed_rollouts):
    """One shared length per successful rollout; no RNG draw or extra state."""
    if type(completed_rollouts) is not int or completed_rollouts < 0:
        raise ConfigError("H3 duration rotation requires a nonnegative completed-rollout count")
    frames = duration_frames(c)
    return duration_config(c, frames[completed_rollouts % len(frames)])


def evaluation_step_directory(c, step):
    mixed = "duration_frames" in (c.algorithm.options.get("post_training") or {})
    return f"step-{step:09d}" + (f"-frames-{c.sampling.frames:03d}" if mixed else "")


def validate_post_training(options, boundaries):
    if (not isinstance(options, dict) or set(options) - (set(POST_TRAINING) | DURATION_OPTIONS)
            or not set(POST_TRAINING) <= set(options)
            or type(options["count"]) is not int or options["count"] != 2
            or options["reward_normalization"] != "prompt_group_std"
            or options["eval_gallery"] is not True):
        raise ConfigError("H3 post-training requires two positions, six-reward group std and AV gallery")
    present = DURATION_OPTIONS & set(options)
    if present and present != DURATION_OPTIONS:
        raise ConfigError("H3 duration buckets and full-duration rewards must be configured together")
    if present:
        frames = options["duration_frames"]
        if (not isinstance(frames, list) or not frames or len(frames) > 8
                or any(type(n) is not int or n % 17 != 5 or not 5 <= n / 24 <= 15 for n in frames)
                or len(set(frames)) != len(frames) or frames != sorted(frames)
                or options["full_duration_rewards"] is not True):
            raise ConfigError("H3 durations require unique ascending native 17n+5 frames within5..15s and full rewards")
    for name in ("video_pool", "audio_pool"):
        if type(options[name]) is not int or not 2 <= options[name] <= boundaries - 2:
            raise ConfigError("H3 candidate pools must contain at least two nonfinal rollout positions")


def selected_positions(seed, attempt, *, video_pool=4, audio_pool=3):
    """One reproducible independent pair per modality, shared by the whole rollout."""
    if type(seed) is not int or seed < 0 or type(attempt) is not int or attempt < 1:
        raise ConfigError("H3 modality selection requires a seed and one-based global attempt")
    if any(type(pool) is not int or pool < 2 for pool in (video_pool, audio_pool)):
        raise ConfigError("H3 modality candidate pools must contain at least two positions")
    result = {}
    for name, pool in (("video", video_pool), ("audio", audio_pool)):
        digest = hashlib.sha256(f"h3/posttraining/v1:{seed}:{attempt}:{name}".encode()).digest()
        generator = random.Random(int.from_bytes(digest, "big"))
        result[name] = tuple(sorted(generator.sample(range(pool), 2)))
    return result
