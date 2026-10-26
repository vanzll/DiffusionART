"""Explicit H3 six-reward high-noise recipe; no rollout RNG consumption."""

import hashlib
import random

from ..config import ConfigError

# Keep legacy defaults for frozen configs; current recipes declare their pools.
POST_TRAINING = {"video_pool": 4, "audio_pool": 3, "count": 2,
                 "reward_normalization": "prompt_group_std", "eval_gallery": True}


def validate_post_training(options, boundaries):
    if (not isinstance(options, dict) or set(options) != set(POST_TRAINING)
            or type(options["count"]) is not int or options["count"] != 2
            or options["reward_normalization"] != "prompt_group_std"
            or options["eval_gallery"] is not True):
        raise ConfigError("H3 post-training requires two positions, six-reward group std and AV gallery")
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
