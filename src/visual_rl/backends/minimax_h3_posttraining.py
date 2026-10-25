"""Explicit H3 six-reward high-noise recipe; no rollout RNG consumption."""

import hashlib
import random

from ..config import ConfigError

POST_TRAINING = {"video_pool": 4, "audio_pool": 3, "count": 2,
                 "reward_normalization": "prompt_group_std", "eval_gallery": True}


def selected_positions(seed, attempt):
    """One reproducible independent pair per modality, shared by the whole rollout."""
    if type(seed) is not int or seed < 0 or type(attempt) is not int or attempt < 1:
        raise ConfigError("H3 modality selection requires a seed and one-based global attempt")
    result = {}
    for name, pool in (("video", 4), ("audio", 3)):
        digest = hashlib.sha256(f"h3/posttraining/v1:{seed}:{attempt}:{name}".encode()).digest()
        generator = random.Random(int.from_bytes(digest, "big"))
        result[name] = tuple(sorted(generator.sample(range(pool), 2)))
    return result
