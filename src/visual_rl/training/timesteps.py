"""ART training-position selection; rollout schedules and update windows stay fixed."""

import hashlib
import random

from ..config import ConfigError


def art_training_positions(training, configured, *, seed=None, attempt=None):
    """Choose cache positions without changing rollout/noise RNG or update clocks.

    ``attempt`` is the one-based global Progress.attempts counter, including
    skipped rollouts. A private, domain-separated RNG chooses one chronological
    high-noise pair for every rank, trajectory, modality and optimizer window.
    It needs no mutable state, so retries and checkpoint continuation reproduce
    the pair. Both Flash modes override the configured pool. Window builders
    count actual sample/time items, including masked items, for normalization.
    """
    if training.timestep_selection == "first":
        return (0,)
    if training.timestep_selection == "high_noise":
        if (type(seed) is not int or seed < 0
                or type(attempt) is not int or attempt < 1):
            raise ConfigError("high_noise requires a global seed and one-based rollout attempt")
        digest = hashlib.sha256(f"diffusionart/high_noise/v1:{seed}:{attempt}".encode()).digest()
        generator = random.Random(int.from_bytes(digest, "big"))
        return tuple(sorted(generator.sample(range(4), 2)))
    if training.timestep_selection == "configured":
        return tuple(configured)
    raise ConfigError("unknown ART training timestep selection")
