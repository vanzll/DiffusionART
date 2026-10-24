"""ART training-position selection; rollout schedules and update windows stay fixed."""

from ..config import ConfigError


def art_training_positions(training, configured):
    """Select rollout index zero for Flash, independently of the configured pool.

    Call before collecting replay caches. Existing window builders then count
    the actual selected sample/time pairs, including masked items, so one
    selected position has temporal denominator one rather than the old count.
    """
    if training.timestep_selection == "first":
        return (0,)
    if training.timestep_selection == "configured":
        return tuple(configured)
    raise ConfigError("unknown ART training timestep selection")
