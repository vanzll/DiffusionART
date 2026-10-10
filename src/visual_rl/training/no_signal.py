"""Process-local liveness guard for globally inactive DiffusionART rollouts."""

import logging


class NoSignalRolloutLimit(RuntimeError):
    """The operational budget for consecutive inactive rollouts was exhausted."""


class NoSignalGuard:
    def __init__(self, maximum=None):
        self.maximum = maximum
        self.consecutive = 0

    def skipped(self, *, warn=True):
        """Call on every rank after the existing global skip decision/publication."""
        self.consecutive += 1
        if self.maximum is not None and self.consecutive >= self.maximum:
            raise NoSignalRolloutLimit(
                "DiffusionART stopped after "
                f"{self.consecutive} consecutive no-signal rollouts: "
                "runtime.max_consecutive_no_signal_rollouts="
                f"{self.maximum} exhausted; skipped rollouts applied no optimizer/EMA updates"
            )
        # Exponential spacing bounds warning volume without another runtime knob.
        if warn and self.consecutive & (self.consecutive - 1) == 0:
            logging.getLogger(__name__).warning(
                "DiffusionART made no optimizer progress for %d consecutive no-signal rollouts; "
                "skipped rollouts applied no optimizer/EMA updates "
                "(runtime.max_consecutive_no_signal_rollouts=%s)",
                self.consecutive, self.maximum,
            )

    def updated(self):
        """Reset only after a useful rollout has actually applied an update."""
        self.consecutive = 0
