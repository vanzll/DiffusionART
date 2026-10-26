"""Operational optimizer boundary for one native attempt, never a new training budget."""
import os
import re
from pathlib import Path

from .minimax_h3_checkpoint_io import CheckpointIO


class AttemptBudget:
    def __init__(self, session, config, root, updates_per_rollout, start_optimizer_steps=0):
        self.session, self.config, self.root = session, config, Path(root)
        self.updates_per_rollout = updates_per_rollout
        requested = os.environ.get('H3_ATTEMPT_STOP_UPDATE')
        session.agree({'H3_attempt_stop_update': requested})
        self.stop = None
        if requested is not None:
            def parse():
                stop = int(requested)
                if (stop % updates_per_rollout or not start_optimizer_steps < stop <= config.training.max_updates):
                    raise ValueError('H3 attempt stop must advance to a complete rollout within the original budget')
                return stop
            self.stop = session.local('H3 operational attempt boundary', parse)

    def should_yield(self, progress):
        if self.stop is None or progress.optimizer_steps < self.stop:
            return False
        if progress.optimizer_steps == self.config.training.max_updates:
            return False
        if progress.optimizer_steps % self.updates_per_rollout:
            raise RuntimeError('H3 attempt stop is not at a resumable complete rollout')
        return True

    def publish(self, progress, checkpoint):
        nonce = os.environ.get('H3_ATTEMPT_NONCE', '')
        if not re.fullmatch(r'[A-Za-z0-9_.-]{1,128}', nonce):
            raise ValueError('H3 boundary handoff requires a safe unique nonce')
        io = CheckpointIO()
        try:
            io.request('boundary', root=str(self.root), checkpoint=str(checkpoint), nonce=nonce,
                config_sha256=self.config.fingerprint(), progress=progress.state_dict(),
                target_optimizer_updates=self.config.training.max_updates,
                world_size=self.session.identity.world_size)
        finally:
            io.close()
