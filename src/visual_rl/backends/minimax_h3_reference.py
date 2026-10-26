"""Immutable initial-policy reference, using local CPU shadows rather than a second DiT.

Create the reference before loading a resumed current policy, then include
``stateful()`` in the checkpoint. Reference inference must finish before any
current-policy forward whose graph will participate in backward.
"""
from contextlib import contextmanager

import torch

from ..training.audio_video_rollout import validate_state
from ..training.audio_video_types import AVState
from ..training.state import PolicyShadow


class ReferenceState:
    """A frozen pretrained parameter snapshot; no EMA/update operation is exposed."""

    def __init__(self, model):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        if isinstance(model, FSDP):
            from ..training.fsdp_shadow import FSDPPolicyShadow

            self._shadow = FSDPPolicyShadow(model)
        else:
            self._shadow = PolicyShadow(model, trainable_only=True)

    def stateful(self):
        return {"h3_reference_policy": self}

    def state_dict(self):
        return {"schema_version": 1, "kind": "frozen_initial_policy",
                "parameters": self._shadow.state_dict()}

    def load_state_dict(self, state):
        if (not isinstance(state, dict)
                or state.keys() != {"schema_version", "kind", "parameters"}
                or state["schema_version"] != 1 or state["kind"] != "frozen_initial_policy"):
            raise ValueError("H3 reference checkpoint schema/identity differs")
        self._shadow.load_state_dict(state["parameters"])

    @contextmanager
    def use_reference(self, session=None):
        """All swap/restore operations are local; only predict calls collectives."""
        context = self._shadow.use()
        if session is None:
            with context, torch.no_grad():
                yield
            return
        session.local("H3 pretrained-reference swap", context.__enter__)
        try:
            with torch.no_grad():
                yield
        except BaseException as error:
            # Collective failures are terminal. Do not enter another vote while
            # another rank may still be inside its failed model forward.
            context.__exit__(type(error), error, error.__traceback__)
            raise
        else:
            session.local("H3 current-policy restore", lambda: context.__exit__(None, None, None))


@torch.no_grad()
def prepare_reference_predictions(cache, predict, device, microbatch):
    """Return detached CPU AV batches, one for each cached original position.

    Caller owns the frozen reference swap around this complete operation.
    Original row IDs and paired modality sigmas/conditioning are preserved.
    No current-policy autograd graph or parameter swap may be live here.
    """
    if type(microbatch) is not int or microbatch < 1:
        raise ValueError("reference prediction microbatch must be positive")
    if not cache.selected or len(cache.states) != len(cache.selected):
        raise ValueError("reference predictions require selected cached AV states")
    output = []
    for slot, step in enumerate(cache.selected):
        source = cache.states[slot]
        validate_state(source)
        values = [torch.empty_like(value, dtype=torch.float32, device="cpu")
                  for value in (source.video, source.audio)]
        for start in range(0, len(source), microbatch):
            stop = min(start + microbatch, len(source))
            rows = torch.arange(start, stop, device=device)
            state = source.select(slice(start, stop)).to(device)
            sv, sa = (schedule[step].to(device).expand(stop - start)
                      for schedule in (cache.video_sigmas, cache.audio_sigmas))
            prediction = predict(state, sv, sa, rows)
            validate_state(prediction, like=state)
            for target, name in zip(values, ("video", "audio")):
                target[start:stop].copy_(getattr(prediction, name).detach().float().cpu())
        output.append(AVState(*values,
            None if source.video_mask is None else source.video_mask.detach().cpu().clone(),
            None if source.audio_mask is None else source.audio_mask.detach().cpu().clone()))
    return tuple(output)
