"""Optional H3 pre-update input identity for continuous/resumed comparisons."""

import os
import re
import time
from dataclasses import fields, is_dataclass
from pathlib import Path

import torch

from ..training.checkpoint import _atomic_json, rng_state
from .minimax_h3_checkpoint_audit import fingerprint
from .minimax_h3_observability import event


def plain_inputs(value):
    if is_dataclass(value):
        return {field.name: plain_inputs(getattr(value, field.name)) for field in fields(value)
                if field.name not in ("predict", "device", "statistics", "diagnostics")}
    if isinstance(value, dict):
        return {key: plain_inputs(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(plain_inputs(child) for child in value)
    if value is None or isinstance(value, (torch.Tensor, str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported H3 input audit value: {type(value).__name__}")


def audit_inputs(session, objective, rows, generator, progress, root):
    nonce = os.environ.get("H3_ATTEMPT_NONCE", "")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", nonce):
        raise ValueError("H3 input audit requires a safe unique attempt nonce")
    destination = (Path(root) / "input-audits" / nonce /
                   f"rollout-{progress.attempts:06d}" / f"rank-{session.identity.rank:05d}.json")
    if destination.exists():
        raise FileExistsError(destination)
    begin = time.monotonic()
    prediction = objective.predict
    bindings = dict(zip(prediction.__code__.co_freevars, prediction.__closure__ or ()))
    if "conditioning" not in bindings:
        raise ValueError("H3 input audit requires the native cached-conditioning callback")
    before_rng = fingerprint(rng_state(cuda_device=session.device))
    before_generator = fingerprint(generator.get_state())
    hashes = dict(objective=fingerprint(plain_inputs(objective)),
                  conditioning=fingerprint(plain_inputs(bindings["conditioning"].cell_contents)),
                  prompts=fingerprint([(row.sample_id, row.text, row.metadata) for row in rows]),
                  rng=before_rng, sampling_generator=before_generator)
    if (fingerprint(rng_state(cuda_device=session.device)) != before_rng or
            fingerprint(generator.get_state()) != before_generator):
        raise ValueError("H3 input audit changed RNG or sampling-generator state")
    record = dict(rank=session.identity.rank, world_size=session.identity.world_size,
                  rollout_attempt=progress.attempts, optimizer_steps=progress.optimizer_steps,
                  hashes=hashes, visible_rng_preserved=True, seconds=time.monotonic() - begin,
                  scope="Cached objective tensors/options and conditioning/prompts/RNG; model/Adam are audited separately. Derived statistics and mutable diagnostics are excluded.")
    _atomic_json(destination, record)
    event("rollout_input_audit_complete", optimizer_steps=progress.optimizer_steps,
          rollout_attempt=progress.attempts, hashes=hashes, seconds=record["seconds"])
    return record
