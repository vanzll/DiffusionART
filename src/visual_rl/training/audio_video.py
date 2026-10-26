"""Owner-step orchestration for opt-in AV ART U2 / NFT U1 / H3 Fast U2."""
from __future__ import annotations

import torch

from .audio_video_objectives import AVARTObjective, AVFastObjective, AVNFTObjective, validate_groups
from .update import accumulated_step
from .video_nft import VideoNFTState


class AVNFTState(VideoNFTState):
    """Existing named/shard-aware old shadow plus resumable refresh count.

    No reference model exists. ``stateful()`` plugs directly into the existing
    checkpoint machinery; a refresh is legal only after a successful U1 step.
    Trainable parameter aliases are validated by the existing shadow class.
    """
    def __init__(self, model):
        super().__init__(model, {"old_policy": {"mode": "slow-ema", "rate": .00125, "maximum": .6}})
        self.successful_updates = 0

    def state_dict(self):
        return {"successful_updates": self.successful_updates}

    def load_state_dict(self, state):
        if set(state) != {"successful_updates"} or type(state["successful_updates"]) is not int or state["successful_updates"] < 0:
            raise ValueError("invalid AV NFT update counter checkpoint")
        self.successful_updates = state["successful_updates"]

    def stateful(self):
        return {**super().stateful(), "av_nft_counters": self}

    def after_update(self, successful_updates=None):
        expected = self.successful_updates + 1
        if successful_updates is not None and successful_updates != expected:
            raise ValueError("AV NFT refresh count must advance exactly once")
        metrics = super().after_update(expected)
        self.successful_updates = expected
        return metrics


def prompt_disjoint_windows(groups: torch.Tensor, selected_count: int):
    """Each prompt contributes K/2 rows to each window, keeping all positions."""
    validate_groups(groups, len(groups), even=True)
    if type(selected_count) is not int or selected_count < 1:
        raise ValueError("U2 requires at least one training position")
    windows = [[], []]
    for group in groups.detach().cpu().unique(sorted=True):
        rows = (groups.detach().cpu() == group).nonzero().flatten().tolist()
        half = len(rows) // 2
        for window, subset in enumerate((rows[:half], rows[half:])):
            windows[window].extend((row, slot) for row in subset for slot in range(selected_count))
    return tuple(tuple(window) for window in windows)


def _has_global_guidance(model, active):
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("no AV trainable parameters")
    flag = torch.tensor(int(active), dtype=torch.int32, device=parameters[0].device)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX, group=getattr(model, "process_group", None))
    return bool(flag.item())


def run_av_updates(model, optimizer, objective, groups, *, algorithm: str, microbatch: int,
                   max_grad_norm: float, state: AVNFTState | None = None, scheduler=None,
                   require_nonzero_grad=False, scaler=None):
    """Actual owner of optimizer.step; never equate forward count with updates.

    Complete-group statistics and old/noise caches already exist on entry.
    Every window uses its original sample*position denominator regardless of
    masking/padding/microbatch size. ``accumulated_step`` owns distributed
    weighting, finite checks and terminal collective-failure policy.
    NFT refreshes old exactly once after its successful step; a failed call
    cannot publish counters/EMA/checkpoint as a successful rollout.
    """
    expected = {"art": AVARTObjective, "nft": AVNFTObjective,
                "flow_grpo_fast": AVFastObjective}.get(algorithm)
    if expected is None or not isinstance(objective, expected):
        raise ValueError("AV algorithm and objective type must agree")
    if algorithm == "nft":
        if not isinstance(state, AVNFTState):
            raise ValueError("NFT U1 requires an AVNFTState owner for one post-update EMA refresh")
        validate_groups(groups, int(objective.rows.max()) + 1)
        windows = (objective.items,)
    else:
        if state is not None:
            raise ValueError("ART/Fast must not construct an NFT old-policy state")
        windows = prompt_disjoint_windows(groups, len(objective.cache.selected))
    # Existing scalar ART alone has a zero-covariance skip rule. NFT's
    # advantage-zero mirror still restores current toward EMA-old; Fast owns
    # its promised U2 even when centered rewards are zero. Reward inactivity
    # is therefore not a valid generic optimizer-skip predicate.
    if algorithm == "art" and not _has_global_guidance(model, objective.has_guidance or getattr(objective, "has_reference", False)):
        return {"train/optimizer_steps_per_rollout": 0., "train/rollout_skipped_zero_signal": 1.}
    objective.diagnostics.clear()
    objective._diagnostic_items = {}
    metrics = {"train/optimizer_steps_per_rollout": 0., "train/rollout_skipped_zero_signal": 0.}
    completed = []
    for index, items in enumerate(windows):
        values = accumulated_step(model, optimizer, items, objective, microbatch=microbatch,
            denominator=len(items), max_grad_norm=max_grad_norm,
            require_nonzero_grad=require_nonzero_grad, scaler=scaler)
        metrics["train/optimizer_steps_per_rollout"] += 1
        completed.append(values)
        for key, value in values.items():
            metrics[f"av/window_{index}/{key.removeprefix('train/')}"] = value
        if algorithm == "nft":
            metrics.update(state.after_update())
        if scheduler is not None:
            scheduler.step()
    for key in completed[0]:
        metrics[key] = sum(value[key] for value in completed) / len(completed)
    # Global and per-window snapshots share ONE ordered post-update reduction.
    # Sum/count/second moments are derived from immutable item identities, so
    # retries overwrite an item and masked items retain their original weight.
    def key(item):
        return tuple(item) if isinstance(item, (tuple, list)) else int(item)
    scopes = [(None, tuple(objective._diagnostic_items.values()))]
    scopes.extend((index, tuple(objective._diagnostic_items[key(item)] for item in items))
                  for index, items in enumerate(windows))
    names = sorted(next(iter(objective._diagnostic_items.values())))
    records, moments = [], []
    for window, items in scopes:
        for name in names:
            values = [item[name] for item in items]
            output = name if window is None else f"av/window_{window}/{name.removeprefix('av/')}"
            records.append((name, output))
            selector = next((f"av/{modality}_selected" for modality in ("video", "audio")
                             if name.startswith(f"av/{modality}_") and not name.endswith("_selected")), None)
            weights = [item.get(selector, 1.) for item in items]
            moments.append([sum(v * w for v, w in zip(values, weights)), sum(weights),
                            sum(v * v * w for v, w in zip(values, weights))])
    totals = torch.tensor(moments, dtype=torch.float64, device=next(model.parameters()).device)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(totals, group=getattr(model, "process_group", None))
    for (name, output), (total, count, square) in zip(records, totals.cpu().tolist()):
        mean = total / count
        if name.endswith("_advantage"):
            metrics[output + "_mean"] = mean
            metrics[output + "_std"] = max(0., square / count - mean * mean) ** .5
        else:
            metrics[output] = mean
    return metrics
