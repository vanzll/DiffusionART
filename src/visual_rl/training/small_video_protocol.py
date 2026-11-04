"""Frozen group-reward, membership and actual-step contracts for small T2V.

These utilities do not choose a model sampler or assert that its replay is ART
compatible. Native transition/mask semantics must be validated separately.
"""

from collections.abc import Mapping

import torch


@torch.no_grad()
def group_video_reward(raw: Mapping[str, torch.Tensor], groups: torch.Tensor, *, k=8):
    """Normalize VQ/MQ independently over COMPLETE K groups, then average once.

    TA is required and checked but only monitored. FP64 statistics avoid false
    signal from summing equal low-precision rewards; model targets are FP32.
    No global/rank-local denominator, fixed scorer scale or second normalization.
    """
    if set(raw) != {"VQ", "MQ", "TA"}:
        raise ValueError("require raw finite VQ, MQ and TA, without missing-score defaults")
    if (
        groups.ndim != 1
        or groups.dtype not in (torch.int32, torch.int64)
        or not groups.numel()
        or (groups < 0).any()
        or type(k) is not int
        or k < 2
    ):
        raise ValueError("require nonempty nonnegative integer prompt groups and K>=2")
    device = groups.device
    values = {}
    for name, value in raw.items():
        if not isinstance(value, torch.Tensor) or value.shape != groups.shape:
            raise ValueError("raw reward shapes must match prompt membership")
        value = value.detach().to(device=device, dtype=torch.float32)
        if not torch.isfinite(value).all():
            raise ValueError(f"nonfinite raw {name} reward")
        values[name] = value
    ids, counts = torch.unique(groups, sorted=True, return_counts=True)
    if not bool((counts == k).all()):
        raise ValueError("reward normalization requires complete K groups before splitting")
    normalized = {name: torch.empty_like(values[name]) for name in ("VQ", "MQ")}
    statistics = {}
    for name in ("VQ", "MQ"):
        means, stds = [], []
        for group in ids:
            selection = groups == group
            members = values[name][selection].double()
            mean, std = members.mean(), members.std(unbiased=False)
            if not torch.isfinite(mean) or not torch.isfinite(std):
                raise ValueError("nonfinite group reward statistics")
            normalized[name][selection] = ((members - mean) / (std + 1e-4)).float()
            means.append(mean)
            stds.append(std)
        statistics[name] = {"mean": torch.stack(means), "population_std": torch.stack(stds)}
    target = (normalized["VQ"] + normalized["MQ"]) / 2
    raw_mean = (values["VQ"] + values["MQ"]) / 2
    metrics = {
        f"reward/raw/{name}_mean": float(value.double().mean()) for name, value in values.items()
    }
    metrics["reward/raw/VQ_MQ_equal_mean"] = float(raw_mean.double().mean())
    metrics["reward/normalized/target_mean"] = float(target.double().mean())
    for name, stats in statistics.items():
        metrics[f"reward/group/{name}_population_std_mean"] = float(stats["population_std"].mean())
        metrics[f"reward/group/{name}_near_zero_fraction"] = float(
            (stats["population_std"] <= 1e-4).double().mean()
        )
    return {
        "target": target,
        "normalized": normalized,
        "raw": values,
        "raw_equal_mean": raw_mean,
        "group_ids": ids,
        "statistics": statistics,
        "metrics": metrics,
    }


def group_windows(groups: torch.Tensor, *, k=8):
    """Return K/2 + K/2 disjoint rows from EVERY complete prompt group."""
    if (
        groups.ndim != 1
        or not groups.numel()
        or groups.dtype not in (torch.int32, torch.int64)
        or (groups < 0).any()
        or type(k) is not int
        or k < 2
        or k % 2
    ):
        raise ValueError("two windows require nonnegative integer groups and positive even K")
    windows = [[], []]
    for group in torch.unique(groups, sorted=True):
        members = torch.where(groups == group)[0]
        if len(members) != k:
            raise ValueError("each prompt must have its complete K samples")
        windows[0].append(members[: k // 2])
        windows[1].append(members[k // 2 :])
    return tuple(torch.cat(rows) for rows in windows)


class RolloutSelection:
    """Private checkpointable selection RNG, independent of generation RNG."""

    def __init__(self, seed=42):
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.rollouts = 0

    def choose(self, eligible, *, flash):
        eligible = tuple(eligible)
        if (
            not eligible
            or any(type(i) is not int or i < 0 for i in eligible)
            or tuple(sorted(set(eligible))) != eligible
        ):
            raise ValueError("eligible native positions must be sorted unique nonnegative indices")
        if flash:
            if len(eligible) < 4:
                raise ValueError("Flash requires four eligible native high-noise positions")
            selected = tuple(
                sorted(eligible[i] for i in torch.randperm(4, generator=self.generator)[:2])
            )
        else:
            selected = eligible
        self.rollouts += 1
        return selected

    def state_dict(self):
        return {"generator": self.generator.get_state().clone(), "rollouts": self.rollouts}

    def load_state_dict(self, state):
        if (
            set(state) != {"generator", "rollouts"}
            or type(state["rollouts"]) is not int
            or state["rollouts"] < 0
        ):
            raise ValueError("invalid selection RNG continuation")
        self.generator.set_state(state["generator"])
        self.rollouts = state["rollouts"]


class ActualStepBudget:
    """Count real optimizer post-step calls, with a hard cumulative resume budget.

    Save only at complete-rollout boundaries; mid-rollout continuation would
    additionally require the entire frozen trajectory/reward/cache and is rejected.
    Attaching hooks neither changes gradients nor adds optimizer calls.
    """

    def __init__(self, optimizer, *, limit=10, completed=0):
        if (
            type(limit) is not int
            or limit <= 0
            or limit % 2
            or type(completed) is not int
            or not 0 <= completed <= limit
            or completed % 2
        ):
            raise ValueError(
                "budget/resume counters must be even complete-rollout optimizer counts"
            )
        self.limit, self.completed = limit, completed
        self._active = None
        self.memberships = []
        self._handles = [
            optimizer.register_step_pre_hook(self._before),
            optimizer.register_step_post_hook(self._after),
        ]

    def begin_window(self, *, rollout, window, rows, groups, selected_positions):
        if self._active is not None or self.completed >= self.limit:
            raise RuntimeError("no new window after budget or while another window is active")
        if (rollout, window) != divmod(self.completed, 2):
            raise ValueError("optimizer count, rollout and window continuation disagree")
        expected = group_windows(groups)[window].detach().cpu().tolist()
        rows = list(rows)
        if sorted(rows) != sorted(expected) or len(set(rows)) != len(rows):
            raise ValueError("window must contain exactly K/2 samples from each prompt group")
        if not selected_positions or len(set(selected_positions)) != len(selected_positions):
            raise ValueError("window requires unique actual selected native transitions")
        self._active = {
            "rollout": rollout,
            "window": window,
            "rows": rows,
            "selected_positions": list(selected_positions),
            "global_loss_denominator": len(rows) * len(selected_positions),
        }

    def _before(self, optimizer, args, kwargs):
        if self._active is None or self.completed >= self.limit:
            raise RuntimeError(
                "optimizer.step outside an admitted window or beyond cumulative budget"
            )

    def _after(self, optimizer, args, kwargs):
        self.completed += 1
        self.memberships.append({**self._active, "actual_global_optimizer_step": self.completed})
        self._active = None

    def state_dict(self):
        if self._active is not None or self.completed % 2:
            raise ValueError("save checkpoint only after both frozen-target windows complete")
        return {
            "limit": self.limit,
            "completed": self.completed,
            "completed_rollouts": self.completed // 2,
            "memberships": list(self.memberships),
        }

    @classmethod
    def from_state_dict(cls, optimizer, state):
        if (
            set(state) != {"limit", "completed", "completed_rollouts", "memberships"}
            or state["completed_rollouts"] != state["completed"] // 2
            or len(state["memberships"]) != state["completed"]
            or [x.get("actual_global_optimizer_step") for x in state["memberships"]]
            != list(range(1, state["completed"] + 1))
        ):
            raise ValueError("checkpoint optimizer counter or membership history is inconsistent")
        result = cls(optimizer, limit=state["limit"], completed=state["completed"])
        result.memberships = list(state["memberships"])
        return result

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
