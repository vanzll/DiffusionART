"""Small-video identity/budget envelope over canonical checkpoint adapters.

The convenience writer is local/unsharded. Distributed adapters store one
envelope and step journal per declared rank; the caller coordinates collectives.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

import torch

from . import checkpoint
from .small_video_protocol import ActualStepBudget, RolloutSelection
from .state import Progress


def _json_value(value):
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    if isinstance(value, set):
        return sorted(_json_value(v) for v in value)
    return value


def _hash(value):
    return hashlib.sha256(
        json.dumps(
            _json_value(value), sort_keys=True, allow_nan=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


@dataclass(frozen=True)
class SmallVideoIdentity:
    model_name: str
    finetune_mode: str
    timestep_mode: str
    native_code_revision: str
    model_revision: str
    scorer_revision: str
    data_revision: str
    data_sha256: str
    protocol_sha256: str
    paired_eval_sha256: str
    config_sha256: str
    step_limit: int = 10
    paired_eval_count: int = 8

    def __post_init__(self):
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model_name must be explicit")
        if self.finetune_mode not in ("lora", "full") or self.timestep_mode not in (
            "original",
            "flash",
        ):
            raise ValueError("small-video fine-tuning/selection identity differs")
        for key in ("native_code_revision", "model_revision", "scorer_revision", "data_revision"):
            value = getattr(self, key)
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
                raise ValueError(f"{key} requires an immutable revision")
        for key in ("data_sha256", "protocol_sha256", "paired_eval_sha256", "config_sha256"):
            value = getattr(self, key)
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"{key} requires a SHA256 identity")
        if type(self.step_limit) is not int or self.step_limit != 10:
            raise ValueError("small-video cumulative actual-step limit must remain10")
        if type(self.paired_eval_count) is not int or self.paired_eval_count != 8:
            raise ValueError("small-video paired evaluation must retain the same8 examples")

    def state_dict(self):
        return asdict(self)

    @property
    def sha256(self):
        return _hash(self.state_dict())


def _policy_identity(model):
    peft = getattr(model, "peft_config", {})
    return {
        "parameters": {
            name: {"shape": list(p.shape), "dtype": str(p.dtype), "trainable": p.requires_grad}
            for name, p in model.named_parameters()
        },
        "peft": _json_value({name: config.to_dict() for name, config in peft.items()}),
    }


def _validate_budget(state, *, world_size=1):
    if (
        not isinstance(state, dict)
        or set(state) != {"limit", "completed", "completed_rollouts", "memberships"}
        or state["limit"] != 10
        or type(state["completed"]) is not int
        or not 0 <= state["completed"] <= 10
        or state["completed"] % 2
        or state["completed_rollouts"] != state["completed"] // 2
        or len(state["memberships"]) != state["completed"]
    ):
        raise ValueError("checkpoint requires the same cumulative10step complete-rollout budget")
    for index, member in enumerate(state["memberships"]):
        if (
            set(member)
            != {
                "rollout",
                "window",
                "rows",
                "selected_positions",
                "global_loss_denominator",
                "actual_global_optimizer_step",
            }
            or (member["rollout"], member["window"]) != divmod(index, 2)
            or member["actual_global_optimizer_step"] != index + 1
            or len(member["rows"]) != 64 // world_size
            or len(set(member["rows"])) != 64 // world_size
            or any(
                type(row) is not int or not 0 <= row < 128 // world_size for row in member["rows"]
            )
            or not member["selected_positions"]
            or len(set(member["selected_positions"])) != len(member["selected_positions"])
            or any(type(slot) is not int or slot < 0 for slot in member["selected_positions"])
            or member["global_loss_denominator"]
            != (64 // world_size) * len(member["selected_positions"])
        ):
            raise ValueError("checkpoint actual-step/window/sample membership differs")
        if index % 2:
            previous = state["memberships"][index - 1]
            if (
                set(previous["rows"]) & set(member["rows"])
                or previous["selected_positions"] != member["selected_positions"]
            ):
                raise ValueError("two frozen-target windows must be disjoint and share selection")


class SmallVideoResumeState:
    """Own the step hooks, selection RNG, data cursor and durable budget fence.

    One writer owns run_root (one journal directory per distributed rank).
    A pre-step journal admission survives a crash;
    unresolved admissions or a journal newer than a checkpoint fail closed.
    Resume supports complete two-window rollout boundaries only.
    """

    def __init__(
        self,
        *,
        run_root,
        identity: SmallVideoIdentity,
        model,
        optimizer,
        data_cursor,
        selection=None,
        budget=None,
        rank=0,
        world_size=1,
        distributed=False,
        policy_identity=None,
        read_only=False,
    ):
        if type(read_only) is not bool:
            raise ValueError("read_only must be an explicit boolean")
        self.read_only = read_only
        if (
            type(rank) is not int
            or type(world_size) is not int
            or world_size < 1
            or 16 % world_size
            or not 0 <= rank < world_size
        ):
            raise ValueError("small-video journal layout requires complete local prompt groups")
        if torch.distributed.is_initialized() and not distributed:
            raise ValueError(
                "distributed checkpoint envelopes require explicit rank-local journal ownership"
            )
        self.layout = {"rank": rank, "world_size": world_size}
        self.identity, self.optimizer = identity, optimizer
        self.selection = RolloutSelection() if selection is None else selection
        self.data_cursor = data_cursor
        self.policy_identity = (
            _policy_identity(model) if policy_identity is None else copy.deepcopy(policy_identity)
        )
        if bool(self.policy_identity["peft"]) != (identity.finetune_mode == "lora"):
            raise ValueError("policy adapter/full mode differs from checkpoint identity")
        cursor = data_cursor.state_dict()
        if cursor.get("dataset_sha256") != identity.data_sha256:
            raise ValueError("data cursor dataset provenance differs")
        self.root = Path(run_root)
        if any(path.is_symlink() for path in (self.root, *self.root.parents)):
            raise ValueError("small-video checkpoint root must not be a symlink")
        self.root.mkdir(parents=True, exist_ok=True)
        self.ledger = self.root / "small-video-step-ledger.json"
        self.budget = ActualStepBudget(optimizer, limit=10) if budget is None else budget
        if self.budget.limit != 10:
            raise ValueError("small-video cumulative actual-step limit must remain10")
        self._journal_handles = []
        if self.ledger.exists():
            self._read_ledger()
            self._ready = False  # Fresh objects cannot train before exact checkpoint restoration.
        else:
            if self.read_only:
                raise ValueError("read-only evaluation requires the original durable step journal")
            if self.budget.completed:
                raise ValueError(
                    "cannot initialize a new journal with already consumed optimizer steps"
                )
            self._write_ledger(completed=0, pending=None, memberships=[])
            self._ready = True
        self._attach_journal()

    def _write_ledger(self, *, completed, pending, memberships):
        checkpoint._atomic_json(
            self.ledger,
            {
                "schema_version": 1,
                "identity_sha256": self.identity.sha256,
                "layout": self.layout,
                "limit": 10,
                "completed": completed,
                "pending": pending,
                "membership_sha256": _hash(memberships),
            },
        )

    def _read_ledger(self):
        if self.ledger.is_symlink() or self.ledger.stat().st_size > 64 * 1024:
            raise ValueError("invalid small-video step journal")
        value = json.loads(self.ledger.read_text())
        if (
            set(value)
            != {
                "schema_version",
                "identity_sha256",
                "layout",
                "limit",
                "completed",
                "pending",
                "membership_sha256",
            }
            or value["schema_version"] != 1
            or value["identity_sha256"] != self.identity.sha256
            or value["layout"] != self.layout
            or value["limit"] != 10
            or type(value["completed"]) is not int
            or not 0 <= value["completed"] <= 10
        ):
            raise ValueError("small-video durable journal identity/budget differs")
        return value

    def _before_step(self, optimizer, args, kwargs):
        if self.read_only:
            raise RuntimeError("read-only checkpoint evaluation forbids every optimizer step")
        current = self._read_ledger()
        if (
            not self._ready
            or self.budget._active is None
            or current["pending"] is not None
            or current["completed"] != self.budget.completed
            or current["membership_sha256"] != _hash(self.budget.memberships)
            or self.budget.completed >= 10
        ):
            raise RuntimeError(
                "actual-step journal denies stale, unadmitted or exhausted optimizer step"
            )
        self._write_ledger(
            completed=self.budget.completed,
            pending=self.budget.completed + 1,
            memberships=self.budget.memberships,
        )

    def _after_step(self, optimizer, args, kwargs):
        current = self._read_ledger()
        if current["pending"] != self.budget.completed:
            raise RuntimeError("actual optimizer post-step and durable admission differ")
        self._write_ledger(
            completed=self.budget.completed, pending=None, memberships=self.budget.memberships
        )

    def _attach_journal(self):
        self._journal_handles = [
            self.optimizer.register_step_pre_hook(self._before_step),
            self.optimizer.register_step_post_hook(self._after_step),
        ]

    def _boundary(self):
        budget = self.budget.state_dict()
        _validate_budget(budget, world_size=self.layout["world_size"])
        ledger = self._read_ledger()
        exact = (
            ledger["pending"] is None
            and ledger["completed"] == budget["completed"]
            and ledger["membership_sha256"] == _hash(budget["memberships"])
        )
        read_only = self.read_only and ledger["completed"] >= budget["completed"]
        if (not self._ready and not self.read_only) or not (exact or read_only):
            raise ValueError("checkpoint does not match the durable completed optimizer boundary")
        if self.selection.rollouts != budget["completed_rollouts"]:
            raise ValueError("selection RNG rollout count differs from actual completed rollouts")
        return budget

    def validate_progress(self, progress):
        Progress.from_state_dict(progress.state_dict())
        budget = self._boundary()
        if (
            progress.optimizer_steps != budget["completed"]
            or progress.completed_rollouts != budget["completed_rollouts"]
        ):
            raise ValueError("progress must retain actual cumulative steps and completed rollouts")

    def state_dict(self):
        if self.read_only:
            raise ValueError("read-only evaluation state cannot publish a training checkpoint")
        cursor = copy.deepcopy(self.data_cursor.state_dict())
        if cursor.get("dataset_sha256") != self.identity.data_sha256:
            raise ValueError("live data cursor provenance changed")
        return {
            "schema_version": 1,
            "identity": self.identity.state_dict(),
            "layout": self.layout,
            "policy_identity": copy.deepcopy(self.policy_identity),
            "budget": copy.deepcopy(self._boundary()),
            "selection": self.selection.state_dict(),
            "data_cursor": cursor,
        }

    def load_state_dict(self, state):
        if (
            set(state)
            != {
                "schema_version",
                "identity",
                "layout",
                "policy_identity",
                "budget",
                "selection",
                "data_cursor",
            }
            or state["schema_version"] != 1
            or state["identity"] != self.identity.state_dict()
            or state["layout"] != self.layout
            or state["policy_identity"] != self.policy_identity
        ):
            raise ValueError("small-video checkpoint frozen identity/policy differs")
        _validate_budget(state["budget"], world_size=self.layout["world_size"])
        budget, ledger = state["budget"], self._read_ledger()
        exact = (
            ledger["pending"] is None
            and ledger["completed"] == budget["completed"]
            and ledger["membership_sha256"] == _hash(budget["memberships"])
        )
        read_only = self.read_only and ledger["completed"] >= budget["completed"]
        if not (exact or read_only):
            raise ValueError("resume refuses older, incomplete or uncertain actual-step accounting")
        if state["selection"].get("rollouts") != budget["completed_rollouts"]:
            raise ValueError("selection RNG rollout count differs from checkpoint budget")
        if state["data_cursor"].get("dataset_sha256") != self.identity.data_sha256:
            raise ValueError("checkpoint data cursor provenance differs")
        # Canonical load treats downstream failures as terminal; never reuse a
        # model/optimizer after any component rejects its schema.
        self.data_cursor.load_state_dict(state["data_cursor"])
        self.selection.load_state_dict(state["selection"])
        for handle in self._journal_handles:
            handle.remove()
        self.budget.close()
        self.budget = ActualStepBudget.from_state_dict(self.optimizer, budget)
        self._attach_journal()  # Post-journal hook must follow the budget's counter hook.
        self._ready = not self.read_only

    def close(self):
        for handle in self._journal_handles:
            handle.remove()
        self._journal_handles.clear()
        self.budget.close()


def save_small_video_checkpoint(
    root, *, model, progress, state: SmallVideoResumeState, generators=None
):
    state.validate_progress(progress)
    if _policy_identity(model) != state.policy_identity:
        raise ValueError("live policy structure/fine-tuning identity changed")
    if Path(root).resolve() != state.root.resolve():
        raise ValueError("checkpoint publication must share the durable step-journal root")
    return checkpoint.save_checkpoint(
        root,
        model=model,
        optimizer=state.optimizer,
        progress=progress,
        config_sha256=state.identity.config_sha256,
        stateful={"small_video": state},
        generators=generators,
        keep_last=2,
    )


def load_small_video_checkpoint(receipt, *, model, state: SmallVideoResumeState, generators=None):
    if Path(receipt).resolve().parent != state.root.resolve():
        raise ValueError("resume requires the original durable step-journal root")
    if _policy_identity(model) != state.policy_identity:
        raise ValueError("fresh policy structure differs from frozen checkpoint identity")
    progress = checkpoint.load_checkpoint(
        receipt,
        model=model,
        optimizer=state.optimizer,
        config_sha256=state.identity.config_sha256,
        stateful={"small_video": state},
        generators=generators,
    )
    state.validate_progress(progress)
    return progress


def export_small_video_final(
    root, *, model, state: SmallVideoResumeState, checkpoint_receipt, name="final"
):
    """Publish a named inference export separately from resumable checkpoints."""
    budget = state._boundary()
    if budget["completed"] != 10:
        raise ValueError("named final export requires the completed cumulative10step budget")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("final export name must be a single safe path component")
    if _policy_identity(model) != state.policy_identity or any(
        p.grad is not None for p in model.parameters()
    ):
        raise ValueError("final export requires unchanged policy identity and cleared gradients")
    receipt = Path(checkpoint_receipt)
    if receipt.resolve().parent != state.root.resolve():
        raise ValueError("final checkpoint must share its durable journal root")
    record = json.loads(receipt.read_text())
    filename = record.get("file")
    if (
        not isinstance(filename, str)
        or Path(filename).name != filename
        or record.get("optimizer_step") != 10
        or record.get("config_sha256") != state.identity.config_sha256
    ):
        raise ValueError("final checkpoint receipt identity differs")
    payload = receipt.parent / filename
    if payload.stat().st_size != record["bytes"] or checkpoint._digest(payload) != record["sha256"]:
        raise ValueError("final checkpoint integrity differs")
    root = Path(root)
    if any(path.is_symlink() for path in (root, *root.parents)):
        raise ValueError("final export root must not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    target = root / name
    if target.exists() or target.is_symlink():
        raise FileExistsError("named final export already exists; never replace previous evidence")
    stage = Path(tempfile.mkdtemp(prefix=f".{name}.pending-", dir=root))
    try:
        if state.identity.finetune_mode == "lora":
            model.save_pretrained(stage / "adapter", safe_serialization=True)
            kind = "peft_adapter"
        else:
            with (stage / "policy.pt").open("wb") as stream:
                torch.save(model.state_dict(), stream)
                stream.flush()
                os.fsync(stream.fileno())
            kind = "state_dict"
        files = {}
        for path in sorted(stage.rglob("*")):
            if path.is_symlink():
                raise ValueError("export produced a symlink")
            if path.is_file():
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
                files[path.relative_to(stage).as_posix()] = {
                    "sha256": checkpoint._digest(path),
                    "bytes": path.stat().st_size,
                }
        checkpoint._atomic_json(
            stage / "complete.json",
            {
                "schema_version": 1,
                "complete": True,
                "kind": kind,
                "identity": state.identity.state_dict(),
                "policy_identity": state.policy_identity,
                "actual_global_optimizer_steps": 10,
                "completed_rollouts": 5,
                "source_checkpoint": record,
                "files": files,
            },
        )
        os.rename(stage, target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return target
