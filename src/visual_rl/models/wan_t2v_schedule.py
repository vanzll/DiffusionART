"""Exact native Wan schedule and frozen-history UniPC replay primitives.

These primitives do not extend the canonical ART objective automatically.
Training must explicitly choose history-conditional mean regression and masks.
"""

import copy
import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class WanT2VSchedule:
    sigmas: torch.Tensor
    timesteps: torch.Tensor
    shift: float

    @property
    def eligible(self):
        # Native starts below1; sigma0 endpoint is not a model-evaluation slot.
        return tuple(range(len(self.timesteps)))


def native_wan_schedule(steps: int = 50, shift: float = 8.0) -> WanT2VSchedule:
    """Reproduce native UniPC's float64 construction and INT64 truncation.

    Native __init__ first rounds 1−1/1000 to float32. set_timesteps then
    interpolates that rounded maximum in NumPy float64, shifts, derives time
    conditions, and finally rounds physical sigmas to float32.
    """
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    if isinstance(shift, bool) or not math.isfinite(shift) or shift <= 0:
        raise ValueError("shift must be finite and positive")
    maximum = torch.tensor(1.0 - 1.0 / 1000, dtype=torch.float32).item()
    raw = np.linspace(maximum, 0, steps + 1).copy()[:-1]
    shifted = shift * raw / (1 + (shift - 1) * raw)
    times = torch.from_numpy(shifted * 1000).to(torch.int64)
    sigmas = torch.from_numpy(np.concatenate([shifted, [0]]).astype(np.float32))
    return WanT2VSchedule(sigmas, times, float(shift))


def require_native_art_replay():
    raise NotImplementedError(
        "Wan T2V native UniPC requires frozen solver history, its actual affine "
        "velocity coefficient, normalized mean regression, and an explicitly "
        "approved corrective-mask contract. Canonical Euler ART replay is incompatible."
    )


def _lambda(sigma):
    return torch.log(1 - sigma) - torch.log(sigma)


def native_unipc_velocity_coefficient(
    sigmas, index: int, *, previous_order: int = 2, corrector_enabled: bool = True
):
    """Analytic scalar a in native next_state=a*v+b, with history frozen.

    Restricted to the official order2/bh2/predict_x0/thresholding=false default.
    Corrector order is the preceding call's this_order, not current order.
    This is a conditional derivative: cached previous predictions never change.
    """
    sigmas = torch.as_tensor(sigmas, dtype=torch.float32, device="cpu")
    if (
        sigmas.ndim != 1
        or len(sigmas) < 2
        or not torch.isfinite(sigmas).all()
        or (sigmas[:-1] <= 0).any()
        or (sigmas >= 1).any()
        or sigmas[-1] != 0
        or not (sigmas[:-1] > sigmas[1:]).all()
    ):
        raise ValueError("native UniPC requires a decreasing (0,1) schedule ending at0")
    if type(index) is not int or not 0 <= index < len(sigmas) - 1:
        raise ValueError("native step index out of range")
    if previous_order not in (1, 2) or (index == 1 and previous_order != 1):
        raise ValueError("previous UniPC order must match native warmup")
    s, sn = sigmas[index : index + 2]
    alpha, alpha_next = 1 - s, 1 - sn
    current_m_in_corrector = torch.tensor(0.0)
    if index > 0 and corrector_enabled:
        h_c = _lambda(s) - _lambda(sigmas[index - 1])
        bh_c = torch.expm1(-h_c)
        if previous_order == 1:
            rho_last = torch.tensor(0.5)
        else:
            if index < 2:
                raise ValueError("order2 corrector requires two previous steps")
            ratio = (_lambda(sigmas[index - 2]) - _lambda(sigmas[index - 1])) / h_c
            phi = bh_c / (-h_c) - 1
            b0 = phi / bh_c
            b1 = (phi / (-h_c) - 0.5) * 2 / bh_c
            matrix = torch.stack((torch.ones(2), torch.stack((ratio, torch.ones_like(ratio)))))
            rho_last = torch.linalg.solve(matrix, torch.stack((b0, b1)))[-1]
        current_m_in_corrector = -alpha * bh_c * rho_last
    # Native warmup and lower_order_final: [1,2,...,2,1].
    order = min(2, index + 1, len(sigmas) - 1 - index)
    h_p = _lambda(sn) - _lambda(s)
    bh_p = torch.expm1(-h_p)
    m_weight = torch.tensor(1.0)
    if order == 2:
        ratio = (_lambda(sigmas[index - 1]) - _lambda(s)) / h_p
        m_weight = 1 - 0.5 / ratio
    coefficient_on_m = sn / s * current_m_in_corrector - alpha_next * bh_p * m_weight
    return -s * coefficient_on_m


_HISTORY_FIELDS = (
    "sigmas",
    "timesteps",
    "model_outputs",
    "timestep_list",
    "last_sample",
    "_step_index",
    "_begin_index",
    "this_order",
    "lower_order_nums",
    "num_inference_steps",
)


def _copy_tree(value, *, device="cpu"):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device).clone()
    if isinstance(value, (list, tuple)):
        return type(value)(_copy_tree(v, device=device) for v in value)
    if isinstance(value, dict):
        return {k: _copy_tree(v, device=device) for k, v in value.items()}
    return copy.deepcopy(value)


@dataclass(frozen=True)
class NativeUniPCSnapshot:
    config: dict
    history: dict
    state: torch.Tensor
    timestep: torch.Tensor

    def state_dict(self):
        return _copy_tree(
            {
                "config": self.config,
                "history": self.history,
                "state": self.state,
                "timestep": self.timestep,
            }
        )

    @classmethod
    def from_state_dict(cls, state):
        if set(state) != {"config", "history", "state", "timestep"}:
            raise ValueError("invalid native UniPC snapshot fields")
        return cls(**_copy_tree(state))


def capture_native_unipc(scheduler, state, timestep) -> NativeUniPCSnapshot:
    """Capture BEFORE step(), including pre-corrector model input and history."""
    config = {
        key: value for key, value in dict(scheduler.config).items() if not key.startswith("_")
    }
    wanted = {
        "solver_order": 2,
        "solver_type": "bh2",
        "predict_x0": True,
        "prediction_type": "flow_prediction",
        "thresholding": False,
        "lower_order_final": True,
        "final_sigmas_type": "zero",
    }
    for key, value in wanted.items():
        if config.get(key) != value:
            raise ValueError(f"unsupported native UniPC config: {key}")
    if getattr(scheduler, "solver_p", None) is not None:
        raise ValueError("external predictor is not supported")
    if not isinstance(state, torch.Tensor) or not torch.isfinite(state).all():
        raise ValueError("native solver input must be a finite tensor")
    timestep = torch.as_tensor(timestep)
    if timestep.numel() != 1 or timestep.dtype != torch.int64:
        raise ValueError("native solver timestep must be one INT64 value")
    history = {
        key: _copy_tree(getattr(scheduler, key))
        for key in _HISTORY_FIELDS
        if hasattr(scheduler, key)
    }
    if history.get("num_inference_steps") is None:
        raise ValueError("native scheduler has not set its schedule")
    # The default may exist as an attribute rather than config in future wrappers.
    history["disable_corrector"] = _copy_tree(scheduler.disable_corrector)
    return NativeUniPCSnapshot(
        _copy_tree(config), history, _copy_tree(state), _copy_tree(timestep.reshape(()))
    )


def replay_native_unipc(snapshot: NativeUniPCSnapshot, velocity, *, scheduler_factory):
    """Call the unmodified native step with detached history; retain gradient in v.

    Factory is injected so importing this module never imports native Wan or
    optional Diffusers. The caller pins and audits that factory's source.
    """
    if velocity.shape != snapshot.state.shape or not torch.isfinite(velocity).all():
        raise ValueError("native replay velocity shape/finiteness mismatch")
    scheduler = scheduler_factory(**copy.deepcopy(snapshot.config))
    for key, value in snapshot.history.items():
        # Native schedules deliberately stay on CPU, including during GPU replay.
        device = "cpu" if key == "sigmas" else velocity.device
        setattr(scheduler, key, _copy_tree(value, device=device))
    return scheduler.step(
        velocity,
        snapshot.timestep.to(velocity.device),
        snapshot.state.to(velocity.device),
        return_dict=False,
    )[0]
