"""SD3 grid parity with the historical model scheduler, independent of method."""

from types import SimpleNamespace

import pytest
import torch

from visual_rl.backends import sd3


@pytest.mark.parametrize("algorithm", [
    "diffusionart", "flow_grpo", "flow_grpo_fast", "flow_dppo", "diffusionnft", "dancegrpo",
])
@pytest.mark.parametrize("steps", [3, 10, 40])
@pytest.mark.parametrize("shift", [1., 3.])
def test_sd3_grid_matches_native_scheduler_for_every_algorithm(algorithm, steps, shift):
    diffusers = pytest.importorskip("diffusers")
    config = SimpleNamespace(
        model=SimpleNamespace(name="sd3"),
        algorithm=SimpleNamespace(name=algorithm),
        sampling=SimpleNamespace(options={"shift": shift}),
    )
    reference = diffusers.FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=shift)
    reference.set_timesteps(steps)

    torch.testing.assert_close(sd3._schedule(config, steps), reference.sigmas, rtol=0, atol=0)
