from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch

from visual_rl.backends.minimax_h3_checkpoint_audit import fingerprint
from visual_rl.backends.minimax_h3_input_audit import audit_inputs, plain_inputs
from visual_rl.models.minimax_h3 import MiniMaxH3Conditioning
from visual_rl.training.audio_video_types import AVState
from visual_rl.training.checkpoint import rng_state
from visual_rl.training.state import Progress


@dataclass
class Objective:
    state: AVState
    predict: object
    device: object
    statistics: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)


def test_audit_covers_latents_conditioning_and_masks_without_consuming_rng(tmp_path, monkeypatch):
    conditioning = MiniMaxH3Conditioning(torch.ones(2, 3, 4), torch.ones(2, 3, dtype=torch.bool))
    def predict(state):
        return state, conditioning
    objective = Objective(AVState(torch.ones(2, 2), torch.zeros(2, 3)), predict, torch.device("cpu"))
    session = SimpleNamespace(device=torch.device("cpu"), identity=SimpleNamespace(rank=0, world_size=1))
    rows = [SimpleNamespace(sample_id="prompt", text="a sound", metadata={"reward_text": "sound"})]
    generator = torch.Generator().manual_seed(73)
    before_rng, before_generator = fingerprint(rng_state(cuda_device="cpu")), fingerprint(generator.get_state())
    def capture(nonce):
        monkeypatch.setenv("H3_ATTEMPT_NONCE", nonce)
        return audit_inputs(session, objective, rows, generator, Progress(attempts=1), tmp_path)
    original = capture("original")
    objective.state.video.add_(1)
    latent_change = capture("latent-change")
    assert latent_change["hashes"]["objective"] != original["hashes"]["objective"]
    assert latent_change["hashes"]["conditioning"] == original["hashes"]["conditioning"]
    conditioning.prompt_embeds.add_(1)
    conditioning_change = capture("conditioning-change")
    assert conditioning_change["hashes"]["conditioning"] != latent_change["hashes"]["conditioning"]
    conditioning.attention_mask[0, 0] = False
    mask_change = capture("mask-change")
    assert mask_change["hashes"]["conditioning"] != conditioning_change["hashes"]["conditioning"]
    assert fingerprint(rng_state(cuda_device="cpu")) == before_rng
    assert fingerprint(generator.get_state()) == before_generator
    with pytest.raises(FileExistsError):
        capture("mask-change")


def test_audit_rejects_unhandled_input_types():
    with pytest.raises(TypeError, match="Unsupported H3 input audit"):
        plain_inputs(object())
