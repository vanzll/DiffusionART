"""Independent AV CPS noise eligibility and exact replay cache contracts."""

import copy

import pytest
import torch

from visual_rl.sampling import flow_transition
from visual_rl.training.audio_video_rollout import collect_av_rollout, h3_cps_transition
from visual_rl.training.audio_video_types import AVState


class CoupledPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, state, sv, sa, rows):
        self.calls.append((sv.clone(), sa.clone(), rows.clone()))
        return AVState(state.video * .1 + state.audio.mean(1)[:, None, None] * .02,
                       state.audio * .2 + state.video.mean((1, 2))[:, None] * .03)


def fixture():
    generator = torch.Generator().manual_seed(73)
    noise = AVState(torch.randn(5, 2, 3, generator=generator), torch.randn(5, 3, generator=generator),
                    torch.tensor([[True, False]] * 5), torch.tensor([[True, True, False]] * 5))
    video = torch.tensor([1., .99, .97, .8, .4, 0.])
    audio = torch.tensor([1., .97, .91, .6, .2, 0.])
    selected = {"video": (0, 3), "audio": (1, 3)}
    union = tuple(sorted(set(selected["video"]) | set(selected["audio"])))
    innovations = {step: AVState(torch.randn(noise.video.shape, generator=generator),
                                 torch.randn(noise.audio.shape, generator=generator)) for step in union}
    return noise, video, audio, selected, union, innovations


def collect(*, microbatch=2, changes=None):
    noise, video, audio, selections, union, innovations = fixture()
    options = dict(selected=union, stochastic=union, innovations=innovations,
                   modality_selected=selections, modality_stochastic=selections,
                   microbatch=microbatch, unselected_kind="cps_zero_noise")
    options.update(changes or {})
    policy = CoupledPolicy()
    return collect_av_rollout(policy, noise, video, audio, **options), policy


def manual(noise, video, audio, stochastic, innovations, selected):
    policy = CoupledPolicy()
    state, cache = noise, {}
    for step in range(len(video) - 1):
        sv, sa = video[step].expand(len(noise)), audio[step].expand(len(noise))
        velocity = policy(state, sv, sa, torch.arange(len(noise)))
        transitions = []
        for name, sigma, schedule in (("video", sv, video), ("audio", sa, audio)):
            if step in stochastic[name]:
                transition = h3_cps_transition(getattr(state, name), getattr(velocity, name),
                    sigma, schedule[step + 1], innovation=getattr(innovations[step], name),
                    mask=getattr(state, name + "_mask"))
            else:
                transition = flow_transition(getattr(state, name), getattr(velocity, name),
                    sigma, schedule[step + 1], kind="cps_zero_noise")
            transitions.append(transition)
        following = AVState(transitions[0].sample, transitions[1].sample,
                            state.video_mask, state.audio_mask)
        if step in selected:
            cache[step] = (state, velocity, following, transitions)
        state = following
    return state, cache


def test_dual_cps_matches_paired_manual_chain_and_replays_ratio_one():
    noise, video, audio, selections, union, innovations = fixture()
    result, policy = collect()
    endpoint, expected = manual(noise, video, audio, selections, innovations, union)
    for name, schedule in (("video", video), ("audio", audio)):
        torch.testing.assert_close(getattr(result.endpoints, name), getattr(endpoint, name), rtol=0, atol=0)
        recorded = getattr(result, "old_" + name + "_log_probs")
        assert recorded.shape == (5, 3) and not recorded.requires_grad
        for slot, step in enumerate(union):
            for storage, index in ((result.states, 0), (result.old_velocities, 1), (result.next_states, 2)):
                torch.testing.assert_close(getattr(storage[slot], name), getattr(expected[step][index], name), rtol=0, atol=0)
            if step not in selections[name]:
                assert torch.equal(recorded[:, slot], torch.zeros(5))
                continue
            state, velocity, following = result.states[slot], result.old_velocities[slot], result.next_states[slot]
            replay = h3_cps_transition(getattr(state, name), getattr(velocity, name), schedule[step],
                schedule[step + 1], observed=getattr(following, name), mask=getattr(state, name + "_mask"))
            torch.testing.assert_close(replay.log_prob, recorded[:, slot], rtol=0, atol=0)
            assert torch.equal(torch.exp(replay.log_prob - recorded[:, slot]), torch.ones(5))
    torch.testing.assert_close(result.old_log_probs,
        .5 * result.old_video_log_probs + .5 * result.old_audio_log_probs, rtol=0, atol=0)
    assert result.video_selected == (0, 3) and result.audio_selected == (1, 3)
    for index, (sv, sa, rows) in enumerate(policy.calls):
        step = index % 5
        assert torch.equal(sv, video[step].expand(len(rows)))
        assert torch.equal(sa, audio[step].expand(len(rows)))
        assert rows.tolist() == ([0, 1], [2, 3], [4])[index // 5]


def test_noise_for_inactive_modality_is_never_injected():
    baseline, _ = collect()
    *_, innovations = fixture()
    innovations[0] = AVState(innovations[0].video, innovations[0].audio + 1000)
    innovations[1] = AVState(innovations[1].video - 1000, innovations[1].audio)
    changed, _ = collect(changes={"innovations": innovations})
    for name in ("video", "audio"):
        torch.testing.assert_close(getattr(baseline.endpoints, name), getattr(changed.endpoints, name), rtol=0, atol=0)
        torch.testing.assert_close(getattr(baseline, "old_" + name + "_log_probs"),
                                   getattr(changed, "old_" + name + "_log_probs"), rtol=0, atol=0)


def test_dual_cps_is_independent_of_physical_microbatch():
    one, _ = collect(microbatch=1)
    many, _ = collect(microbatch=4)
    for name in ("video", "audio"):
        torch.testing.assert_close(getattr(one.endpoints, name), getattr(many.endpoints, name), rtol=0, atol=0)
        torch.testing.assert_close(getattr(one, "old_" + name + "_log_probs"),
                                   getattr(many, "old_" + name + "_log_probs"), rtol=0, atol=0)


def test_legacy_shared_fast_cache_is_unchanged_and_keeps_optional_fields_empty():
    noise, video, audio, _, union, innovations = fixture()
    legacy = collect_av_rollout(CoupledPolicy(), noise, video, audio, selected=union, stochastic=union,
        innovations=innovations, microbatch=2, unselected_kind="cps_zero_noise")
    shared = {"video": union, "audio": union}
    explicit = collect_av_rollout(CoupledPolicy(), noise, video, audio, selected=union, stochastic=union,
        innovations=innovations, microbatch=2, unselected_kind="cps_zero_noise",
        modality_selected=shared, modality_stochastic=shared)
    assert legacy.old_video_log_probs is legacy.old_audio_log_probs is None
    for name in ("video", "audio"):
        torch.testing.assert_close(getattr(legacy.endpoints, name), getattr(explicit.endpoints, name), rtol=0, atol=0)
    torch.testing.assert_close(legacy.old_log_probs, explicit.old_log_probs, rtol=0, atol=0)


@pytest.mark.parametrize("changes", [
    {"modality_selected": None},
    {"modality_stochastic": {"video": (0, 3)}},
    {"modality_stochastic": {"video": (0, 1), "audio": (1, 3)}},
    {"modality_stochastic": {"video": (0., 3.), "audio": (1, 3)}},
    {"modality_stochastic": {"video": (0, "3"), "audio": (1, 3)}},
    {"modality_stochastic": {"video": None, "audio": (1, 3)}},
    {"stochastic": (0, 1, 2, 3)},
    {"selected": (0, 1)},
    {"unselected_kind": "ode"},
    {"modality_selected": {"video": (3, 0), "audio": (1, 3)}},
])
def test_dual_cps_rejects_inconsistent_noise_and_replay_contracts(changes):
    with pytest.raises(ValueError):
        collect(changes=changes)


def test_dual_cps_rejects_missing_or_nonfinite_innovations():
    *_, innovations = fixture()
    missing = dict(innovations)
    missing.pop(0)
    with pytest.raises(ValueError, match="innovation"):
        collect(changes={"innovations": missing})
    nonfinite = copy.deepcopy(innovations)
    nonfinite[0].audio[0, 0] = torch.nan  # Even unused branch metadata cannot be invalid.
    with pytest.raises(ValueError, match="finite"):
        collect(changes={"innovations": nonfinite})
