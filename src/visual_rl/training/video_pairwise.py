"""Frozen Hunyuan binary feedback with rank-global tie-anchor semantics."""

import torch

from ..algorithms import art_pairwise_covariance
from ..config import ConfigError
from .objectives import ARTObjective, prompt_disjoint_windows


def enabled(c):
    return c.algorithm.name == "diffusionart" and "pairwise" in c.algorithm.options


def validate_options(c):
    if c.task != "t2v" or c.model.name != "hunyuan_public":
        raise ConfigError("pairwise video feedback currently supports Hunyuan T2V")
    options = c.algorithm.options["pairwise"]
    if (not isinstance(options, dict) or set(options) != {"seed"}
            or type(options["seed"]) is not int or options["seed"] < 0):
        raise ConfigError("pairwise requires a nonnegative integer seed")
    if c.sampling.samples_per_prompt not in (2, 4, 8):
        raise ConfigError("pairwise requires K=2/4/8")
    if c.reward.weights != {"VQ": 1., "MQ": 0., "TA": 0.} or c.reward.normalization != "fixed":
        raise ConfigError("pairwise comparisons require fixed-normalized VQ only")
    if not c.algorithm.options["restore_zero_signal"]:
        raise ConfigError("pairwise requires the historical global tie-anchor rule")
    covariance = c.algorithm.options["covariance"]
    expected = {"reduction": "unbiased", "direction_mode": "historical",
                "scale_mode": "unit_normalized_covariance", "guidance_gain": 1.,
                "zero_signal_mode": "historical", "zero_signal_atol": 0., "eps": 1e-5}
    if not isinstance(covariance, dict) or any(value != expected.get(key) for key, value in covariance.items()):
        raise ConfigError("pairwise requires the historical unit-covariance recipe")


def prepare(c, cache, scores, *, attempt, group_offset=0):
    # Scalar magnitudes stop here. Neither the objective nor covariance sees them.
    if scores.space != "fixed_normalized" or scores.fixed_normalized is None:
        raise ValueError("pairwise scorer must return fixed-normalized VQ")
    return art_pairwise_covariance(
        cache.endpoints, scores.fixed_normalized["VQ"], c.sampling.samples_per_prompt,
        seed=c.algorithm.options["pairwise"]["seed"], round_index=attempt,
        group_offset=group_offset,
    )


def activity(feedback, device):
    preference = feedback["row_preference_active"]
    return torch.tensor([bool(preference.any()), bool(feedback["row_active"].any()),
                         bool((~preference).any())], dtype=torch.int32, device=device)


def finalize(c, cache, predict, feedback, global_activity, device):
    """Call after MAX-reducing activity, outside any local-error collective."""
    any_preference, any_guidance, any_tie = global_activity.cpu().tolist()
    preference = feedback["row_preference_active"]
    anchors = ~preference & bool(any_preference)
    trainable = bool(any_guidance or (any_preference and any_tie))
    metrics = {"train/pairwise/inactive_rollout": int(not trainable)}
    if not trainable:
        return None, (), metrics
    rows = len(cache.endpoints)
    groups = torch.arange(rows) // c.sampling.samples_per_prompt
    objective = ARTObjective(
        cache, predict, feedback["group_directions"][groups],
        feedback["row_guidance_magnitudes"], anchors, device,
        options=c.algorithm.options["loss"], reconstruct_old_from_mean=True,
    )
    # No filtering or reweighting: inactive rows still occupy the original mean.
    return objective, prompt_disjoint_windows(groups, len(cache.selected)), metrics
