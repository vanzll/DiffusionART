"""Config-independent, pure-PyTorch algorithm kernels.

All loss APIs return graph-connected ``per_item_loss`` with shape [N].
Callers own global denominators, sample weights, and optimizer boundaries.
See docs/algorithm-provenance.md for supported variants and source attribution.
"""

from .art import art_loss
from .covariance import art_pairwise_covariance, art_scalar_covariance
from .nft import nft_loss, old_policy_decay
from .policy import clipped_grpo_loss, flowdppo_loss, transition_kl_per_sample
from .refl import refl_loss

__all__ = [
    "art_loss",
    "art_pairwise_covariance",
    "art_scalar_covariance",
    "nft_loss",
    "old_policy_decay",
    "clipped_grpo_loss",
    "flowdppo_loss",
    "transition_kl_per_sample",
    "refl_loss",
]
