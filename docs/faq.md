# DiffusionART FAQ

## What is DiffusionART, and which models does it apply to?

DiffusionART is a group-based reinforcement learning algorithm for
diffusion-based foundation models. It generates multiple samples per prompt
and aggregates their rewards into shared guidance. Our experiments cover
DiT-based image and video generation with SD3.5-M, FLUX.1-dev,
HunyuanVideo-13B, and a CFG- and step-distilled Wan2.1-I2V-14B model.
Flow-matching vision-language-action (VLA) policies are a potential future
application, not evaluated in this work.

## Does DiffusionART require backpropagation through the reward model?

No. DiffusionART uses reward outputs, not reward-model gradients. It can
therefore use black-box feedback, including scores or pairwise judgments
from a closed-source vision-language model (VLM) API.

## Is DiffusionART forward-process or reverse-process RL?

It is a mixed forward- and reverse-process RL approach: its regression-based
learning principle is inspired by forward-process RL, but its updates act on
states visited during reverse-process ODE rollouts. Aggregating reward
guidance first avoids the forward-target consistency force, while direct
regression avoids reverse-transition likelihood estimation.

## Does DiffusionART support pairwise feedback?

Yes. DiffusionART-Pairwise assigns 1 to the winner and 0 to the loser, or
0.5 to both in a tie, and uses these labels in the covariance guidance.
With one pair per prompt (K = 2), the covariance is proportional to the
winner's latent endpoint minus the loser's: guidance points from the
lower-reward sample toward the higher-reward sample. A tie gives zero
guidance. For larger groups, the pairwise evidence is aggregated before
normalization. Pairwise judgments can be more natural for video generation,
where comparing two outputs is often easier than assigning calibrated
absolute quality scores.

## Is DiffusionART online or offline RL?

DiffusionART is online RL. It repeatedly generates and scores fresh samples
as the model changes, rather than training only on a fixed dataset of
previously collected outputs and preferences.

## Is DiffusionART on-policy or off-policy RL?

DiffusionART uses on-policy rollout-and-update training: each iteration
collects fresh samples from the current model, performs two updates on
disjoint minibatches with a frozen rollout reference and target, then
refreshes the rollouts. As in PPO-style training, multiple updates can use
one fresh rollout batch; the policy has already moved by the second update.
Restoration and trust-region masking control drift within the iteration,
without an EMA-lagged rollout policy or a historical replay buffer.

## Resources

- [Paper](https://arxiv.org/abs/2610.12572)
- [Project website](https://vanzll.github.io/diffusionart-project/#faq)
- [Training and inference documentation](README.md)
