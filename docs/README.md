# Documentation

New to the method? Start with the [DiffusionART FAQ](faq.md) for model
coverage, black-box rewards, pairwise feedback, and online/on-policy training.
The guides below cover installation and execution; reproduction and
development-status documents are listed separately.

## Start Here

| Goal | Guide |
| --- | --- |
| Install and check the framework | [Quick start](../README.md#quick-start) |
| Train SD3.5 | [SD3 setup and training](sd3-runner.md) |
| Train FLUX | [FLUX setup and training](flux-runner.md) |
| Install a video environment | [Pinned video installation](video-installation.md) |
| Train HunyuanVideo | [HunyuanVideo runner](hunyuan-runner.md) |
| Train Wan I2V | [Wan runner](wan-runner.md) |
| Integrate MiniMax-H3 audio/video | [H3 setup, six recipes and acceptance status](minimax-h3.md) |
| Run across GPUs or nodes | [Distributed execution](distributed-execution.md) |
| Resume or evaluate a checkpoint | [Checkpoint guide](checkpoint-resume.md) |

## Algorithms

Train only at the initial denoising position with [DiffusionART-Flash](art-flash.md).

Shared equations, source revisions, and adaptations:
[algorithm contracts](algorithm-provenance.md).

| Model family | Method guides |
| --- | --- |
| SD3 | [DiffusionART](sd3-runner.md) / [DiffusionNFT](sd3-nft.md) / [GRPO, Fast, DPPO, Dance](sd3-grpo.md) |
| FLUX | [DiffusionART](flux-runner.md) / [DiffusionNFT](flux-nft.md) / [GRPO, Fast, DPPO](flux-grpo.md) |
| HunyuanVideo | [Method selection](hunyuan-runner.md#algorithm-selection) / [Pairwise feedback](hunyuan-runner.md#pairwise-feedback) / [ReFL](hunyuan-runner.md#refl-gradients-and-reporting) |
| Wan I2V | [Recipes and execution](wan-runner.md#configs-and-execution) |

## Data, Rewards & Logging

- [Prompt manifests and reference images](prompt-data.md)
- [PickScore](sd3-runner.md), [GenEval](geneval.md), [OCR](sd3-ocr.md),
  [Aesthetic](sd3-aesthetic.md), [CLIPScore](sd3-clipscore.md)
- [Reward composition and normalization](composite-rewards.md)
- [VideoReward and differentiable scoring](video-reward.md)
- [W&B, local metrics, and GPU-hour definitions](logging.md)

## Reproduction

- [Image recipes](paper-image.md) and [video recipes](paper-video.md)
- [Run and source inventory](reproduction-inventory.json)
- [Release scope and validation status](release-scope.md)

## Extend the Framework

- [Development and tests](development.md)
- [Model adapter contract](model-contract.md)
- [Reward adapter contract](reward-contract.md)
- [GenEval deployment and migration](geneval-migration.md)
- [Third-party notices](../THIRD_PARTY_NOTICES.md)
