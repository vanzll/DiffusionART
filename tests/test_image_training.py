"""Real tiny SD3 -> VAE -> CLIP -> ART optimizer integration; no downloads."""

import pytest
import torch

from visual_rl.algorithms import art_scalar_covariance
from visual_rl.models import SD3Adapter, SD3Conditioning, SD3Config
from visual_rl.models.finetune import FineTuneConfig, prepare_policy
from visual_rl.rewards.pickscore import PickScore
from visual_rl.sampling import shifted_schedule
from visual_rl.training.image import ConditionedImagePolicy, decode_and_score_images
from visual_rl.training.objectives import ARTObjective, disjoint_windows
from visual_rl.training.rollout import collect_rollout
from visual_rl.training.update import accumulated_step


def test_fp16_cfg_promotes_predictions_before_guidance_mixing(tmp_path):
    from test_sd3_runner import config_file, tiny_parts

    from visual_rl.config import load_config

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        parts = tiny_parts(load_config(config_file(tmp_path)))
        adapter = parts.adapter
        adapter.config = SD3Config(2, .25, latent_channels=2, cfg_scale=3.7)
        c = SD3Conditioning(torch.randn(2, 3, 12), torch.randn(2, 8),
                           torch.randn(2, 3, 12), torch.randn(2, 8))
        policy = ConditionedImagePolicy(adapter, c, autocast_dtype=torch.float16)
        z, t = torch.randn(2, 2, 4, 4), torch.tensor([.4, .7])
        actual = policy(z, t, torch.arange(2))
        with torch.autocast("cpu", dtype=torch.float16):
            raw = adapter.transformer(
                hidden_states=torch.cat([z, z]), timestep=torch.cat([t, t]) * 1000,
                encoder_hidden_states=torch.cat([c.negative_prompt_embeds, c.prompt_embeds]),
                pooled_projections=torch.cat([c.negative_pooled_prompt_embeds, c.pooled_prompt_embeds]),
                return_dict=False,
            )[0]
        assert raw.dtype == torch.float16 and actual.dtype == torch.float32
        negative, positive = raw.float().chunk(2)
        expected = negative + 3.7 * (positive - negative)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        params = tuple(adapter.transformer.parameters())
        grads = torch.autograd.grad(actual.square().sum(), params, allow_unused=True)
        oracle = torch.autograd.grad(expected.square().sum(), params, allow_unused=True)
        for a, b in zip(grads, oracle):
            if a is None:
                assert b is None
            else:
                torch.testing.assert_close(a, b, rtol=0, atol=0)
    finally:
        torch.set_num_threads(old_threads)


def test_sd3_pickscore_lora_two_disjoint_updates():
    diffusers = pytest.importorskip("diffusers")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from test_pickscore import Processor

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng():
            torch.manual_seed(19)
            transformer = diffusers.SD3Transformer2DModel(
                sample_size=4, patch_size=2, in_channels=2, out_channels=2, num_layers=1,
                attention_head_dim=8, num_attention_heads=2, joint_attention_dim=12,
                caption_projection_dim=16, pooled_projection_dim=8, pos_embed_max_size=4,
            ).eval()
            transformer = prepare_policy(transformer, FineTuneConfig(
                mode="lora", rank=2, alpha=4, target_modules=("attn.to_q", "attn.to_v")
            ))
            vae = diffusers.AutoencoderKL(
                in_channels=3, out_channels=3, down_block_types=("DownEncoderBlock2D",),
                up_block_types=("UpDecoderBlock2D",), block_out_channels=(8,), layers_per_block=1,
                latent_channels=2, norm_num_groups=4, sample_size=4, scaling_factor=2,
                shift_factor=0.25,
            )
            adapter = SD3Adapter(transformer, vae, SD3Config(2, 0.25, latent_channels=2))
            conditioning = SD3Conditioning(torch.randn(2, 3, 12).repeat_interleave(2, 0),
                                           torch.randn(2, 8).repeat_interleave(2, 0))
            predict = ConditionedImagePolicy(adapter, conditioning)
            noise = torch.randn(4, 2, 4, 4)
            sigmas = shifted_schedule(3, 1.0)
            cache = collect_rollout(predict, noise, sigmas, selected=(0, 1), microbatch=2)
            clip = transformers.CLIPModel(transformers.CLIPConfig(
                projection_dim=4,
                text_config=dict(vocab_size=16, hidden_size=8, intermediate_size=16,
                                 num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=8),
                vision_config=dict(hidden_size=8, intermediate_size=16, num_hidden_layers=1,
                                   num_attention_heads=2, image_size=4, patch_size=2),
            ))
            scorer = PickScore(clip, Processor(), space="fixed_normalized")
            images = decode_and_score_images(adapter, cache.endpoints, ["2", "2", "4", "4"],
                                             scorer, decode_microbatch=1, score_microbatch=2)
            assert images.pixels.shape == (4, 3, 4, 4)
            expected_pixels = (vae.decode(cache.endpoints / 2 + .25, return_dict=False)[0] / 2 + .5).clamp(0, 1)
            torch.testing.assert_close(images.pixels, expected_pixels, atol=2e-6, rtol=2e-5)
            direct_scores = scorer(expected_pixels, ["2", "2", "4", "4"], microbatch=4)
            torch.testing.assert_close(images.scores.weighted, direct_scores.weighted, atol=1e-6, rtol=1e-5)
            groups = torch.tensor([0, 0, 1, 1])
            cov = art_scalar_covariance(cache.endpoints, images.scores.weighted, groups, 2)
            objective = ARTObjective(cache, predict, cov["group_directions"][groups],
                                     cov["group_guidance_magnitudes"][groups],
                                     torch.zeros(4, dtype=torch.bool), torch.device("cpu"))
            old_predictions = cache.old_velocities.clone()
            optimizer = torch.optim.AdamW([p for p in transformer.parameters() if p.requires_grad],
                                          lr=1e-3, weight_decay=0)
            trainable = {n: p.detach().clone() for n, p in transformer.named_parameters() if p.requires_grad}
            for window in disjoint_windows(4, 2):
                metrics = accumulated_step(transformer, optimizer, window, objective,
                                            microbatch=3, denominator=len(window), max_grad_norm=1)
                assert metrics["train/grad_norm/before_clip"] > 0
                assert torch.isfinite(torch.tensor(metrics["train/loss"]))
            assert any(not torch.equal(trainable[n], p) for n, p in transformer.named_parameters()
                       if n in trainable)
            torch.testing.assert_close(cache.old_velocities, old_predictions, atol=0, rtol=0)
            assert all(p.grad is None and not p.requires_grad for p in vae.parameters())
            assert all(p.grad is None and not p.requires_grad for p in clip.parameters())
    finally:
        torch.set_num_threads(old_threads)
