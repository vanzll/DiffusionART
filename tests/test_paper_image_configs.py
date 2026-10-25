"""Paper candidates: backend validation plus independently pinned science checks."""

import re
from pathlib import Path

import pytest

from visual_rl.config import load_config
from visual_rl.registry import get_backend

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "configs/paper/image"
PROFILES_EXPECTED = ('sd3_geneval_art.yaml',
 'sd3_geneval_grpo.yaml',
 'sd3_geneval_dance.yaml',
 'sd3_geneval_fast.yaml',
 'sd3_geneval_nft_ema.yaml',
 'sd3_geneval_nft_onpolicy.yaml',
 'sd3_geneval_dppo.yaml',
 'sd3_ocr_art.yaml',
 'sd3_ocr_art_one_update.yaml',
 'sd3_ocr_grpo.yaml',
 'sd3_ocr_dance.yaml',
 'sd3_ocr_fast.yaml',
 'sd3_ocr_nft_ema.yaml',
 'sd3_ocr_nft_onpolicy.yaml',
 'sd3_ocr_dppo.yaml',
 'sd3_pickscore_art.yaml',
 'sd3_pickscore_grpo.yaml',
 'sd3_pickscore_dance.yaml',
 'sd3_pickscore_fast.yaml',
 'sd3_pickscore_nft_ema.yaml',
 'sd3_pickscore_nft_onpolicy.yaml',
 'sd3_pickscore_dppo.yaml',
 'flux_geneval_art.yaml',
 'flux_geneval_fast.yaml',
 'flux_geneval_dppo.yaml',
 'flux_geneval_nft_ema.yaml')
# Independently pinned numerical expectations for optimizer, batch and loss contracts.
SAMPLE_EXPECTATIONS = [{'num_steps': 10,
  'num_image_per_prompt': 24,
  'train_batch_size': 9,
  'num_batches_per_epoch': 16,
  'guidance_scale': 1,
  'eval_num_steps': 40,
  'noise_level': 0.8,
  'sde_window_size': 0,
  'sde_window_range': [0, 10],
  'same_latent': False},
 {'num_steps': 10,
  'num_image_per_prompt': 24,
  'train_batch_size': 9,
  'num_batches_per_epoch': 16,
  'guidance_scale': 1,
  'eval_num_steps': 40,
  'noise_level': 0.8,
  'sde_window_size': 0,
  'sde_window_range': [0, 10],
  'same_latent': True},
 {'num_steps': 10,
  'num_image_per_prompt': 24,
  'train_batch_size': 9,
  'num_batches_per_epoch': 16,
  'guidance_scale': 1,
  'eval_num_steps': 40,
  'noise_level': 0.8,
  'sde_window_size': 3,
  'sde_window_range': [0, 5],
  'same_latent': False},
 {'num_steps': 10,
  'num_image_per_prompt': 24,
  'train_batch_size': 9,
  'num_batches_per_epoch': 16,
  'guidance_scale': 1,
  'eval_num_steps': 40,
  'solver': 'dpm2',
  'noise_level': 0.7,
  'sde_window_size': 0,
  'sde_window_range': [0, 10],
  'same_latent': False},
 {'num_steps': 10,
  'num_image_per_prompt': 16,
  'train_batch_size': 8,
  'num_batches_per_epoch': 16,
  'guidance_scale': 1,
  'eval_num_steps': 40,
  'noise_level': 0.8,
  'sde_window_size': 0,
  'sde_window_range': [0, 10],
  'same_latent': False},
 {'num_steps': 10,
  'num_image_per_prompt': 16,
  'train_batch_size': 8,
  'num_batches_per_epoch': 16,
  'guidance_scale': 1,
  'eval_num_steps': 40,
  'noise_level': 0.8,
  'sde_window_size': 3,
  'sde_window_range': [0, 5],
  'same_latent': False},
 {'num_steps': 10,
  'num_image_per_prompt': 16,
  'train_batch_size': 8,
  'num_batches_per_epoch': 16,
  'guidance_scale': 1,
  'eval_num_steps': 40,
  'solver': 'dpm2',
  'noise_level': 0.7,
  'sde_window_size': 0,
  'sde_window_range': [0, 5],
  'same_latent': False},
 {'num_steps': 10,
  'num_image_per_prompt': 18,
  'train_batch_size': 9,
  'num_batches_per_epoch': 16,
  'guidance_scale': 1,
  'eval_num_steps': 40,
  'noise_level': 0.8,
  'sde_window_size': 0,
  'sde_window_range': [0, 10],
  'same_latent': False},
 {'num_steps': 10,
  'num_image_per_prompt': 18,
  'train_batch_size': 9,
  'num_batches_per_epoch': 16,
  'guidance_scale': 1,
  'eval_num_steps': 40,
  'noise_level': 0.8,
  'sde_window_size': 3,
  'sde_window_range': [0, 5],
  'same_latent': False},
 {'num_steps': 10,
  'num_image_per_prompt': 18,
  'train_batch_size': 9,
  'num_batches_per_epoch': 16,
  'guidance_scale': 1,
  'eval_num_steps': 40,
  'solver': 'dpm2',
  'noise_level': 0.7,
  'sde_window_size': 0,
  'sde_window_range': [0, 5],
  'same_latent': False},
 {'num_steps': 6,
  'num_image_per_prompt': 24,
  'train_batch_size': 3,
  'num_batches_per_epoch': 48,
  'guidance_scale': 3.5,
  'eval_num_steps': 28,
  'noise_level': 0.8,
  'sde_window_size': 3,
  'sde_window_range': [0, 3],
  'same_latent': False},
 {'num_steps': 6,
  'num_image_per_prompt': 24,
  'train_batch_size': 3,
  'num_batches_per_epoch': 48,
  'guidance_scale': 3.5,
  'eval_num_steps': 28,
  'noise_level': 0.7,
  'sde_window_size': 3,
  'sde_window_range': [0, 3],
  'same_latent': False}]
TRAIN_EXPECTATIONS = [{'learning_rate': 0.0003,
  'batch_size': 9,
  'adam_weight_decay': 0.0001,
  'adam_beta1': 0.9,
  'adam_beta2': 0.999,
  'adam_epsilon': 1e-08,
  'max_grad_norm': 1,
  'gradient_accumulation_steps': 8,
  'timestep_fraction': 1,
  'timestep_shift': 3,
  'covariance_guidance_scale_mode': 'unit_normalized_covariance',
  'covariance_guidance_gain': 1,
  'covariance_guidance_loss_space': 'velocity_linear_sigma',
  'x0_covariance_mask_space': 'velocity',
  'x0_covariance_velocity_mask_metric': 'scheduler_reference_metric',
  'x0_covariance_corrective_mask_enabled': True,
  'x0_covariance_q2_reference_sde_noise_level': 0.8,
  'x0_covariance_q2_reference_kl_threshold': 1e-05,
  'adv_clip_max': 5,
  'ema': False,
  'clip_range': 1e-05},
 {'learning_rate': 0.0003,
  'batch_size': 9,
  'adam_weight_decay': 0.0001,
  'adam_beta1': 0.9,
  'adam_beta2': 0.999,
  'adam_epsilon': 1e-08,
  'max_grad_norm': 1,
  'gradient_accumulation_steps': 8,
  'timestep_fraction': 0.99,
  'timestep_shift': 3,
  'adv_clip_max': 5,
  'ema': True,
  'clip_range': 1e-05},
 {'learning_rate': 0.0003,
  'batch_size': 9,
  'adam_weight_decay': 0.0001,
  'adam_beta1': 0.9,
  'adam_beta2': 0.999,
  'adam_epsilon': 1e-08,
  'max_grad_norm': 1,
  'gradient_accumulation_steps': 8,
  'timestep_fraction': 1,
  'timestep_shift': 3,
  'adv_clip_max': 5,
  'ema': True,
  'clip_range': 1e-05},
 {'learning_rate': 0.0003,
  'batch_size': 9,
  'adam_weight_decay': 0.0001,
  'adam_beta1': 0.9,
  'adam_beta2': 0.999,
  'adam_epsilon': 1e-08,
  'max_grad_norm': 1,
  'gradient_accumulation_steps': 16,
  'timestep_fraction': 0.99,
  'timestep_shift': 3,
  'adv_clip_max': 5,
  'adv_mode': 'all',
  'ema': True,
  'clip_range': 1e-05},
 {'learning_rate': 0.0003,
  'batch_size': 9,
  'adam_weight_decay': 0.0001,
  'adam_beta1': 0.9,
  'adam_beta2': 0.999,
  'adam_epsilon': 1e-08,
  'max_grad_norm': 1,
  'gradient_accumulation_steps': 8,
  'timestep_fraction': 0.99,
  'timestep_shift': 3,
  'adv_clip_max': 5,
  'ema': True,
  'flow_dppo_kl_threshold': 1e-05,
  'clip_range': 1e-05},
 {'learning_rate': 0.0003,
  'batch_size': 8,
  'adam_weight_decay': 0.0001,
  'adam_beta1': 0.9,
  'adam_beta2': 0.999,
  'adam_epsilon': 1e-08,
  'max_grad_norm': 1,
  'gradient_accumulation_steps': 8,
  'timestep_fraction': 1,
  'timestep_shift': 3,
  'covariance_guidance_scale_mode': 'unit_normalized_covariance',
  'covariance_guidance_gain': 1,
  'covariance_guidance_loss_space': 'velocity_linear_sigma',
  'x0_covariance_mask_space': 'velocity',
  'x0_covariance_velocity_mask_metric': 'scheduler_reference_metric',
  'x0_covariance_corrective_mask_enabled': True,
  'x0_covariance_q2_reference_sde_noise_level': 0.8,
  'x0_covariance_q2_reference_kl_threshold': 1e-05,
  'adv_clip_max': 5,
  'ema': False,
  'clip_range': 1e-05},
 {'learning_rate': 0.0003,
  'batch_size': 8,
  'adam_weight_decay': 0.0001,
  'adam_beta1': 0.9,
  'adam_beta2': 0.999,
  'adam_epsilon': 1e-08,
  'max_grad_norm': 1,
  'gradient_accumulation_steps': 16,
  'timestep_fraction': 1,
  'timestep_shift': 3,
  'covariance_guidance_scale_mode': 'unit_normalized_covariance',
  'covariance_guidance_gain': 1,
  'covariance_guidance_loss_space': 'velocity_linear_sigma',
  'x0_covariance_mask_space': 'velocity',
  'x0_covariance_velocity_mask_metric': 'scheduler_reference_metric',
  'x0_covariance_corrective_mask_enabled': True,
  'x0_covariance_q2_reference_sde_noise_level': 0.8,
  'x0_covariance_q2_reference_kl_threshold': 1e-05,
  'adv_clip_max': 5,
  'ema': False,
  'clip_range': 1e-05},
 {'learning_rate': 0.0003,
  'batch_size': 8,
  'adam_weight_decay': 0.0001,
  'adam_beta1': 0.9,
  'adam_beta2': 0.999,
  'adam_epsilon': 1e-08,
  'max_grad_norm': 1,
  'gradient_accumulation_steps': 8,
  'timestep_fraction': 1,
  'timestep_shift': 3,
  'adv_clip_max': 5,
  'ema': True,
  'clip_range': 1e-05},
 {'learning_rate': 0.0003,
  'batch_size': 8,
  'adam_weight_decay': 0.0001,
  'adam_beta1': 0.9,
  'adam_beta2': 0.999,
  'adam_epsilon': 1e-08,
  'max_grad_norm': 1,
  'gradient_accumulation_steps': 8,
  'timestep_fraction': 1,
  'timestep_shift': 3,
  'adv_clip_max': 5,
  'adv_mode': 'all',
  'ema': True,
  'clip_range': 1e-05},
 {'learning_rate': 0.0003,
  'batch_size': 8,
  'adam_weight_decay': 0.0001,
  'adam_beta1': 0.9,
  'adam_beta2': 0.999,
  'adam_epsilon': 1e-08,
  'max_grad_norm': 1,
  'gradient_accumulation_steps': 16,
  'timestep_fraction': 1,
  'timestep_shift': 3,
  'adv_clip_max': 5,
  'adv_mode': 'all',
  'ema': True,
  'clip_range': 1e-05},
 {'learning_rate': 0.0003,
  'batch_size': 9,
  'adam_weight_decay': 0.0001,
  'adam_beta1': 0.9,
  'adam_beta2': 0.999,
  'adam_epsilon': 1e-08,
  'max_grad_norm': 1,
  'gradient_accumulation_steps': 16,
  'timestep_fraction': 1,
  'timestep_shift': 3,
  'adv_clip_max': 5,
  'adv_mode': 'all',
  'ema': True,
  'clip_range': 1e-05},
 {'learning_rate': 0.0003,
  'batch_size': 3,
  'adam_weight_decay': 0.0001,
  'adam_beta1': 0.9,
  'adam_beta2': 0.999,
  'adam_epsilon': 1e-08,
  'max_grad_norm': 1,
  'gradient_accumulation_steps': 24,
  'timestep_fraction': 1,
  'timestep_shift': 3,
  'covariance_guidance_scale_mode': 'unit_normalized_covariance',
  'covariance_guidance_gain': 1,
  'covariance_guidance_loss_space': 'velocity_linear_sigma',
  'x0_covariance_mask_space': 'velocity',
  'x0_covariance_velocity_mask_metric': 'scheduler_reference_metric',
  'x0_covariance_corrective_mask_enabled': True,
  'x0_covariance_q2_reference_sde_noise_level': 0.8,
  'x0_covariance_q2_reference_kl_threshold': 1e-05,
  'adv_clip_max': 5,
  'ema': False,
  'clip_range': 1e-05},
 {'learning_rate': 0.0003,
  'batch_size': 3,
  'adam_weight_decay': 0.0001,
  'adam_beta1': 0.9,
  'adam_beta2': 0.999,
  'adam_epsilon': 1e-08,
  'max_grad_norm': 1,
  'gradient_accumulation_steps': 24,
  'timestep_fraction': 1,
  'timestep_shift': 3,
  'adv_clip_max': 5,
  'ema': True,
  'clip_range': 1e-05},
 {'learning_rate': 0.0003,
  'batch_size': 3,
  'adam_weight_decay': 0.0001,
  'adam_beta1': 0.9,
  'adam_beta2': 0.999,
  'adam_epsilon': 1e-08,
  'max_grad_norm': 1,
  'gradient_accumulation_steps': 24,
  'timestep_fraction': 0.5,
  'timestep_shift': 3,
  'adv_clip_max': 5,
  'ema': False,
  'flow_dppo_kl_threshold': 1e-05,
  'clip_range': 1e-05}]
SCIENCE_EXPECTATIONS = {'sd3_geneval_art.yaml': {'seed': 42,
                          'mixed_precision': 'fp16',
                          'allow_tf32': True,
                          'resolution': 512,
                          'reward_fn': {'geneval': 1},
                          'activation_checkpointing': False,
                          'sample': 0,
                          'train': 0},
 'sd3_geneval_grpo.yaml': {'seed': 42,
                           'mixed_precision': 'fp16',
                           'allow_tf32': True,
                           'resolution': 512,
                           'reward_fn': {'geneval': 1},
                           'activation_checkpointing': False,
                           'sample': 0,
                           'train': 1},
 'sd3_geneval_dance.yaml': {'seed': 42,
                            'mixed_precision': 'fp16',
                            'allow_tf32': True,
                            'resolution': 512,
                            'reward_fn': {'geneval': 1},
                            'activation_checkpointing': False,
                            'sample': 1,
                            'train': 1},
 'sd3_geneval_fast.yaml': {'seed': 42,
                           'mixed_precision': 'fp16',
                           'allow_tf32': True,
                           'resolution': 512,
                           'reward_fn': {'geneval': 1},
                           'activation_checkpointing': False,
                           'sample': 2,
                           'train': 2},
 'sd3_geneval_nft_ema.yaml': {'seed': 42,
                              'mixed_precision': 'fp16',
                              'allow_tf32': True,
                              'resolution': 512,
                              'reward_fn': {'geneval': 1},
                              'beta': 1,
                              'decay_type': 1,
                              'activation_checkpointing': False,
                              'sample': 3,
                              'train': 3},
 'sd3_geneval_nft_onpolicy.yaml': {'seed': 42,
                                   'mixed_precision': 'fp16',
                                   'allow_tf32': True,
                                   'resolution': 512,
                                   'reward_fn': {'geneval': 1},
                                   'beta': 1,
                                   'decay_type': 0,
                                   'activation_checkpointing': False,
                                   'sample': 3,
                                   'train': 3},
 'sd3_geneval_dppo.yaml': {'seed': 42,
                           'mixed_precision': 'fp16',
                           'allow_tf32': True,
                           'resolution': 512,
                           'reward_fn': {'geneval': 1},
                           'activation_checkpointing': False,
                           'sample': 0,
                           'train': 4},
 'sd3_ocr_art.yaml': {'seed': 42,
                      'mixed_precision': 'fp16',
                      'allow_tf32': True,
                      'resolution': 512,
                      'reward_fn': {'ocr': 1},
                      'activation_checkpointing': False,
                      'sample': 4,
                      'train': 5},
 'sd3_ocr_art_one_update.yaml': {'seed': 42,
                                 'mixed_precision': 'fp16',
                                 'allow_tf32': True,
                                 'resolution': 512,
                                 'reward_fn': {'ocr': 1},
                                 'activation_checkpointing': False,
                                 'sample': 4,
                                 'train': 6},
 'sd3_ocr_grpo.yaml': {'seed': 42,
                       'mixed_precision': 'fp16',
                       'allow_tf32': True,
                       'resolution': 512,
                       'reward_fn': {'ocr': 1},
                       'activation_checkpointing': False,
                       'sample': 0,
                       'train': 1},
 'sd3_ocr_dance.yaml': {'seed': 42,
                        'mixed_precision': 'fp16',
                        'allow_tf32': True,
                        'resolution': 512,
                        'reward_fn': {'ocr': 1},
                        'activation_checkpointing': False,
                        'sample': 1,
                        'train': 1},
 'sd3_ocr_fast.yaml': {'seed': 42,
                       'mixed_precision': 'fp16',
                       'allow_tf32': True,
                       'resolution': 512,
                       'reward_fn': {'ocr': 1},
                       'activation_checkpointing': False,
                       'sample': 5,
                       'train': 7},
 'sd3_ocr_nft_ema.yaml': {'seed': 42,
                          'mixed_precision': 'fp16',
                          'allow_tf32': True,
                          'resolution': 512,
                          'reward_fn': {'ocr': 1},
                          'beta': 0.1,
                          'decay_type': 2,
                          'activation_checkpointing': False,
                          'sample': 6,
                          'train': 8},
 'sd3_ocr_nft_onpolicy.yaml': {'seed': 42,
                               'mixed_precision': 'fp16',
                               'allow_tf32': True,
                               'resolution': 512,
                               'reward_fn': {'ocr': 1},
                               'beta': 1,
                               'decay_type': 0,
                               'activation_checkpointing': False,
                               'sample': 6,
                               'train': 9},
 'sd3_ocr_dppo.yaml': {'seed': 42,
                       'mixed_precision': 'fp16',
                       'allow_tf32': True,
                       'resolution': 512,
                       'reward_fn': {'ocr': 1},
                       'activation_checkpointing': False,
                       'sample': 0,
                       'train': 4},
 'sd3_pickscore_art.yaml': {'seed': 42,
                            'mixed_precision': 'fp16',
                            'allow_tf32': True,
                            'resolution': 512,
                            'reward_fn': {'pickscore': 1},
                            'activation_checkpointing': False,
                            'sample': 7,
                            'train': 0},
 'sd3_pickscore_grpo.yaml': {'seed': 42,
                             'mixed_precision': 'fp16',
                             'allow_tf32': True,
                             'resolution': 512,
                             'reward_fn': {'pickscore': 1},
                             'activation_checkpointing': False,
                             'sample': 0,
                             'train': 1},
 'sd3_pickscore_dance.yaml': {'seed': 42,
                              'mixed_precision': 'fp16',
                              'allow_tf32': True,
                              'resolution': 512,
                              'reward_fn': {'pickscore': 1},
                              'activation_checkpointing': False,
                              'sample': 1,
                              'train': 1},
 'sd3_pickscore_fast.yaml': {'seed': 42,
                             'mixed_precision': 'fp16',
                             'allow_tf32': True,
                             'resolution': 512,
                             'reward_fn': {'pickscore': 1},
                             'activation_checkpointing': False,
                             'sample': 8,
                             'train': 2},
 'sd3_pickscore_nft_ema.yaml': {'seed': 42,
                                'mixed_precision': 'fp16',
                                'allow_tf32': True,
                                'resolution': 512,
                                'reward_fn': {'pickscore': 1},
                                'beta': 1,
                                'decay_type': 1,
                                'activation_checkpointing': False,
                                'sample': 9,
                                'train': 10},
 'sd3_pickscore_nft_onpolicy.yaml': {'seed': 42,
                                     'mixed_precision': 'fp16',
                                     'allow_tf32': True,
                                     'resolution': 512,
                                     'reward_fn': {'pickscore': 1},
                                     'beta': 1,
                                     'decay_type': 0,
                                     'activation_checkpointing': False,
                                     'sample': 9,
                                     'train': 10},
 'sd3_pickscore_dppo.yaml': {'seed': 42,
                             'mixed_precision': 'fp16',
                             'allow_tf32': True,
                             'resolution': 512,
                             'reward_fn': {'pickscore': 1},
                             'activation_checkpointing': False,
                             'sample': 0,
                             'train': 4},
 'flux_geneval_art.yaml': {'seed': 42,
                           'mixed_precision': 'bf16',
                           'allow_tf32': True,
                           'resolution': 512,
                           'reward_fn': {'geneval': 1},
                           'activation_checkpointing': False,
                           'sample': 10,
                           'train': 11},
 'flux_geneval_fast.yaml': {'seed': 42,
                            'mixed_precision': 'bf16',
                            'allow_tf32': True,
                            'resolution': 512,
                            'reward_fn': {'geneval': 1},
                            'activation_checkpointing': False,
                            'sample': 10,
                            'train': 12},
 'flux_geneval_dppo.yaml': {'seed': 42,
                            'mixed_precision': 'bf16',
                            'allow_tf32': True,
                            'resolution': 512,
                            'reward_fn': {'geneval': 1},
                            'activation_checkpointing': False,
                            'sample': 11,
                            'train': 13}}


def scientific_expectation(name):
    row = SCIENCE_EXPECTATIONS[name]
    return {**row, "sample": SAMPLE_EXPECTATIONS[row["sample"]], "train": TRAIN_EXPECTATIONS[row["train"]]}

METHODS = ("art", "grpo", "fast", "dppo", "nft_ema", "nft_onpolicy", "dance")
FLUX_METHODS = ("art", "fast", "dppo", "nft_ema")
FLUX_LORA_TARGETS = (
    "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
    "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
    "ff.net.0.proj", "ff.net.2", "ff_context.net.0.proj", "ff_context.net.2",
)


@pytest.fixture(autouse=True)
def asset_environment(monkeypatch):
    names = set()
    for directory in (PROFILES, ROOT / "configs/examples"):
        for path in directory.glob("*.yaml"):
            names.update(re.findall(r"\$\{([A-Z_0-9]+)\}", path.read_text()))
    for name in names:
        value = "a" * 64 if name.endswith(("SHA256", "SHA")) else f"/public-assets/{name.lower()}"
        monkeypatch.setenv(name, value)




def test_candidate_coverage():
    assert set(PROFILES_EXPECTED) <= {p.name for p in PROFILES.glob("*.yaml")}
    for benchmark in ("geneval", "ocr", "pickscore"):
        for method in METHODS:
            assert (PROFILES / f"sd3_{benchmark}_{method}.yaml").is_file()
    assert (PROFILES / "flux_geneval_nft_ema.yaml").is_file()


@pytest.mark.parametrize("method", FLUX_METHODS)
def test_flux_lora_matches_pinned_comparison(method):
    c = load_config(PROFILES / f"flux_geneval_{method}.yaml")
    finetune = c.model.options["finetune"]
    assert finetune["mode"] == "lora"
    assert finetune["rank"] == 32
    assert finetune["alpha"] == 64
    assert tuple(finetune["target_modules"]) == FLUX_LORA_TARGETS


def test_flux_comparison_has_identical_trainable_parameter_inventory():
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    pytest.importorskip("peft")
    from visual_rl.models.finetune import FineTuneConfig, prepare_policy

    inventories = []
    with torch.random.fork_rng():
        for method in FLUX_METHODS:
            c = load_config(PROFILES / f"flux_geneval_{method}.yaml")
            model = diffusers.FluxTransformer2DModel(
                patch_size=1, in_channels=8, out_channels=8,
                num_layers=1, num_single_layers=1,
                attention_head_dim=16, num_attention_heads=2,
                joint_attention_dim=24, pooled_projection_dim=8,
                guidance_embeds=True, axes_dims_rope=(4, 6, 6),
            )
            model = prepare_policy(model, FineTuneConfig(**c.model.options["finetune"]))
            trainable = {name: (tuple(p.shape), p.dtype)
                         for name, p in model.named_parameters() if p.requires_grad}
            assert trainable and all("lora_" in name for name in trainable)
            assert all(dtype == torch.float32 for _, dtype in trainable.values())
            inventories.append(trainable)
    assert all(inventory == inventories[0] for inventory in inventories[1:])


@pytest.mark.parametrize("filename", sorted(PROFILES_EXPECTED))
def test_backend_and_public_candidate_contract(filename):
    c = load_config(PROFILES / filename)
    assert get_backend(c).name == c.model.name
    assert c.runtime.distributed == "ddp"
    assert c.runtime.log == "wandb"
    assert c.training.max_updates == 1000
    assert c.provenance["kind"] == "training_candidate_not_fullhistorical_reproduction"
    assert c.provenance["semantic_gaps"]
    assert c.training.allow_tf32 is True
    assert c.evaluation.every_updates == 50
    assert c.evaluation.at_updates == ()
    assert c.runtime.checkpoint_every == 200
    assert all(c.runtime.checkpoint_due(step, 1000) for step in (200, 400, 600, 800, 1000))
    assert c.evaluation.enabled
    assert c.evaluation.due(1000, 1000)
    assert c.runtime.checkpoint_due(1000, 1000)
    assert re.fullmatch(r"[0-9a-f]{40}", c.model.revision)
    assert len(c.data.sha256) == len(c.evaluation.manifest_sha256) == 64
    if c.algorithm.name == "diffusionart":
        assert c.runtime.checkpoint_at_updates == (1000,)
    if c.algorithm.name == "dancegrpo":
        assert c.algorithm.options["advantage_mode"] == "prompt_global_std"
        assert c.sampling.options["endpoint_kind"] == "last_x0"
        assert "source unavailable" in " ".join(c.provenance["semantic_gaps"])


@pytest.mark.parametrize("filename", sorted(set(PROFILES_EXPECTED) - {"flux_geneval_nft_ema.yaml"}))
def test_scientific_parameters_against_independent_constants(filename):
    c = load_config(PROFILES / filename)
    source = scientific_expectation(filename)
    sample, train = source["sample"], source["train"]
    assert c.seed == source["seed"]
    assert c.training.precision == source["mixed_precision"]
    assert c.training.allow_tf32 == source["allow_tf32"]
    assert c.training.learning_rate == train["learning_rate"]
    assert c.training.microbatch == train["batch_size"]
    assert c.training.weight_decay == train["adam_weight_decay"]
    assert c.training.betas == (train["adam_beta1"], train["adam_beta2"])
    assert c.training.epsilon == train["adam_epsilon"]
    assert c.training.grad_clip == train["max_grad_norm"]
    assert c.sampling.steps == sample["num_steps"]
    assert c.sampling.samples_per_prompt == sample["num_image_per_prompt"]
    assert c.sampling.prompts_per_rollout * sample["num_image_per_prompt"] == (
        sample["train_batch_size"] * sample["num_batches_per_epoch"] * 8
    )
    assert c.sampling.microbatch == sample["train_batch_size"]
    assert c.sampling.guidance_scale == sample["guidance_scale"]
    assert c.sampling.height == c.sampling.width == source["resolution"]
    assert c.evaluation.sampling_steps == sample["eval_num_steps"]
    assert c.reward.weights == source["reward_fn"]
    assert c.evaluation.every_updates == 50 and c.evaluation.at_updates == ()
    a, s = c.algorithm.options, c.sampling.options
    count = int(sample["num_steps"] * train["timestep_fraction"])
    if c.model.name == "sd3":
        assert s["shift"] == train["timestep_shift"]
    if c.algorithm.name == "diffusionart":
        assert s["train_timesteps"] == list(range(count))
        assert a.get("updates_per_rollout", 2) == (
            sample["num_batches_per_epoch"] // train["gradient_accumulation_steps"]
        )
        assert a["covariance"]["scale_mode"] == train["covariance_guidance_scale_mode"]
        assert a["covariance"]["guidance_gain"] == train["covariance_guidance_gain"]
        for target, original in (
            ("loss_space", "covariance_guidance_loss_space"),
            ("mask_space", "x0_covariance_mask_space"),
            ("velocity_mask_metric", "x0_covariance_velocity_mask_metric"),
            ("corrective_mask_enabled", "x0_covariance_corrective_mask_enabled"),
            ("q2_reference_sde_noise_level", "x0_covariance_q2_reference_sde_noise_level"),
            ("q2_reference_kl_threshold", "x0_covariance_q2_reference_kl_threshold"),
        ):
            assert a["loss"][target] == train[original]
    elif c.algorithm.name == "diffusionnft":
        assert a["loss"]["nft_beta"] == source["beta"]
        assert a["loss"]["adv_clip_max"] == train["adv_clip_max"]
        assert a["loss"]["adv_mode"] == train["adv_mode"]
        assert s["solver"] == sample["solver"]
        assert s["train_timestep_count"] == count
        expected_decay = {
            0: {"rate": 0, "maximum": 0, "delay": 0},
            1: {"rate": 0.001, "maximum": 0.5, "delay": 0},
            2: {"rate": 0.0075, "maximum": 0.999, "delay": 75},
        }
        assert a["old_policy"] == expected_decay[source["decay_type"]]
        assert a["evaluation_ema"]["enabled"] == train["ema"]
    else:
        assert s["noise_level"] == sample["noise_level"]
        assert s["logical_batch_size"] == sample["train_batch_size"]
        assert a["training_batches"] == sample["num_batches_per_epoch"]
        assert a["evaluation_ema"]["enabled"] == train["ema"]
        assert a["loss"]["advantage_clip"] == train["adv_clip_max"]
        if c.algorithm.name == "flow_dppo":
            assert a["loss"]["kl_mask_threshold"] == train["flow_dppo_kl_threshold"]
        else:
            assert a["loss"]["ratio_clip"] == train["clip_range"]
        if sample["sde_window_size"] > 0:
            start, stop = sample["sde_window_range"]
            assert s["pool"] == list(range(start, stop))
            assert s["count"] == sample["sde_window_size"]
        else:
            assert s["count"] == count
        if c.algorithm.name == "dancegrpo":
            assert s["same_latent"] == sample["same_latent"]


def test_ocr_art_one_update_preserves_restoration_and_full_window():
    from visual_rl.training.objectives import disjoint_windows

    one = load_config(PROFILES / "sd3_ocr_art_one_update.yaml")
    two = load_config(PROFILES / "sd3_ocr_art.yaml")
    assert get_backend(one).name == "sd3"
    options = dict(one.algorithm.options)
    assert options.pop("updates_per_rollout") == 1
    assert options == two.algorithm.options
    for field in ("model", "reward", "data", "sampling", "training", "evaluation"):
        assert getattr(one, field) == getattr(two, field)
    assert one.runtime.output_dir != two.runtime.output_dir
    assert one.provenance["candidate_updates_per_rollout"] == 1
    assert one.provenance["semantic_gaps"] == two.provenance["semantic_gaps"]
    rows = one.sampling.prompts_per_rollout * one.sampling.samples_per_prompt
    slots = len(one.sampling.options["train_timesteps"])
    windows = disjoint_windows(rows, slots, updates=one.algorithm.options["updates_per_rollout"])
    assert len(windows) == 1
    assert len(windows[0]) == rows * slots
    assert {row for row, _ in windows[0]} == set(range(rows))


def test_known_unresolved_semantics_remain_declared():
    c = load_config(PROFILES / "sd3_ocr_nft_ema.yaml")
    assert c.provenance["candidate_updates_per_rollout"] == 2
    assert c.algorithm.options["updates_per_rollout"] == 2
    for filename in ("sd3_geneval_dppo.yaml", "sd3_ocr_dppo.yaml", "sd3_pickscore_dppo.yaml"):
        c = load_config(PROFILES / filename)
        assert "checkpoint unavailable" in " ".join(c.provenance["semantic_gaps"])








def test_flux_nft_pinned_scientific_contract():
    c = load_config(PROFILES / "flux_geneval_nft_ema.yaml")
    assert c.sampling.prompts_per_rollout == 48
    assert c.sampling.samples_per_prompt == 24
    assert c.sampling.microbatch == c.training.microbatch == 6
    assert c.training.max_updates == 1000
    assert c.evaluation.every_updates == 50 and c.evaluation.at_updates == ()
    assert c.runtime.checkpoint_at_updates == (100, 250, 500, 800, 1000)
    assert c.sampling.steps == 6
    assert c.sampling.options["train_timestep_count"] == 5
    assert c.training.precision == "bf16"
    assert c.training.learning_rate == 0.0003
    assert c.algorithm.options["loss"]["nft_beta"] == 1
    assert c.algorithm.options["updates_per_rollout"] == 1
    assert c.model.options["gradient_checkpointing"] is True
