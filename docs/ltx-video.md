# Native LTX-Video 2B 0.9.6 distilled

This component adapter targets only
`ltxv-2b-0.9.6-distilled-04-25.safetensors`, from
[`Lightricks/LTX-Video` revision8984fa25007f376c1a299016d0957a37a2f797bb](https://huggingface.co/Lightricks/LTX-Video/tree/8984fa25007f376c1a299016d0957a37a2f797bb).
Native implementation reference:
[`Lightricks/LTX-Video` commit50f0aef1297574a4b29acbdb874e3a363a6d5f2e](https://github.com/Lightricks/LTX-Video/tree/50f0aef1297574a4b29acbdb874e3a363a6d5f2e).
It does not load LTX-2, LTX-2.5 or later/larger model variants.

The adapters and algebra tests do not establish GPU parity or a validated
eight-GPU training recipe. Native ART integration requires the replay and mask
contract below; a generic Euler rollout cannot substitute for this sampler.

## Frozen native profile

The [official0.9.6 configuration](https://github.com/Lightricks/LTX-Video/blob/50f0aef1297574a4b29acbdb874e3a363a6d5f2e/configs/ltxv-2b-0.9.6-distilled.yaml)
specifies eight steps, CFG1, STG0, `sampler=from_checkpoint`, stochastic sampling,
decode timestep0.05 and decode noise scale0.025. Prompt enhancement is disabled,
so neither Florence nor an extra language model is loaded. The T5 text encoder
and tokenizer come from the `text_encoder` and `tokenizer` subfolders of
`PixArt-alpha/PixArt-XL-2-1024-MS`, at immutable revision
`b89adadeccd9ead2adcb9fa2825d3fabec48d404`.

Use704×1216,153frames at30fps (5.1seconds), with latent shape
`[B,128,20,22,38]`. The0.9.6 release recommends1216×704 at30fps; its supported
grid requires spatial dimensions divisible by32 and frames of the form8n+1.
153frames satisfy the documented recommendation of fewer than257frames.
The native decoder yields8F−7frames because each temporal upsampling stage
removes the first duplicated frame. Its native `target_shape` argument uses
the upstream F×8 hint.

The native `LinearQuadratic(8)` sampler computes the following physical schedule
in FP32, including terminal zero. Checkpoint `allowed_inference_steps` values
are only a rounded four-decimal validation allowlist, not the solver table:

| Position | sigma | next sigma | velocity coefficient a | noise std |
| --- | --- | --- | --- | --- |
|0|1.000000|0.993750|−0.006250000|0.993750|
|1|0.993750|0.987500|−0.012421875|0.987500|
|2|0.987500|0.981250|−0.018515625|0.981250|
|3|0.981250|0.975000|−0.024531250|0.975000|
|4|0.975000|0.909375|−0.088359375|0.909375|
|5|0.909375|0.725000|−0.250078125|0.725000|
|6|0.725000|0.421875|−0.419140625|0.421875|
|7|0.421875|0.000000|−0.421875000|0.000000|

All eight native transitions have sigma>0 and a≠0. The terminal sigma0 is a
boundary, never a training input. The first four positions have sigma≥0.9812;
a Flash selection can select two distinct positions among those four while
retaining the entire native stochastic trajectory.

## Native stochastic transition and replay

The [official RF scheduler](https://github.com/Lightricks/LTX-Video/blob/50f0aef1297574a4b29acbdb874e3a363a6d5f2e/ltx_video/schedulers/rf.py)
uses the following transition, with a fresh standard-normal innovation epsilon:

```text
x0 = x - sigma * velocity
mean = (1 - next_sigma) * x0
next_state = mean + next_sigma * epsilon
a = -sigma * (1 - next_sigma)
b = (1 - next_sigma) * x
mean = a * velocity + b
```

This sampler has no multistep history. Cache the actual state, old velocity,
old mean, next state, sigma/next sigma, a/b coefficients, noise standard
deviation and innovation. `LTXVideoReplayStep` captures detached snapshots of
these quantities. Cache the final innovation even though its weight is zero:
upstream still draws it, changing the RNG state used for decode.

Initial noise uses `randn_tensor([B,F*H*W,C],generator=...)`, then unpacks to
`[B,C,F,H,W]`; decode noise is drawn on unpacked latents. Transition noise
uses `torch.randn_like` on packed `[B,F*H*W,C]` tensors, preserving strides.
`draw_ltx_transition_innovation` follows that storage order. Upstream scheduler
and decode calls draw from the global Torch RNG; merely supplying its pipeline
generator does not control those draws. An official parity comparison must
align the global RNG as well as initial noise, or inject/cache every draw.
FP32 timesteps promote the initially BF16 sampler state to FP32; later native
states remain FP32 while transformer forwards can remain BF16. Do not round
states back to BF16 after each transition.

For a frozen affine transition, dividing per-item mean-space squared error by
a² gives the corresponding velocity-space regression gradient. The same
frozen velocity target, timestep weights, inactive/restoration decisions,
corrective mask and global denominator must be used. The utility
`ltx_video_normalized_mean_mse` establishes this algebra only; it does not
choose or certify an ART mask. The existing `scheduler_reference_metric` is
a reference Q2 SDE metric, not KL under this LTX transition kernel. Calling
it native LTX KL or replacing its coefficient with a without an explicit
scientific decision changes the method. In particular, the final LTX
transition has zero variance, so native Gaussian KL is singular there.

## Component behavior

`LTXVideoAdapter.predict_velocity` packs the full latent batch, makes one
physical transformer call and unpacks its result. It preserves the binary
T5 attention mask for different prompt lengths. The native transformer receives
sigma as `[B,1]`; it applies its own factor1000. Spatial coordinates use32×
latent positions, and causal temporal coordinates are `max(8*f-7,0)/fps`.
No per-sample transformer loop is hidden inside this adapter.

The [official inference path](https://github.com/Lightricks/LTX-Video/blob/50f0aef1297574a4b29acbdb874e3a363a6d5f2e/ltx_video/inference.py)
enables **external per-channel latent normalization**, independently of the
VAE's internal `normalize_latent_channels=False` configuration. Conditioned
decode therefore performs:

```text
decode_latent = 0.975 * endpoint + 0.025 * decode_noise
vae_latent = decode_latent * vae.std_of_means + vae.mean_of_means
pixels = vae.decode(vae_latent, timestep=0.05, target_shape=native_shape)
```

The VAE is frozen but remains differentiable with respect to input latents.
Native pixels are returned before presentation clipping; nonfinite values
raise an error. Decode noise can be supplied explicitly for paired base/post
evaluation. Upstream tiled decode branches omit timestep conditioning, so
this adapter rejects `use_hw_tiling` and `use_z_tiling`; enabling those branches
requires a separate verified conditioning-preserving implementation.

`load_ltx_video_components` lazily loads the optional native classes, freezes
the T5 encoder/VAE, and requires the exact pinned repository/revision for remote
T5 assets. Supply an explicit `native_root` pointing to a clean checkout at the
code revision above; the loader checks Git identity and rejects conflicting
package or concrete component origins before loading weights. Local text assets
use the caller's verified asset receipt. The loader does not alter the
environment or select an optimizer. The common
`prepare_policy` contract supplies full FP32 optimizer parameters or
rank32/alpha32/dropout0 LoRA. Native attention projection suffixes are
`to_q`, `to_k`, `to_v`, `to_out.0`; validate their concrete module matches and
parameter inventory before constructing the optimizer.

## Licenses and attribution

The [pinned native source LICENSE](https://github.com/Lightricks/LTX-Video/blob/50f0aef1297574a4b29acbdb874e3a363a6d5f2e/LICENSE)
is Apache-2.0. The weights use a separate license: the
[pinned HF model card](https://huggingface.co/Lightricks/LTX-Video/blob/8984fa25007f376c1a299016d0957a37a2f797bb/README.md)
declares `license: other` and explicitly maps **2B0.9.6-distilled** to the
[LTXV Open Weights License0.X](https://huggingface.co/Lightricks/LTX-Video/blob/8984fa25007f376c1a299016d0957a37a2f797bb/LTX-Video-Open-Weights-License-0.X.txt),
dated April15,2025. Its SHA256 is
`30eabbcc090dd3bd70cffaa14338142df82f348f7bd91b2f63f0c6099ccf50e0`.

The exact distilled safetensors header additionally embeds a model-specific
LTXV Open Weights License identifying **LTX Video-2B-Distilled-04-25**, dated
April17,2025 (UTF8 text SHA256
`98223c8f76ebda6bcf9b12a739d397a68979b3b57510b81ab8768609759e4d62`).
Both model-license texts include use restrictions and redistribution conditions;
section2 addresses commercial entities with annual revenue of at least
USD10million and a separate commercial-use agreement. Consult the linked
model license for its terms; Apache-2.0 source licensing does not replace it.
Retain the applicable model license and Lightricks attribution with downloaded
or redistributed assets. This repository does not redistribute model weights.
