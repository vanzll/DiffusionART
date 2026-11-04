# Wan2.1 T2V1.3B components

This backend is separate from the distilled Wan I2V14B adapter. Its native
components use the official Wan2.1 source revision
`9737cba9c1c3c4d04b33fcad41c111989865d315` and native model revision
`37ec512624d61f7aa208f7ea8140a131f93afc9a`.

The [official480P1.3B example](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/README.md#text-to-video)
uses832×480,81frames,16fps,50samplingsteps,CFG6,shift8 and UniPC. Its latent
shape is16×21×60×104. This is a native order2 predictor/corrector solver;
replacing it with Euler changes the sampler.

`visual_rl.models.wan_t2v.WanT2VAdapter` accepts injected native `WanModel`
and the native `WanVAE.model` module. Its lazy `from_pretrained` factory reads
an existing local snapshot and requires explicit `native_root` pointing to
the official source checkout at the pinned commit. Tracked source changes,
another cached `wan` module origin, a different weight revision, or another
model architecture fail before loading. Install the pinned official source separately.
The factory does not download assets or install optional dependencies.

Pass `WanT2VConditioning` with positive embeddings and their binary tokenizer
attention mask, negative embeddings/mask forCFG, and the native scheduler's
INT64 `timesteps` for every row. The physical sigma and integer time condition
are different quantities. DiT text embeddings are zero padded before projection
to reproduce official padding behavior. The transformer executes one physical
batch including bothCFGbranches; upstream patch/RoPE/unpatchify helpers retain
their native per-row operations.

For native attention LoRA the projection suffixes are `q,k,v,o`; both self
and text cross attention are included. Use the common `prepare_policy` API
with rank32,alpha32,dropout0 or full weights and FP32optimizer parameters;
forward autocast and distributed wrapping belong to the caller. The VAE stays
frozen and its affine normalization is applied exactly once.

Native schedule, detached history snapshots and replay utilities live in
`wan_t2v_schedule.py`. They preserve the native corrector, predictor history
and integer time table. The canonical ART objective currently assumes
`x+delta_sigma*v` and cannot replay that history directly. An exact conditional
native mean is affine `a*v+b` under frozen history; its velocity coefficient
generally differs from `delta_sigma`. Mean regression requires division by
`a²` plus matching weights, restoration and masks to preserve velocity ART.
The Q2 reference SDE mask is not a native UniPC transition KL.

The explicit compatibility gate fails until the training replay contract is
approved and integrated. These component interfaces and CPU test definitions
do not establish pretrained GPU parity,8GPUtrainability, memory requirements
or reward improvements.

The pinned [official code license](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/LICENSE.txt)
and [native model license](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/blob/37ec512624d61f7aa208f7ea8140a131f93afc9a/LICENSE.txt)
are Apache2.0; the pinned [model card](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/blob/37ec512624d61f7aa208f7ea8140a131f93afc9a/README.md)
declares `apache-2.0`. Credit the Alibaba Wan Team Authors. Native UniPC source
also identifies its derivation from Hugging Face Diffusers v0.31.0.

For reproducibility, obtain the native transformer, `Wan2.1_VAE.pth`,
`models_t5_umt5-xxl-enc-bf16.pth`, and bundled `google/umt5-xxl` tokenizer
from the same pinned model revision. The upstream dependency file contains
version ranges, including `numpy<2`, and is not a reproducible environment
lock. The training recipe must pin its resolved optional dependencies.

The native factory verifies source Git revision and tracked cleanliness, repository
root and imported module origins, expected immutable model revision, local component
files and full1.3B root configuration. Native transformer parameters remainFP32,
matching official loading; useBF16 forward autocast and the common fine-tuning
setup forFP32optimizer parameters. A blanketBF16 parameter cast breaks native
precision-sensitive time embedding/projection semantics and is rejected.

The launch master must separately hash weight/VAE/text/tokenizer files against
the pinned snapshot once and persist an asset receipt before validated execution.
Every rank may pass `asset_preflight(path,revision)` to validate that master receipt
before loading; raise on receipt mismatch. The factory deliberately avoids
rehashing17GB per rank. An expected revision argument and matching model config
alone do not verify local weight bytes.
