# Four-Reward Image Training

The SD3.5-M recipe scores GenEval, PickScore, CLIPScore and HPSv2.1.
HPS uses the official `hpsv2==1.2.0` ViT-H-14 implementation and the pinned
`xswu/HPSv2` v2.1 checkpoint. Its score is the paired normalized-feature dot
product, not a CLIP logit or an aesthetic-predictor score. No weights are bundled.

## Setup

```bash
uv sync --locked --python 3.10 --extra diffusers --extra image-rewards --extra hpsv2
```

The lock selects an [asset-complete HPS wheel](../third_party/hpsv2/README.md):
the PyPI wheel and source distribution omit the tokenizer vocabulary. Only
that resource and wheel packaging metadata are repaired, plus removal of an
unused GUI import for headless hosts. Model, preprocessing and scoring code
are unchanged. No installed files are patched. The lock also overrides
HPS's obsolete upstream **test-tool** requirements. HPS's `protobuf<4`
dependency requires a compatible W&B version, pinned by the lock.

For a pip installation, install that wheel before this project's extras:

```bash
python -m pip install third_party/hpsv2/hpsv2-1.2.0-2-py3-none-any.whl
python -m pip install '.[diffusers,image-rewards,hpsv2]'
```

Prepare the separate [GenEval worker and asset bundle](geneval.md), then set
`GENEVAL_PYTHON`, `GENEVAL_MANIFEST` and `GENEVAL_MANIFEST_SHA256`. Supply the full
structured benchmark manifests through `GENEVAL_TRAIN_MANIFEST` and
`GENEVAL_EVAL_MANIFEST`; optional `GENEVAL_TRAIN_SHA256`/`GENEVAL_EVAL_SHA256`
provide expected dataset hashes. Do not substitute the four-row example dataset.
Every row must retain GenEval's prompt and task annotations.

Accept the SD3 model's Hub license and configure Hub/W&B credentials as in the
[quick start](quickstart.md). Then run:

```bash
export WANDB_PROJECT=my-diffusion-experiments
export VISUAL_RL_ASSET_ROOT=/absolute/writable/path/assets
export PAPER_OUTPUT_ROOT=/absolute/writable/path/outputs

uv run --locked --extra diffusers --extra image-rewards --extra hpsv2 visual-rl \
  --config configs/paper/image/sd3_multireward_art.yaml --nproc-per-node 8
```

The parent prepares pinned SD3, PickScore, processor, CLIP and HPS snapshots
before launching GPU workers. HPS downloads only `HPS_v2.1_compressed.pt`;
its local SHA256 is checked before safe, strict state-dict loading. Loading the
complete HPS state does not require downloading a second LAION checkpoint.
Existing local asset bindings take precedence. GenEval's separate worker and
benchmark manifests still require explicit provisioning.

## Score And Update Contracts

The logged/evaluated scalar is the **sum**:

```text
GenEval + PickScore raw logit / 26 + CLIPScore raw logit / 30 + HPSv2.1 cosine
```

Raw components remain available under `train/reward/raw/<component>` and
`eval/reward/raw/<component>`; scaled components use `fixed_normalized`.
`train/reward` and `eval/reward` report the sum, not the average advantage.

ART independently centers and population-std normalizes each scaled reward
within a prompt group, using `std + 0.0001`, then averages the four advantages.
That combined signal supplies the endpoint covariance; it is not clipped or
normalized again before covariance estimation. The usual unit-RMS covariance
direction and two disjoint MSE updates follow. Scoring weights and training
advantage weights therefore have distinct roles.

The explicit `algorithm.options.multi_reward_advantage` option also supports
joint advantages in the native image GRPO/NFT update paths. When absent, their
existing scalar-reward normalization is unchanged. With it enabled, clipping
remains in the original algorithm objective, after advantage combination.
The supplied GRPO-family recipes enable joint advantages. The NFT recipes
instead normalize the summed scalar.

## Training And Acceptance

ART uses P48/K24, ten ODE sampling steps, ten update positions, two disjoint updates,
LR 0.0003, FP16, no CFG, no reference KL and no evaluation EMA.
The training budget is 1000 successful optimizer updates.

The author's later-approved public scalar zero-signal rule is retained:
no restoration on exact-zero-signal rows, with the original window denominator.
This is not advertised as bitwise identity with historical training. Historical
asset hashes, sampler/RNG order and GPU result parity remain unverified.

Configuration CPU tests check score pairing, quantization, component
scales, advantage composition, configuration boundaries and update integration.
They do not certify pretrained HPS GPU inference or full paper-curve reproduction.
