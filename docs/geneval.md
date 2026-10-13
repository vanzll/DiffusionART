# Native GenEval Integration

SD3 and FLUX image runners accept `reward.name: geneval`, alone or as a
component of `composite`. The scorer uses the historical JPEG input and dense
training reward. Accuracy and strict accuracy are diagnostics, not substitute
optimization targets. See [migration provenance](geneval-migration.md).

**Acceptance status:** scoring rules have CPU fixtures and pinned-upstream
comparisons; process transport has actual subprocess failure tests. Clean-Linux
installation, compiled detector operators and pretrained GPU/result parity are
not yet accepted across all supported environments. A successful short run
does not certify full paper-performance reproduction.

## Separate Worker Environment

The trainer keeps its current environment. `GENEVAL_PYTHON` selects a separate
Python 3.10 interpreter containing the detector and OpenCLIP dependencies. The
worker imports this checkout directly, so installing the entire training
dependency stack into that interpreter is unnecessary.

Versions read from the working reference deployment:

| Package | Version |
| --- | --- |
| torch | 2.1.2 |
| torchvision | 0.16.2 |
| mmcv-full | 1.7.2 |
| mmdet | 2.28.2 |
| open-clip-torch | 2.31.0 |
| clip-benchmark | 1.6.1 |
| numpy | 1.26.0 |
| Pillow | 12.2.0 |

The deployment's MMDetection source revision is
`e9cae2d0787cd5c2fc6165a6061f92fa09e48fb1`. Start from the
[reference installation instructions](https://github.com/yifan123/reward-server/blob/8245af22b75fe640950a880ad31fe702a5d9a014/README.md),
but pin these versions instead of following its moving `2.x` branch. Flask and
Gunicorn are not required by this adapter. MMCV CUDA operators must be built
for the actual GPU architecture and compatible PyTorch/CUDA toolchain; a
successful Python import alone does not validate them. The reproducible build
recipe/container is still an outstanding release gate, not a verified one-line
installation command.

## Prepare Local Assets

Prepare an immutable local bundle:

```text
assets/geneval/
  detector_config.py
  detector.pth
  clip.pt
  object_names.txt
```

- `detector.pth`: the Swin-S instance-segmentation checkpoint in the
  [pinned MMDetection model table](https://github.com/open-mmlab/mmdetection/blob/e9cae2d0787cd5c2fc6165a6061f92fa09e48fb1/configs/mask2former/README.md),
  named `mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth`.
- `clip.pt`: OpenAI ViT-L-14, obtained through the official OpenCLIP/OpenAI asset
  registry. The loader does not fetch weights at training time.
- `object_names.txt`: the exact ordered
  [reference class list](https://github.com/yifan123/reward-server/blob/8245af22b75fe640950a880ad31fe702a5d9a014/reward_server/object_names.txt).
- `detector_config.py`: a flattened version of the matching pinned config,
  produced in the worker environment with
  `mmcv.Config.fromfile(source_config).dump(output_config)`.
  Configs containing `_base_` references are rejected, since those extra files
  would otherwise fall outside the manifest's hashes.

Verified current reference asset SHA256 values:

| Asset | SHA256 |
| --- | --- |
| Detector | `743b7d99015f1224c6d57fd4b14d04b15cc8ec72ae7ee7831e7c71d8873b7a54` |
| OpenAI CLIP | `b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836` |
| Class list | `608f6a0e5c8ca1c7a92b818430141fe268c85b7930b50f4160f1d65893b5bacd` |

Obtain assets from trusted publishers and review their terms. Checkpoint loading
and MMDetection Python configs are trusted-code operations; checksumming a file
does not establish that its original source was safe. No weights are bundled.

Record the bundle and exact package versions using the intended interpreter:

```bash
export GENEVAL_PYTHON=/absolute/path/to/geneval-environment/bin/python
"$GENEVAL_PYTHON" -I src/visual_rl/rewards/geneval_loader.py --bundle assets/geneval
```

The command prints the absolute manifest path and SHA256. Assign them to
`GENEVAL_MANIFEST` and `GENEVAL_MANIFEST_SHA256`. It does not overwrite an
existing manifest. Every declared file and package version is checked again
when loading; the class vocabulary is additionally checked against its reference
hash. Hashing is synchronous local I/O, not a network-filesystem deadline.

## Full Training

After preparing the worker and exporting `GENEVAL_PYTHON`, `GENEVAL_MANIFEST`
and `GENEVAL_MANIFEST_SHA256`, accept the base-model license and configure
HF/W&B credentials. Launch from the trainer environment:

```bash
uv run --locked --extra diffusers visual-rl \
  --config configs/paper/image/sd3_geneval_art.yaml --nproc-per-node 8
# For FLUX, use configs/paper/image/flux_geneval_art.yaml instead.
```

These full-budget recipes prepare base models and benchmark prompts from HF
automatically; they do not install the separate detector worker. No manual
prompt download or conversion is needed. The public evaluation manifest is
already expanded four times per prompt; do not expand it again. See
[image recipes](paper-image.md) for budgets and other algorithms.

The optional `configs/examples/sd3_geneval.yaml` and `flux_geneval.yaml` are
small integration checks with illustrative prompts and manually supplied model
assets, not the recommended full-experiment entrypoints.

## Execution And Metrics

The worker shares the trainer's selected device but has its own interpreter and
CUDA context. Budget memory for both policy and reward models. The example uses
`detector_microbatch: 1` and `color_workers: 0`; color crops still use the original
batch size 16, without extra DataLoader processes. CPU color transforms are
deterministic. Increasing loader workers requires separate lifecycle/platform
validation; it is not a free GPU-memory optimization.

`microbatch` bounds each scoring request. The trainer rounds RGB to uint8 and
JPEG-encodes with Pillow defaults; the worker decodes JPEG before detection.
Keep both environments' Pillow/JPEG implementations fixed for pixel parity.
Startup and request deadlines default to 300 seconds, configurable up to 600.
A failed, hung or malformed worker stops scoring with an explicit error; it
does not turn an infrastructure failure into a reward, restart or replay it.

- `train/reward/raw/geneval`: dense score used for optimization, possibly negative.
- `train/diagnostics/geneval/strict_accuracy`: strict binary correctness, averaged
  over images. Training skips ordinary correctness, as the historical client did.
- `eval/diagnostics/geneval/accuracy` and `strict_accuracy`: per-image means.
- `eval/diagnostics/geneval/accuracy_macro` and `strict_accuracy_macro`: mean over
  represented task tags, with corresponding `*_tags_present` counts. A full
  benchmark should contain all six tags; a subset is not the same evaluation.
- `eval/diagnostics/geneval/tag/<tag>/...`: per-tag accuracy; nonmember `-10`
  sentinels are excluded, never averaged as scores.

Per-image diagnostics are persisted in the evaluation manifest, separately
from weighted rewards. Combination with other scorers preserves that separation.
