# SD3 OCR Integration

The native image runner accepts `reward.name: ocr`. This adapter preserves the
research image OCR reward, not the different video-OCR frame aggregation.
It uses PaddleOCR for recognition on CPU, as the original `ocr_score` wrapper
did even when policy training used a GPU.

The engine runs once per scorer in a persistent fresh-interpreter child. This
isolates PaddleOCR 2.x's `tools` module and import-path modifications from the
trainer. RGB bytes and native OCR results cross a dedicated pipe; recognition
order and the reward rule are unchanged. Startup is limited to 120 seconds,
each request to 60 seconds, and input RGB to 64 MiB. A hung/dead worker or
protocol failure is fatal, never interpreted as a low-quality image and never
silently restarted. Normal completion and exceptions close the worker.

**Current validation:** score semantics and SD3 train/eval integration have
offline CPU tests with an injected OCR engine. The native Paddle engine and
its full dependency environment still require Linux/model acceptance. This
guide does not certify an end-to-end paper reproduction yet.

## Environment And Assets

The loader currently supports exactly `paddleocr==2.9.1` and
`paddlepaddle==2.6.2` (CPU), with an explicit `PP-OCRv3` or `PP-OCRv4` model
bundle. These selected versions are not inferred historical paper versions.
Recover and verify the original environment and model hashes before labeling
a config as a paper reproduction.

Use a separate Linux Python 3.10 environment for Paddle's older dependencies;
do not add Paddle to the main lock by hand. The package's `ocr` extra supplies
Pillow and RapidFuzz, not Paddle. A candidate environment, pending acceptance:

```bash
python3.10 -m venv .venv-ocr
.venv-ocr/bin/python -m pip install '.[diffusers,ocr]' \
  'numpy<2' 'paddleocr==2.9.1' 'paddlepaddle==2.6.2'
```

The full training recipe downloads SD3 automatically. Prepare the OCR bundle
from the exact selected model artifacts, in this layout:

```text
assets/ocr/
  det/inference.pdmodel
  det/inference.pdiparams
  rec/inference.pdmodel
  rec/inference.pdiparams
  cls/inference.pdmodel
  cls/inference.pdiparams
  en_dict.txt
```

The [PaddleOCR 2.9.1 model registry](https://github.com/PaddlePaddle/PaddleOCR/blob/v2.9.1/paddleocr.py)
identifies English detector/recognizer assets and the corresponding character
dictionary. The English PP-OCRv4 profile uses the v3 English detector and v4
English recognizer. Do not substitute model generations based on filenames.
The disabled angle classifier must also be present: this constructor checks
its assets regardless of whether angle classification is enabled. No model
weights or upstream dictionaries are redistributed here.

Record the prepared bundle with the portable helper (it does not download or
certify the origin of the supplied assets):

```bash
.venv-ocr/bin/python -m visual_rl.rewards.paddle_loader \
  --bundle assets/ocr --ocr-version PP-OCRv4
```

It prints the absolute manifest path and SHA256. Existing manifests are not
overwritten. Set `OCR_MANIFEST` and `OCR_MANIFEST_SHA256` to those two values.
Use an immutable, locally readable bundle; the loader hashes every required
file before constructing the engine. Missing files or dependency drift fail
instead of selecting another OCR version. These are local synchronous asset
checks, not network-filesystem deadlines.

## Full Training

After setting `OCR_MANIFEST` and `OCR_MANIFEST_SHA256`, accept the base-model
license and configure HF/W&B credentials. Run with the OCR-capable interpreter:

```bash
.venv-ocr/bin/visual-rl \
  --config configs/paper/image/sd3_ocr_art.yaml --nproc-per-node 8
```

This runs 1000 updates and prepares the base model and benchmark prompts from
HF automatically. The OCR subprocess uses the trainer's Python interpreter,
so Paddle must be installed in this environment; unlike GenEval, there is no
separate worker-interpreter setting. No manual prompt download is needed.

The optional `configs/examples/sd3_ocr.yaml` has four updates and two
illustrative prompts with manually supplied SD3 assets; it is not the full
experiment. `--validate` checks configuration, not native models or GPU memory.
Train/eval artifacts and checkpoints use the same runner as other image rewards.

## Exact Reward Rule

1. Extract the first complete double-quoted substring from each prompt.
2. Convert tensor images to RGB uint8 by rounding `pixels * 255`. Do not swap
   RGB and BGR channels to match a different OCR convention.
3. Call the engine separately for each image with `cls=False`. Concatenate
   text in its returned order, retaining entries with confidence greater than
   zero. The engine's own detection/confidence filtering remains unchanged.
4. Remove ASCII spaces and lowercase both target and recognized text. Tabs,
   punctuation and other whitespace are not removed.
5. If the target is a substring of recognized text, return 1. Otherwise return
   `1 - min(edit_distance, target_length) / target_length`.

Scores are raw FP32 values in [0,1], without batch normalization or another
fixed divisor. `train/reward/raw/ocr` and `eval/reward/raw/ocr` retain that
meaning. Physical score microbatch bounds conversion memory; it does not batch
the sequential native OCR calls or change the formula. A CPU engine can
return scores on the policy device for composition with other scorers.

The error policy is required: `zero` reproduces the historical exception-to-zero
behavior and emits a warning; `raise` stops on recognition/parsing failures.
An ordinary no-text result remains a valid zero under either policy. Worker
startup, timeout and transport failures are not governed by that policy. Invalid
or empty quoted targets are rejected before model loading for both training
and validation datasets. Malformed prompts are not silently converted to zero.

## Provenance

Research reference: `flow_grpo_alg/ocr.py` at
`3ca2a36f6b20c53a06813013234666d9c033a660`, blob
`9223f65cef2a407779c84a4867331b1ff56d4650`, plus the quantization wrapper in
`flow_grpo_alg/rewards.py`. RapidFuzz's unit-cost Levenshtein distance implements
the same edit-distance definition. No differentiable OCR path is implied.
