# SD3 With CLIPScore

This integration example switches the [SD3 runner](sd3-runner.md) to the
research repository's frozen CLIP scorer. It is not a paper reproduction
profile. ART, NFT and the native GRPO-family runners use the same scoring API.

## Prepare And Run

Follow the SD3 guide for model access, model assets and GPU requirements.
Install the tensor image transforms and fetch the pinned scorer assets:

```bash
uv sync --locked --extra diffusers --extra image-rewards
uv run --locked --extra diffusers --extra image-rewards hf download openai/clip-vit-large-patch14 \
  --revision 32bd64288804d66eefd0ccbe215aa642df71cc41 \
  --include '*.json' '*.txt' '*.safetensors'
uv run --locked --extra diffusers --extra image-rewards visual-rl \
  --config configs/examples/sd3_clipscore.yaml --validate
uv run --locked --extra diffusers --extra image-rewards visual-rl \
  --config configs/examples/sd3_clipscore.yaml
```

Run from the repository root. Downloads need network access; training loads
assets offline by default. Configuration validation does not load weights or
prove GPU capacity. Training, evaluation and checkpoint outputs go to
`outputs/sd35-clipscore-integration`.

## Scoring Contract

CLIPScore consumes floating RGB tensors in [0,1] without a uint8 or PIL
roundtrip. It applies tensor bilinear resize, center crop and normalization
using the pinned processor's sizes and channel statistics. In contrast,
PickScore uses its own quantized PIL preprocessing. Sharing a CLIP model
interface does not make these preprocessing protocols interchangeable.

The configuration explicitly selects `antialias: true`. The historical scorer
used torchvision's default, which changed to true in version 0.17; binding an
older paper environment must verify that setting rather than infer it from
the model asset. See the [torchvision Resize reference](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.Resize.html).
The processor's PIL resampling and rescaling switches do not override this
tensor transform. Text uses maximum-length padding and truncation with the
pinned tokenizer's context length.

The model remains frozen in evaluation mode. The native factory uses FP32,
independently of policy forward precision. `reward/raw/clipscore` is the
paired image/text logit, including the model's learned logit scale.
`reward/fixed_normalized/clipscore` is that logit divided by 30, matching the
historical training wrapper. This is not a clamped cosine score. Both score
spaces are retained; `normalization: fixed` selects the latter for training.

In a [composite reward](composite-rewards.md), the component supplies raw
logits; use explicit top-level fixed statistics with mean 0 and std 30 to
reproduce this scale. It is applied once, never once per child and again in
the composite. ART joint advantages select their input score space separately.

## Verification

The source reference is `flow_grpo_alg/clip_scorer.py` at research commit
`3ca2a36f6b20c53a06813013234666d9c033a660`, blob
`65fdcb395c62d046474228526e78fb4d06336ef5`.
Offline tests compare tensor transforms and logits with an independent
reference and run real tiny-model loaders through train/eval CLI paths.
These checks do not establish pretrained GPU parity or reproduce a paper
curve; those acceptance runs remain required before release.
