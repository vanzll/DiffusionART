# Native Aesthetic Reward

The SD3 runner accepts `reward.name: aesthetic` independently of its ART, NFT,
Flow-GRPO, Fast or Flow-DPPO algorithm switch. The scorer is frozen and returns
native aesthetic scores, without a sigmoid, division by ten or group scaling.
It is not a differentiable ReFL reward adapter.

## Assets And Run

First install the dependencies and prepare SD3 as described in the
[runner guide](sd3-runner.md). From the repository root, choose a persistent
asset location; the example below uses the user's cache directory:

```bash
uv run --locked --extra diffusers hf download openai/clip-vit-large-patch14 --revision 32bd64288804d66eefd0ccbe215aa642df71cc41 --include '*.json' --include '*.txt' --include '*.safetensors'
export AESTHETIC_HEAD_PATH="$HOME/.cache/visual-rl/sac+logos+ava1-l14-linearMSE.pth"
curl --fail --location --max-time 120 --create-dirs \
  'https://raw.githubusercontent.com/christophschuhmann/improved-aesthetic-predictor/fe88a163f4661b4ddabba0751ff645e2e620746e/sac%2Blogos%2Bava1-l14-linearMSE.pth' \
  --output "$AESTHETIC_HEAD_PATH"
uv run --locked --extra diffusers visual-rl --config configs/examples/sd3_aesthetic.yaml --validate
uv run --locked --extra diffusers visual-rl --config configs/examples/sd3_aesthetic.yaml
```

The head's expected SHA256 is pinned in the config. The loader verifies it
before loading its state dictionary with `weights_only=True` and strict key
matching. Assets remain outside the repository. The run defaults to offline
loading; an absent or incompatible asset is an error, not a fallback model.

Four optimizer updates, evaluation at 0/2/4 and full checkpoints at 2/4 use the
same lifecycle as the PickScore example. `train/reward/raw/aesthetic` and
`eval/reward/raw/aesthetic` contain the native score; their corresponding total
reward is identical for this single-scorer weight-one configuration. No
`pickscore` metric is emitted. Other algorithms can use this same reward block.
Only `normalization: raw` is accepted for this standalone scorer; a fixed scale
must not be guessed from an unrelated benchmark.

## Scoring Contract

The historical research wrapper rounds decoded `[0,1]` RGB to uint8 before
CLIP preprocessing. The adapter preserves that quantization and uses the slow
PIL-based CLIP image processor, projected image features with unit L2 norm and
the original 768-1024-128-64-16-1 head. Its dropout layers stay in evaluation
mode, and it has no inserted activation functions. Text is not a model input,
but the API validates one prompt per image to preserve sample identities.

The [upstream architecture](https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/fe88a163f4661b4ddabba0751ff645e2e620746e/simple_inference.py)
uses OpenAI CLIP. The research implementation instead uses Transformers CLIP;
this adapter follows the research input path. It does not claim the two library
implementations are bitwise interchangeable. Reference research source:
`flow_grpo_alg/aesthetic_scorer.py`, blob
`a36046f2a3ea7b063aa1997864b322384eca42be`, under canonical commit `3ca2a36`.
The downloaded upstream head and the research asset have identical SHA256
`21dd590f3ccdc646f0d53120778b296013b096a035a2718c9cb0d511bff0f1e0`.

Tiny local CLIP/SD3 assets test the real offline loader, train/eval and metric
identities. Pretrained GPU score parity and paper result reproduction remain
separate acceptance work; this example is not a four-reward paper profile.
