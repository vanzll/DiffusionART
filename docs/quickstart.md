# Full Experiment Quick Start

Automatic preparation downloads tokenizer, image-preprocessing configuration
and license files for the pinned PickScore processor. `CLIPProcessor` does not
use the LAION model weights; the scorer loads its separate pinned PickScore
model. This keeps the PIL preprocessing and every model revision unchanged.

Install from the repository root:

```bash
uv sync --locked --python 3.10 --extra diffusers
```

Accept the SD3.5 model license on Hugging Face. Supply `HF_TOKEN` and
`WANDB_API_KEY` from your environment/secret store, or log in with the Hub and
W&B CLIs. Tokens do not belong in YAML or shell scripts.

```bash
export WANDB_PROJECT=my-diffusion-experiments
# Optional: export WANDB_ENTITY=my-team
export VISUAL_RL_ASSET_ROOT=/absolute/writable/path/assets
export PAPER_OUTPUT_ROOT=/absolute/writable/path/outputs

uv run --locked --extra diffusers visual-rl \
  --config configs/paper/image/sd3_pickscore_art.yaml --nproc-per-node 8
```

This runs 1000 successful optimizer updates, not a smoke test. The parent
process downloads pinned SD3, PickScore and processor snapshots using the Hub
cache/download APIs. It downloads the pinned public Flow-GRPO PickScore prompt
lists (25,432 train / 2,048 eval), checks their SHA256, converts them without
changing order and verifies the resulting manifest SHA256. GPU workers start
only after preparation completes. A gated-model error requires granting model
access; the launcher cannot accept licenses for you.

Downloads are resumable through the Hub's existing cache machinery. The asset
root defaults to `~/.cache/visual-rl`; `HF_HOME` can separately select Hub cache
and credential storage. No environment copying, GPU lease or extra supervisor
is involved: the parent replaces itself with ordinary `torchrun`.

Existing model/manifests environment variables take precedence. Custom JSONL
data needs only its path; SHA256 is recorded automatically. Supplied hashes
remain strict integrity checks. Using custom data does not certify it as the
paper dataset. The public default dataset is not claimed byte-identical to an
unavailable historical training snapshot.

The seven SD3 PickScore paper configs share this asset preparation. Other paper
configs also default to online W&B, but keep their task-specific data setup.
After [video environment installation](video-installation.md), the same
entrypoint downloads video models and training texts, prepares Hunyuan
conditioning, or converts the distilled Wan policy. Initial Hunyuan text
encoding uses a GPU and may take hours, with row-level progress. Wan's fixed
evaluation images currently need one user-supplied manifest path; they are not
replaced with newly generated images.

`--validate` and `--print-config` plan default paths without downloads.
`--prepare-only` prepares assets without starting workers or requiring W&B
credentials. Model licenses and Hub credentials may still be needed.

For multiple nodes, use the same writable shared asset/output roots and recipe
on every node. Set each node's rank and a reachable master address:

```bash
visual-rl --config configs/paper/video/hunyuan_art_721.yaml \
  --nproc-per-node 8 --nnodes 4 --node-rank "$NODE_RANK" \
  --master-addr "$MASTER_ADDR" --master-port 29500
```

The parent processes coordinate shared preparation before starting ordinary
torchrun workers. No cluster scheduler, Ray service or extra supervisor is
required. See [distributed execution](distributed-execution.md).
