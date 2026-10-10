# Trained Policy Export

Export is separate from resume: a complete checkpoint retains optimizer, RNG,
data cursors and algorithm/shadow state. An export contains only the **current
trained transformer policy**, never optimizer state, EMA/old policy, a reward
model, VAE, scheduler, text encoder or a complete inference pipeline.

## Supported Scope

- SD3/SD3.5 (`model.name: sd3`) and FLUX (`model.name: flux`).
- Full-weight FP32 transformers and the single default PEFT LoRA constructed by
  `models/finetune.py`. No merging, key conversion, quantization or dtype casting
  of checkpoint tensors. Adapter-only export rejects a changed frozen base.
- Schema-v1 local receipt `.json` and complete DDP `manifest.json`. All listed
  rank files are checked, even though optimizer/RNG are not exported. Different
  rank-local buffers are rejected rather than silently choosing a rank.
- **Not supported:** FSDP local shards (require process-group-aware consolidation),
  Hunyuan native/FastVideo, Wan video, arbitrary model/backend formats, raw `.pt`
  files, unfinished saves and `latest.json` pointers. This utility does not
  implement resharding or reinterpret video model architectures.

## Export

Use an environment with the project's Diffusers/PEFT optional dependencies.
Supply the original experiment config and its original resolved environment
variables: its fingerprint must match the checkpoint. Runtime CUDA/DDP settings
in that config do not launch training or select a GPU here.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m visual_rl.export \
  --config configs/my-experiment.yaml \
  --checkpoint /absolute/path/to/published-receipt.json \
  --output /absolute/path/to/new-policy-export
```

For DDP use `/absolute/path/to/step-.../manifest.json` instead. Resolve
`latest.json` yourself to a concrete publication before invoking the command.
No `torchrun`, CUDA or model download is needed. The configured base model and
revision must already be available locally/in the Hugging Face cache; the
constructor uses standard `from_pretrained(..., subfolder="transformer",
revision=..., local_files_only=True, torch_dtype=torch.float32)` APIs and does
not load other pipeline components. A full checkpoint and base transformer must
fit CPU memory; optimizer tensors are read for completeness but never applied.

Only export trusted, immutable inputs. SHA256 and byte counts detect corruption,
not malicious replacement. Loading uses PyTorch `weights_only=True` on CPU;
unsupported serialized objects are rejected, not retried with unsafe pickle.
Schema, config, progress, rank identities and strict policy keys/shapes/dtypes
are checked. Nonfinite policy tensors are rejected.

The output parent must exist; the destination must be new and outside source
checkpoint/base-model directories. Parent traversal and symlink paths are
rejected. Existing directories, even empty ones, are never overwritten. Normal
save failures remove the new output directory. A process crash can leave a
partial directory: only a successful final `export.json` marks completion;
remove/inspect such a directory manually or choose a new destination on retry.
There is no network-mount deadline, automatic storage failover, upload or push.
Choose an available storage root explicitly, and use the actual absolute output
path printed by the command and recorded in `export.json`.

## Standard Loading

Full export uses Diffusers `save_pretrained` with safetensors and model config:

```python
from diffusers import SD3Transformer2DModel  # or FluxTransformer2DModel

policy = SD3Transformer2DModel.from_pretrained(
    "/absolute/path/to/new-policy-export", local_files_only=True
)
```

LoRA export uses PEFT `save_pretrained`, producing `adapter_config.json` and
`adapter_model.safetensors`. Load the exact original base transformer first,
then attach the adapter (this is PEFT format, not pipeline `load_lora_weights`):

```python
from diffusers import FluxTransformer2DModel  # or SD3Transformer2DModel
from peft import PeftModel

base = FluxTransformer2DModel.from_pretrained(
    "/original/base/pipeline", subfolder="transformer",
    revision="original-revision", local_files_only=True,
)
policy = PeftModel.from_pretrained(
    base, "/absolute/path/to/new-policy-export", local_files_only=True
)
```

The adapter records base path/revision, but does not bundle that base. If moving
machines, map the recorded base to the same assets at their new location before
loading. `export.json` records source publication path/hash, config fingerprint,
optimizer step, model/base identity, format and actual output path. It contains
no training config dump or credentials. Exports cannot resume training.

API references: [Diffusers models](https://huggingface.co/docs/diffusers/api/models/overview)
and [PEFT model saving/loading](https://huggingface.co/docs/peft/package_reference/peft_model).

## Offline Verification

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q tests/test_export_model.py
```

Tests construct tiny real SD3/FLUX/PEFT components, use native local and CPU-Gloo
DDP checkpoint writers, and reload standard exports with exact tensor equality.
They cover corrupt/missing/incomplete saves, config and tensor mismatches,
unsupported backends, path safety, no-overwrite and failed-save cleanup. No GPU
or pretrained model downloads are required.
