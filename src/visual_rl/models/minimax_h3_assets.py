"""Lazy, offline-by-default native H3 component loading, without service stacks."""

import json
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch

from .minimax_h3 import (
    DIFFUSERS_REVISION,
    MODEL_ID,
    MODEL_REVISION,
    MiniMaxH3Adapter,
    MiniMaxH3Conditioning,
)

# Deliberately exclude transformer_ref, original checkpoint formats and examples.
H3_ASSET_ALLOW_PATTERNS = (
    "model_index.json", "README.md", "LICENSE*",
    "transformer/*.json", "transformer/*.safetensors",
    "text_encoder/*.json", "text_encoder/*.safetensors",
    "tokenizer/*", "processor/*",
    "vae/*.json", "vae/*.safetensors", "audio_vae/*.json", "audio_vae/*.safetensors",
    "scheduler/*.json", "audio_scheduler/*.json",
)


@dataclass
class MiniMaxH3Conditioner:
    text_encoder: object
    tokenizer: object
    processor: object
    text_encoder_layer: int = 50

    def staged(self, device):
        return stage_native_components((self.text_encoder,), device)

    @torch.no_grad()
    def encode(self, prompts):
        from diffusers.modular_pipelines.minimax_h3.encoders import get_qwen3vl_prompt_embeds
        if not prompts or any(not isinstance(p, str) or not p for p in prompts):
            raise ValueError("H3 prompts must be nonempty strings")
        parameter = next(self.text_encoder.parameters())
        embeds = []
        for prompt in prompts:
            ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            if not ids:
                raise ValueError("H3 prompt tokenized to an empty sequence")
            embeds.append(get_qwen3vl_prompt_embeds(
                self.text_encoder, self.processor, ids, {}, self.text_encoder_layer,
                device=parameter.device, dtype=parameter.dtype,
            )[0])
        lengths = torch.tensor([len(e) for e in embeds], device=parameter.device)
        padded = torch.nn.utils.rnn.pad_sequence(embeds, batch_first=True)
        mask = torch.arange(padded.shape[1], device=parameter.device)[None] < lengths[:, None]
        return MiniMaxH3Conditioning(padded, mask)


@contextmanager
def stage_native_components(components, device):
    """Temporarily place frozen native modules, then offload ALL back to CPU.

    No dtype conversion or trainable policy staging is permitted. Callers must
    budget policy/optimizer residency separately: a staged conditioner can still
    exceed device capacity when optimizer state is resident.
    """
    components = tuple(dict.fromkeys(components))
    if not components or any(not isinstance(m, torch.nn.Module) for m in components):
        raise ValueError("staging requires frozen torch modules")
    for component in components:
        if any(p.requires_grad for p in component.parameters()):
            raise ValueError("never stage a trainable policy with frozen native components")
        if getattr(component, "hf_device_map", None) or any(
            hasattr(m, "_hf_hook") or hasattr(m, "_diffusers_hook") for m in component.modules()
        ):
            raise ValueError("remove automatic placement/offload hooks before explicit staging")
    try:
        for component in components:
            component.to(device)
        yield
    finally:
        error = None
        for component in components:
            try:
                component.to("cpu")
            except Exception as exc:
                error = error or exc
        if error is not None:
            raise RuntimeError("frozen H3 offload failed; residency is unknown") from error


def resolve_minimax_h3_assets(model_id=MODEL_ID, revision=MODEL_REVISION, *,
                             cache_dir=None, local_files_only=True):
    """Resolve a pinned repository or a prepared snapshot with explicit manifest.

    Local prepared snapshots must retain the Hub snapshots/<commit> identity or
    contain h3_asset_manifest.json written by the preparation layer.
    """
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("H3 requires an immutable forty-character Hub revision")
    local = Path(model_id).expanduser()
    if local.is_dir():
        root = local.resolve()
        manifest_path = root / "h3_asset_manifest.json"
        if root.name != revision:
            if not manifest_path.is_file():
                raise ValueError("local H3 assets need explicit revision provenance")
            record = json.loads(manifest_path.read_text())
            if record.get("revision") != revision:
                raise ValueError("local H3 asset revision does not match requested revision")
    else:
        from huggingface_hub import snapshot_download
        root = Path(snapshot_download(
            repo_id=model_id, revision=revision, cache_dir=cache_dir,
            local_files_only=local_files_only, allow_patterns=list(H3_ASSET_ALLOW_PATTERNS),
        ))
    index_path = root / "model_index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"H3 model_index.json missing in {root}")
    index = json.loads(index_path.read_text())
    expected = {
        "transformer": "MiniMaxH3Transformer3DModel", "vae": "AutoencoderKLMiniMaxH3",
        "audio_vae": "AutoencoderKLMiniMaxH3Audio",
        "text_encoder": "Qwen3VLForConditionalGeneration", "tokenizer": "Qwen2TokenizerFast",
        "processor": "Qwen3VLProcessor",
    }
    components = {}
    for name, cls in expected.items():
        entry = index.get(name)
        if not isinstance(entry, list) or len(entry) < 2 or entry[1] != cls:
            raise ValueError(f"Unsupported native H3 component: {name}")
        if len(entry) > 2:
            source = entry[2]
            if (source.get("pretrained_model_name_or_path") not in (None, MODEL_ID, model_id)
                    or source.get("revision") not in (None, revision)
                    or source.get("subfolder", name) != name):
                raise ValueError(f"H3 component has unresolved foreign identity: {name}")
        path = root / name
        if not path.is_dir():
            raise FileNotFoundError(f"H3 component missing: {path}")
        components[name] = {"class": cls, "path": str(path), "revision": revision}
    return root, {"model_id": str(model_id), "revision": revision,
                  "diffusers_revision": DIFFUSERS_REVISION, "components": components}


def load_minimax_h3(model_id=MODEL_ID, revision=MODEL_REVISION, *, device="cpu",
                   dtype=torch.bfloat16, finetune=None, cache_dir=None,
                   local_files_only=True):
    if torch.device(device).type != "cpu":
        raise ValueError("load H3 components on CPU, then shard policy and stage frozen modules explicitly")
    root, manifest = resolve_minimax_h3_assets(
        model_id, revision, cache_dir=cache_dir, local_files_only=local_files_only,
    )
    # Actual loading requires a prepared, validated inventory. A matching folder
    # name or model_index structure is sufficient only for the low-level resolver.
    from .prepare_minimax_h3 import validate_assets
    integrity = validate_assets(root)
    manifest["asset_integrity"] = integrity

    import diffusers

    from .check_minimax_h3_environment import check_native_sources
    actual_sources, source_errors = check_native_sources(Path(diffusers.__file__).parent)
    if source_errors:
        raise RuntimeError("H3 requires exact pinned native source files: " + "; ".join(source_errors))
    manifest["native_source_sha256"] = actual_sources

    from diffusers import (
        AutoencoderKLMiniMaxH3,
        AutoencoderKLMiniMaxH3Audio,
        MiniMaxH3Transformer3DModel,
    )
    from transformers import Qwen2TokenizerFast, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    def load(cls, name):
        return cls.from_pretrained(root / name, torch_dtype=dtype, local_files_only=True)
    transformer = load(MiniMaxH3Transformer3DModel, "transformer")
    vae = load(AutoencoderKLMiniMaxH3, "vae").to(device)
    audio_vae = load(AutoencoderKLMiniMaxH3Audio, "audio_vae").to(device)
    # Mixed native precision is defined by _keep_in_fp32_modules; never cast the
    # whole component again with .to(dtype), which would discard that contract.
    transformer = transformer.to(device)
    adapter = MiniMaxH3Adapter(transformer, vae, audio_vae)
    if finetune is not None:
        from .finetune import prepare_policy
        adapter.set_trainable_module(prepare_policy(transformer, finetune))
    text_encoder = load(Qwen3VLForConditionalGeneration, "text_encoder").to(device)
    text_encoder.requires_grad_(False).eval()
    tokenizer = Qwen2TokenizerFast.from_pretrained(root / "tokenizer", local_files_only=True)
    processor = Qwen3VLProcessor.from_pretrained(root / "processor", local_files_only=True)
    return adapter, MiniMaxH3Conditioner(text_encoder, tokenizer, processor), manifest
