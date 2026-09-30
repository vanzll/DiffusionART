"""Optional, frozen FLUX text preparation; no sampling or conditioning cache.

Conventions inspected at research revision
3ca2a36f6b20c53a06813013234666d9c033a660: train_flux.py::compute_text_embeddings
and train_dreambooth_lora_flux.py::encode_prompt (blob
1100e280c247293707be5bbe2c5f08a27c6793b9). CLIPTextModel supplies its pooled
output, without a projection head; T5 supplies tokens. This delegates to public
Diffusers, not the historical implementation. CLIP retains its tokenizer limit
(77 for FLUX); T5 length is caller owned (the historical recipe used 128).
Historical checkpoint/mixed-precision parity is not claimed.
"""

import os

import torch
from torch import Tensor, nn

from .flux import FluxConditioning
from .sd3_text import _placement, _prompts


class FluxTextEncoder:
    """Own placement/freeze state of an injected pipeline's CLIP and T5.

    Both encoders and tokenizers must be present. Other assets are untouched.
    Do not share encoders with text training or install offload hooks: placement
    is explicit and resident. No negative prompts, CFG, LoRA, clip-skip, cached
    embeddings, or per-encoder overrides are exposed. FluxAdapter owns embedded
    guidance. Import directly from ``visual_rl.models.flux_text``.
    """

    def __init__(self, pipeline, *, device, torch_dtype):
        self.device = _placement(device, torch_dtype)
        self.torch_dtype = torch_dtype
        if not callable(getattr(pipeline, "encode_prompt", None)):
            raise ValueError("pipeline must provide encode_prompt")
        encoders = []
        for suffix in ("", "_2"):
            encoder = getattr(pipeline, "text_encoder" + suffix, None)
            tokenizer = getattr(pipeline, "tokenizer" + suffix, None)
            if not isinstance(encoder, nn.Module) or not callable(tokenizer):
                raise ValueError("both CLIP/T5 encoders and tokenizers are required")
            encoders.append(encoder)
        self.pipeline = pipeline
        self._encoders = tuple(encoders)
        for encoder in self._encoders:
            encoder.to(device=self.device, dtype=self.torch_dtype)
        self._freeze()

    def _freeze(self):
        for encoder in self._encoders:
            encoder.requires_grad_(False).eval()
            for parameter in encoder.parameters():
                parameter.grad = None

    @classmethod
    def from_pretrained(cls, model_id_or_path, *, revision, device, torch_dtype,
                        local_files_only=True):
        """Load text assets only, offline by default, at an explicit revision.

        Use a commit hash for reproducibility. Hugging Face's normal credential
        and cache configuration applies. Downloads require local_files_only=False.
        Transformer and VAE weights are never loaded.
        """
        if not isinstance(model_id_or_path, (str, os.PathLike)):
            raise ValueError("model_id_or_path must be a nonempty string or path")
        model_id_or_path = os.fspath(model_id_or_path)
        if not isinstance(model_id_or_path, str) or not model_id_or_path.strip():
            raise ValueError("model_id_or_path must be a nonempty string or path")
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError("revision must be explicit and nonempty")
        if not isinstance(local_files_only, bool):
            raise ValueError("local_files_only must be a bool")
        device = _placement(device, torch_dtype)

        from diffusers import FluxPipeline

        pipeline = FluxPipeline.from_pretrained(
            model_id_or_path, revision=revision, local_files_only=local_files_only,
            torch_dtype=torch_dtype, transformer=None, vae=None,
        )
        return cls(pipeline, device=device, torch_dtype=torch_dtype)

    def encode(self, prompt, *, max_sequence_length,
               num_images_per_prompt=1) -> FluxConditioning:
        """Return detached tokens [B*K,L,D], pools [B*K,Dclip], IDs [L,3].

        Rows are prompt-major: [p0,p0,p1,p1] for two prompts and K=2.
        T5 sequence length must be in 1..512; CLIP uses its tokenizer limit.
        IDs are unbatched zeros. Each tensor owns fresh ordinary storage usable
        by a trainable transformer, even under outer inference mode/autocast.
        """
        prompts = _prompts(prompt, "prompt")
        if type(max_sequence_length) is not int or not 1 <= max_sequence_length <= 512:
            raise ValueError("max_sequence_length must be an integer in [1, 512]")
        if type(num_images_per_prompt) is not int or num_images_per_prompt < 1:
            raise ValueError("num_images_per_prompt must be a positive integer")

        self._freeze()
        # Disable outer inference/autocast for both encoding and fresh storage.
        with torch.inference_mode(False), torch.no_grad(), torch.autocast(
                device_type=self.device.type, enabled=False):
            tokens, pooled, raw_text_ids = self.pipeline.encode_prompt(
                prompt=prompts, prompt_2=None, device=self.device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )
            batch = len(prompts) * num_images_per_prompt

            def checked(value, prefix, ndim, name):
                if (not isinstance(value, Tensor) or value.ndim != ndim
                        or tuple(value.shape[:len(prefix)]) != prefix
                        or any(size == 0 for size in value.shape)
                        or not value.is_floating_point()
                        or value.layout != torch.strided or value.device.type == "meta"):
                    raise ValueError(f"{name} has invalid shape, floating dtype, or storage")
                return value.detach().to(device=self.device, dtype=self.torch_dtype).clone()

            tokens = checked(tokens, (batch, max_sequence_length), 3, "prompt_embeds")
            pooled = checked(pooled, (batch,), 2, "pooled_prompt_embeds")
            text_ids = checked(raw_text_ids, (max_sequence_length, 3), 2, "text_ids")
            if torch.count_nonzero(raw_text_ids).item() != 0:
                raise ValueError("text_ids must be all zeros")
        return FluxConditioning(
            prompt_embeds=tokens, pooled_prompt_embeds=pooled, text_ids=text_ids,
        )
