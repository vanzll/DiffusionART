"""Instance-scoped batched extension of the immutable native MiniMax-H3 ABI.

The released ABI shares layout/modulation indices over B. These subclasses keep
all native modules and state-dict keys, but allow per-example rotary coordinates
and AdaLN indices, and mask padded text in both attention stacks. No upstream
class, function, file, or other loaded model is modified.

The algebra below follows Diffusers' pinned MiniMax-H3 implementation (Apache
2.0, Copyright 2025 The MiniMax Team and The HuggingFace Team).
"""

import torch
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.modeling_utils import get_parameter_dtype
from diffusers.models.transformers import transformer_minimax_h3 as native
from diffusers.utils import apply_lora_scale


def _rotary(x, cos, sin):
    if cos.ndim == 2:
        cos, sin = cos[None], sin[None]
    width = cos.shape[-1]
    leading, trailing = x[..., :width], x[..., width:]
    first, second = leading.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    return torch.cat((leading * cos.to(x.dtype)[:, :, None]
                      + rotated * sin.to(x.dtype)[:, :, None], trailing), dim=-1).contiguous()


def _padding_attention_mask(valid):
    # SDPA treats a None mask as all-visible. Keeping a redundant bool mask
    # can exclude fused backends even when every token is valid.
    return None if bool(valid.all()) else valid[:, None, None, :]


class MiniMaxH3BatchedAttnProcessor(native.MiniMaxH3AttnProcessor):
    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
        if attn.fused_projections:
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            query, key, value = (projection(hidden_states)
                                 for projection in (attn.to_q, attn.to_k, attn.to_v))
        query, key, value = (x.unflatten(-1, (attn.heads, -1)) for x in (query, key, value))
        query, key = attn.norm_q(query), attn.norm_k(key)
        if rotary_emb is not None:
            query, key = (_rotary(x, *rotary_emb) for x in (query, key))
        result = dispatch_attention_fn(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0,
            is_causal=False, backend=self._attention_backend, parallel_config=self._parallel_config,
        ).flatten(2, 3).type_as(query)
        return attn.to_out[1](attn.to_out[0](result))


# Keep class names intact: the H3 FSDP wrapper identifies dtype-query units by
# their native names. Subclassing also preserves native block isinstance checks.
class MiniMaxH3TokenRefinerBlock(native.MiniMaxH3TokenRefinerBlock):
    def forward(self, hidden_states, attention_mask=None):
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), attention_mask=attention_mask)
        return hidden_states + self.ff(self.norm2(hidden_states))


class MiniMaxH3TokenRefiner(native.MiniMaxH3TokenRefiner):
    def forward(self, hidden_states, attention_mask=None):
        for block in self.refiner_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(block, hidden_states, attention_mask)
            else:
                hidden_states = block(hidden_states, attention_mask)
        return self.final_norm(hidden_states)


class MiniMaxH3TransformerBlock(native.MiniMaxH3TransformerBlock):
    def forward(self, hidden_states, temb, adaln_indices, rotary_emb, attention_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(temb)
        # Tensor indexing handles either native [S] or batched [B,S] indices.
        residual = hidden_states
        normalized = self.norm1(hidden_states) * (1.0 + scale_msa[adaln_indices]) + shift_msa[adaln_indices]
        hidden_states = residual + gate_msa[adaln_indices] * self.attn(normalized, rotary_emb, attention_mask)
        normalized = self.norm2(hidden_states) * (1.0 + scale_mlp[adaln_indices]) + shift_mlp[adaln_indices]
        return hidden_states + gate_mlp[adaln_indices] * self.ff(normalized)


class MiniMaxH3AdaLayerNormOut(native.MiniMaxH3AdaLayerNormOut):
    def forward(self, hidden_states, temb, timestep_indices):
        shift, scale = self.linear(torch.nn.functional.silu(temb).to(get_parameter_dtype(self.linear))).chunk(2, -1)
        return self.norm(hidden_states) * (1.0 + scale[timestep_indices]) + shift[timestep_indices]


class MiniMaxH3Transformer3DModel(native.MiniMaxH3Transformer3DModel):
    @apply_lora_scale("attention_kwargs")
    def forward(self, hidden_states, audio_hidden_states, encoder_hidden_states, timestep,
                timestep_indices, token_tags, position_ids, video_indices, audio_indices,
                text_indices, attention_kwargs=None, return_dict=True):
        options = dict(attention_kwargs or {})
        text_mask = options.pop("h3_text_mask", None)
        if options:
            raise ValueError(f"Unsupported H3 attention options: {sorted(options)}")
        if text_mask is None:
            # Retain the public native ABI for direct singleton/reference callers.
            return super().forward(
                hidden_states, audio_hidden_states, encoder_hidden_states, timestep,
                timestep_indices, token_tags, position_ids, video_indices, audio_indices,
                text_indices, return_dict=return_dict,
            )
        batch, length = hidden_states.shape[0], token_tags.numel()
        if (position_ids.shape != (batch, length, 3)
                or timestep_indices.shape != (batch, length)
                or text_mask.shape != encoder_hidden_states.shape[:2]
                or text_mask.dtype != torch.bool):
            raise ValueError("Invalid batched H3 layout, timestep indices, or text mask")
        refiner_mask = _padding_attention_mask(text_mask)
        rotary_emb = tuple(x.reshape(batch, length, -1)
                           for x in self.rope(position_ids.reshape(-1, 3)))
        video = self.proj_in(hidden_states.to(get_parameter_dtype(self.proj_in)))
        audio = self.audio_proj_in(audio_hidden_states.to(get_parameter_dtype(self.audio_proj_in)))
        # Invalid padded values cannot enter a projection or receive gradients.
        text = encoder_hidden_states.masked_fill(~text_mask[..., None], 0)
        text = self.context_embedder(text.to(get_parameter_dtype(self.context_embedder)))
        text = self.token_refiner(text, refiner_mask)
        packed = text.new_zeros((batch, length, text.shape[-1]))
        packed = packed.index_copy(1, text_indices, text)
        packed = packed.index_copy(1, video_indices, video.to(text.dtype))
        packed = packed.index_copy(1, audio_indices, audio.to(text.dtype))
        valid = torch.ones((batch, length), dtype=torch.bool, device=packed.device)
        valid[:, text_indices] = text_mask
        attention_mask = None if refiner_mask is None else valid[:, None, None, :]
        temb = self.time_embedder(self.time_proj(timestep).to(get_parameter_dtype(self.time_embedder)))
        adaln_indices = timestep_indices * native.MINIMAX_H3_MODALITY_NUM + token_tags[None]
        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                packed = self._gradient_checkpointing_func(
                    block, packed, temb, adaln_indices, rotary_emb, attention_mask)
            else:
                packed = block(packed, temb, adaln_indices, rotary_emb, attention_mask)
        packed = self.norm_out(packed, temb, timestep_indices).to(get_parameter_dtype(self.proj_out))
        video = self.proj_out(packed).index_select(1, video_indices)
        audio = self.audio_proj_out(packed).index_select(1, audio_indices)
        if not return_dict:
            return video, audio
        return native.MiniMaxH3TransformerOutput(sample=video, audio_sample=audio)


def enable_batched_h3(model):
    """Upgrade only this native instance, before PEFT/FSDP wraps its modules.

    No parameters/modules are added or replaced. Original objects, trainability,
    dtype-query boundaries, state keys, and pretrained source identity survive.
    """
    if isinstance(model, MiniMaxH3Transformer3DModel):
        return model
    if type(model) is not native.MiniMaxH3Transformer3DModel:
        raise TypeError("Batched H3 extension requires the pinned native transformer instance")
    # Validate before mutating any instance; incompatible processors fail atomically.
    for module in model.modules():
        if (isinstance(module, native.MiniMaxH3Attention)
                and type(module.processor) is not native.MiniMaxH3AttnProcessor):
            raise TypeError("Install batched H3 before changing native attention processors")
    subclasses = {
        native.MiniMaxH3Transformer3DModel: MiniMaxH3Transformer3DModel,
        native.MiniMaxH3TokenRefiner: MiniMaxH3TokenRefiner,
        native.MiniMaxH3TokenRefinerBlock: MiniMaxH3TokenRefinerBlock,
        native.MiniMaxH3TransformerBlock: MiniMaxH3TransformerBlock,
        native.MiniMaxH3AdaLayerNormOut: MiniMaxH3AdaLayerNormOut,
    }
    for module in model.modules():
        subclass = subclasses.get(type(module))
        if subclass is not None:
            module.__class__ = subclass
        if isinstance(module, native.MiniMaxH3Attention):
            previous = module.processor
            processor = MiniMaxH3BatchedAttnProcessor()
            processor._attention_backend = previous._attention_backend
            processor._parallel_config = previous._parallel_config
            module.set_processor(processor)
    return model
