"""BD3LM model for Hugging Face.

Pure Transformer for Block Diffusion (LLaMA-style).
No timestep conditioning. Standard pre-norm transformer with
RMSNorm, SwiGLU MLP, and RoPE. Uses block diffusion via FlexAttention masks.
Ported from WindowDiffusion implementation.
"""

import typing
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import modeling_outputs, AutoModel

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    flex_attention_compiled = torch.compile(flex_attention, dynamic=False)
    FLEX_ATTN_AVAILABLE = True
except ImportError:
    flex_attention_compiled = None
    FLEX_ATTN_AVAILABLE = False

from .configuration_bd3lm import BD3LMConfig


# ── Block Diffusion Attention Mask ──────────────────────────────────

def block_diff_mask(b, h, q_idx, kv_idx, block_size=None, n=None):
    """Block diffusion attention mask for [xt | x0] layout.

    Layout: [xt_0..xt_{n-1} | x0_0..x0_{n-1}], total 2n tokens.

    Four quadrants:
      xt->xt: Block diagonal — same block, bidirectional within block.
      xt->x0: Offset block causal (strict >) — noisy block i sees
              clean blocks 0..i-1 only (NOT block i).
      x0->x0: Block causal (>=) — clean block i sees clean blocks 0..i.
      x0->xt: Always blocked.
    """
    x0_flag_q = (q_idx >= n)
    x0_flag_kv = (kv_idx >= n)

    block_q = torch.where(x0_flag_q, (q_idx - n) // block_size, q_idx // block_size)
    block_kv = torch.where(x0_flag_kv, (kv_idx - n) // block_size, kv_idx // block_size)

    # xt->xt: same block, bidirectional
    block_diagonal = (block_q == block_kv) & (~x0_flag_q) & (~x0_flag_kv)
    # xt->x0: noisy block i sees clean blocks 0..i-1
    offset_block_causal = (block_q > block_kv) & x0_flag_kv & (~x0_flag_q)
    # x0->x0: clean block i sees clean blocks 0..i
    block_causal = (block_q >= block_kv) & x0_flag_kv & x0_flag_q

    return block_diagonal | offset_block_causal | block_causal


# ── Model Components ────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


def precompute_rope_freqs(head_dim, max_len, theta=10000.0):
    """Precompute RoPE cos/sin tables. Returns (max_len, head_dim//2) each."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    """Apply RoPE. x: (B, n_heads, S, head_dim), cos/sin: (S, head_dim//2)."""
    dtype = x.dtype
    half = x.shape[-1] // 2
    x1, x2 = x.float()[..., :half], x.float()[..., half:]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return out.to(dtype)


class SwiGLUMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Attention(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.hidden_size = hidden_size
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = dropout

    def forward(self, x, rope_cos, rope_sin, mask=None):
        B, S, _ = x.shape
        qkv = self.qkv_proj(x).reshape(B, S, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each: (B, S, n_heads, head_dim)

        q = q.transpose(1, 2)  # (B, n_heads, S, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        # Dispatch to attention backend
        if mask is not None and FLEX_ATTN_AVAILABLE and not isinstance(mask, torch.Tensor):
            out = flex_attention_compiled(q, k, v, block_mask=mask)
        elif mask is not None and isinstance(mask, torch.Tensor):
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, is_causal=False,
                dropout_p=self.dropout if self.training else 0.0)
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=False,
                dropout_p=self.dropout if self.training else 0.0)

        out = out.transpose(1, 2).reshape(B, S, self.hidden_size)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, n_heads, intermediate_size, dropout=0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = Attention(hidden_size, n_heads, dropout)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = SwiGLUMLP(hidden_size, intermediate_size)
        self.gradient_checkpointing = False

    def forward(self, x, rope_cos, rope_sin, mask=None):
        if self.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward_body, x, rope_cos, rope_sin, mask,
                use_reentrant=False)
        return self._forward_body(x, rope_cos, rope_sin, mask)

    def _forward_body(self, x, rope_cos, rope_sin, mask=None):
        x = x + self.attn(self.norm1(x), rope_cos, rope_sin, mask)
        x = x + self.mlp(self.norm2(x))
        return x


# ── Backbone ────────────────────────────────────────────────────────

class PureTransformerBackbone(nn.Module):
    """Pure LLaMA-style transformer backbone for block diffusion.

    No timestep conditioning (following LLaDA/RADD design).
    Input during training: [xt | x0] layout (2L tokens).
    Returns logits for xt portion only (first L tokens).
    """

    def __init__(self, config: BD3LMConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_dim
        self.n_layers = config.n_blocks
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.intermediate_size = config.intermediate_size
        self.vocab_size = config.vocab_size
        self.seq_len = config.model_length
        self.block_size = config.block_size
        self.cross_attn = config.cross_attn
        self.attn_backend = config.attn_backend

        # Token embedding
        self.tok_embed = nn.Embedding(self.vocab_size, self.hidden_size)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                self.hidden_size, self.n_heads,
                self.intermediate_size, config.dropout)
            for _ in range(self.n_layers)
        ])

        # Final norm + output projection
        self.final_norm = RMSNorm(self.hidden_size)
        if not config.tie_word_embeddings:
            self.output_proj = nn.Linear(
                self.hidden_size, self.vocab_size, bias=False)
        else:
            self.output_proj = None

        # Precompute RoPE (reused per stream in [xt|x0] layout)
        rope_cos, rope_sin = precompute_rope_freqs(
            self.head_dim, self.seq_len, config.rope_theta)
        self.register_buffer('rope_cos', rope_cos, persistent=False)
        self.register_buffer('rope_sin', rope_sin, persistent=False)

        # Create block diffusion attention mask
        if self.cross_attn:
            self._gen_mask(self.seq_len, self.block_size)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _gen_mask(self, seqlen, block_size):
        """Generate block diffusion attention mask."""
        if self.attn_backend == 'flex' and FLEX_ATTN_AVAILABLE:
            self.mask = create_block_mask(
                partial(block_diff_mask, block_size=block_size, n=seqlen),
                B=None, H=None, Q_LEN=seqlen * 2, KV_LEN=seqlen * 2)
        else:
            mask = block_diff_mask(
                b=None, h=None,
                q_idx=torch.arange(seqlen * 2)[:, None],
                kv_idx=torch.arange(seqlen * 2)[None, :],
                block_size=block_size, n=seqlen)
            self.register_buffer('mask', mask, persistent=False)

    def forward(self, input_ids, sample_mode=False, output_hidden_states=False):
        B, S = input_ids.shape
        L = self.seq_len

        all_hidden_states = []
        x = self.tok_embed(input_ids)

        if output_hidden_states:
            all_hidden_states.append(x)

        # Determine mask and RoPE based on layout
        if self.cross_attn and not sample_mode:
            # Training: [xt | x0] layout
            n = S // 2
            if n != self.seq_len:
                # Handle variable-length sequences
                if self.attn_backend == 'flex' and FLEX_ATTN_AVAILABLE:
                    mask = create_block_mask(
                        partial(block_diff_mask,
                                block_size=self.block_size, n=n),
                        B=None, H=None, Q_LEN=n * 2, KV_LEN=n * 2)
                else:
                    mask = block_diff_mask(
                        b=None, h=None,
                        q_idx=torch.arange(n * 2, device=x.device)[:, None],
                        kv_idx=torch.arange(n * 2, device=x.device)[None, :],
                        block_size=self.block_size, n=n)
            else:
                mask = self.mask.to(x.device) if isinstance(
                    self.mask, torch.Tensor) else self.mask

            # RoPE: reuse [0..n-1] positions for both xt and x0 halves
            rope_cos = torch.cat(
                [self.rope_cos[:n], self.rope_cos[:n]], dim=0)
            rope_sin = torch.cat(
                [self.rope_sin[:n], self.rope_sin[:n]], dim=0)
        else:
            # Sampling: xt only
            mask = None
            rope_cos = self.rope_cos[:S]
            rope_sin = self.rope_sin[:S]

        # Forward through transformer layers
        for layer in self.layers:
            x = layer(x, rope_cos, rope_sin, mask)
            if output_hidden_states:
                all_hidden_states.append(x)

        x = self.final_norm(x)

        # Output logits
        if self.output_proj is not None:
            logits = self.output_proj(x)
        else:
            logits = F.linear(x, self.tok_embed.weight)

        # Return only xt portion logits in training mode
        if self.cross_attn and not sample_mode:
            n = S // 2
            logits = logits[:, :n]
            all_hidden_states = [h[:, :n] for h in all_hidden_states]

        return logits, all_hidden_states


# ── HF-Compatible Model ────────────────────────────────────────────

class BD3LM(transformers.PreTrainedModel):
    """HF-compatible BD3LM model.

    Pure transformer block diffusion with no timestep conditioning.
    Ported from WindowDiffusion implementation.
    """
    config_class = BD3LMConfig
    base_model_prefix = "bd3lm"
    supports_gradient_checkpointing = True

    def __init__(self, config: BD3LMConfig):
        super().__init__(config)
        self.config = config
        self.backbone = PureTransformerBackbone(config)
        if config.var_min:
            self.register_buffer(
                'sampling_eps_min',
                torch.tensor(config.sampling_eps_min))
            self.register_buffer(
                'sampling_eps_max',
                torch.tensor(config.sampling_eps_max))

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, TransformerBlock):
            module.gradient_checkpointing = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        timesteps: torch.FloatTensor = None,  # accepted but ignored
        sample_mode: typing.Optional[bool] = None,
        output_hidden_states: typing.Optional[bool] = None,
        return_dict: typing.Optional[bool] = None,
    ) -> typing.Union[
        torch.Tensor, typing.Tuple,
        modeling_outputs.MaskedLMOutput]:
        """HF-compatible forward method.

        Args:
            input_ids: (B, 2L) for [xt|x0] training, (B, S) for sampling.
            timesteps: Ignored. Kept for interface compatibility.
            sample_mode: If True, treat input as xt-only for sampling.
            output_hidden_states: Return intermediate hidden states.
            return_dict: Return MaskedLMOutput instead of tuple.
        """
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states)
        return_dict = (
            return_dict
            if return_dict is not None
            else self.config.use_return_dict)

        logits, all_hidden_states = self.backbone(
            input_ids=input_ids,
            sample_mode=sample_mode or False,
            output_hidden_states=output_hidden_states or False)

        if return_dict:
            return modeling_outputs.MaskedLMOutput(
                logits=logits,
                hidden_states=(
                    all_hidden_states if output_hidden_states else None),
                loss=None)
        elif output_hidden_states:
            return logits, all_hidden_states
        else:
            return logits


AutoModel.register(BD3LMConfig, BD3LM)
