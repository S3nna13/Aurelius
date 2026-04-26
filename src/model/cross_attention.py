"""Cross-attention module for multimodal fusion (TriBEv2-inspired).

Enables Aurelius decoder layers to attend to external modality embeddings
(image tokens, audio frames, structured data) via a cross-attention sublayer
inserted after self-attention.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class CrossAttentionConfig:
    d_model: int = 2048
    n_heads: int = 16
    head_dim: int = 128
    context_dim: int = 768  # external feature dimension (e.g. CLIP = 768, Wav2Vec = 1024)
    dropout: float = 0.0
    use_layer_norm: bool = True  # pre-norm before cross-attn


class CrossAttentionLayer(nn.Module):
    """Cross-attention: query from text, key/value from external context.

    Implements the pattern from encoder-decoder attention and TriBEv2's
    multimodal fusion: text tokens attend to external modality embeddings.

    Architecture:
        1. Pre-norm x (if use_layer_norm)
        2. q = q_proj(x),  k = k_proj(context),  v = v_proj(context)
        3. scaled_dot_product_attention(q, k, v, attn_mask from context_mask)
        4. o_proj → residual: x + output
    """

    def __init__(self, cfg: CrossAttentionConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.dropout = cfg.dropout

        inner_dim = cfg.n_heads * cfg.head_dim

        self.norm = nn.LayerNorm(cfg.d_model) if cfg.use_layer_norm else None

        # Query from text hidden states
        self.q_proj = nn.Linear(cfg.d_model, inner_dim, bias=False)
        # Key/Value from external context
        self.k_proj = nn.Linear(cfg.context_dim, inner_dim, bias=False)
        self.v_proj = nn.Linear(cfg.context_dim, inner_dim, bias=False)
        # Output projection back to d_model
        self.o_proj = nn.Linear(inner_dim, cfg.d_model, bias=False)

    def forward(
        self,
        x: Tensor,  # (B, S, d_model) — text hidden states (queries)
        context: Tensor,  # (B, C, context_dim) — external embeddings (keys/values)
        context_mask: Tensor | None = None,  # (B, C) bool mask, True = valid
    ) -> Tensor:  # (B, S, d_model) — updated text hidden states
        B, S, _ = x.shape
        C = context.shape[1]

        residual = x

        # 1. Pre-norm
        if self.norm is not None:
            x = self.norm(x)

        # 2. Project queries, keys, values
        q = self.q_proj(x)  # (B, S, n_heads * head_dim)
        k = self.k_proj(context)  # (B, C, n_heads * head_dim)
        v = self.v_proj(context)  # (B, C, n_heads * head_dim)

        # Reshape to (B, n_heads, seq, head_dim)
        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, C, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, C, self.n_heads, self.head_dim).transpose(1, 2)

        # 3. Build attention mask from context_mask
        # context_mask: (B, C) bool, True = valid token
        # SDPA expects attn_mask: additive float mask (B, 1, 1, C) or (B, n_heads, S, C)
        attn_mask = None
        if context_mask is not None:
            # True = valid → 0.0; False = invalid → -inf
            float_mask = torch.zeros(B, 1, 1, C, dtype=q.dtype, device=q.device)
            float_mask = float_mask.masked_fill(~context_mask.view(B, 1, 1, C), float("-inf"))
            attn_mask = float_mask

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )

        # Reshape back: (B, n_heads, S, head_dim) -> (B, S, n_heads * head_dim)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)

        # 4. Output projection + residual
        return residual + self.o_proj(out)


class MultiModalTransformerBlock(nn.Module):
    """Standard TransformerBlock + optional CrossAttentionLayer.

    If context is provided, runs cross-attention after self-attention.
    If context is None, behaves identically to the base TransformerBlock.
    """

    def __init__(
        self,
        self_attn: nn.Module,
        ffn: nn.Module,
        norm1: nn.Module,
        norm2: nn.Module,
        cross_attn: CrossAttentionLayer | None = None,
        norm_cross: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.self_attn = self_attn
        self.ffn = ffn
        self.norm1 = norm1
        self.norm2 = norm2
        self.cross_attn = cross_attn
        self.norm_cross = norm_cross

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        mask: Tensor | None = None,
        past_kv=None,
        context: Tensor | None = None,
        context_mask: Tensor | None = None,
    ) -> tuple[Tensor, tuple]:
        """
        Returns:
            (hidden_states, kv_cache) — same signature as TransformerBlock.
        """
        # Self-attention (pre-norm pattern, same as TransformerBlock)
        attn_out, kv = self.self_attn(self.norm1(x), freqs_cis, mask, past_kv)
        x = x + attn_out

        # Optional cross-attention
        if self.cross_attn is not None and context is not None:
            x = self.cross_attn(x, context, context_mask)

        # FFN
        x = x + self.ffn(self.norm2(x))

        return x, kv


def scaled_dot_product_cross_attention(
    q: torch.Tensor,  # (B, H, T_q, D)
    k: torch.Tensor,  # (B, H, T_kv, D)
    v: torch.Tensor,  # (B, H, T_kv, D)
    mask: torch.Tensor | None = None,  # (B, 1, 1, T_kv) or (B, H, T_q, T_kv)
) -> torch.Tensor:
    """Scaled dot-product cross-attention (no causal mask).

    mask: True positions are VALID (not masked). False → -inf.
    Returns: (B, H, T_q, D)
    """
    if hasattr(F, "scaled_dot_product_attention"):
        # Convert boolean valid-mask to additive float mask expected by SDPA
        attn_mask = None
        if mask is not None:
            attn_mask = torch.zeros_like(mask, dtype=q.dtype)
            attn_mask = attn_mask.masked_fill(~mask, float("-inf"))
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
    else:
        # Manual fallback
        scale = q.size(-1) ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, T_q, T_kv)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, v)


class CrossAttention(nn.Module):
    """Multi-head cross-attention.

    Q from x (decoder/query stream), K/V from context (encoder/retrieved docs).

    Args:
        config: AureliusConfig (for d_model, n_heads, head_dim)
        d_context: int | None — if None, uses config.d_model (self-attention mode)
    """

    def __init__(self, config, d_context: int | None = None) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.d_context = d_context if d_context is not None else config.d_model
        self.dropout = getattr(config, "dropout", 0.0)

        inner_dim = self.n_heads * self.head_dim

        # W_q: d_model → n_heads * head_dim
        self.w_q = nn.Linear(self.d_model, inner_dim, bias=False)
        # W_k: d_context → n_heads * head_dim
        self.w_k = nn.Linear(self.d_context, inner_dim, bias=False)
        # W_v: d_context → n_heads * head_dim
        self.w_v = nn.Linear(self.d_context, inner_dim, bias=False)
        # W_o: n_heads * head_dim → d_model
        self.w_o = nn.Linear(inner_dim, self.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,  # (B, T_q, d_model) — queries
        context: torch.Tensor,  # (B, T_kv, d_context) — keys and values
        context_mask: torch.Tensor | None = None,  # (B, T_kv) bool mask, True=valid
        **kwargs,
    ) -> torch.Tensor:
        """Returns: (B, T_q, d_model)

        No causal mask — can attend to full context.
        context_mask: False positions get -inf before softmax.
        """
        B, T_q, _ = x.shape
        T_kv = context.shape[1]

        q = self.w_q(x).view(B, T_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(context).view(B, T_kv, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(context).view(B, T_kv, self.n_heads, self.head_dim).transpose(1, 2)

        # Build mask for scaled_dot_product_cross_attention: (B, 1, 1, T_kv) bool
        attn_mask = None
        if context_mask is not None:
            attn_mask = context_mask.view(B, 1, 1, T_kv)

        out = scaled_dot_product_cross_attention(q, k, v, mask=attn_mask)

        # (B, n_heads, T_q, head_dim) → (B, T_q, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T_q, -1)
        return self.w_o(out)


class CrossAttentionBlock(nn.Module):
    """Transformer block with cross-attention.

    Architecture:
        x = x + SelfAttention(RMSNorm(x))      # standard causal self-attention
        x = x + CrossAttention(RMSNorm(x), context)  # cross-attend to context
        x = x + FFN(RMSNorm(x))

    Args:
        config: AureliusConfig
        d_context: int
    """

    def __init__(self, config, d_context: int) -> None:
        super().__init__()
        from .attention import GroupedQueryAttention
        from .ffn import SwiGLUFFN
        from .rms_norm import RMSNorm

        self.self_attn = GroupedQueryAttention(config)
        self.cross_attn = CrossAttention(config, d_context=d_context)
        self.ffn = SwiGLUFFN(config)
        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)
        self.norm3 = RMSNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        from .attention import precompute_rope_frequencies

        B, T, _ = x.shape
        freqs = precompute_rope_frequencies(self.self_attn.head_dim, T, device=x.device)

        # Self-attention (causal)
        attn_out, _ = self.self_attn(self.norm1(x), freqs)
        x = x + attn_out

        # Cross-attention
        x = x + self.cross_attn(self.norm2(x), context, context_mask=context_mask)

        # FFN
        x = x + self.ffn(self.norm3(x))
        return x


class RAGAttentionLayer(nn.Module):
    """RAG-style cross-attention: attend to a set of retrieved document embeddings.

    Retrieved documents are projected to the model's KV space, then the model
    can cross-attend to them at every layer (FiD-style) or at specific layers.

    Args:
        config: AureliusConfig
        n_docs: int (number of retrieved documents, default 5)
        doc_embed_dim: int (dimension of document embeddings, default d_model)
    """

    def __init__(self, config, n_docs: int = 5, doc_embed_dim: int | None = None) -> None:
        super().__init__()
        doc_embed_dim = doc_embed_dim if doc_embed_dim is not None else config.d_model
        self.doc_proj = nn.Linear(doc_embed_dim, config.d_model, bias=False)
        self.cross_attn = CrossAttention(config, d_context=config.d_model)
        self.norm = nn.LayerNorm(config.d_model)  # pre-norm on query

    def forward(
        self,
        x: torch.Tensor,  # (B, T, d_model)
        doc_embeddings: torch.Tensor,  # (B, n_docs, doc_embed_dim)
        doc_mask: torch.Tensor | None = None,  # (B, n_docs) bool
    ) -> torch.Tensor:
        """Attend to documents, add to x via residual."""
        docs = self.doc_proj(doc_embeddings)  # (B, n_docs, d_model)
        attn_out = self.cross_attn(self.norm(x), docs, context_mask=doc_mask)
        return x + attn_out


def add_cross_attention_to_model(
    model,  # AureliusTransformer
    context_dim: int,
    layer_indices: list[int] | None = None,  # None = all layers
) -> nn.ModuleList:
    """Create CrossAttentionLayer instances for specified transformer layers.

    Args:
        model: AureliusTransformer instance.
        context_dim: Dimensionality of the external context embeddings.
        layer_indices: Indices of layers to add cross-attention to.
                       None means all layers.

    Returns:
        nn.ModuleList of CrossAttentionLayer instances, one per specified layer.
        Caller is responsible for storing them and passing context in forward.
    """
    cfg = model.config
    n_layers = cfg.n_layers

    if layer_indices is None:
        layer_indices = list(range(n_layers))

    cross_attn_cfg = CrossAttentionConfig(
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        head_dim=cfg.head_dim,
        context_dim=context_dim,
        dropout=cfg.dropout,
        use_layer_norm=True,
    )

    cross_attn_layers = nn.ModuleList([CrossAttentionLayer(cross_attn_cfg) for _ in layer_indices])

    return cross_attn_layers
