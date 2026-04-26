"""nGPT: Normalized Transformer with Representation Learning on the Hypersphere.

All weight matrices and hidden states live on the unit hypersphere. Key ideas:
  - Hidden states are L2-normalized after every residual add
  - Weight matrices (QKV, FFN) are L2-normalized along their output dimension
  - Eigenlearning rates αₐ and αₘ (per-dimension learnable scalars) control how
    fast weights move along the sphere at each sub-block
  - Attention uses normalized Q/K with √d_k scaling (not 1/√d_k)
  - Output logit head is a plain linear — NOT normalized

Reference:
    Loshchilov et al., "nGPT: Normalized Transformer with Representation
    Learning on the Hypersphere", NVIDIA 2024.
    https://arxiv.org/abs/2410.01131
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class NGPTConfig:
    """Configuration for the nGPT model.

    Args:
        d_model:     Hidden / embedding dimension.
        n_heads:     Number of attention heads.
        head_dim:    Dimension per head.  Total QKV dim = n_heads * head_dim.
        d_ff:        Feed-forward hidden dimension.
        n_layers:    Number of nGPT blocks.
        vocab_size:  Vocabulary size.
        max_seq_len: Maximum sequence length.
        alpha_attn:  Initial value for the attention eigenlearning-rate vector.
        alpha_mlp:   Initial value for the MLP eigenlearning-rate vector.
    """

    d_model: int = 512
    n_heads: int = 8
    head_dim: int = 64
    d_ff: int = 2048
    n_layers: int = 6
    vocab_size: int = 32000
    max_seq_len: int = 2048
    alpha_attn: float = 0.05
    alpha_mlp: float = 0.05


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    """Return x normalized to unit L2 norm along *dim*."""
    return x / (x.norm(dim=dim, keepdim=True).clamp(min=eps))


def normalize_weights(module: nn.Module, eps: float = 1e-12) -> None:
    """Normalize all Linear weight matrices in *module* to unit L2 norm per
    output neuron (i.e. along dim=1, the input-feature dimension).

    Call this after every optimizer step to keep weight rows on the sphere.
    """
    for m in module.modules():
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                norms = m.weight.norm(dim=1, keepdim=True).clamp(min=eps)
                m.weight.div_(norms)


# ---------------------------------------------------------------------------
# Normalized Embedding
# ---------------------------------------------------------------------------


class NormalizedEmbedding(nn.Module):
    """Token embedding table whose output vectors are L2-normalized.

    Shape: (vocab_size, d_model) — same as nn.Embedding.
    """

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, d_model))
        nn.init.normal_(self.weight, std=1.0 / math.sqrt(d_model))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return L2-normalized embedding vectors.

        Args:
            input_ids: (B, T) long tensor.

        Returns:
            (B, T, d_model) unit-norm vectors.
        """
        emb = F.embedding(input_ids, self.weight)
        return _l2_normalize(emb, dim=-1)


# ---------------------------------------------------------------------------
# nGPT Attention
# ---------------------------------------------------------------------------


class NGPTAttention(nn.Module):
    """Multi-head self-attention on the unit hypersphere.

    Differences from standard attention:
      - Q and K are L2-normalized per head before computing scores
      - Dot-product scores are scaled by √d_k  (not 1/√d_k)
      - A learnable per-head scale s_qk can further adjust the score magnitude
      - Weight matrices W_q, W_k, W_v, W_o are kept on the hypersphere via
        post-step weight normalization (done externally via normalize_weights)
    """

    def __init__(self, config: NGPTConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model

        inner_dim = config.n_heads * config.head_dim

        self.q_proj = nn.Linear(config.d_model, inner_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, inner_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, inner_dim, bias=False)
        self.out_proj = nn.Linear(inner_dim, config.d_model, bias=False)

        # Learnable per-head QK scale (initialized to sqrt(head_dim))
        # Shape: (1, n_heads, 1, 1) for broadcasting over (B, H, T, T)
        init_scale = math.sqrt(config.head_dim)
        self.sqk_scale = nn.Parameter(torch.full((config.n_heads,), init_scale))

        self._init_weights()

    def _init_weights(self) -> None:
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.normal_(proj.weight, std=0.02)
            # Normalize rows immediately so we start on the hypersphere
            with torch.no_grad():
                norms = proj.weight.norm(dim=1, keepdim=True).clamp(min=1e-12)
                proj.weight.div_(norms)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute normalized multi-head self-attention.

        Args:
            x:    (B, T, d_model) — inputs are assumed to be on the hypersphere.
            mask: Optional additive attention mask (B, 1, T, T) or (1, 1, T, T).
                  Use -inf for positions to be masked out.

        Returns:
            (B, T, d_model)
        """
        B, T, _ = x.shape
        H, D = self.n_heads, self.head_dim

        # Project and reshape → (B, H, T, D)
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        # Normalize Q and K onto the unit sphere per position per head
        q = _l2_normalize(q, dim=-1)
        k = _l2_normalize(k, dim=-1)

        # Attention scores: scale by √d_k * s_qk
        # sqk_scale: (H,) → (1, H, 1, 1)
        scale = self.sqk_scale.view(1, H, 1, 1) * math.sqrt(D)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, T, T)

        if mask is not None:
            scores = scores + mask

        attn_weights = F.softmax(scores, dim=-1)  # (B, H, T, T)

        # Aggregate values
        out = torch.matmul(attn_weights, v)  # (B, H, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, H * D)

        return self.out_proj(out)  # (B, T, d_model)


# ---------------------------------------------------------------------------
# nGPT MLP
# ---------------------------------------------------------------------------


class _NGPTMLP(nn.Module):
    """SwiGLU-style MLP used inside nGPTBlock.

    Weight matrices are kept on the sphere externally via normalize_weights.
    """

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

        for proj in (self.gate_proj, self.up_proj, self.down_proj):
            nn.init.normal_(proj.weight, std=0.02)
            with torch.no_grad():
                norms = proj.weight.norm(dim=1, keepdim=True).clamp(min=1e-12)
                proj.weight.div_(norms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# nGPT Block
# ---------------------------------------------------------------------------


class NGPTBlock(nn.Module):
    """Single nGPT block.

    Update rule (eigenlearning-rate form):
        h_A  = Norm(Attention(h))
        h    = Norm(h + α_A ⊙ (h_A − h))

        h_M  = Norm(MLP(h))
        h    = Norm(h + α_M ⊙ (h_M − h))

    The learnable α vectors are initialized to config.alpha_attn / config.alpha_mlp
    and scaled up to [0, 1] via sigmoid during the forward pass so they stay
    in a valid range.
    """

    def __init__(self, config: NGPTConfig) -> None:
        super().__init__()
        self.attn = NGPTAttention(config)
        self.mlp = _NGPTMLP(config.d_model, config.d_ff)

        # Eigenlearning rates: one scalar per hidden dimension
        # Stored as raw logits, converted with sigmoid to (0, 1)
        alpha_a_init = math.log(config.alpha_attn / (1.0 - config.alpha_attn + 1e-7))
        alpha_m_init = math.log(config.alpha_mlp / (1.0 - config.alpha_mlp + 1e-7))

        self.alpha_attn = nn.Parameter(torch.full((config.d_model,), alpha_a_init))
        self.alpha_mlp = nn.Parameter(torch.full((config.d_model,), alpha_m_init))

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:    (B, T, d_model) — unit-norm hidden states.
            mask: Optional (B, 1, T, T) additive attention mask.

        Returns:
            (B, T, d_model) — unit-norm hidden states.
        """
        # --- Attention sub-block ---
        h_a = _l2_normalize(self.attn(x, mask), dim=-1)
        alpha_a = torch.sigmoid(self.alpha_attn)  # (d_model,)
        x = _l2_normalize(x + alpha_a * (h_a - x), dim=-1)

        # --- MLP sub-block ---
        h_m = _l2_normalize(self.mlp(x), dim=-1)
        alpha_m = torch.sigmoid(self.alpha_mlp)  # (d_model,)
        x = _l2_normalize(x + alpha_m * (h_m - x), dim=-1)

        return x


# ---------------------------------------------------------------------------
# Full nGPT Model
# ---------------------------------------------------------------------------


class NGPTModel(nn.Module):
    """Full nGPT language model.

    Token embeddings and all hidden states reside on the unit hypersphere.
    The final projection to vocabulary logits is a plain (un-normalized)
    linear layer so that logit magnitudes can vary freely.

    Usage::

        config = NGPTConfig(d_model=512, n_heads=8, head_dim=64,
                            d_ff=2048, n_layers=6, vocab_size=32000)
        model = NGPTModel(config)
        logits = model(input_ids)          # (B, T, vocab_size)

        # After each optimizer step, renormalize weights back to the sphere:
        normalize_weights(model)
    """

    def __init__(self, config: NGPTConfig) -> None:
        super().__init__()
        self.config = config

        self.embedding = NormalizedEmbedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([NGPTBlock(config) for _ in range(config.n_layers)])
        # Plain linear head — no normalization on logits
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        nn.init.normal_(self.lm_head.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the nGPT forward pass.

        Args:
            input_ids: (B, T) long tensor of token indices.
            mask:      Optional additive attention mask (B, 1, T, T) or
                       (1, 1, T, T) with 0 at valid positions and -inf at
                       positions to be masked.

        Returns:
            logits: (B, T, vocab_size) — raw un-normalized scores.
        """
        x = self.embedding(input_ids)  # (B, T, d_model), unit norm

        for block in self.blocks:
            x = block(x, mask)  # (B, T, d_model), unit norm

        return self.lm_head(x)  # (B, T, vocab_size), NOT normalized
