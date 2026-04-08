"""Multi-head Latent Attention (MLA) — DeepSeek-V2 style KV cache compression.

Standard GQA KV cache: 2 * n_kv_heads * head_dim per token
MLA KV cache:          kv_lora_rank per token (much smaller)

Architecture:
  c_KV = W_DKV @ x          (down-projection to latent, kv_lora_rank)
  K    = W_UK @ c_KV         (up-project to key)
  V    = W_UV @ c_KV         (up-project to value)
  Q    = W_Q @ x             (standard query, optionally with Q compression too)

Cache stores c_KV instead of K, V.

Reference: DeepSeek-V2 Technical Report (arXiv:2405.04434)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AureliusConfig


@dataclass
class MLAConfig:
    kv_lora_rank: int = 512    # latent KV dimension (much smaller than 2*n_kv_heads*head_dim)
    q_lora_rank: int = 1536    # query compression rank (0 = no Q compression)
    rope_head_dim: int = 64    # head dim for RoPE subset (decoupled RoPE)


class MultiHeadLatentAttention(nn.Module):
    """MLA with compressed KV cache via low-rank latent projection.

    KV cache stores c_KV of shape (batch, seq, kv_lora_rank) instead of
    full K, V of shape (batch, seq, n_kv_heads, head_dim) each.

    Memory savings = 2 * n_kv_heads * head_dim / kv_lora_rank
    E.g., for n_kv_heads=8, head_dim=128, kv_lora_rank=512:
    savings = 2048 / 512 = 4x KV cache reduction
    """

    def __init__(self, config: AureliusConfig, mla_cfg: MLAConfig | None = None) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.mla_cfg = mla_cfg or MLAConfig()

        kv_rank = self.mla_cfg.kv_lora_rank

        # KV compression: x → c_KV (latent)
        self.W_DKV = nn.Linear(config.d_model, kv_rank, bias=False)                              # down-project
        self.W_UK = nn.Linear(kv_rank, config.n_kv_heads * config.head_dim, bias=False)          # up-project K
        self.W_UV = nn.Linear(kv_rank, config.n_kv_heads * config.head_dim, bias=False)          # up-project V

        # Query: standard projection
        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)

        # Output
        self.o_proj = nn.Linear(config.n_heads * config.head_dim, config.d_model, bias=False)

        # GQA repetition factor
        self.n_kv_heads = config.n_kv_heads
        self.n_rep = config.n_heads // config.n_kv_heads

        self.attn_dropout = config.dropout

    def forward(
        self,
        x: torch.Tensor,                          # (B, S, D)
        freqs_cis: torch.Tensor,                  # (S, head_dim//2)
        mask: torch.Tensor | None = None,
        past_kv_latent: torch.Tensor | None = None,  # (B, past_S, kv_lora_rank)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (output, kv_latent_cache)
            - output: (B, S, D)
            - kv_latent_cache: (B, past_S + S, kv_lora_rank)  — compact KV cache
        """
        B, S, _ = x.shape

        # Compute KV latent for new tokens
        c_KV_new = self.W_DKV(x)  # (B, S, kv_lora_rank)

        # Concatenate with past latent if provided
        if past_kv_latent is not None:
            c_KV = torch.cat([past_kv_latent, c_KV_new], dim=1)
        else:
            c_KV = c_KV_new

        # Project latent to K, V for all positions
        S_total = c_KV.shape[1]
        K = self.W_UK(c_KV).view(B, S_total, self.n_kv_heads, self.head_dim)
        V = self.W_UV(c_KV).view(B, S_total, self.n_kv_heads, self.head_dim)

        # Queries for new tokens only
        Q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim)

        # Apply RoPE to Q and to the new K positions only
        from .attention import apply_rope
        Q = apply_rope(Q, freqs_cis)

        # RoPE on new K tokens; cached K tokens already had RoPE applied when they
        # were originally processed (we re-derive from latent here, so apply to new slice)
        K_new = K[:, -S:, :, :]
        K_new = apply_rope(K_new, freqs_cis)
        if S_total > S:
            K = torch.cat([K[:, :-S, :, :], K_new], dim=1)
        else:
            K = K_new

        # GQA expansion: repeat KV heads to match Q heads
        if self.n_rep > 1:
            K = K.unsqueeze(3).expand(B, S_total, self.n_kv_heads, self.n_rep, self.head_dim)
            K = K.reshape(B, S_total, self.n_heads, self.head_dim)
            V = V.unsqueeze(3).expand(B, S_total, self.n_kv_heads, self.n_rep, self.head_dim)
            V = V.reshape(B, S_total, self.n_heads, self.head_dim)

        # Transpose to (B, H, S, D) for SDPA
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # is_causal only during full prefill (no mask, no past cache)
        is_causal = mask is None and past_kv_latent is None

        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out), c_KV

    def kv_cache_memory_ratio(self) -> float:
        """Compare MLA cache to standard GQA cache size.

        Returns standard / MLA ratio (higher = more compression).
        """
        standard = 2 * self.n_kv_heads * self.head_dim
        mla = self.mla_cfg.kv_lora_rank
        return standard / mla
