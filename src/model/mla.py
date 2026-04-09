"""Multi-head Latent Attention (MLA): compress KV into low-rank latent for efficient KV cache (DeepSeek-V2)."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rms_norm import RMSNorm


@dataclass
class MLAConfig:
    d_model: int = 512
    n_heads: int = 8
    head_dim: int = 64
    kv_lora_rank: int = 64  # latent dimension for compressed KV
    q_lora_rank: int = 0  # if > 0, also compress queries (0 = no Q compression)
    rope_dim: int = 32  # portion of head_dim that gets RoPE (decoupled RoPE)
    dropout: float = 0.0


class DownProjectKV(nn.Module):
    """Compress KV: d_model -> kv_lora_rank."""

    def __init__(self, d_model: int, kv_lora_rank: int) -> None:
        super().__init__()
        self.down_proj = nn.Linear(d_model, kv_lora_rank, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_model) -> (B, T, kv_lora_rank)."""
        return self.down_proj(x)


class UpProjectKV(nn.Module):
    """Expand latent back to K and V heads."""

    def __init__(self, kv_lora_rank: int, n_heads: int, head_dim: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.k_up = nn.Linear(kv_lora_rank, n_heads * head_dim, bias=False)
        self.v_up = nn.Linear(kv_lora_rank, n_heads * head_dim, bias=False)

    def forward(self, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """c: (B, T, kv_lora_rank) -> (K, V) each (B, n_heads, T, head_dim)."""
        B, T, _ = c.shape
        K = self.k_up(c).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_up(c).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        return K, V


class MultiHeadLatentAttention(nn.Module):
    """Full MLA layer with compressed KV cache."""

    def __init__(self, config: MLAConfig) -> None:
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim

        # Q projection
        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)

        # KV down-projection and up-projection
        self.kv_down = DownProjectKV(config.d_model, config.kv_lora_rank)
        self.kv_up = UpProjectKV(config.kv_lora_rank, config.n_heads, config.head_dim)

        # Output projection
        self.o_proj = nn.Linear(config.n_heads * config.head_dim, config.d_model, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.scale = config.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        past_kv: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, d_model)
            mask: optional attention mask
            past_kv: optional past latent cache (B, T_past, kv_lora_rank)

        Returns:
            (output, current_latent_c):
              - output: (B, T, d_model)
              - current_latent_c: (B, T_total, kv_lora_rank) — cache the latent, not K/V
        """
        B, T, _ = x.shape

        # Project Q
        Q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        # (B, n_heads, T, head_dim)

        # Down-project x -> latent c
        c_new = self.kv_down(x)  # (B, T, kv_lora_rank)

        # Concat with past latent if provided
        if past_kv is not None:
            c = torch.cat([past_kv, c_new], dim=1)
        else:
            c = c_new

        # Up-project c -> K, V
        K, V = self.kv_up(c)  # each (B, n_heads, T_total, head_dim)

        # Scaled dot-product attention
        is_causal = mask is None and past_kv is None

        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=mask,
            dropout_p=self.config.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        # (B, n_heads, T, head_dim) -> (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out), c


class MLABlock(nn.Module):
    """Transformer block using MLA + SwiGLU FFN."""

    def __init__(self, config: MLAConfig, d_ff: int) -> None:
        super().__init__()
        # Pre-norm -> MLA
        self.norm1 = RMSNorm(config.d_model)
        self.attn = MultiHeadLatentAttention(config)

        # Pre-norm -> SwiGLU FFN
        self.norm2 = RMSNorm(config.d_model)
        self.gate_proj = nn.Linear(config.d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, config.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        past_kv: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (output, latent_cache)
        """
        # Pre-norm -> MLA -> residual
        h, latent_cache = self.attn(self.norm1(x), mask=mask, past_kv=past_kv)
        x = x + h

        # Pre-norm -> SwiGLU FFN -> residual
        h = self.norm2(x)
        h = self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))
        x = x + h

        return x, latent_cache


def compute_kv_cache_savings(config: MLAConfig) -> dict[str, int | float]:
    """Calculate memory per token: standard vs MLA.

    Returns:
        {"standard_per_token": int, "mla_per_token": int, "compression_ratio": float}
    """
    standard_per_token = 2 * config.n_heads * config.head_dim
    mla_per_token = config.kv_lora_rank
    compression_ratio = standard_per_token / mla_per_token
    return {
        "standard_per_token": standard_per_token,
        "mla_per_token": mla_per_token,
        "compression_ratio": compression_ratio,
    }
