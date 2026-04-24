from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GQAConfig:
    n_heads: int = 32
    n_kv_heads: int = 8
    head_dim: int = 128
    dropout: float = 0.0
    causal: bool = True

    def __post_init__(self) -> None:
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
            )


class GroupedQueryAttention(nn.Module):
    """Multi-head attention with grouped key-value heads (GQA/MQA)."""

    def __init__(self, config: GQAConfig, d_model: int = 4096) -> None:
        super().__init__()
        self.config = config
        self.d_model = d_model

        self.q_proj = nn.Linear(d_model, config.n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * config.head_dim, d_model, bias=False)

    @property
    def n_groups(self) -> int:
        return self.config.n_heads // self.config.n_kv_heads

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch, seq_len, _ = x.shape
        cfg = self.config

        q = self.q_proj(x).view(batch, seq_len, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)

        if kv_cache is not None:
            k_cached, v_cached = kv_cache
            k = torch.cat([k_cached, k], dim=2)
            v = torch.cat([v_cached, v], dim=2)

        k_full_len = k.size(2)

        k_expanded = k.repeat_interleave(self.n_groups, dim=1)
        v_expanded = v.repeat_interleave(self.n_groups, dim=1)

        scale = 1.0 / math.sqrt(cfg.head_dim)
        scores = torch.matmul(q, k_expanded.transpose(-2, -1)) * scale

        if cfg.causal:
            causal_mask = torch.ones(seq_len, k_full_len, device=x.device, dtype=torch.bool).tril(
                diagonal=k_full_len - seq_len
            )
            scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        if attention_mask is not None:
            scores = scores + attention_mask

        weights = torch.softmax(scores, dim=-1)

        if cfg.dropout > 0.0 and self.training:
            weights = F.dropout(weights, p=cfg.dropout)

        out = torch.matmul(weights, v_expanded)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, cfg.n_heads * cfg.head_dim)
        out = self.o_proj(out)

        return out, (k, v)
