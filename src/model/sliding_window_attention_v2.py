"""
Sliding window attention: each token attends only to the W tokens before it
(causal local attention), reducing O(L^2) to O(L*W).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class SWAConfig:
    d_model: int
    n_heads: int
    head_dim: int
    window_size: int = 512
    dropout: float = 0.0


class SlidingWindowAttention(nn.Module):
    def __init__(self, config: SWAConfig) -> None:
        super().__init__()
        self.config = config
        inner_dim = config.n_heads * config.head_dim

        self.q_proj = nn.Linear(config.d_model, inner_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, inner_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, inner_dim, bias=False)
        self.o_proj = nn.Linear(inner_dim, config.d_model, bias=False)

        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else None

    def _build_window_mask(self, T: int, window_size: int, device: torch.device) -> Tensor:
        """Return BoolTensor (T, T) where True means query i can attend to key j.

        Token i can attend to j when j <= i (causal) and i - j <= window_size.
        """
        i_idx = torch.arange(T, device=device).unsqueeze(1)  # (T, 1)
        j_idx = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
        causal = j_idx <= i_idx
        in_window = (i_idx - j_idx) <= window_size
        return causal & in_window  # (T, T)

    def forward(self, x: Tensor, causal: bool = True) -> Tensor:
        """
        Args:
            x:      (B, T, d_model)
            causal: if True apply causal local window mask

        Returns:
            (B, T, d_model)
        """
        B, T, _ = x.shape
        H = self.config.n_heads
        d = self.config.head_dim
        scale = math.sqrt(d)

        q = self.q_proj(x).view(B, T, H, d).transpose(1, 2)  # (B, H, T, d)
        k = self.k_proj(x).view(B, T, H, d).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, d).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, H, T, T)

        if causal:
            window_mask = self._build_window_mask(T, self.config.window_size, x.device)
            # Mask out positions outside the window by setting to -inf
            attn_scores = attn_scores.masked_fill(~window_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        if self.dropout is not None and self.training:
            attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)  # (B, H, T, d)
        out = out.transpose(1, 2).reshape(B, T, H * d)
        return self.o_proj(out)
