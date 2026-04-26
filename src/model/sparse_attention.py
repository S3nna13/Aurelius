"""Sparse attention: local window + strided global patterns."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class SparseAttentionConfig:
    d_model: int = 256
    n_heads: int = 4
    window_size: int = 32  # local attention window (each side)
    stride: int = 8  # global strided attention step
    dropout: float = 0.0


def build_local_mask(
    seq_len: int,
    window_size: int,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """Build (T, T) boolean mask where mask[i,j]=True if |i-j| <= window_size AND j<=i (causal).

    Used as attention mask: True = attend, False = mask out.
    """
    rows = torch.arange(seq_len, device=device).unsqueeze(1)  # (T, 1)
    cols = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, T)
    causal = cols <= rows
    local = (rows - cols).abs() <= window_size
    return causal & local


def build_strided_mask(
    seq_len: int,
    stride: int,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """Build (T, T) boolean mask where mask[i,j]=True if j % stride == 0 AND j<=i (causal global)."""  # noqa: E501
    rows = torch.arange(seq_len, device=device).unsqueeze(1)  # (T, 1)
    cols = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, T)
    causal = cols <= rows
    strided = (cols % stride) == 0
    return causal & strided


def build_sparse_mask(
    seq_len: int,
    window_size: int,
    stride: int,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """Combine local + strided masks: True where EITHER mask is True."""
    local = build_local_mask(seq_len, window_size, device=device)
    strided = build_strided_mask(seq_len, stride, device=device)
    return local | strided


class SparseAttention(nn.Module):
    """Multi-head attention with sparse (local+strided) pattern.

    Args:
        d_model: model dimension
        n_heads: number of heads
        window_size: local window radius (attend to window_size tokens each side)
        stride: strided global attention step
    """

    def __init__(self, config: SparseAttentionConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.window_size = config.window_size
        self.stride = config.stride
        self.dropout = config.dropout

        assert config.d_model % config.n_heads == 0, (  # noqa: S101
            f"d_model ({config.d_model}) must be divisible by n_heads ({config.n_heads})"
        )
        self.head_dim = config.d_model // config.n_heads

        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args: x (B, T, d_model)
        Returns: (B, T, d_model)

        Applies sparse attention mask before softmax.
        Positions where mask=False get -inf before softmax.
        """
        B, T, C = x.shape

        qkv = self.qkv_proj(x)  # (B, T, 3*d_model)
        q, k, v = qkv.split(self.d_model, dim=-1)

        # Reshape to (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Build sparse mask (T, T) — True = attend
        sparse_mask = build_sparse_mask(T, self.window_size, self.stride, device=x.device)

        # Compute attention scores
        scale = self.head_dim**-0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, n_heads, T, T)

        # Apply sparse mask: set False positions to -inf
        # sparse_mask: (T, T) -> broadcast over (B, n_heads, T, T)
        attn = attn.masked_fill(~sparse_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        # Replace NaN from rows that are all -inf (shouldn't happen for causal since diagonal is True)  # noqa: E501
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)  # (B, n_heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class SparseTransformerLayer(nn.Module):
    """Transformer layer with sparse attention + FFN."""

    def __init__(self, config: SparseAttentionConfig) -> None:
        super().__init__()
        self.attn = SparseAttention(config)
        self.attn_norm = nn.LayerNorm(config.d_model)
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class SparseTransformer(nn.Module):
    """Sparse transformer: embed -> N sparse layers -> lm_head."""

    def __init__(
        self,
        config: SparseAttentionConfig,
        n_layers: int,
        vocab_size: int,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, config.d_model)
        self.layers = nn.ModuleList([SparseTransformerLayer(config) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, vocab_size, bias=False)

    def forward(self, input_ids: Tensor) -> Tensor:
        """Returns logits (B, T, vocab_size)."""
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)


def sparsity_ratio(seq_len: int, window_size: int, stride: int) -> float:
    """Compute fraction of attention entries that are attended (not masked).

    Lower = sparser.
    """
    mask = build_sparse_mask(seq_len, window_size, stride)
    total = seq_len * seq_len
    attended = mask.sum().item()
    return attended / total
