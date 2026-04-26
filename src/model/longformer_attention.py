"""Longformer-style attention: sliding window local + global tokens with optional dilation."""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class LongformerConfig:
    window_size: int = 64  # one-sided window (attend to w tokens left and right)
    n_global_tokens: int = 1  # first n_global_tokens tokens attend globally
    dilation: int = 1  # stride between attended tokens (1 = no dilation)
    d_model: int = 64
    n_heads: int = 2


def create_local_attention_mask(T: int, window_size: int, dilation: int = 1) -> Tensor:
    """Build causal local attention mask with optional dilation.

    Token i can attend to token j if:
      - j <= i (causal)
      - |i - j| <= window_size * dilation
      - (i - j) % dilation == 0

    Returns:
        (T, T) bool tensor, True = can attend.
    """
    rows = torch.arange(T).unsqueeze(1)  # (T, 1)
    cols = torch.arange(T).unsqueeze(0)  # (1, T)
    diff = rows - cols  # positive when j < i

    causal = cols <= rows
    within_window = diff <= window_size * dilation
    dilated = (diff % dilation) == 0

    return causal & within_window & dilated


def create_global_attention_mask(T: int, n_global: int) -> Tensor:
    """Build global attention mask.

    Global tokens (0..n_global-1) attend to ALL tokens and all tokens attend to them.

    Returns:
        (T, T) bool tensor, True = can attend.
    """
    mask = torch.zeros(T, T, dtype=torch.bool)
    if n_global > 0:
        g = min(n_global, T)
        mask[:g, :] = True  # global tokens attend to all
        mask[:, :g] = True  # all tokens attend to global tokens
    return mask


def merge_attention_masks(local_mask: Tensor, global_mask: Tensor) -> Tensor:
    """Merge local and global attention masks via logical OR.

    Returns:
        (T, T) bool tensor.
    """
    return local_mask | global_mask


class LongformerSelfAttention(nn.Module):
    """Longformer self-attention: local window + global tokens with optional dilation."""

    def __init__(self, config: LongformerConfig):
        super().__init__()
        self.config = config
        self.head_dim = config.d_model // config.n_heads
        self.n_heads = config.n_heads
        self.d_model = config.d_model

        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.o_proj = nn.Linear(config.d_model, config.d_model)

    def forward(self, x: Tensor, global_token_ids: list[int] | None = None) -> Tensor:
        """
        Args:
            x: (B, T, d_model)
            global_token_ids: optional list of additional global token indices

        Returns:
            (B, T, d_model)
        """
        B, T, _ = x.shape
        device = x.device

        # Build masks
        local_mask = create_local_attention_mask(
            T, self.config.window_size, self.config.dilation
        ).to(device)

        global_mask = create_global_attention_mask(T, self.config.n_global_tokens).to(device)

        # Add any extra global tokens passed at runtime
        if global_token_ids:
            extra = torch.zeros(T, T, dtype=torch.bool, device=device)
            for g in global_token_ids:
                if 0 <= g < T:
                    extra[g, :] = True
                    extra[:, g] = True
            global_mask = global_mask | extra

        combined_mask = merge_attention_masks(local_mask, global_mask)  # (T, T)

        # Project queries, keys, values
        Q = self.q_proj(x)  # (B, T, d_model)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape to (B, n_heads, T, head_dim)
        def split_heads(t: Tensor) -> Tensor:
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        # Compute scaled dot-product scores: (B, n_heads, T, T)
        scale = self.head_dim**-0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

        # Apply mask: zero out disallowed pairs with large negative value
        # combined_mask shape: (T, T) — broadcast over B and n_heads
        scores = scores.masked_fill(~combined_mask.unsqueeze(0).unsqueeze(0), -1e9)

        attn_weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, V)  # (B, n_heads, T, head_dim)

        # Merge heads: (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.o_proj(out)


def compute_attention_sparsity(T: int, window_size: int, n_global: int) -> dict:
    """Compute theoretical sparsity of the combined Longformer attention mask.

    Returns:
        dict with keys:
          - "total_pairs": T * T
          - "attended_pairs": number of True entries in merged mask
          - "sparsity_ratio": 1 - attended_pairs / total_pairs
          - "vs_full_attention": attended_pairs / total_pairs
    """
    local_mask = create_local_attention_mask(T, window_size)
    global_mask = create_global_attention_mask(T, n_global)
    merged = merge_attention_masks(local_mask, global_mask)

    total_pairs = T * T
    attended_pairs = int(merged.sum().item())
    vs_full = attended_pairs / total_pairs
    sparsity_ratio = 1.0 - vs_full

    return {
        "total_pairs": total_pairs,
        "attended_pairs": attended_pairs,
        "sparsity_ratio": sparsity_ratio,
        "vs_full_attention": vs_full,
    }
