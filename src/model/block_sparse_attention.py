"""Block-sparse attention patterns: efficient attention over long sequences using BigBird/block-sparse structure."""  # noqa: E501

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class BlockSparseConfig:
    block_size: int = 64  # size of each block
    n_global_tokens: int = 4  # number of global tokens (attend everywhere)
    n_random_blocks: int = 2  # random block connections per query block
    window_size: int = 1  # local sliding window (in blocks)
    d_model: int = 512
    n_heads: int = 8
    head_dim: int = 64
    dropout: float = 0.0


def create_block_sparse_mask(
    seq_len: int,
    config: BlockSparseConfig,
    device: torch.device = None,
) -> Tensor:
    """Create BigBird-style block sparse attention mask.

    Returns a boolean mask of shape (seq_len, seq_len) where True means
    attend and False means mask out.

    Pattern:
    - Global tokens (0..n_global_tokens-1): attend to all, all attend to them
    - Local window: each token attends to blocks within ±window_size
    - Random blocks: each query block attends to n_random_blocks randomly chosen blocks

    Args:
        seq_len: sequence length (must be divisible by block_size)
        config: BlockSparseConfig with pattern parameters
        device: target device

    Returns:
        Boolean tensor (seq_len, seq_len)
    """
    assert seq_len % config.block_size == 0, (  # noqa: S101
        f"seq_len ({seq_len}) must be divisible by block_size ({config.block_size})"
    )

    n_blocks = seq_len // config.block_size
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

    # Global tokens: rows 0..n_global_tokens-1 attend to all, all attend to them
    n_global = min(config.n_global_tokens, seq_len)
    if n_global > 0:
        mask[:n_global, :] = True  # global token rows attend everywhere
        mask[:, :n_global] = True  # all columns attend to global tokens

    # Local sliding window (in blocks)
    for block_q in range(n_blocks):
        q_start = block_q * config.block_size
        q_end = q_start + config.block_size
        for offset in range(-config.window_size, config.window_size + 1):
            block_k = block_q + offset
            if 0 <= block_k < n_blocks:
                k_start = block_k * config.block_size
                k_end = k_start + config.block_size
                mask[q_start:q_end, k_start:k_end] = True

    # Random block connections
    if config.n_random_blocks > 0 and n_blocks > 0:
        torch.manual_seed(0)  # reproducible random pattern
        for block_q in range(n_blocks):
            q_start = block_q * config.block_size
            q_end = q_start + config.block_size
            # Choose random blocks (avoid choosing the same block repeatedly)
            n_rand = min(config.n_random_blocks, n_blocks)
            perm = torch.randperm(n_blocks, device=device)
            chosen = perm[:n_rand]
            for block_k in chosen.tolist():
                k_start = block_k * config.block_size
                k_end = k_start + config.block_size
                mask[q_start:q_end, k_start:k_end] = True

    return mask


def compute_block_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor,
    dropout_p: float = 0.0,
) -> Tensor:
    """Scaled dot-product attention with block-sparse boolean mask.

    Args:
        query: (B, H, T, D)
        key:   (B, H, T, D)
        value: (B, H, T, D)
        mask:  (T, T) boolean — True positions are attended, False → -inf
        dropout_p: attention dropout probability

    Returns:
        (B, H, T, D)
    """
    scale = math.sqrt(query.size(-1))
    scores = torch.matmul(query, key.transpose(-2, -1)) / scale  # (B, H, T, T)

    # Apply boolean mask: False positions get -inf
    if mask is not None:
        # mask: (T, T) → broadcast over (B, H, T, T)
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    attn_weights = F.softmax(scores, dim=-1)

    # Replace any NaN rows (all -inf) with zeros to avoid NaN propagation
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p)

    return torch.matmul(attn_weights, value)


class BlockSparseAttention(nn.Module):
    """Attention layer using BigBird-style block sparse patterns."""

    def __init__(self, config: BlockSparseConfig):
        super().__init__()
        self.config = config
        inner_dim = config.n_heads * config.head_dim
        self.q_proj = nn.Linear(config.d_model, inner_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, inner_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, inner_dim, bias=False)
        self.out_proj = nn.Linear(inner_dim, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Forward pass.

        Args:
            x:    (B, T, d_model)
            mask: optional (T, T) boolean mask; created if not provided

        Returns:
            (B, T, d_model)
        """
        B, T, _ = x.shape
        H = self.config.n_heads
        D = self.config.head_dim

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        if mask is None:
            mask = create_block_sparse_mask(T, self.config, device=x.device)

        out = compute_block_attention(q, k, v, mask, dropout_p=self.config.dropout)
        out = self.dropout(out)
        out = out.transpose(1, 2).contiguous().view(B, T, H * D)  # (B, T, inner_dim)
        return self.out_proj(out)


class BigBirdBlock(nn.Module):
    """Full transformer block with block-sparse attention (pre-norm residual)."""

    def __init__(self, config: BlockSparseConfig, d_ff: int):
        super().__init__()
        self.attn = BlockSparseAttention(config)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, config.d_model),
        )
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with pre-norm residual connections.

        Args:
            x: (B, T, D)

        Returns:
            (B, T, D)
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


def compute_sparsity(mask: Tensor) -> float:
    """Return fraction of False (masked-out) positions in the mask.

    Args:
        mask: boolean tensor of any shape

    Returns:
        Float in [0, 1] — fraction of zeros (masked positions)
    """
    total = mask.numel()
    if total == 0:
        return 0.0
    n_false = (~mask).sum().item()
    return float(n_false) / float(total)


def estimate_flop_reduction(seq_len: int, config: BlockSparseConfig) -> float:
    """Estimate attention FLOPs reduction vs dense attention.

    Counts attended pairs from the block-sparse mask and compares to dense.

    Args:
        seq_len: sequence length (must be divisible by block_size)
        config: BlockSparseConfig

    Returns:
        Ratio dense_pairs / sparse_pairs (>= 1.0)
    """
    mask = create_block_sparse_mask(seq_len, config)
    sparse_pairs = mask.sum().item()
    dense_pairs = seq_len * seq_len
    if sparse_pairs == 0:
        return float("inf")
    return float(dense_pairs) / float(sparse_pairs)
