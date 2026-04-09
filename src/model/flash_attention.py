"""Memory-efficient attention: tiled computation, online softmax, and memory analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class FlashConfig:
    """Configuration for flash attention simulation.

    Attributes:
        block_size: Tile size for Q/K/V blocking.
        use_causal_mask: If True, apply causal masking (tokens cannot attend to future).
        dropout_p: Attention dropout probability.
    """

    block_size: int = 64
    use_causal_mask: bool = True
    dropout_p: float = 0.0


def online_softmax(scores: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Numerically stable online softmax over the last dimension.

    Args:
        scores: Attention scores of arbitrary shape (..., S).

    Returns:
        Tuple of (m, l, p):
            m: Running max, shape (..., 1).
            l: Running sum of exp, shape (..., 1).
            p: Softmax probabilities, same shape as scores.
    """
    # Stabilise by subtracting the row-wise max
    m = scores.max(dim=-1, keepdim=True).values          # (..., 1)
    exp_shifted = torch.exp(scores - m)                   # (..., S)
    l = exp_shifted.sum(dim=-1, keepdim=True)             # (..., 1)
    p = exp_shifted / l                                   # (..., S)
    return m, l, p


def tiled_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    block_size: int,
    causal: bool = True,
) -> Tensor:
    """Compute scaled dot-product attention in tiles over the sequence dimension.

    Processes Q in blocks of ``block_size`` rows.  For each Q-block every K/V
    block is visited, a causal mask is applied within the tile and the output is
    accumulated using standard (non-online) softmax for correctness simplicity.

    Args:
        Q: Query tensor, shape (B, H, T, head_dim).
        K: Key tensor,   shape (B, H, T, head_dim).
        V: Value tensor, shape (B, H, T, head_dim).
        block_size: Number of query rows processed per tile.
        causal: If True, future key positions receive score −1e9.

    Returns:
        Output tensor of shape (B, H, T, head_dim).
    """
    B, H, T, head_dim = Q.shape
    scale = 1.0 / math.sqrt(head_dim)
    device = Q.device

    output = torch.zeros_like(Q)

    for q_start in range(0, T, block_size):
        q_end = min(q_start + block_size, T)
        q_block = Q[:, :, q_start:q_end, :]          # (B, H, Tq, D)
        Tq = q_end - q_start

        # Accumulate scores over all K positions for this Q-block
        # scores: (B, H, Tq, T)
        scores = scale * torch.matmul(q_block, K.transpose(-2, -1))

        if causal:
            # q row indices (absolute): shape (Tq, 1)
            q_idx = torch.arange(q_start, q_end, device=device).view(-1, 1)
            # k col indices (absolute): shape (1, T)
            k_idx = torch.arange(T, device=device).view(1, -1)
            # mask where key position is strictly after query position
            future_mask = k_idx > q_idx                  # (Tq, T)
            scores = scores.masked_fill(
                future_mask.unsqueeze(0).unsqueeze(0), -1e9
            )

        # Standard softmax + weighted sum — correctness over complexity
        attn_weights = F.softmax(scores, dim=-1)          # (B, H, Tq, T)
        output[:, :, q_start:q_end, :] = torch.matmul(attn_weights, V)

    return output


class FlashAttentionSimulator(nn.Module):
    """Multi-head attention computed tile-by-tile to limit peak memory usage.

    This is a pedagogical simulator that mirrors the memory-access pattern of
    FlashAttention using pure PyTorch operations.

    Args:
        d_model: Total model dimension.
        n_heads: Number of attention heads. Must divide d_model evenly.
        config: FlashConfig controlling tile size and masking.
    """

    def __init__(self, d_model: int, n_heads: int, config: FlashConfig) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.config = config

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor, shape (B, T, d_model).
            mask: Optional attention mask (currently unused; causal masking is
                  controlled by ``config.use_causal_mask``).

        Returns:
            Output tensor, shape (B, T, d_model).
        """
        B, T, _ = x.shape

        # Project and reshape to (B, H, T, head_dim)
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        out = tiled_attention(
            q, k, v,
            block_size=self.config.block_size,
            causal=self.config.use_causal_mask,
        )

        # (B, H, T, head_dim) -> (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.o_proj(out)


def compute_memory_footprint(
    B: int,
    H: int,
    T: int,
    head_dim: int,
    block_size: int,
) -> dict:
    """Compute theoretical memory usage in bytes (float32 = 4 bytes).

    Args:
        B: Batch size.
        H: Number of attention heads.
        T: Sequence length.
        head_dim: Dimension per head.
        block_size: Q-block tile size used in tiled attention.

    Returns:
        Dictionary with keys:
            ``standard_attention_bytes``: Full (B, H, T, T) attention matrix.
            ``tiled_attention_bytes``:    One Q-block worth, shape (B, H, block_size, T).
            ``memory_reduction_factor``:  standard / tiled (float).
    """
    bytes_per_elem = 4  # float32

    standard_attention_bytes = B * H * T * T * bytes_per_elem
    tiled_attention_bytes = B * H * block_size * T * bytes_per_elem
    memory_reduction_factor = (
        standard_attention_bytes / tiled_attention_bytes
        if tiled_attention_bytes > 0
        else 1.0
    )

    return {
        "standard_attention_bytes": standard_attention_bytes,
        "tiled_attention_bytes": tiled_attention_bytes,
        "memory_reduction_factor": memory_reduction_factor,
    }


def benchmark_attention_equivalence(
    B: int,
    H: int,
    T: int,
    head_dim: int,
    block_size: int,
) -> dict:
    """Check numerical equivalence between standard and tiled attention.

    Generates random Q, K, V tensors and compares:
    - Standard attention: ``softmax(QKᵀ / sqrt(d)) @ V`` (causal)
    - Tiled attention via :func:`tiled_attention`

    Args:
        B: Batch size.
        H: Number of attention heads.
        T: Sequence length.
        head_dim: Dimension per head.
        block_size: Tile size for tiled attention.

    Returns:
        Dictionary with keys:
            ``max_diff``:   Maximum absolute difference (float).
            ``mean_diff``:  Mean absolute difference (float).
            ``equivalent``: True when ``max_diff < 1e-4``.
    """
    torch.manual_seed(0)
    Q = torch.randn(B, H, T, head_dim)
    K = torch.randn(B, H, T, head_dim)
    V = torch.randn(B, H, T, head_dim)

    # Standard causal attention — materialise the full T×T matrix
    scale = 1.0 / math.sqrt(head_dim)
    scores = scale * torch.matmul(Q, K.transpose(-2, -1))  # (B, H, T, T)
    # Build a causal mask
    causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), -1e9)
    standard_out = torch.matmul(F.softmax(scores, dim=-1), V)

    # Tiled attention
    tiled_out = tiled_attention(Q, K, V, block_size=block_size, causal=True)

    diff = (standard_out - tiled_out).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    return {
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "equivalent": max_diff < 1e-4,
    }
