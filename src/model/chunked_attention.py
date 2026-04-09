"""Memory-efficient chunked attention for the Aurelius LLM.

Processes attention in query chunks to reduce peak memory from O(T^2) to
O(T * chunk_size), while remaining numerically equivalent to standard attention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class ChunkedAttnConfig:
    chunk_size: int = 64        # query chunk size
    causal: bool = True
    scale: float | None = None  # if None, use 1/sqrt(head_dim)


def chunked_attention(
    q: Tensor,  # (B, H, T, D)
    k: Tensor,  # (B, H, S, D)
    v: Tensor,  # (B, H, S, D)
    chunk_size: int,
    causal: bool = True,
    scale: float | None = None,
) -> Tensor:
    """Compute attention in query chunks to reduce peak memory.

    For each query chunk [i*chunk : i*chunk+chunk]:
      1. Compute scores = q_chunk @ k.T * scale -> (B, H, chunk, S)
      2. If causal: mask out future positions (j > actual_q_pos)
      3. Apply softmax (with max subtraction for numerical stability)
      4. Weighted sum with v
      5. Concatenate chunks

    Returns: (B, H, T, D) — numerically equivalent to standard attention.
    """
    B, H, T, D = q.shape
    S = k.shape[2]

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    outputs = []
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        q_chunk = q[:, :, start:end, :]  # (B, H, chunk, D)

        # Scores: (B, H, chunk, S)
        scores = torch.matmul(q_chunk, k.transpose(-2, -1)) * scale

        if causal:
            # query positions: [start, end)
            # key positions:   [0, S)
            # mask out positions where key_pos > query_pos
            chunk_len = end - start
            q_positions = torch.arange(start, end, device=q.device).unsqueeze(1)   # (chunk, 1)
            k_positions = torch.arange(0, S, device=q.device).unsqueeze(0)          # (1, S)
            causal_mask = k_positions > q_positions  # (chunk, S) — True where masked
            # Broadcast to (1, 1, chunk, S)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(causal_mask, float('-inf'))

        # Numerical stability: subtract max
        scores_max = scores.amax(dim=-1, keepdim=True)
        # Replace -inf max with 0 to avoid NaN when entire row is masked
        scores_max = torch.where(torch.isinf(scores_max), torch.zeros_like(scores_max), scores_max)
        scores = scores - scores_max

        attn_weights = torch.softmax(scores, dim=-1)  # (B, H, chunk, S)

        # Weighted sum: (B, H, chunk, D)
        out_chunk = torch.matmul(attn_weights, v)
        outputs.append(out_chunk)

    return torch.cat(outputs, dim=2)  # (B, H, T, D)


def standard_attention(
    q: Tensor,  # (B, H, T, D)
    k: Tensor,  # (B, H, S, D)
    v: Tensor,  # (B, H, S, D)
    causal: bool = True,
    scale: float | None = None,
) -> Tensor:
    """Reference implementation for correctness comparison."""
    B, H, T, D = q.shape
    S = k.shape[2]

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    # (B, H, T, S)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    if causal:
        # T x S causal mask
        q_positions = torch.arange(T, device=q.device).unsqueeze(1)   # (T, 1)
        k_positions = torch.arange(S, device=q.device).unsqueeze(0)    # (1, S)
        mask = k_positions > q_positions  # (T, S)
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

    attn_weights = torch.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, v)  # (B, H, T, D)


def compute_memory_ratio(T: int, chunk_size: int) -> float:
    """Return chunk_size/T — ratio of peak attn map memory to full attention."""
    return chunk_size / T


class ChunkedMultiHeadAttention(nn.Module):
    """Multi-head attention using chunked computation."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        cfg: ChunkedAttnConfig,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.cfg = cfg

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """x: (B, T, D) -> (B, T, D).

        Projects to Q/K/V, reshapes to (B, H, T, head_dim), applies
        chunked_attention, reshapes back to (B, T, D), applies out_proj.
        """
        B, T, D = x.shape

        def _project_and_reshape(proj: nn.Linear) -> Tensor:
            out = proj(x)  # (B, T, D)
            return out.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, head_dim)

        q = _project_and_reshape(self.q_proj)
        k = _project_and_reshape(self.k_proj)
        v = _project_and_reshape(self.v_proj)

        scale = self.cfg.scale if self.cfg.scale is not None else 1.0 / math.sqrt(self.head_dim)

        out = chunked_attention(
            q, k, v,
            chunk_size=self.cfg.chunk_size,
            causal=self.cfg.causal,
            scale=scale,
        )  # (B, H, T, head_dim)

        # Reshape back: (B, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


def estimate_attention_memory_mb(
    batch_size: int,
    n_heads: int,
    seq_len: int,
    head_dim: int,
    chunk_size: int | None = None,
    bytes_per_element: int = 4,
) -> dict[str, float]:
    """Return dict with full_mb, chunked_mb, savings_fraction.

    full:    B * H * T * T * bytes
    chunked: B * H * chunk_size * T * bytes  (if chunk_size provided, else same as full)
    """
    full_elements = batch_size * n_heads * seq_len * seq_len
    full_mb = full_elements * bytes_per_element / (1024 ** 2)

    if chunk_size is not None:
        chunked_elements = batch_size * n_heads * chunk_size * seq_len
    else:
        chunked_elements = full_elements
    chunked_mb = chunked_elements * bytes_per_element / (1024 ** 2)

    savings_fraction = 1.0 - (chunked_mb / full_mb) if full_mb > 0 else 0.0

    return {
        "full_mb": full_mb,
        "chunked_mb": chunked_mb,
        "savings_fraction": savings_fraction,
    }


class ChunkedAttentionBenchmark:
    """Benchmark chunked vs standard attention equivalence and speed."""

    def __init__(self, cfg: ChunkedAttnConfig) -> None:
        self.cfg = cfg

    def verify_equivalence(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        atol: float = 1e-4,
    ) -> bool:
        """Return True if chunked_attention matches standard_attention within atol."""
        scale = self.cfg.scale
        ref = standard_attention(q, k, v, causal=self.cfg.causal, scale=scale)
        chunked = chunked_attention(
            q, k, v,
            chunk_size=self.cfg.chunk_size,
            causal=self.cfg.causal,
            scale=scale,
        )
        return torch.allclose(ref, chunked, atol=atol)

    def estimate_speedup(self, T: int, chunk_size: int) -> float:
        """Theoretical speedup estimate: always returns 1.0 (same flops, less memory)."""
        return 1.0
