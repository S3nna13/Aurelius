"""Memory-efficient attention implementations.

Implements chunked attention (Rabe & Staats 2021 style) to avoid O(N^2) memory
by computing attention in chunks over the K/V dimension using online softmax.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import AureliusConfig


def online_softmax_update(
    prev_max: torch.Tensor,  # (B, H, Q)
    prev_sum: torch.Tensor,  # (B, H, Q)
    prev_out: torch.Tensor,  # (B, H, Q, D)
    new_scores: torch.Tensor,  # (B, H, Q, K_chunk)
    new_v: torch.Tensor,  # (B, H, K_chunk, D)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Update running (max, sum, out) with a new chunk of K/V pairs.

    Uses the log-sum-exp trick to maintain numerical stability without
    materializing the full attention matrix.

    Args:
        prev_max: Running maximum of attention scores, shape (B, H, Q).
        prev_sum: Running sum of exp(scores - max), shape (B, H, Q).
        prev_out: Running weighted value output, shape (B, H, Q, D).
        new_scores: New chunk of attention scores, shape (B, H, Q, K_chunk).
        new_v: New chunk of values, shape (B, H, K_chunk, D).

    Returns:
        Tuple of updated (max, sum, out).
    """
    # 1. chunk_max: (B, H, Q)
    chunk_max = new_scores.max(dim=-1).values

    # 2. new_max: element-wise max using torch.maximum
    new_max = torch.maximum(prev_max, chunk_max)

    # 3. exp_new: (B, H, Q, K_chunk)
    exp_new = torch.exp(new_scores - new_max.unsqueeze(-1))

    # 4. chunk_sum: (B, H, Q)
    chunk_sum = exp_new.sum(dim=-1)

    # 5. scale_prev: correction factor for previous accumulator (B, H, Q)
    scale_prev = torch.exp(prev_max - new_max)

    # 6. new_sum: (B, H, Q)
    new_sum = scale_prev * prev_sum + chunk_sum

    # 7. new_out: (B, H, Q, D)
    # exp_new @ new_v: (B, H, Q, K_chunk) x (B, H, K_chunk, D) -> (B, H, Q, D)
    new_out = scale_prev.unsqueeze(-1) * prev_out + exp_new @ new_v

    return new_max, new_sum, new_out


def chunked_attention(
    q: torch.Tensor,  # (B, H, T_q, D)
    k: torch.Tensor,  # (B, H, T_k, D)
    v: torch.Tensor,  # (B, H, T_k, D)
    chunk_size: int = 128,
    causal: bool = True,
    scale: float | None = None,
) -> torch.Tensor:
    """Compute attention in chunks over K/V dimension.

    Avoids materializing the full (T_q, T_k) attention matrix.
    Memory usage: O(T_q * chunk_size) instead of O(T_q * T_k).

    Args:
        q: Query tensor, shape (B, H, T_q, D).
        k: Key tensor, shape (B, H, T_k, D).
        v: Value tensor, shape (B, H, T_k, D).
        chunk_size: Number of K/V positions to process per chunk.
        causal: If True, apply causal masking (query i cannot attend to key j > i).
        scale: Attention scale factor. Defaults to 1/sqrt(D).

    Returns:
        Output tensor of shape (B, H, T_q, D).
    """
    B, H, T_q, D = q.shape
    T_k = k.shape[2]

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    # Initialize running accumulators
    # Use float32 for numerical stability
    running_max = torch.full((B, H, T_q), float("-inf"), dtype=torch.float32, device=q.device)
    running_sum = torch.zeros((B, H, T_q), dtype=torch.float32, device=q.device)
    running_out = torch.zeros((B, H, T_q, D), dtype=torch.float32, device=q.device)

    q_f = q.float()
    k_f = k.float()
    v_f = v.float()

    # Query position indices for causal masking: (1, 1, T_q, 1)
    q_pos = torch.arange(T_q, device=q.device).view(1, 1, T_q, 1)

    kv_start = 0
    while kv_start < T_k:
        kv_end = min(kv_start + chunk_size, T_k)
        actual_chunk = kv_end - kv_start

        k_chunk = k_f[:, :, kv_start:kv_end, :]  # (B, H, chunk, D)
        v_chunk = v_f[:, :, kv_start:kv_end, :]  # (B, H, chunk, D)

        # scores: (B, H, T_q, chunk)
        scores = scale * torch.matmul(q_f, k_chunk.transpose(-2, -1))

        if causal:
            # k_j positions: (1, 1, 1, chunk)
            k_pos = torch.arange(kv_start, kv_end, device=q.device).view(1, 1, 1, actual_chunk)
            # Mask: True where k_j > q_i (future positions)
            causal_mask = k_pos > q_pos  # (1, 1, T_q, chunk) broadcast
            scores = scores.masked_fill(causal_mask, float("-inf"))

        running_max, running_sum, running_out = online_softmax_update(
            running_max, running_sum, running_out, scores, v_chunk
        )

        kv_start = kv_end

    # Normalize: output = out / sum
    output = running_out / running_sum.unsqueeze(-1)

    return output.to(q.dtype)


class MemoryEfficientAttention(nn.Module):
    """Multi-head attention using chunked computation.

    Drop-in replacement for standard attention that avoids materializing
    the full N×N attention matrix by processing K/V in chunks.

    Args:
        config: AureliusConfig with model hyperparameters.
        chunk_size: Number of K/V positions per chunk (default 128).
    """

    def __init__(self, config: AureliusConfig, chunk_size: int = 128) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model
        self.chunk_size = chunk_size
        self.n_rep = self.n_heads // self.n_kv_heads  # GQA repetition factor

        # Projections (no bias, matching GroupedQueryAttention convention)
        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * config.head_dim, config.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,  # (B, T, D)
        past_key_values=None,
        use_cache: bool = False,
        **kwargs,
    ) -> torch.Tensor | tuple:
        """Forward pass using chunked attention.

        Args:
            x: Input tensor of shape (B, T, D).
            past_key_values: Optional cached (k, v) tensors as a tuple.
            use_cache: If True, return (output, present_kv); else return output.
            **kwargs: Additional arguments (ignored, for API compatibility).

        Returns:
            If use_cache=False: output tensor of shape (B, T, D).
            If use_cache=True: tuple of (output, (k_cache, v_cache)).
        """
        B, T, _ = x.shape

        # Project
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)

        # Handle KV cache
        if past_key_values is not None:
            past_k, past_v = past_key_values
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)

        present_kv = (k, v)
        T_k = k.shape[1]

        # Expand KV heads for GQA: (B, T, n_kv_heads, D) -> (B, T, n_heads, D)
        if self.n_rep > 1:
            k = k.unsqueeze(3).expand(B, T_k, self.n_kv_heads, self.n_rep, self.head_dim)
            k = k.reshape(B, T_k, self.n_heads, self.head_dim)
            v = v.unsqueeze(3).expand(B, T_k, self.n_kv_heads, self.n_rep, self.head_dim)
            v = v.reshape(B, T_k, self.n_heads, self.head_dim)

        # Transpose to (B, H, T, D) for chunked_attention
        q = q.transpose(1, 2)  # (B, n_heads, T_q, head_dim)
        k = k.transpose(1, 2)  # (B, n_heads, T_k, head_dim)
        v = v.transpose(1, 2)  # (B, n_heads, T_k, head_dim)

        # Chunked attention (causal by default)
        out = chunked_attention(
            q,
            k,
            v,
            chunk_size=self.chunk_size,
            causal=True,
        )

        # Reshape: (B, n_heads, T, head_dim) -> (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.o_proj(out)

        if use_cache:
            return out, present_kv
        return out


def compute_attention_memory(
    seq_len: int,
    n_heads: int,
    head_dim: int,
    dtype_bytes: int = 2,
    chunk_size: int | None = None,
) -> dict:
    """Estimate attention memory usage in megabytes.

    Args:
        seq_len: Sequence length (T).
        n_heads: Number of attention heads (H).
        head_dim: Dimension per head (D).
        dtype_bytes: Bytes per element (2 for fp16/bf16, 4 for fp32).
        chunk_size: If provided, compute chunked attention memory.

    Returns:
        Dictionary with keys:
            'standard_attn_mb': Memory for full N×N attention matrix (MB).
            'chunked_attn_mb': Memory for chunked attention (MB).
            'qkv_mb': Memory for Q, K, V tensors (MB).
            'reduction_factor': standard_attn_mb / chunked_attn_mb.
    """
    bytes_per_mb = 1024**2

    # Standard attention: N^2 attention matrix per head
    # Shape: (n_heads, seq_len, seq_len)
    standard_attn_elements = n_heads * seq_len * seq_len
    standard_attn_mb = (standard_attn_elements * dtype_bytes) / bytes_per_mb

    # Chunked attention: only O(seq_len * chunk_size) per head
    if chunk_size is not None:
        chunked_elements = n_heads * seq_len * chunk_size
        chunked_attn_mb = (chunked_elements * dtype_bytes) / bytes_per_mb
    else:
        # No chunking: same as standard
        chunked_attn_mb = standard_attn_mb

    # QKV tensors: Q, K, V each of shape (seq_len, n_heads, head_dim)
    qkv_elements = 3 * seq_len * n_heads * head_dim
    qkv_mb = (qkv_elements * dtype_bytes) / bytes_per_mb

    # Reduction factor
    reduction_factor = standard_attn_mb / chunked_attn_mb if chunked_attn_mb > 0 else 1.0

    return {
        "standard_attn_mb": standard_attn_mb,
        "chunked_attn_mb": chunked_attn_mb,
        "qkv_mb": qkv_mb,
        "reduction_factor": reduction_factor,
    }
