"""Attention memory analysis and block tiling utilities.

Provides theoretical estimates for:
- Standard attention memory footprint (O(n²) attention matrix)
- Flash Attention memory footprint (O(n) streaming)
- Optimal block sizes for a given SRAM budget
- Hardware utilization estimates
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AttentionMemoryStats:
    """Memory footprint analysis for a single attention layer."""

    seq_len: int
    n_heads: int
    head_dim: int
    dtype_bytes: int  # 2 for bf16/fp16, 4 for fp32

    @property
    def qkv_bytes(self) -> int:
        """Memory for Q, K, V tensors: 3 * B * H * S * D * dtype_bytes."""
        # (Assumes batch=1 for per-layer analysis)
        return 3 * self.n_heads * self.seq_len * self.head_dim * self.dtype_bytes

    @property
    def attention_matrix_bytes(self) -> int:
        """Standard attention matrix S = Q @ K^T: B * H * S * S * dtype_bytes."""
        return self.n_heads * self.seq_len * self.seq_len * self.dtype_bytes

    @property
    def flash_attention_bytes(self) -> int:
        """Flash attention extra memory (only stores O, l, m per block).

        Flash attention uses O(S) not O(S²) for the attention scores.
        Extra memory: output O (same as QKV), plus log-sum-exp l and max m
        both of shape (H, S) — negligible vs QKV.
        """
        output_bytes = self.n_heads * self.seq_len * self.head_dim * self.dtype_bytes
        lm_bytes = 2 * self.n_heads * self.seq_len * self.dtype_bytes  # l and m
        return output_bytes + lm_bytes

    @property
    def memory_ratio(self) -> float:
        """Ratio of standard to Flash attention extra memory.

        = attention_matrix_bytes / flash_attention_bytes
        Higher = bigger win from Flash Attention.
        """
        return self.attention_matrix_bytes / max(1, self.flash_attention_bytes)


@dataclass
class BlockTilingConfig:
    """Block tiling configuration for Flash Attention."""

    block_q: int  # query block size
    block_kv: int  # key/value block size
    n_blocks_q: int
    n_blocks_kv: int
    sram_usage_bytes: int
    fits_in_sram: bool


def compute_optimal_block_size(
    head_dim: int,
    dtype_bytes: int,
    sram_budget_bytes: int,
    max_block: int = 256,
) -> BlockTilingConfig:
    """Find the largest square block size that fits in SRAM.

    Flash Attention SRAM requirement per block:
        (block_q + block_kv) * head_dim * dtype_bytes   # for Q and K/V slices
        + block_q * block_kv * dtype_bytes              # for S block

    Find largest block_size (power of 2) such that total fits in sram_budget.
    block_q = block_kv = block_size.

    Args:
        head_dim: Attention head dimension
        dtype_bytes: Bytes per element (2 for fp16/bf16, 4 for fp32)
        sram_budget_bytes: Available SRAM (e.g., 96KB = 98304 bytes for A100 SM)
        max_block: Maximum block size to consider

    Returns:
        BlockTilingConfig with optimal block sizes.
    """
    best_block = 1
    # Try powers of 2 from max_block down to 1
    block = 1
    powers = []
    while block <= max_block:
        powers.append(block)
        block *= 2

    for b in reversed(powers):
        # SRAM needed: (b + b) * head_dim * dtype_bytes + b * b * dtype_bytes
        sram_needed = (2 * b * head_dim * dtype_bytes) + (b * b * dtype_bytes)
        if sram_needed <= sram_budget_bytes:
            best_block = b
            break

    block_size = best_block
    sram_usage = (2 * block_size * head_dim * dtype_bytes) + (block_size * block_size * dtype_bytes)
    fits = sram_usage <= sram_budget_bytes

    # n_blocks_q and n_blocks_kv are symbolic (seq_len not provided), use 1 as placeholder
    # The config is per-head; callers can multiply by ceil(seq_len / block_size)
    return BlockTilingConfig(
        block_q=block_size,
        block_kv=block_size,
        n_blocks_q=1,
        n_blocks_kv=1,
        sram_usage_bytes=sram_usage,
        fits_in_sram=fits,
    )


def attention_flops(
    seq_len: int,
    n_heads: int,
    head_dim: int,
    causal: bool = True,
) -> int:
    """Compute attention FLOPs.

    Q @ K^T: (S, D) @ (D, S) = S² * D * 2 FLOPs per head
    Softmax: ~5 * S² FLOPs per head (exp, sum, div, etc.) — approximate
    Scores @ V: S² * D * 2 per head
    Total ≈ (4 * D + 5) * S² * H  FLOPs
    For causal: multiply by 0.5 (half the attention matrix is masked)
    """
    s2 = seq_len * seq_len
    # Per head: QK^T matmul + softmax + SV matmul
    flops_per_head = (4 * head_dim + 5) * s2
    total = flops_per_head * n_heads
    if causal:
        total = total // 2
    return int(total)


def memory_bandwidth_bound_at_seqlen(
    seq_len: int,
    n_heads: int,
    head_dim: int,
    dtype_bytes: int = 2,
    memory_bandwidth_gbps: float = 2000.0,  # A100 HBM bandwidth
    flops_per_second: float = 312e12,  # A100 bf16 Tensor Core FLOPS
) -> dict[str, float]:
    """Estimate whether attention is compute-bound or memory-bound at a given seq_len.

    Returns dict with:
    - flops: attention FLOPs
    - bytes_moved: bytes read/written (QKV + attention matrix or flash alternative)
    - arithmetic_intensity: flops / bytes
    - ridge_point: device flops / bandwidth (ops/byte threshold)
    - is_compute_bound: True if arithmetic_intensity > ridge_point
    - standard_attention_time_ms: estimated wall time with standard attention
    - flash_attention_time_ms: estimated wall time with Flash Attention
    """
    stats = AttentionMemoryStats(
        seq_len=seq_len,
        n_heads=n_heads,
        head_dim=head_dim,
        dtype_bytes=dtype_bytes,
    )

    flops = attention_flops(seq_len, n_heads, head_dim, causal=True)

    # bytes moved for standard attention: QKV reads + attention matrix + output
    standard_bytes = (
        stats.qkv_bytes
        + stats.attention_matrix_bytes
        + (
            n_heads * seq_len * head_dim * dtype_bytes  # output O
        )
    )

    # bytes moved for flash attention: QKV reads + output (no full attn matrix materialized)
    flash_bytes = stats.qkv_bytes + stats.flash_attention_bytes

    bandwidth_bytes_per_sec = memory_bandwidth_gbps * 1e9

    arithmetic_intensity = flops / max(1, standard_bytes)
    ridge_point = flops_per_second / bandwidth_bytes_per_sec

    is_compute_bound = arithmetic_intensity > ridge_point

    standard_attention_time_ms = (standard_bytes / bandwidth_bytes_per_sec) * 1000.0
    flash_attention_time_ms = (flash_bytes / bandwidth_bytes_per_sec) * 1000.0

    return {
        "flops": float(flops),
        "bytes_moved": float(standard_bytes),
        "arithmetic_intensity": arithmetic_intensity,
        "ridge_point": ridge_point,
        "is_compute_bound": is_compute_bound,
        "standard_attention_time_ms": standard_attention_time_ms,
        "flash_attention_time_ms": flash_attention_time_ms,
    }
