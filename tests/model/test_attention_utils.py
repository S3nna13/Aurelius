"""Tests for attention memory analysis and block tiling utilities."""
import math
import pytest

from src.model.attention_utils import (
    AttentionMemoryStats,
    BlockTilingConfig,
    compute_optimal_block_size,
    attention_flops,
    memory_bandwidth_bound_at_seqlen,
)


# ---------------------------------------------------------------------------
# AttentionMemoryStats
# ---------------------------------------------------------------------------

def test_attention_memory_stats_qkv_bytes():
    stats = AttentionMemoryStats(seq_len=512, n_heads=8, head_dim=64, dtype_bytes=2)
    expected = 3 * 8 * 512 * 64 * 2
    assert stats.qkv_bytes == expected


def test_attention_matrix_bytes_quadratic():
    """Attention matrix bytes should grow quadratically with seq_len."""
    stats_512 = AttentionMemoryStats(seq_len=512, n_heads=8, head_dim=64, dtype_bytes=2)
    stats_1024 = AttentionMemoryStats(seq_len=1024, n_heads=8, head_dim=64, dtype_bytes=2)
    # Doubling seq_len should quadruple attention matrix bytes
    assert stats_1024.attention_matrix_bytes == 4 * stats_512.attention_matrix_bytes


def test_flash_attention_bytes_linear():
    """Flash attention bytes should grow linearly with seq_len."""
    stats_512 = AttentionMemoryStats(seq_len=512, n_heads=8, head_dim=64, dtype_bytes=2)
    stats_1024 = AttentionMemoryStats(seq_len=1024, n_heads=8, head_dim=64, dtype_bytes=2)
    # Doubling seq_len should double flash attention bytes
    assert stats_1024.flash_attention_bytes == 2 * stats_512.flash_attention_bytes


def test_memory_ratio_large_seqlen():
    """At long sequences, the memory ratio should be >> 1."""
    stats = AttentionMemoryStats(seq_len=4096, n_heads=16, head_dim=64, dtype_bytes=2)
    assert stats.memory_ratio > 10.0


# ---------------------------------------------------------------------------
# compute_optimal_block_size
# ---------------------------------------------------------------------------

def test_compute_optimal_block_size_returns_config():
    config = compute_optimal_block_size(
        head_dim=64,
        dtype_bytes=2,
        sram_budget_bytes=98304,  # 96 KB
    )
    assert isinstance(config, BlockTilingConfig)


def test_compute_optimal_block_size_fits_in_sram():
    sram_budget = 98304  # 96 KB
    config = compute_optimal_block_size(
        head_dim=64,
        dtype_bytes=2,
        sram_budget_bytes=sram_budget,
    )
    assert config.fits_in_sram is True
    assert config.sram_usage_bytes <= sram_budget


def test_compute_optimal_block_size_power_of_2():
    config = compute_optimal_block_size(
        head_dim=64,
        dtype_bytes=2,
        sram_budget_bytes=98304,
    )
    # Power of 2 check: n & (n-1) == 0 for n > 0
    assert config.block_q > 0
    assert config.block_kv > 0
    assert (config.block_q & (config.block_q - 1)) == 0
    assert (config.block_kv & (config.block_kv - 1)) == 0


# ---------------------------------------------------------------------------
# attention_flops
# ---------------------------------------------------------------------------

def test_attention_flops_causal_half():
    """Causal attention should use approximately half the FLOPs of non-causal."""
    seq_len, n_heads, head_dim = 1024, 8, 64
    flops_causal = attention_flops(seq_len, n_heads, head_dim, causal=True)
    flops_full = attention_flops(seq_len, n_heads, head_dim, causal=False)
    # Causal should be ~half of full (integer division may cause slight difference)
    assert flops_causal == flops_full // 2


# ---------------------------------------------------------------------------
# memory_bandwidth_bound_at_seqlen
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {
    "flops",
    "bytes_moved",
    "arithmetic_intensity",
    "ridge_point",
    "is_compute_bound",
    "standard_attention_time_ms",
    "flash_attention_time_ms",
}


def test_memory_bandwidth_bound_returns_dict():
    result = memory_bandwidth_bound_at_seqlen(
        seq_len=512,
        n_heads=8,
        head_dim=64,
    )
    assert isinstance(result, dict)
    assert REQUIRED_KEYS.issubset(result.keys())


def test_memory_bandwidth_bound_short_seq_memory_bound():
    """At very short sequences, attention is memory-bound (low arithmetic intensity)."""
    result = memory_bandwidth_bound_at_seqlen(
        seq_len=64,
        n_heads=8,
        head_dim=64,
    )
    # At short seq_len, arithmetic intensity should be below the ridge point
    assert result["arithmetic_intensity"] < result["ridge_point"]
    assert result["is_compute_bound"] is False
