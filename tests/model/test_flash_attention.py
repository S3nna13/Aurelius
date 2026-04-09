"""Tests for src/model/flash_attention.py — minimum 12 tests."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from src.model.flash_attention import (
    FlashAttentionSimulator,
    FlashConfig,
    benchmark_attention_equivalence,
    compute_memory_footprint,
    online_softmax,
    tiled_attention,
)

# ---------------------------------------------------------------------------
# Shared test dimensions (small to keep tests fast)
# ---------------------------------------------------------------------------
B, H, T, HD, BS = 2, 2, 16, 16, 8  # batch, heads, seq_len, head_dim, block_size
D_MODEL = H * HD  # 32


# ---------------------------------------------------------------------------
# 1. FlashConfig defaults
# ---------------------------------------------------------------------------
def test_flash_config_defaults():
    cfg = FlashConfig()
    assert cfg.block_size == 64
    assert cfg.use_causal_mask is True
    assert cfg.dropout_p == 0.0


# ---------------------------------------------------------------------------
# 2. online_softmax — output shapes
# ---------------------------------------------------------------------------
def test_online_softmax_shapes():
    scores = torch.randn(B, H, T, T)
    m, l, p = online_softmax(scores)
    assert m.shape == (B, H, T, 1), f"m shape mismatch: {m.shape}"
    assert l.shape == (B, H, T, 1), f"l shape mismatch: {l.shape}"
    assert p.shape == (B, H, T, T), f"p shape mismatch: {p.shape}"


# ---------------------------------------------------------------------------
# 3. online_softmax — probabilities sum to 1
# ---------------------------------------------------------------------------
def test_online_softmax_sums_to_one():
    scores = torch.randn(B, H, T, T)
    _, _, p = online_softmax(scores)
    row_sums = p.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), (
        f"Row sums not 1: max deviation {(row_sums - 1).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# 4. tiled_attention — output shape (B, H, T, head_dim)
# ---------------------------------------------------------------------------
def test_tiled_attention_output_shape():
    Q = torch.randn(B, H, T, HD)
    K = torch.randn(B, H, T, HD)
    V = torch.randn(B, H, T, HD)
    out = tiled_attention(Q, K, V, block_size=BS, causal=True)
    assert out.shape == (B, H, T, HD), f"Unexpected shape: {out.shape}"


# ---------------------------------------------------------------------------
# 5. tiled_attention — causal masking (output at t must not depend on t+1..T)
# ---------------------------------------------------------------------------
def test_tiled_attention_causal_masking():
    torch.manual_seed(42)
    Q = torch.randn(B, H, T, HD)
    K = torch.randn(B, H, T, HD)
    V = torch.randn(B, H, T, HD)

    out_before = tiled_attention(Q, K, V, block_size=BS, causal=True)

    # Perturb V at position t=T-1 (the last token) and check that earlier outputs
    # don't change — i.e., position 0's output must remain identical.
    V_perturbed = V.clone()
    V_perturbed[:, :, T - 1, :] += 100.0  # large delta at the last position

    out_after = tiled_attention(Q, K_perturbed := K, V_perturbed, block_size=BS, causal=True)

    # Output at position 0 should be unchanged (position 0 never attends to position T-1)
    assert torch.allclose(out_before[:, :, 0, :], out_after[:, :, 0, :], atol=1e-5), (
        "Causal masking violated: position 0 output changed when future V was perturbed"
    )


# ---------------------------------------------------------------------------
# 6. tiled_attention — equivalence with standard attention (max_diff < 1e-3)
# ---------------------------------------------------------------------------
def test_tiled_attention_equivalence_with_standard():
    torch.manual_seed(7)
    Q = torch.randn(B, H, T, HD)
    K = torch.randn(B, H, T, HD)
    V = torch.randn(B, H, T, HD)

    # Standard causal attention
    scale = 1.0 / math.sqrt(HD)
    scores = scale * torch.matmul(Q, K.transpose(-2, -1))
    causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), -1e9)
    standard_out = torch.matmul(F.softmax(scores, dim=-1), V)

    tiled_out = tiled_attention(Q, K, V, block_size=BS, causal=True)

    max_diff = (standard_out - tiled_out).abs().max().item()
    assert max_diff < 1e-3, f"max_diff={max_diff:.2e} exceeds 1e-3"


# ---------------------------------------------------------------------------
# 7. FlashAttentionSimulator — forward output shape (B, T, d_model)
# ---------------------------------------------------------------------------
def test_flash_attention_simulator_output_shape():
    cfg = FlashConfig(block_size=BS)
    model = FlashAttentionSimulator(d_model=D_MODEL, n_heads=H, config=cfg)
    x = torch.randn(B, T, D_MODEL)
    out = model(x)
    assert out.shape == (B, T, D_MODEL), f"Unexpected shape: {out.shape}"


# ---------------------------------------------------------------------------
# 8. FlashAttentionSimulator — works with different block sizes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("block_size", [4, 8, 16, T])
def test_flash_attention_simulator_different_block_sizes(block_size):
    cfg = FlashConfig(block_size=block_size)
    model = FlashAttentionSimulator(d_model=D_MODEL, n_heads=H, config=cfg)
    x = torch.randn(B, T, D_MODEL)
    out = model(x)
    assert out.shape == (B, T, D_MODEL), (
        f"block_size={block_size}: unexpected shape {out.shape}"
    )


# ---------------------------------------------------------------------------
# 9. compute_memory_footprint — required keys present
# ---------------------------------------------------------------------------
def test_compute_memory_footprint_keys():
    result = compute_memory_footprint(B, H, T, HD, BS)
    assert "standard_attention_bytes" in result
    assert "tiled_attention_bytes" in result
    assert "memory_reduction_factor" in result


# ---------------------------------------------------------------------------
# 10. compute_memory_footprint — reduction factor > 1 when T > block_size
# ---------------------------------------------------------------------------
def test_compute_memory_footprint_reduction_factor():
    assert T > BS, "Test assumes T > block_size"
    result = compute_memory_footprint(B, H, T, HD, BS)
    factor = result["memory_reduction_factor"]
    assert factor > 1.0, f"Expected reduction_factor > 1, got {factor}"


# ---------------------------------------------------------------------------
# 11. benchmark_attention_equivalence — correct keys returned
# ---------------------------------------------------------------------------
def test_benchmark_attention_equivalence_keys():
    result = benchmark_attention_equivalence(B, H, T, HD, BS)
    assert "max_diff" in result
    assert "mean_diff" in result
    assert "equivalent" in result


# ---------------------------------------------------------------------------
# 12. benchmark_attention_equivalence — small T (T <= block_size) should be equivalent
# ---------------------------------------------------------------------------
def test_benchmark_attention_equivalence_small_T():
    small_T = 8
    small_BS = 8  # block_size == T  → single tile, no tiling artefacts
    result = benchmark_attention_equivalence(B, H, small_T, HD, small_BS)
    assert result["equivalent"], (
        f"Expected equivalence for T={small_T}, block_size={small_BS}; "
        f"max_diff={result['max_diff']:.2e}"
    )


# ---------------------------------------------------------------------------
# Extra: verify benchmark equivalence also holds for the main test dimensions
# ---------------------------------------------------------------------------
def test_benchmark_attention_equivalence_standard_dims():
    result = benchmark_attention_equivalence(B, H, T, HD, BS)
    assert result["equivalent"], (
        f"Tiled attention not equivalent for B={B} H={H} T={T} HD={HD} BS={BS}; "
        f"max_diff={result['max_diff']:.2e}"
    )
