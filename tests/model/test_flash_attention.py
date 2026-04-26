"""Tests for src/model/flash_attention.py — minimum 16 tests."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from src.model.flash_attention import (
    # New components
    FlashAttentionConfig,
    FlashAttentionLayer,
    FlashAttentionSimulator,
    FlashConfig,
    benchmark_attention_equivalence,
    chunked_attention,
    compute_memory_footprint,
    flash_attention_forward,
    memory_efficiency_ratio,
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
    m, item, p = online_softmax(scores)
    assert m.shape == (B, H, T, 1), f"m shape mismatch: {m.shape}"
    assert item.shape == (B, H, T, 1), f"l shape mismatch: {item.shape}"
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

    out_after = tiled_attention(Q, _K_perturbed := K, V_perturbed, block_size=BS, causal=True)

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
    assert out.shape == (B, T, D_MODEL), f"block_size={block_size}: unexpected shape {out.shape}"


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


# ===========================================================================
# New tests for FlashAttentionConfig, chunked_attention, flash_attention_forward,
# FlashAttentionLayer, and memory_efficiency_ratio
# ===========================================================================


# ---------------------------------------------------------------------------
# N1. FlashAttentionConfig defaults
# ---------------------------------------------------------------------------
def test_flash_attention_config_defaults():
    cfg = FlashAttentionConfig()
    assert cfg.block_size == 64
    assert cfg.causal is True
    assert cfg.dropout == 0.0
    assert cfg.scale is None


# ---------------------------------------------------------------------------
# N2. FlashAttentionConfig custom values round-trip
# ---------------------------------------------------------------------------
def test_flash_attention_config_custom():
    cfg = FlashAttentionConfig(block_size=32, causal=False, dropout=0.1, scale=0.5)
    assert cfg.block_size == 32
    assert cfg.causal is False
    assert cfg.dropout == 0.1
    assert cfg.scale == 0.5


# ---------------------------------------------------------------------------
# N3. chunked_attention output shape (B, H, T, d)
# ---------------------------------------------------------------------------
def test_chunked_attention_output_shape():
    Q = torch.randn(B, H, T, HD)
    K = torch.randn(B, H, T, HD)
    V = torch.randn(B, H, T, HD)
    out = chunked_attention(Q, K, V, block_size=BS, causal=True)
    assert out.shape == (B, H, T, HD), f"Unexpected shape: {out.shape}"


# ---------------------------------------------------------------------------
# N4. chunked_attention mathematically equivalent to standard attention (tol 1e-4)
# ---------------------------------------------------------------------------
def test_chunked_attention_equivalence_with_standard():
    torch.manual_seed(17)
    Q = torch.randn(B, H, T, HD)
    K = torch.randn(B, H, T, HD)
    V = torch.randn(B, H, T, HD)

    # Standard causal attention
    scale = 1.0 / math.sqrt(HD)
    scores = scale * torch.matmul(Q, K.transpose(-2, -1))
    causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), -1e9)
    standard_out = torch.matmul(F.softmax(scores, dim=-1), V)

    chunked_out = chunked_attention(Q, K, V, block_size=BS, causal=True)

    max_diff = (standard_out - chunked_out).abs().max().item()
    assert max_diff < 1e-4, f"max_diff={max_diff:.2e} exceeds 1e-4"


# ---------------------------------------------------------------------------
# N5. chunked_attention causal masking: future tokens have zero attention weight
# ---------------------------------------------------------------------------
def test_chunked_attention_causal_masking():
    torch.manual_seed(42)
    Q = torch.randn(B, H, T, HD)
    K = torch.randn(B, H, T, HD)
    V = torch.randn(B, H, T, HD)

    out_before = chunked_attention(Q, K, V, block_size=BS, causal=True)

    # Perturb V at last position — position 0 must be unaffected
    V_perturbed = V.clone()
    V_perturbed[:, :, T - 1, :] += 100.0

    out_after = chunked_attention(Q, K, V_perturbed, block_size=BS, causal=True)

    assert torch.allclose(out_before[:, :, 0, :], out_after[:, :, 0, :], atol=1e-5), (
        "Causal masking violated: position 0 output changed when future V was perturbed"
    )


# ---------------------------------------------------------------------------
# N6. chunked_attention non-causal: all positions can attend to all others
# ---------------------------------------------------------------------------
def test_chunked_attention_non_causal_equivalence():
    torch.manual_seed(99)
    Q = torch.randn(B, H, T, HD)
    K = torch.randn(B, H, T, HD)
    V = torch.randn(B, H, T, HD)

    scale = 1.0 / math.sqrt(HD)
    scores = scale * torch.matmul(Q, K.transpose(-2, -1))
    standard_out = torch.matmul(F.softmax(scores, dim=-1), V)

    chunked_out = chunked_attention(Q, K, V, block_size=BS, causal=False)
    max_diff = (standard_out - chunked_out).abs().max().item()
    assert max_diff < 1e-4, f"non-causal max_diff={max_diff:.2e} exceeds 1e-4"


# ---------------------------------------------------------------------------
# N7. block_size=1 gives same result as block_size=T
# ---------------------------------------------------------------------------
def test_chunked_attention_block_size_1_vs_full():
    torch.manual_seed(55)
    Q = torch.randn(B, H, T, HD)
    K = torch.randn(B, H, T, HD)
    V = torch.randn(B, H, T, HD)

    out_bs1 = chunked_attention(Q, K, V, block_size=1, causal=True)
    out_bsT = chunked_attention(Q, K, V, block_size=T, causal=True)

    max_diff = (out_bs1 - out_bsT).abs().max().item()
    assert max_diff < 1e-4, f"block_size=1 vs block_size=T max_diff={max_diff:.2e}"


# ---------------------------------------------------------------------------
# N8. chunked_attention with explicit scale matches scaled standard attention
# ---------------------------------------------------------------------------
def test_chunked_attention_explicit_scale():
    torch.manual_seed(33)
    Q = torch.randn(B, H, T, HD)
    K = torch.randn(B, H, T, HD)
    V = torch.randn(B, H, T, HD)

    explicit_scale = 0.25
    scores = explicit_scale * torch.matmul(Q, K.transpose(-2, -1))
    causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), -1e9)
    standard_out = torch.matmul(F.softmax(scores, dim=-1), V)

    chunked_out = chunked_attention(Q, K, V, block_size=BS, causal=True, scale=explicit_scale)
    max_diff = (standard_out - chunked_out).abs().max().item()
    assert max_diff < 1e-4, f"explicit scale max_diff={max_diff:.2e} exceeds 1e-4"


# ---------------------------------------------------------------------------
# N9. flash_attention_forward wrapper produces correct shape
# ---------------------------------------------------------------------------
def test_flash_attention_forward_shape():
    cfg = FlashAttentionConfig(block_size=BS, causal=True)
    Q = torch.randn(B, H, T, HD)
    K = torch.randn(B, H, T, HD)
    V = torch.randn(B, H, T, HD)
    out = flash_attention_forward(Q, K, V, cfg)
    assert out.shape == (B, H, T, HD), f"Unexpected shape: {out.shape}"


# ---------------------------------------------------------------------------
# N10. flash_attention_forward agrees with chunked_attention directly
# ---------------------------------------------------------------------------
def test_flash_attention_forward_agrees_with_chunked():
    torch.manual_seed(77)
    cfg = FlashAttentionConfig(block_size=BS, causal=True)
    Q = torch.randn(B, H, T, HD)
    K = torch.randn(B, H, T, HD)
    V = torch.randn(B, H, T, HD)

    out_wrapper = flash_attention_forward(Q, K, V, cfg)
    out_direct = chunked_attention(Q, K, V, block_size=BS, causal=True)

    assert torch.allclose(out_wrapper, out_direct, atol=1e-6), (
        "flash_attention_forward and chunked_attention disagree"
    )


# ---------------------------------------------------------------------------
# N11. FlashAttentionLayer output shape (B, T, d_model)
# ---------------------------------------------------------------------------
def test_flash_attention_layer_output_shape():
    cfg = FlashAttentionConfig(block_size=BS)
    layer = FlashAttentionLayer(d_model=D_MODEL, n_heads=H, config=cfg)
    x = torch.randn(B, T, D_MODEL)
    out = layer(x)
    assert out.shape == (B, T, D_MODEL), f"Unexpected shape: {out.shape}"


# ---------------------------------------------------------------------------
# N12. FlashAttentionLayer forward with various seq lengths
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seq_len", [4, 8, 16, 32])
def test_flash_attention_layer_various_seq_lengths(seq_len):
    cfg = FlashAttentionConfig(block_size=4)
    layer = FlashAttentionLayer(d_model=D_MODEL, n_heads=H, config=cfg)
    x = torch.randn(B, seq_len, D_MODEL)
    out = layer(x)
    assert out.shape == (B, seq_len, D_MODEL), f"seq_len={seq_len}: unexpected shape {out.shape}"


# ---------------------------------------------------------------------------
# N13. memory_efficiency_ratio > 1 for seq_len > block_size
# ---------------------------------------------------------------------------
def test_memory_efficiency_ratio_greater_than_one():
    ratio = memory_efficiency_ratio(seq_len=512, block_size=64)
    assert ratio > 1.0, f"Expected ratio > 1, got {ratio}"


# ---------------------------------------------------------------------------
# N14. memory_efficiency_ratio correct value (seq_len / block_size)
# ---------------------------------------------------------------------------
def test_memory_efficiency_ratio_correct_value():
    ratio = memory_efficiency_ratio(seq_len=256, block_size=32)
    expected = 256 / 32  # = 8.0
    assert abs(ratio - expected) < 1e-9, f"Expected {expected}, got {ratio}"


# ---------------------------------------------------------------------------
# N15. memory_efficiency_ratio = 1 when block_size == seq_len
# ---------------------------------------------------------------------------
def test_memory_efficiency_ratio_equals_one_when_full():
    ratio = memory_efficiency_ratio(seq_len=64, block_size=64)
    assert abs(ratio - 1.0) < 1e-9, f"Expected 1.0, got {ratio}"


# ---------------------------------------------------------------------------
# N16. Gradient flows through chunked_attention
# ---------------------------------------------------------------------------
def test_chunked_attention_gradient_flows():
    Q = torch.randn(2, 2, 8, 8, requires_grad=True)
    K = torch.randn(2, 2, 8, 8, requires_grad=True)
    V = torch.randn(2, 2, 8, 8, requires_grad=True)

    out = chunked_attention(Q, K, V, block_size=4, causal=True)
    loss = out.sum()
    loss.backward()

    assert Q.grad is not None, "No gradient for Q"
    assert K.grad is not None, "No gradient for K"
    assert V.grad is not None, "No gradient for V"
    assert not torch.isnan(Q.grad).any(), "NaN gradient in Q"
    assert not torch.isnan(K.grad).any(), "NaN gradient in K"
    assert not torch.isnan(V.grad).any(), "NaN gradient in V"
