"""Unit tests for :class:`src.model.parallel_attention.ParallelAttentionBlock`."""

from __future__ import annotations

import time

import pytest
import torch

from src.model.parallel_attention import ParallelAttentionBlock


D_MODEL = 32
N_HEADS = 4
HEAD_DIM = 8
N_KV_HEADS = 2
D_FF = 64
B = 2
S = 16


def _block(dropout: float = 0.0) -> ParallelAttentionBlock:
    return ParallelAttentionBlock(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        head_dim=HEAD_DIM,
        n_kv_heads=N_KV_HEADS,
        d_ff=D_FF,
        dropout=dropout,
    )


def test_output_shape():
    block = _block()
    x = torch.randn(B, S, D_MODEL)
    y = block(x)
    assert y.shape == (B, S, D_MODEL)


def test_output_dtype_preserved():
    block = _block().to(torch.float32)
    x = torch.randn(B, S, D_MODEL, dtype=torch.float32)
    y = block(x)
    assert y.dtype == x.dtype


def test_gradient_flow_to_all_params():
    block = _block()
    x = torch.randn(B, S, D_MODEL, requires_grad=True)
    y = block(x)
    y.pow(2).mean().backward()
    missing = [n for n, p in block.named_parameters() if p.grad is None]
    assert not missing, f"No gradient for: {missing}"
    for n, p in block.named_parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {n}"


def test_determinism_with_seeded_input():
    torch.manual_seed(0)
    block_a = _block()
    torch.manual_seed(0)
    block_b = _block()
    x = torch.randn(B, S, D_MODEL)
    block_a.eval()
    block_b.eval()
    y_a = block_a(x)
    y_b = block_b(x)
    assert torch.allclose(y_a, y_b, atol=0.0, rtol=0.0)


def test_no_nan_or_inf():
    block = _block()
    x = torch.randn(B, S, D_MODEL) * 5.0
    y = block(x)
    assert torch.isfinite(y).all()


def test_gqa_broadcast_works():
    block = ParallelAttentionBlock(
        d_model=D_MODEL, n_heads=4, head_dim=8, n_kv_heads=1, d_ff=D_FF
    )
    x = torch.randn(B, S, D_MODEL)
    y = block(x)
    assert y.shape == (B, S, D_MODEL)
    assert block.k_proj.weight.shape[0] == 1 * 8
    assert block.q_proj.weight.shape[0] == 4 * 8


def test_residual_preserved_when_branches_zeroed():
    """Zeroing both output projections must reduce the block to identity."""
    block = _block()
    with torch.no_grad():
        block.o_proj.weight.zero_()
        block.w_down.weight.zero_()
    x = torch.randn(B, S, D_MODEL)
    y = block(x)
    assert torch.allclose(y, x, atol=1e-6, rtol=1e-6)


def test_invalid_head_dim_raises():
    with pytest.raises(ValueError, match="head_dim"):
        ParallelAttentionBlock(
            d_model=32, n_heads=4, head_dim=9, n_kv_heads=2, d_ff=64
        )


def test_invalid_kv_heads_raises():
    with pytest.raises(ValueError, match="divisible"):
        ParallelAttentionBlock(
            d_model=32, n_heads=4, head_dim=8, n_kv_heads=3, d_ff=64
        )


def test_batch1_seq1_edge_case():
    block = _block()
    x = torch.randn(1, 1, D_MODEL)
    y = block(x)
    assert y.shape == (1, 1, D_MODEL)
    assert torch.isfinite(y).all()


def test_train_vs_eval_dropout_distinguishable():
    torch.manual_seed(42)
    block = _block(dropout=0.5)
    x = torch.randn(B, S, D_MODEL)

    block.eval()
    y_eval_1 = block(x)
    y_eval_2 = block(x)
    assert torch.allclose(y_eval_1, y_eval_2)

    block.train()
    torch.manual_seed(1)
    y_train_1 = block(x)
    torch.manual_seed(2)
    y_train_2 = block(x)
    assert not torch.allclose(y_train_1, y_train_2)
    assert not torch.allclose(y_eval_1, y_train_1)


def test_large_seq_runs_fast():
    block = _block()
    block.eval()
    x = torch.randn(1, 1024, D_MODEL)
    t0 = time.perf_counter()
    with torch.no_grad():
        y = block(x)
    elapsed = time.perf_counter() - t0
    assert y.shape == (1, 1024, D_MODEL)
    assert elapsed < 1.0, f"large-seq forward took {elapsed:.3f}s (>=1s)"


def test_param_count_envelope():
    block = _block()
    total = sum(p.numel() for p in block.parameters())
    # LN: 64; qkvo: 1024+512+512+1024; ffn: 4096+2048 = 9280
    expected = 9280
    assert total == expected, f"expected {expected} params, got {total}"


def test_wrong_input_shape_raises():
    block = _block()
    with pytest.raises(ValueError):
        block(torch.randn(B, S, D_MODEL + 1))


def test_parallel_equivalence_formula():
    """out == x + attn(LN(x)) + ffn(LN(x)) (parallel, not sequential)."""
    torch.manual_seed(0)
    block = _block()
    block.eval()
    x = torch.randn(B, S, D_MODEL)
    normed = block.norm(x)
    expected = x + block._attention(normed) + block._ffn(normed)
    got = block(x)
    assert torch.allclose(got, expected, atol=1e-6, rtol=1e-6)
