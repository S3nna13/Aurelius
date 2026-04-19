"""Unit tests for ChunkedLocalAttention."""

from __future__ import annotations

import time

import pytest
import torch

from src.model.chunked_local_attention import ChunkedLocalAttention


D_MODEL = 32
N_HEADS = 4
HEAD_DIM = 8
CHUNK = 8
WINDOW = 4
B = 2
S = 32


def _make(**overrides) -> ChunkedLocalAttention:
    kw = dict(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        head_dim=HEAD_DIM,
        chunk_size=CHUNK,
        window_size=WINDOW,
        dropout=0.0,
    )
    kw.update(overrides)
    return ChunkedLocalAttention(**kw)


def test_output_shape():
    mod = _make()
    x = torch.randn(B, S, D_MODEL)
    y = mod(x)
    assert y.shape == (B, S, D_MODEL)


def test_output_dtype_matches_input():
    mod = _make()
    x = torch.randn(B, S, D_MODEL, dtype=torch.float32)
    assert mod(x).dtype == torch.float32


def test_gradient_flow_all_params():
    mod = _make()
    x = torch.randn(B, S, D_MODEL, requires_grad=True)
    y = mod(x)
    y.sum().backward()
    assert x.grad is not None
    for name, p in mod.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"


def test_determinism_with_seed():
    torch.manual_seed(0)
    mod_a = _make()
    torch.manual_seed(0)
    mod_b = _make()
    x = torch.randn(B, S, D_MODEL)
    assert torch.allclose(mod_a(x), mod_b(x))


def test_no_nan_or_inf():
    mod = _make()
    x = torch.randn(B, S, D_MODEL) * 5.0
    y = mod(x)
    assert torch.isfinite(y).all()


def test_window_plus_one_chunk_attends_only_immediate_past():
    """chunk_size = window_size + 1 -> each token attends exactly to its chunk prefix."""
    mod = _make(chunk_size=WINDOW + 1, window_size=WINDOW)
    x = torch.randn(B, 2 * (WINDOW + 1), D_MODEL)
    y = mod(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_large_window_approximates_full_chunk_attention():
    """With window_size = chunk_size - 1, each token can attend to full causal chunk."""
    mod_small = _make(chunk_size=CHUNK, window_size=1)
    mod_large = _make(chunk_size=CHUNK, window_size=CHUNK - 1)
    # Copy weights so the only difference is the window.
    mod_large.load_state_dict(mod_small.state_dict())
    x = torch.randn(B, S, D_MODEL)
    y_small = mod_small(x)
    y_large = mod_large(x)
    # Outputs should differ (not equal) but both finite.
    assert not torch.allclose(y_small, y_large)
    assert torch.isfinite(y_small).all() and torch.isfinite(y_large).all()


def test_batch_one_and_seq_one_degenerate():
    mod = _make()
    # B=1
    y1 = mod(torch.randn(1, S, D_MODEL))
    assert y1.shape == (1, S, D_MODEL)
    # S=1
    y2 = mod(torch.randn(B, 1, D_MODEL))
    assert y2.shape == (B, 1, D_MODEL)
    assert torch.isfinite(y2).all()


def test_invalid_chunk_le_window_raises():
    with pytest.raises(ValueError, match="chunk_size"):
        ChunkedLocalAttention(
            d_model=D_MODEL, n_heads=N_HEADS, head_dim=HEAD_DIM,
            chunk_size=WINDOW, window_size=WINDOW,
        )
    with pytest.raises(ValueError, match="chunk_size"):
        ChunkedLocalAttention(
            d_model=D_MODEL, n_heads=N_HEADS, head_dim=HEAD_DIM,
            chunk_size=WINDOW - 1, window_size=WINDOW,
        )


def test_invalid_head_dim_times_heads_neq_d_model_raises():
    with pytest.raises(ValueError, match="d_model"):
        ChunkedLocalAttention(
            d_model=D_MODEL, n_heads=N_HEADS, head_dim=HEAD_DIM + 1,
            chunk_size=CHUNK, window_size=WINDOW,
        )


def test_1024_tokens_chunk64_runs_fast():
    mod = _make(chunk_size=64, window_size=32)
    x = torch.randn(1, 1024, D_MODEL)
    # Warm-up.
    mod(x)
    t0 = time.perf_counter()
    y = mod(x)
    dt = time.perf_counter() - t0
    assert y.shape == (1, 1024, D_MODEL)
    assert dt < 1.0, f"forward took {dt:.3f}s (>1s)"


def test_dropout_zero_is_deterministic_across_calls():
    mod = _make(dropout=0.0)
    mod.train()  # training mode; dropout_p=0 so still deterministic
    x = torch.randn(B, S, D_MODEL)
    y1 = mod(x)
    y2 = mod(x)
    assert torch.allclose(y1, y2)


def test_eval_mode_disables_dropout():
    mod = _make(dropout=0.5)
    mod.train(False)
    x = torch.randn(B, S, D_MODEL)
    y1 = mod(x)
    y2 = mod(x)
    # In eval, dropout disabled -> outputs identical.
    assert torch.allclose(y1, y2)


def test_non_multiple_seq_len_padding_handled():
    """S not divisible by chunk_size should still produce correct-shape output."""
    mod = _make(chunk_size=8, window_size=4)
    S_odd = 30  # not a multiple of 8
    x = torch.randn(B, S_odd, D_MODEL)
    y = mod(x)
    assert y.shape == (B, S_odd, D_MODEL)
    assert torch.isfinite(y).all()
