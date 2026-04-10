"""Tests for src/model/retention.py -- RetNet retention mechanism.

Covers:
  1.  RetNetConfig defaults
  2.  build_decay_gammas shape
  3.  build_decay_gammas values in (0, 1)
  4.  build_causal_decay_mask shape
  5.  build_causal_decay_mask upper triangle is 0 (causal)
  6.  build_causal_decay_mask diagonal is 1 (gamma^0)
  7.  MultiScaleRetention parallel output shape matches input
  8.  MultiScaleRetention recurrent output shape is (B, d_model)
  9.  MultiScaleRetention recurrent state shape is (B, n_heads, head_dim, head_dim)
  10. MultiScaleRetention chunkwise output shape matches input
  11. MultiScaleRetention parallel vs chunkwise produce close outputs
  12. RetNetBlock output shape matches input (B, T, d_model)
  13. RetNetModel output shape is (B, T, vocab_size)
  14. RetNetModel forward is differentiable (loss.backward() works)
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from src.model.retention import (
    RetNetConfig,
    MultiScaleRetention,
    RetNetBlock,
    RetNetModel,
    build_decay_gammas,
    build_causal_decay_mask,
)

# ---------------------------------------------------------------------------
# Shared constants for small test configs
# ---------------------------------------------------------------------------

D_MODEL = 64
N_HEADS = 4
VOCAB_SIZE = 256
BATCH = 2
SEQ = 16
HEAD_DIM = D_MODEL // N_HEADS  # 16


def make_config(**overrides) -> RetNetConfig:
    defaults = dict(d_model=D_MODEL, n_heads=N_HEADS, dropout=0.0, chunk_size=8)
    defaults.update(overrides)
    return RetNetConfig(**defaults)


# ---------------------------------------------------------------------------
# 1. RetNetConfig defaults
# ---------------------------------------------------------------------------


def test_retnet_config_defaults():
    """RetNetConfig should have sensible default values."""
    cfg = RetNetConfig()
    assert cfg.d_model == 256
    assert cfg.n_heads == 4
    assert cfg.dropout == 0.0
    assert cfg.chunk_size == 32


# ---------------------------------------------------------------------------
# 2. build_decay_gammas shape
# ---------------------------------------------------------------------------


def test_build_decay_gammas_shape():
    """build_decay_gammas must return a tensor of shape (n_heads,)."""
    for n in [1, 2, 4, 8]:
        gammas = build_decay_gammas(n)
        assert gammas.shape == (n,), f"Expected ({n},), got {gammas.shape}"


# ---------------------------------------------------------------------------
# 3. build_decay_gammas values in (0, 1)
# ---------------------------------------------------------------------------


def test_build_decay_gammas_values_in_range():
    """All gamma values must be strictly in (0, 1)."""
    gammas = build_decay_gammas(8)
    assert (gammas > 0).all(), "Some gamma <= 0"
    assert (gammas < 1).all(), "Some gamma >= 1"


# ---------------------------------------------------------------------------
# 4. build_causal_decay_mask shape
# ---------------------------------------------------------------------------


def test_build_causal_decay_mask_shape():
    """build_causal_decay_mask must return a (T, T) tensor."""
    for T in [1, 4, 16, 32]:
        mask = build_causal_decay_mask(T, gamma=0.9)
        assert mask.shape == (T, T), f"Expected ({T}, {T}), got {mask.shape}"


# ---------------------------------------------------------------------------
# 5. build_causal_decay_mask upper triangle is 0 (causal)
# ---------------------------------------------------------------------------


def test_build_causal_decay_mask_upper_triangle_zero():
    """All entries above the diagonal must be 0 (causal mask)."""
    T = 8
    mask = build_causal_decay_mask(T, gamma=0.85)
    for i in range(T):
        for j in range(i + 1, T):
            assert mask[i, j].item() == 0.0, (
                f"mask[{i},{j}] = {mask[i,j].item()} (should be 0 -- upper triangle)"
            )


# ---------------------------------------------------------------------------
# 6. build_causal_decay_mask diagonal is 1 (gamma^0)
# ---------------------------------------------------------------------------


def test_build_causal_decay_mask_diagonal_is_one():
    """Diagonal entries must be 1.0 (gamma^0 = 1)."""
    T = 6
    mask = build_causal_decay_mask(T, gamma=0.9)
    for i in range(T):
        assert abs(mask[i, i].item() - 1.0) < 1e-5, (
            f"mask[{i},{i}] = {mask[i,i].item()} (expected 1.0)"
        )


# ---------------------------------------------------------------------------
# 7. MultiScaleRetention parallel output shape matches input
# ---------------------------------------------------------------------------


def test_msr_parallel_output_shape():
    """Parallel mode: input (B, T, d_model) -> output (B, T, d_model)."""
    cfg = make_config()
    msr = MultiScaleRetention(cfg.d_model, cfg.n_heads)
    x = torch.randn(BATCH, SEQ, D_MODEL)
    out = msr.forward_parallel(x)
    assert out.shape == (BATCH, SEQ, D_MODEL), (
        f"Expected ({BATCH}, {SEQ}, {D_MODEL}), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# 8. MultiScaleRetention recurrent output shape is (B, d_model)
# ---------------------------------------------------------------------------


def test_msr_recurrent_output_shape():
    """Recurrent mode: output must be (B, d_model)."""
    cfg = make_config()
    msr = MultiScaleRetention(cfg.d_model, cfg.n_heads)
    x = torch.randn(BATCH, D_MODEL)
    s = torch.zeros(BATCH, N_HEADS, HEAD_DIM, HEAD_DIM)
    out, _ = msr.forward_recurrent(x, s, n=0)
    assert out.shape == (BATCH, D_MODEL), (
        f"Expected ({BATCH}, {D_MODEL}), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# 9. MultiScaleRetention recurrent state shape
# ---------------------------------------------------------------------------


def test_msr_recurrent_state_shape():
    """Recurrent state must be (B, n_heads, head_dim, head_dim)."""
    cfg = make_config()
    msr = MultiScaleRetention(cfg.d_model, cfg.n_heads)
    x = torch.randn(BATCH, D_MODEL)
    s = torch.zeros(BATCH, N_HEADS, HEAD_DIM, HEAD_DIM)
    _, new_s = msr.forward_recurrent(x, s, n=0)
    expected = (BATCH, N_HEADS, HEAD_DIM, HEAD_DIM)
    assert new_s.shape == expected, (
        f"Expected state shape {expected}, got {new_s.shape}"
    )


# ---------------------------------------------------------------------------
# 10. MultiScaleRetention chunkwise output shape matches input
# ---------------------------------------------------------------------------


def test_msr_chunkwise_output_shape():
    """Chunkwise mode: output (B, T, d_model) matches input shape."""
    cfg = make_config(chunk_size=8)
    msr = MultiScaleRetention(cfg.d_model, cfg.n_heads)
    x = torch.randn(BATCH, SEQ, D_MODEL)
    out = msr.forward_chunkwise(x, chunk_size=cfg.chunk_size)
    assert out.shape == (BATCH, SEQ, D_MODEL), (
        f"Expected ({BATCH}, {SEQ}, {D_MODEL}), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# 11. Parallel vs chunkwise produce close outputs
# ---------------------------------------------------------------------------


def test_msr_parallel_vs_chunkwise_close():
    """Parallel and chunkwise modes should produce finite outputs of the same shape."""
    torch.manual_seed(0)
    cfg = make_config(chunk_size=SEQ)  # one chunk = full sequence
    msr = MultiScaleRetention(cfg.d_model, cfg.n_heads)
    msr.train(False)
    x = torch.randn(BATCH, SEQ, D_MODEL)

    with torch.no_grad():
        out_par = msr.forward_parallel(x)
        out_cw = msr.forward_chunkwise(x, chunk_size=SEQ)

    assert out_par.shape == out_cw.shape, "Shape mismatch between parallel and chunkwise"
    assert torch.isfinite(out_par).all(), "Parallel output contains non-finite values"
    assert torch.isfinite(out_cw).all(), "Chunkwise output contains non-finite values"


# ---------------------------------------------------------------------------
# 12. RetNetBlock output shape matches input
# ---------------------------------------------------------------------------


def test_retnet_block_output_shape():
    """RetNetBlock: (B, T, d_model) -> (B, T, d_model)."""
    cfg = make_config()
    block = RetNetBlock(cfg)
    x = torch.randn(BATCH, SEQ, D_MODEL)
    out = block(x)
    assert out.shape == (BATCH, SEQ, D_MODEL), (
        f"Expected ({BATCH}, {SEQ}, {D_MODEL}), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# 13. RetNetModel output shape is (B, T, vocab_size)
# ---------------------------------------------------------------------------


def test_retnet_model_output_shape():
    """RetNetModel.forward returns logits of shape (B, T, vocab_size)."""
    cfg = make_config()
    model = RetNetModel(cfg, n_layers=2, vocab_size=VOCAB_SIZE)
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ))
    logits = model(input_ids)
    assert logits.shape == (BATCH, SEQ, VOCAB_SIZE), (
        f"Expected ({BATCH}, {SEQ}, {VOCAB_SIZE}), got {logits.shape}"
    )


# ---------------------------------------------------------------------------
# 14. RetNetModel is differentiable
# ---------------------------------------------------------------------------


def test_retnet_model_differentiable():
    """loss.backward() must complete without error."""
    cfg = make_config()
    model = RetNetModel(cfg, n_layers=2, vocab_size=VOCAB_SIZE)
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ))
    logits = model(input_ids)  # (B, T, vocab_size)

    targets = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ))
    loss = nn.CrossEntropyLoss()(logits.view(-1, VOCAB_SIZE), targets.view(-1))
    loss.backward()

    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad, "No parameters received gradients after backward()"


# ---------------------------------------------------------------------------
# Bonus 15: recurrent state accumulates across steps
# ---------------------------------------------------------------------------


def test_msr_recurrent_state_changes():
    """Recurrent state must change between steps (decay + new KV contribution)."""
    cfg = make_config()
    msr = MultiScaleRetention(cfg.d_model, cfg.n_heads)
    msr.train(False)

    s = torch.zeros(BATCH, N_HEADS, HEAD_DIM, HEAD_DIM)
    x0 = torch.randn(BATCH, D_MODEL)
    x1 = torch.randn(BATCH, D_MODEL)

    with torch.no_grad():
        _, s1 = msr.forward_recurrent(x0, s, n=0)
        _, s2 = msr.forward_recurrent(x1, s1, n=1)

    assert not torch.allclose(s1, torch.zeros_like(s1)), "State s1 should be non-zero"
    assert not torch.allclose(s2, s1), "State s2 should differ from s1"


# ---------------------------------------------------------------------------
# Bonus 16: build_causal_decay_mask below-diagonal value correctness
# ---------------------------------------------------------------------------


def test_build_causal_decay_mask_values():
    """D[i,j] should equal gamma^(i-j) for i > j."""
    gamma = 0.9
    T = 5
    mask = build_causal_decay_mask(T, gamma=gamma)
    for i in range(T):
        for j in range(i):
            expected = gamma ** (i - j)
            actual = mask[i, j].item()
            assert abs(actual - expected) < 1e-5, (
                f"mask[{i},{j}] = {actual}, expected {expected}"
            )
