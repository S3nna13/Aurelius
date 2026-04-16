"""Tests for the Griffin hybrid RG-LRU + Local Attention model.

Reference: De et al., 2024 -- "Griffin: Mixing Gated Linear Recurrences with
Local Attention for Efficient Language Models". https://arxiv.org/abs/2402.19427

Tiny config: d_model=64, n_heads=4, head_dim=16, d_ff=128,
             n_layers=3, vocab_size=256, local_window=8, lru_per_attn=2
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.model.griffin import (
    GriffinConfig,
    GriffinModel,
    GriffinBlock,
    LocalSlidingWindowAttention,
    RGLRULayer,
)


# ---------------------------------------------------------------------------
# Shared fixture -- tiny config for fast tests
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg() -> GriffinConfig:
    return GriffinConfig(
        d_model=64,
        n_heads=4,
        head_dim=16,
        d_ff=128,
        n_layers=3,
        vocab_size=256,
        local_window=8,
        lru_per_attn=2,
    )


@pytest.fixture
def rglru(cfg: GriffinConfig) -> RGLRULayer:
    torch.manual_seed(0)
    return RGLRULayer(cfg)


@pytest.fixture
def lswa(cfg: GriffinConfig) -> LocalSlidingWindowAttention:
    torch.manual_seed(0)
    return LocalSlidingWindowAttention(cfg)


@pytest.fixture
def model(cfg: GriffinConfig) -> GriffinModel:
    torch.manual_seed(42)
    return GriffinModel(cfg)


# ---------------------------------------------------------------------------
# Test 1 -- RGLRULayer output shape (B, T, d_model)
# ---------------------------------------------------------------------------

def test_rglru_output_shape(rglru: RGLRULayer, cfg: GriffinConfig):
    """RGLRULayer must return output of shape (B, T, d_model)."""
    B, T = 2, 16
    x = torch.randn(B, T, cfg.d_model)
    out, _ = rglru(x)
    assert out.shape == (B, T, cfg.d_model), (
        f"Expected ({B}, {T}, {cfg.d_model}), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# Test 2 -- RGLRULayer returns new_state of shape (B, d_model)
# ---------------------------------------------------------------------------

def test_rglru_state_shape(rglru: RGLRULayer, cfg: GriffinConfig):
    """RGLRULayer must return new_state of shape (B, d_model)."""
    B, T = 3, 10
    x = torch.randn(B, T, cfg.d_model)
    _, new_state = rglru(x)
    assert new_state.shape == (B, cfg.d_model), (
        f"Expected ({B}, {cfg.d_model}), got {new_state.shape}"
    )


# ---------------------------------------------------------------------------
# Test 3 -- RGLRULayer alpha values are in (0, 1)
# ---------------------------------------------------------------------------

def test_rglru_alpha_bounded(rglru: RGLRULayer):
    """All alpha values (per-channel decay) must be strictly in (0, 1)."""
    alpha = rglru.alpha
    assert alpha.shape == (rglru.gate_r.out_features,), "alpha shape mismatch"
    assert torch.all(alpha > 0), "alpha contains values <= 0"
    assert torch.all(alpha < 1), "alpha contains values >= 1"


# ---------------------------------------------------------------------------
# Test 4 -- Sequential step-by-step matches batch forward (recurrent consistency)
# ---------------------------------------------------------------------------

def test_rglru_recurrent_consistency(rglru: RGLRULayer, cfg: GriffinConfig):
    """Processing one token at a time must produce the same output as processing
    the whole sequence at once (recurrent consistency)."""
    torch.manual_seed(7)
    rglru.eval()
    B, T = 2, 8
    x = torch.randn(B, T, cfg.d_model)

    # Batch forward
    with torch.no_grad():
        batch_out, _ = rglru(x)

    # Step-by-step forward
    state = None
    step_outputs = []
    with torch.no_grad():
        for t in range(T):
            x_t = x[:, t : t + 1, :]  # (B, 1, d_model)
            out_t, state = rglru(x_t, state)
            step_outputs.append(out_t)

    step_out = torch.cat(step_outputs, dim=1)  # (B, T, d_model)

    assert torch.allclose(batch_out, step_out, atol=1e-5), (
        f"Max diff between batch and step-by-step: {(batch_out - step_out).abs().max():.6f}"
    )


# ---------------------------------------------------------------------------
# Test 5 -- LocalSlidingWindowAttention output shape correct
# ---------------------------------------------------------------------------

def test_lswa_output_shape(lswa: LocalSlidingWindowAttention, cfg: GriffinConfig):
    """LocalSlidingWindowAttention must return output of shape (B, T, d_model)."""
    B, T = 2, 20
    x = torch.randn(B, T, cfg.d_model)
    out, _ = lswa(x)
    assert out.shape == (B, T, cfg.d_model), (
        f"Expected ({B}, {T}, {cfg.d_model}), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# Test 6 -- Tokens beyond window don't affect position 0 output
# ---------------------------------------------------------------------------

def test_lswa_window_isolation(lswa: LocalSlidingWindowAttention, cfg: GriffinConfig):
    """Changing tokens beyond the window size must not alter position-0 output.

    Position 0 can only attend to itself (causal + window=8).
    Perturbing positions >= window should have zero effect on position-0 output.
    """
    torch.manual_seed(3)
    lswa.eval()
    B, T = 1, 32
    x = torch.randn(B, T, cfg.d_model)

    with torch.no_grad():
        out1, _ = lswa(x)

    # Perturb tokens beyond the window (positions >= window)
    x2 = x.clone()
    x2[:, cfg.local_window:, :] = torch.randn_like(x2[:, cfg.local_window:, :]) * 10.0

    with torch.no_grad():
        out2, _ = lswa(x2)

    # Position 0 output must be unchanged
    assert torch.allclose(out1[:, 0, :], out2[:, 0, :], atol=1e-5), (
        f"Position-0 output changed after perturbing tokens beyond window. "
        f"Max diff: {(out1[:, 0, :] - out2[:, 0, :]).abs().max():.6f}"
    )


# ---------------------------------------------------------------------------
# Test 7 -- GriffinModel output logits shape (B, T, vocab_size)
# ---------------------------------------------------------------------------

def test_model_logits_shape(model: GriffinModel, cfg: GriffinConfig):
    """GriffinModel must return logits of shape (B, T, vocab_size)."""
    B, T = 2, 12
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))
    logits, _ = model(input_ids)
    assert logits.shape == (B, T, cfg.vocab_size), (
        f"Expected ({B}, {T}, {cfg.vocab_size}), got {logits.shape}"
    )


# ---------------------------------------------------------------------------
# Test 8 -- GriffinModel returns list of states
# ---------------------------------------------------------------------------

def test_model_returns_states(model: GriffinModel, cfg: GriffinConfig):
    """GriffinModel must return a list of states (one per block).

    LRU block states must be tensors of shape (B, d_model).
    Attn block states are KV-cache tuples (not checked for shape here).
    """
    B, T = 2, 8
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))
    logits, states = model(input_ids)

    n_lru = sum(1 for t in model.block_types if t == "lru")
    n_blocks = len(model.block_types)

    assert isinstance(states, list), "states must be a list"
    # states has one entry per block
    assert len(states) == n_blocks, (
        f"Expected {n_blocks} states (one per block), got {len(states)}"
    )
    # LRU states must be tensors of shape (B, d_model)
    lru_states = [s for s, t in zip(states, model.block_types) if t == "lru"]
    assert len(lru_states) == n_lru, f"Expected {n_lru} LRU states"
    for s in lru_states:
        assert isinstance(s, torch.Tensor), "LRU state must be a tensor"
        assert s.shape == (B, cfg.d_model), (
            f"Each LRU state must have shape ({B}, {cfg.d_model}), got {s.shape}"
        )


# ---------------------------------------------------------------------------
# Test 9 -- No NaN/Inf in forward pass
# ---------------------------------------------------------------------------

def test_no_nan_inf(model: GriffinModel, cfg: GriffinConfig):
    """Forward pass must produce finite (no NaN/Inf) logits and LRU states."""
    torch.manual_seed(99)
    B, T = 2, 16
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))
    logits, states = model(input_ids)

    assert torch.isfinite(logits).all(), "logits contain NaN or Inf"
    # Check LRU states (tensors); attn states are KV-cache tuples
    for s, bt in zip(states, model.block_types):
        if bt == "lru" and s is not None:
            assert torch.isfinite(s).all(), "LRU state contains NaN or Inf"
        elif bt == "attn" and s is not None:
            K, V = s
            assert torch.isfinite(K).all(), "Attn K cache contains NaN or Inf"
            assert torch.isfinite(V).all(), "Attn V cache contains NaN or Inf"


# ---------------------------------------------------------------------------
# Test 10 -- loss.backward() succeeds
# ---------------------------------------------------------------------------

def test_backward_pass(model: GriffinModel, cfg: GriffinConfig):
    """loss.backward() must complete without error and produce non-None grads."""
    torch.manual_seed(5)
    B, T = 2, 8
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))
    labels    = torch.randint(0, cfg.vocab_size, (B, T))

    logits, _ = model(input_ids)
    loss = nn.CrossEntropyLoss()(logits.view(-1, cfg.vocab_size), labels.view(-1))
    loss.backward()

    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad, "No parameter received a gradient after backward()"


# ---------------------------------------------------------------------------
# Test 11 -- Different input sequences give different outputs
# ---------------------------------------------------------------------------

def test_different_inputs_different_outputs(model: GriffinModel, cfg: GriffinConfig):
    """Two different input sequences must produce different logits."""
    torch.manual_seed(11)
    model.eval()
    B, T = 1, 10
    ids_a = torch.randint(0, cfg.vocab_size, (B, T))
    ids_b = torch.randint(0, cfg.vocab_size, (B, T))
    # Make sure they are actually different
    while torch.equal(ids_a, ids_b):
        ids_b = torch.randint(0, cfg.vocab_size, (B, T))

    with torch.no_grad():
        logits_a, _ = model(ids_a)
        logits_b, _ = model(ids_b)

    assert not torch.allclose(logits_a, logits_b, atol=1e-6), (
        "Different inputs produced identical logits -- model may be broken."
    )


# ---------------------------------------------------------------------------
# Test 12 -- Stateful generation: token-by-token matches batch forward
# ---------------------------------------------------------------------------

def test_stateful_generation_matches_batch(model: GriffinModel, cfg: GriffinConfig):
    """Feeding one token at a time with carried states must yield the same logits
    as a single batch forward over the whole sequence."""
    torch.manual_seed(21)
    model.eval()
    B, T = 2, 6
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))

    # Batch forward
    with torch.no_grad():
        batch_logits, _ = model(input_ids)

    # Token-by-token forward with state threading
    states = None
    step_logits = []
    with torch.no_grad():
        for t in range(T):
            tok = input_ids[:, t : t + 1]          # (B, 1)
            logits_t, states = model(tok, states)   # (B, 1, vocab_size)
            step_logits.append(logits_t)

    step_logits_cat = torch.cat(step_logits, dim=1)  # (B, T, vocab_size)

    assert torch.allclose(batch_logits, step_logits_cat, atol=1e-4), (
        f"Max diff between batch and token-by-token logits: "
        f"{(batch_logits - step_logits_cat).abs().max():.6f}"
    )
