"""Tests for RWKV (Receptance Weighted Key Value) linear recurrence layer.

Reference: Peng et al. 2023, "RWKV: Reinventing RNNs for the Transformer Era".

All tests use small configs to keep CI fast:
  d_model=32, d_ff=64, n_heads=1, batch=2, seq_len=8
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.model.rwkv import (
    RWKVTimeMixing,
    RWKVChannelMixing,
    RWKVBlock,
    RWKVLayer,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

D_MODEL  = 32
D_FF     = 64
N_HEADS  = 1
BATCH    = 2
SEQ_LEN  = 8


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def time_mix():
    return RWKVTimeMixing(D_MODEL, n_heads=N_HEADS)


@pytest.fixture
def channel_mix():
    return RWKVChannelMixing(D_MODEL, D_FF)


@pytest.fixture
def block():
    return RWKVBlock(D_MODEL, D_FF, n_heads=N_HEADS)


@pytest.fixture
def layer():
    return RWKVLayer(D_MODEL, D_FF, n_layers=2, n_heads=N_HEADS)


@pytest.fixture
def x():
    torch.manual_seed(42)
    return torch.randn(BATCH, SEQ_LEN, D_MODEL)


# ---------------------------------------------------------------------------
# 1. RWKVTimeMixing forward, state=None → output shape (B, T, d_model)
# ---------------------------------------------------------------------------

def test_time_mixing_output_shape(time_mix, x):
    out, _ = time_mix(x, state=None)
    assert out.shape == (BATCH, SEQ_LEN, D_MODEL), (
        f"Expected ({BATCH}, {SEQ_LEN}, {D_MODEL}), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# 2. RWKVTimeMixing returns new_state of shape (B, d_model)
# ---------------------------------------------------------------------------

def test_time_mixing_state_shape(time_mix, x):
    _, new_state = time_mix(x, state=None)
    assert new_state.shape == (BATCH, D_MODEL), (
        f"Expected ({BATCH}, {D_MODEL}), got {new_state.shape}"
    )


# ---------------------------------------------------------------------------
# 3. RWKVTimeMixing recurrent mode: step-by-step produces same sequence shape
#    as parallel mode.
# ---------------------------------------------------------------------------

def test_time_mixing_recurrent_vs_parallel_shape(time_mix):
    torch.manual_seed(0)
    x_full = torch.randn(BATCH, SEQ_LEN, D_MODEL)

    # Parallel (all at once)
    parallel_out, _ = time_mix(x_full, state=None)

    # Recurrent (one token at a time)
    state = None
    recurrent_outs = []
    for t in range(SEQ_LEN):
        x_t = x_full[:, t : t + 1, :]  # (B, 1, D)
        out_t, state = time_mix(x_t, state=state)
        recurrent_outs.append(out_t)
    recurrent_out = torch.cat(recurrent_outs, dim=1)  # (B, T, D)

    assert recurrent_out.shape == parallel_out.shape, (
        f"Shape mismatch: recurrent {recurrent_out.shape} vs parallel {parallel_out.shape}"
    )


# ---------------------------------------------------------------------------
# 4. RWKVChannelMixing forward returns correct shape
# ---------------------------------------------------------------------------

def test_channel_mixing_output_shape(channel_mix, x):
    out, _ = channel_mix(x, state=None)
    assert out.shape == (BATCH, SEQ_LEN, D_MODEL), (
        f"Expected ({BATCH}, {SEQ_LEN}, {D_MODEL}), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# 5. RWKVChannelMixing new_state has correct shape
# ---------------------------------------------------------------------------

def test_channel_mixing_state_shape(channel_mix, x):
    _, new_state = channel_mix(x, state=None)
    assert new_state.shape == (BATCH, D_MODEL), (
        f"Expected ({BATCH}, {D_MODEL}), got {new_state.shape}"
    )


# ---------------------------------------------------------------------------
# 6. RWKVBlock forward, no state → (output, time_state, channel_state)
# ---------------------------------------------------------------------------

def test_block_forward_no_state(block, x):
    out, ts, cs = block(x, time_state=None, channel_state=None)
    assert out.shape == (BATCH, SEQ_LEN, D_MODEL), (
        f"Block output shape mismatch: {out.shape}"
    )
    assert ts.shape == (BATCH, D_MODEL), f"time_state shape: {ts.shape}"
    assert cs.shape == (BATCH, D_MODEL), f"channel_state shape: {cs.shape}"


# ---------------------------------------------------------------------------
# 7. RWKVBlock with explicit states runs without error
# ---------------------------------------------------------------------------

def test_block_with_explicit_states(block, x):
    ts_init = torch.zeros(BATCH, D_MODEL)
    cs_init = torch.zeros(BATCH, D_MODEL)
    out, ts, cs = block(x, time_state=ts_init, channel_state=cs_init)
    assert out.shape == (BATCH, SEQ_LEN, D_MODEL)
    assert ts.shape  == (BATCH, D_MODEL)
    assert cs.shape  == (BATCH, D_MODEL)


# ---------------------------------------------------------------------------
# 8. RWKVLayer forward, states=None → output (B, T, d_model)
# ---------------------------------------------------------------------------

def test_layer_output_shape(layer, x):
    out, _ = layer(x, states=None)
    assert out.shape == (BATCH, SEQ_LEN, D_MODEL), (
        f"Expected ({BATCH}, {SEQ_LEN}, {D_MODEL}), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# 9. RWKVLayer returns list of states, length == n_layers
# ---------------------------------------------------------------------------

def test_layer_states_length(layer, x):
    _, new_states = layer(x, states=None)
    assert len(new_states) == layer.n_layers, (
        f"Expected {layer.n_layers} states, got {len(new_states)}"
    )
    for i, (ts, cs) in enumerate(new_states):
        assert ts.shape == (BATCH, D_MODEL), f"Layer {i} time_state shape: {ts.shape}"
        assert cs.shape == (BATCH, D_MODEL), f"Layer {i} channel_state shape: {cs.shape}"


# ---------------------------------------------------------------------------
# 10. Recurrent single-step: RWKVBlock with T=1 runs correctly
# ---------------------------------------------------------------------------

def test_block_single_step(block):
    torch.manual_seed(7)
    x_single = torch.randn(BATCH, 1, D_MODEL)
    out, ts, cs = block(x_single, time_state=None, channel_state=None)
    assert out.shape == (BATCH, 1, D_MODEL), f"Single-step output shape: {out.shape}"
    assert ts.shape  == (BATCH, D_MODEL)
    assert cs.shape  == (BATCH, D_MODEL)


# ---------------------------------------------------------------------------
# 11. Time decay params (w) have correct shape after init
# ---------------------------------------------------------------------------

def test_time_decay_param_shape(time_mix):
    assert time_mix.w_log.shape == (D_MODEL,), (
        f"w_log shape expected ({D_MODEL},), got {time_mix.w_log.shape}"
    )
    # Actual decay values must be in (0, 1) — strict exponential decay.
    w = torch.exp(-torch.exp(time_mix.w_log))
    assert (w > 0).all() and (w < 1).all(), "Decay values must be in (0, 1)"


# ---------------------------------------------------------------------------
# 12. Sequential single-token inference matches parallel shape
# ---------------------------------------------------------------------------

def test_sequential_single_token_shape(time_mix):
    torch.manual_seed(3)
    x_single = torch.randn(BATCH, 1, D_MODEL)
    out_seq, state_seq = time_mix(x_single, state=None)

    x_parallel = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    out_par, _ = time_mix(x_parallel, state=None)

    # Both should have the same number of dimensions and last two dims consistent.
    assert out_seq.shape == (BATCH, 1, D_MODEL), f"seq shape: {out_seq.shape}"
    assert out_par.shape == (BATCH, SEQ_LEN, D_MODEL), f"par shape: {out_par.shape}"


# ---------------------------------------------------------------------------
# 13. RWKVLayer gradients flow (loss.backward() no error)
# ---------------------------------------------------------------------------

def test_layer_gradients_flow(layer, x):
    x_grad = x.clone().requires_grad_(True)
    out, _ = layer(x_grad, states=None)
    loss = out.sum()
    loss.backward()
    assert x_grad.grad is not None, "Gradient did not flow back to input"
    assert not torch.isnan(x_grad.grad).any(), "NaN in gradients"


# ---------------------------------------------------------------------------
# 14. Different seq lengths work: T=1, T=4, T=16
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seq_len", [1, 4, 16])
def test_different_seq_lengths(block, seq_len):
    torch.manual_seed(seq_len)
    x_var = torch.randn(BATCH, seq_len, D_MODEL)
    out, ts, cs = block(x_var, time_state=None, channel_state=None)
    assert out.shape == (BATCH, seq_len, D_MODEL), (
        f"seq_len={seq_len}: expected ({BATCH}, {seq_len}, {D_MODEL}), got {out.shape}"
    )
    assert ts.shape == (BATCH, D_MODEL)
    assert cs.shape == (BATCH, D_MODEL)
