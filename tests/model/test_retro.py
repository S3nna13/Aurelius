"""Tests for src/model/retro.py — RETRO (Borgeaud et al., 2021).

Tiny config used throughout:
    d_model=64, n_heads=4, n_layers=4, chunk_size=8
    K=2, neighbor_len=8, n_retrieval_layers=[2]  (only layer index 2 is RETRO)
"""

from __future__ import annotations

import pytest
import torch

from src.model.retro import (
    RETRODecoder,
    RETROEncoder,
    RETROCrossChunkAttention,
    RETROBlock,
    StandardBlock,
)

# ---------------------------------------------------------------------------
# Tiny config constants (paper notation: L=chunk_size, K=neighbors per chunk)
# ---------------------------------------------------------------------------
D = 64          # d_model
NH = 4          # n_heads
NL = 4          # n_layers
CS = 8          # chunk_size  L (paper)
K = 2           # neighbors per chunk
NL_RETR = 8    # neighbor_len
RETR_LAYERS = [2]  # only layer 2 is a RETRO block

B = 2           # batch size
T = 16          # sequence length (= 2 * chunk_size — two chunks)


def make_decoder(**kwargs) -> RETRODecoder:
    defaults = dict(
        d_model=D,
        n_heads=NH,
        n_layers=NL,
        chunk_size=CS,
        n_retrieval_layers=RETR_LAYERS,
    )
    defaults.update(kwargs)
    return RETRODecoder(**defaults)


def make_neighbors(batch: int = B, n_chunks: int = T // CS) -> torch.Tensor:
    """Random neighbor embeddings: (B, n_chunks, K, neighbor_len, d_model)."""
    return torch.randn(batch, n_chunks, K, NL_RETR, D)


# ---------------------------------------------------------------------------
# Test 1: Output shape matches input (B, T, d_model)
# ---------------------------------------------------------------------------
def test_output_shape_no_retrieval():
    model = make_decoder()
    x = torch.randn(B, T, D)
    out = model(x)
    assert out.shape == (B, T, D), f"Expected {(B, T, D)}, got {out.shape}"


def test_output_shape_with_retrieval():
    model = make_decoder()
    x = torch.randn(B, T, D)
    neighbors = make_neighbors()
    out = model(x, neighbors)
    assert out.shape == (B, T, D), f"Expected {(B, T, D)}, got {out.shape}"


# ---------------------------------------------------------------------------
# Test 2: Gradient flow — backward produces finite grads on all params
# ---------------------------------------------------------------------------
def test_gradient_flow():
    model = make_decoder()
    x = torch.randn(B, T, D, requires_grad=True)
    neighbors = make_neighbors()
    out = model(x, neighbors)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "Input gradient is None"
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient is None for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"


# ---------------------------------------------------------------------------
# Test 3: Determinism under fixed seed
# ---------------------------------------------------------------------------
def test_determinism():
    def run(seed: int):
        torch.manual_seed(seed)
        model = make_decoder()
        model.train(False)
        torch.manual_seed(seed + 1)
        x = torch.randn(B, T, D)
        neighbors = make_neighbors()
        with torch.no_grad():
            return model(x, neighbors)

    out1 = run(42)
    out2 = run(42)
    assert torch.allclose(out1, out2), "Outputs differ under the same seed"


# ---------------------------------------------------------------------------
# Test 4: neighbors=None -> pure transformer, no crash
# ---------------------------------------------------------------------------
def test_no_retrieval_no_crash():
    model = make_decoder()
    x = torch.randn(B, T, D)
    out = model(x, neighbors=None)
    assert out.shape == (B, T, D)


# ---------------------------------------------------------------------------
# Test 5: With neighbors -> output differs from no-retrieval output
# ---------------------------------------------------------------------------
def test_retrieval_changes_output():
    torch.manual_seed(0)
    model = make_decoder()
    model.train(False)
    x = torch.randn(B, T, D)
    neighbors = make_neighbors()

    with torch.no_grad():
        out_plain = model(x, neighbors=None)
        out_retro = model(x, neighbors)

    assert not torch.allclose(out_plain, out_retro), (
        "Retrieval output should differ from no-retrieval output"
    )


# ---------------------------------------------------------------------------
# Test 6: T not divisible by chunk_size -> raises ValueError
# ---------------------------------------------------------------------------
def test_non_divisible_T_raises():
    model = make_decoder()
    T_bad = CS + 3   # not divisible by chunk_size
    x = torch.randn(B, T_bad, D)
    n_chunks_fake = (T_bad // CS) + 1
    neighbors = make_neighbors(n_chunks=n_chunks_fake)

    with pytest.raises(ValueError, match="divisible by chunk_size"):
        model(x, neighbors)


# ---------------------------------------------------------------------------
# Test 7: K=1 (single neighbor) works
# ---------------------------------------------------------------------------
def test_k1_single_neighbor():
    model = make_decoder()
    x = torch.randn(B, T, D)
    n_chunks = T // CS
    neighbors_k1 = torch.randn(B, n_chunks, 1, NL_RETR, D)
    out = model(x, neighbors_k1)
    assert out.shape == (B, T, D)


# ---------------------------------------------------------------------------
# Test 8: K=2, n_chunks=T//chunk_size -> correct shape handling
# ---------------------------------------------------------------------------
def test_k2_full_chunks():
    model = make_decoder()
    n_chunks = T // CS
    x = torch.randn(B, T, D)
    neighbors = make_neighbors(n_chunks=n_chunks)
    assert neighbors.shape == (B, n_chunks, K, NL_RETR, D)
    out = model(x, neighbors)
    assert out.shape == (B, T, D)


# ---------------------------------------------------------------------------
# Test 9: No NaN/Inf on zeros input
# ---------------------------------------------------------------------------
def test_no_nan_on_zero_input():
    model = make_decoder()
    x = torch.zeros(B, T, D)
    neighbors = torch.zeros(B, T // CS, K, NL_RETR, D)
    out = model(x, neighbors)
    assert torch.isfinite(out).all(), "NaN or Inf detected on zero input"


# ---------------------------------------------------------------------------
# Test 10: No NaN/Inf on large inputs
# ---------------------------------------------------------------------------
def test_no_nan_on_large_input():
    model = make_decoder()
    x = torch.randn(B, T, D) * 100.0
    neighbors = torch.randn(B, T // CS, K, NL_RETR, D) * 100.0
    out = model(x, neighbors)
    assert torch.isfinite(out).all(), "NaN or Inf detected on large input"


# ---------------------------------------------------------------------------
# Test 11: RETROEncoder produces shape (B * n_chunks * K, neighbor_len, d_model)
# ---------------------------------------------------------------------------
def test_encoder_output_shape():
    encoder = RETROEncoder(d_model=D, n_heads=NH, n_layers=2)
    n_chunks = T // CS
    flat_batch = B * n_chunks * K
    x = torch.randn(flat_batch, NL_RETR, D)
    out = encoder(x)
    assert out.shape == (flat_batch, NL_RETR, D), (
        f"Expected {(flat_batch, NL_RETR, D)}, got {out.shape}"
    )


# ---------------------------------------------------------------------------
# Test 12: CCA cross-attention output shape matches input chunk shape
# ---------------------------------------------------------------------------
def test_cca_output_shape():
    cca = RETROCrossChunkAttention(d_model=D, n_heads=NH)
    # H_i: (B, chunk_size, d_model)  — queries
    H_i = torch.randn(B, CS, D)
    # E_i: (B, K * neighbor_len, d_model)  — keys and values
    E_i = torch.randn(B, K * NL_RETR, D)
    out = cca(H_i, E_i)
    assert out.shape == (B, CS, D), (
        f"CCA output shape mismatch: expected {(B, CS, D)}, got {out.shape}"
    )


# ---------------------------------------------------------------------------
# Test 13: Long sequence T = 4 * chunk_size -> 4 chunks processed correctly
# ---------------------------------------------------------------------------
def test_long_sequence_four_chunks():
    T_long = 4 * CS
    n_chunks = T_long // CS
    model = make_decoder()
    x = torch.randn(B, T_long, D)
    neighbors = torch.randn(B, n_chunks, K, NL_RETR, D)
    out = model(x, neighbors)
    assert out.shape == (B, T_long, D), (
        f"Expected {(B, T_long, D)}, got {out.shape}"
    )


# ---------------------------------------------------------------------------
# Test 14: Non-retrieval layers ignore neighbors entirely (StandardBlock has no CCA)
# ---------------------------------------------------------------------------
def test_standard_block_has_no_cca():
    block = StandardBlock(d_model=D, n_heads=NH)
    assert not hasattr(block, "cca"), "StandardBlock must not have a CCA sub-layer"

    x = torch.randn(B, T, D)
    out = block(x)
    assert out.shape == (B, T, D)


def test_retro_layers_assignment():
    """Verify that the correct layer indices are RETROBlock vs StandardBlock."""
    model = make_decoder()
    for i, layer in enumerate(model.layers):
        if i in RETR_LAYERS:
            assert isinstance(layer, RETROBlock), (
                f"Layer {i} should be RETROBlock, got {type(layer)}"
            )
        else:
            assert isinstance(layer, StandardBlock), (
                f"Layer {i} should be StandardBlock, got {type(layer)}"
            )
