"""Tests for src/model/hierarchical_attention.py.

Covers:
 1. HierAttnConfig default values
 2. chunk_sequence chunked shape: (B*n_chunks, chunk_size, D)
 3. chunk_sequence handles T not divisible by chunk_size
 4. summarize_chunks mean: output shape (B, n_chunks, D)
 5. summarize_chunks first: output shape (B, n_chunks, D)
 6. summarize_chunks last: output shape (B, n_chunks, D) with correct values
 7. cross_chunk_attention output shape (B, n_chunks, D)
 8. broadcast_chunk_context output shape (B*n_chunks, chunk_size, D)
 9. LocalAttention output shape (B, T, D)
10. HierarchicalAttention output shape (B, T, D)
11. HierarchicalAttention output is finite (no NaN / Inf)
12. HierAttnBlock output shape (B, T, D)
13. HierAttnBlock residual connection changes output
"""

import torch

from src.model.hierarchical_attention import (
    HierarchicalAttention,
    HierAttnBlock,
    HierAttnConfig,
    LocalAttention,
    broadcast_chunk_context,
    chunk_sequence,
    cross_chunk_attention,
    summarize_chunks,
)

# ---------------------------------------------------------------------------
# Shared tiny constants
# ---------------------------------------------------------------------------

D = 16
N_HEADS = 2
CHUNK = 4
B = 2
T = 8  # 2 chunks of size 4

TINY_CFG = HierAttnConfig(d_model=D, n_heads=N_HEADS, chunk_size=CHUNK)


# ---------------------------------------------------------------------------
# 1. test_hierattn_config_defaults
# ---------------------------------------------------------------------------


def test_hierattn_config_defaults():
    """HierAttnConfig instantiates with documented defaults."""
    cfg = HierAttnConfig()
    assert cfg.d_model == 512
    assert cfg.n_heads == 8
    assert cfg.chunk_size == 512
    assert cfg.n_global_tokens == 1
    assert cfg.causal is False


# ---------------------------------------------------------------------------
# 2. test_chunk_sequence_shape
# ---------------------------------------------------------------------------


def test_chunk_sequence_shape():
    """chunk_sequence returns (B*n_chunks, chunk_size, D) and correct n_chunks."""
    x = torch.randn(B, T, D)
    chunked, n_chunks = chunk_sequence(x, CHUNK)
    assert n_chunks == T // CHUNK, f"Expected {T // CHUNK} chunks, got {n_chunks}"
    assert chunked.shape == (B * n_chunks, CHUNK, D), (
        f"Expected ({B * n_chunks}, {CHUNK}, {D}), got {chunked.shape}"
    )


# ---------------------------------------------------------------------------
# 3. test_chunk_sequence_non_divisible
# ---------------------------------------------------------------------------


def test_chunk_sequence_non_divisible():
    """chunk_sequence pads and chunks correctly when T % chunk_size != 0."""
    T_odd = 7  # not divisible by CHUNK=4; needs 1 token of padding
    x = torch.randn(B, T_odd, D)
    chunked, n_chunks = chunk_sequence(x, CHUNK)
    # Ceiling division: ceil(7/4) = 2 chunks
    expected_n_chunks = 2
    assert n_chunks == expected_n_chunks, f"Expected {expected_n_chunks} chunks, got {n_chunks}"
    assert chunked.shape == (B * n_chunks, CHUNK, D), (
        f"Expected ({B * n_chunks}, {CHUNK}, {D}), got {chunked.shape}"
    )
    # Padding amount should be stored on the tensor
    assert hasattr(chunked, "pad"), "chunked should have a .pad attribute"
    assert chunked.pad == 1, f"Expected pad=1, got {chunked.pad}"


# ---------------------------------------------------------------------------
# 4. test_summarize_chunks_mean_shape
# ---------------------------------------------------------------------------


def test_summarize_chunks_mean_shape():
    """summarize_chunks('mean') produces (B, n_chunks, D) after reshape."""
    n_chunks = T // CHUNK
    chunk_outputs = torch.randn(B * n_chunks, CHUNK, D)
    flat = summarize_chunks(chunk_outputs, method="mean")
    assert flat.shape == (B * n_chunks, D), f"Expected ({B * n_chunks}, {D}), got {flat.shape}"
    summaries = flat.view(B, n_chunks, D)
    assert summaries.shape == (B, n_chunks, D), (
        f"Expected ({B}, {n_chunks}, {D}), got {summaries.shape}"
    )


# ---------------------------------------------------------------------------
# 5. test_summarize_chunks_first_shape
# ---------------------------------------------------------------------------


def test_summarize_chunks_first_shape():
    """summarize_chunks('first') produces (B*n_chunks, D)."""
    n_chunks = T // CHUNK
    chunk_outputs = torch.randn(B * n_chunks, CHUNK, D)
    flat = summarize_chunks(chunk_outputs, method="first")
    assert flat.shape == (B * n_chunks, D), f"Expected ({B * n_chunks}, {D}), got {flat.shape}"
    # Verify 'first' really picks the first token
    assert torch.allclose(flat, chunk_outputs[:, 0, :]), (
        "'first' summary should equal the first token of each chunk"
    )


# ---------------------------------------------------------------------------
# 6. test_summarize_chunks_last_shape_and_values
# ---------------------------------------------------------------------------


def test_summarize_chunks_last_shape_and_values():
    """summarize_chunks('last') picks the last token of each chunk."""
    n_chunks = T // CHUNK
    chunk_outputs = torch.randn(B * n_chunks, CHUNK, D)
    flat = summarize_chunks(chunk_outputs, method="last")
    assert flat.shape == (B * n_chunks, D)
    assert torch.allclose(flat, chunk_outputs[:, -1, :]), (
        "'last' summary should equal the last token of each chunk"
    )


# ---------------------------------------------------------------------------
# 7. test_cross_chunk_attention_shape
# ---------------------------------------------------------------------------


def test_cross_chunk_attention_shape():
    """cross_chunk_attention returns (B, n_chunks, D)."""
    n_chunks = T // CHUNK
    summaries = torch.randn(B, n_chunks, D)
    out = cross_chunk_attention(summaries, n_heads=N_HEADS, d_model=D)
    assert out.shape == (B, n_chunks, D), f"Expected ({B}, {n_chunks}, {D}), got {out.shape}"


# ---------------------------------------------------------------------------
# 8. test_broadcast_chunk_context_shape
# ---------------------------------------------------------------------------


def test_broadcast_chunk_context_shape():
    """broadcast_chunk_context returns (B*n_chunks, chunk_size, D)."""
    n_chunks = T // CHUNK
    summaries = torch.randn(B, n_chunks, D)
    chunk_outputs = torch.randn(B * n_chunks, CHUNK, D)
    enriched = broadcast_chunk_context(summaries, chunk_outputs, n_chunks)
    assert enriched.shape == (B * n_chunks, CHUNK, D), (
        f"Expected ({B * n_chunks}, {CHUNK}, {D}), got {enriched.shape}"
    )


# ---------------------------------------------------------------------------
# 9. test_local_attention_output_shape
# ---------------------------------------------------------------------------


def test_local_attention_output_shape():
    """LocalAttention returns (B, T, D)."""
    layer = LocalAttention(d_model=D, n_heads=N_HEADS)
    x = torch.randn(B, T, D)
    with torch.no_grad():
        out = layer(x)
    assert out.shape == (B, T, D), f"Expected ({B}, {T}, {D}), got {out.shape}"


# ---------------------------------------------------------------------------
# 10. test_hierarchical_attention_output_shape
# ---------------------------------------------------------------------------


def test_hierarchical_attention_output_shape():
    """HierarchicalAttention returns (B, T, D)."""
    model = HierarchicalAttention(TINY_CFG)
    x = torch.randn(B, T, D)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (B, T, D), f"Expected ({B}, {T}, {D}), got {out.shape}"


# ---------------------------------------------------------------------------
# 11. test_hierarchical_attention_output_finite
# ---------------------------------------------------------------------------


def test_hierarchical_attention_output_finite():
    """HierarchicalAttention output contains no NaN or Inf values."""
    model = HierarchicalAttention(TINY_CFG)
    x = torch.randn(B, T, D)
    with torch.no_grad():
        out = model(x)
    assert torch.isfinite(out).all(), "Output should be fully finite (no NaN or Inf)"


# ---------------------------------------------------------------------------
# 12. test_hierattn_block_output_shape
# ---------------------------------------------------------------------------


def test_hierattn_block_output_shape():
    """HierAttnBlock returns (B, T, D)."""
    block = HierAttnBlock(TINY_CFG)
    x = torch.randn(B, T, D)
    with torch.no_grad():
        out = block(x)
    assert out.shape == (B, T, D), f"Expected ({B}, {T}, {D}), got {out.shape}"


# ---------------------------------------------------------------------------
# 13. test_hierattn_block_residual_changes_output
# ---------------------------------------------------------------------------


def test_hierattn_block_residual_changes_output():
    """HierAttnBlock output differs from its input (residual is non-trivial)."""
    torch.manual_seed(0)
    block = HierAttnBlock(TINY_CFG)
    x = torch.randn(B, T, D)
    with torch.no_grad():
        out = block(x)
    # The block adds attention output to x; unless attention is zero, they differ.
    assert not torch.allclose(out, x), (
        "Block output should differ from input (residual + attention should modify values)"
    )
