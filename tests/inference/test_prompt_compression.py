"""Tests for src/inference/prompt_compression.py.

Uses a lightweight mock model so tests run quickly without loading
Aurelius weights.  The mock model contains a single nn.MultiheadAttention
layer so that _collect_attention_weights can capture real attention tensors.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.inference.prompt_compression import (
    AttentionBasedCompressor,
    SelectiveContextCompressor,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

VOCAB_SIZE = 64
EMBED_DIM = 32
N_HEADS = 4
SEQ_LEN = 20


class _MockModel(nn.Module):
    """Tiny transformer-like model with a real MHA layer for hook capture."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.attn = nn.MultiheadAttention(
            embed_dim=EMBED_DIM,
            num_heads=N_HEADS,
            batch_first=True,
        )
        self.lm_head = nn.Linear(EMBED_DIM, VOCAB_SIZE)

    def forward(self, input_ids: torch.Tensor):
        # input_ids: (B, seq_len)
        x = self.embedding(input_ids)  # (B, seq_len, embed_dim)
        # need_weights=True so hooks capture (B, seq_len, seq_len) avg weights
        x, _ = self.attn(x, x, x, need_weights=True, average_attn_weights=True)
        logits = self.lm_head(x)  # (B, seq_len, vocab_size)
        return logits


@pytest.fixture(scope="module")
def mock_model() -> _MockModel:
    torch.manual_seed(0)
    m = _MockModel()
    m.train(False)
    return m


@pytest.fixture(scope="module")
def input_ids() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))


# ---------------------------------------------------------------------------
# AttentionBasedCompressor tests
# ---------------------------------------------------------------------------


class TestAttentionBasedCompressor:
    def test_score_tokens_shape(self, mock_model, input_ids):
        """score_tokens returns a 1-D tensor of length seq_len."""
        compressor = AttentionBasedCompressor(mock_model, keep_ratio=0.5)
        scores = compressor.score_tokens(input_ids)
        assert scores.shape == (SEQ_LEN,), f"Expected shape ({SEQ_LEN},), got {scores.shape}"

    def test_score_tokens_nonneg(self, mock_model, input_ids):
        """All importance scores must be >= 0."""
        compressor = AttentionBasedCompressor(mock_model, keep_ratio=0.5)
        scores = compressor.score_tokens(input_ids)
        assert (scores >= 0).all(), "Found negative importance scores"

    def test_compress_output_shorter(self, mock_model, input_ids):
        """Compressed sequence is shorter than the original."""
        compressor = AttentionBasedCompressor(mock_model, keep_ratio=0.5)
        compressed_ids, kept_indices = compressor.compress(input_ids)
        assert compressed_ids.shape[1] < SEQ_LEN, (
            f"Compressed length {compressed_ids.shape[1]} is not < {SEQ_LEN}"
        )

    def test_compress_keeps_ratio(self, mock_model, input_ids):
        """n_kept is approximately keep_ratio * seq_len (within plus or minus 1)."""
        keep_ratio = 0.6
        compressor = AttentionBasedCompressor(mock_model, keep_ratio=keep_ratio)
        compressed_ids, kept_indices = compressor.compress(input_ids)
        expected = int(keep_ratio * SEQ_LEN)
        assert abs(kept_indices.shape[0] - expected) <= 1, (
            f"Kept {kept_indices.shape[0]} tokens, expected ~{expected}"
        )

    def test_kept_indices_valid(self, mock_model, input_ids):
        """All kept_indices are in [0, seq_len)."""
        compressor = AttentionBasedCompressor(mock_model, keep_ratio=0.5)
        _, kept_indices = compressor.compress(input_ids)
        assert (kept_indices >= 0).all() and (kept_indices < SEQ_LEN).all(), (
            f"kept_indices out of range [0, {SEQ_LEN}): {kept_indices}"
        )

    def test_uniform_strategy(self, mock_model, input_ids):
        """Uniform strategy gives equal scores for every token."""
        compressor = AttentionBasedCompressor(mock_model, keep_ratio=0.5, strategy="uniform")
        scores = compressor.score_tokens(input_ids)
        expected = 1.0 / SEQ_LEN
        assert torch.allclose(scores, torch.full_like(scores, expected)), (
            "Uniform strategy should give identical scores for all tokens"
        )


# ---------------------------------------------------------------------------
# SelectiveContextCompressor
# ---------------------------------------------------------------------------


class TestSelectiveContextCompressor:
    def test_selective_context_compress(self, mock_model, input_ids):
        """Output is shorter than input (with keep_ratio < 1 and >= 2 chunks)."""
        # chunk_size=8 -> 20 tokens -> 3 chunks; keep_ratio=0.5 -> keep 1-2 chunks
        compressor = SelectiveContextCompressor(mock_model, chunk_size=8, keep_ratio=0.5)
        compressed = compressor.compress(input_ids)
        assert compressed.shape[1] < SEQ_LEN, (
            f"Compressed length {compressed.shape[1]} is not < {SEQ_LEN}"
        )
        assert compressed.shape[0] == 1, "Batch dimension must be preserved"


# ---------------------------------------------------------------------------
# compression_ratio
# ---------------------------------------------------------------------------


class TestCompressionRatio:
    def test_compression_ratio_calculation(self, mock_model):
        """compression_ratio(100, 50) == 0.5."""
        compressor = AttentionBasedCompressor(mock_model)
        ratio = compressor.compression_ratio(100, 50)
        assert ratio == pytest.approx(0.5), f"Expected 0.5, got {ratio}"
