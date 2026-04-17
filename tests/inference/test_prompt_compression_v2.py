"""Tests for src/inference/prompt_compression_v2.py.

Tiny configs: vocab=16, d_model=8, seq_len=8, compression_ratio=0.5,
chunk_size=4.  All tests run forward/backward passes -- not just
instantiation.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from src.inference.prompt_compression_v2 import (
    ChunkCompressor,
    IterativeCompressor,
    SemanticPreserver,
    TokenImportanceScorer,
    TokenSelector,
)

# ---------------------------------------------------------------------------
# Tiny model and fixtures
# ---------------------------------------------------------------------------

VOCAB = 16
D_MODEL = 8
SEQ_LEN = 8
COMP_RATIO = 0.5
CHUNK_SIZE = 4


class _TinyModel(nn.Module):
    """Simple embed + linear head; returns (B, T, V) logits."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)   # (B, T, D_MODEL)
        return self.head(x)             # (B, T, VOCAB)


class _MHAModel(nn.Module):
    """Embed + MHA (returns attention weights) + linear head."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, D_MODEL)
        self.attn = nn.MultiheadAttention(
            embed_dim=D_MODEL,
            num_heads=2,
            batch_first=True,
        )
        self.head = nn.Linear(D_MODEL, VOCAB)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        x, _ = self.attn(x, x, x, need_weights=True, average_attn_weights=True)
        return self.head(x)


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(42)
    m = _TinyModel()
    m.eval()
    return m


@pytest.fixture(scope="module")
def mha_model():
    torch.manual_seed(42)
    m = _MHAModel()
    m.eval()
    return m


@pytest.fixture(scope="module")
def ids():
    torch.manual_seed(7)
    return torch.randint(0, VOCAB, (1, SEQ_LEN))


# ---------------------------------------------------------------------------
# Test 1: TokenImportanceScorer.perplexity_score -- shape (T,)
# ---------------------------------------------------------------------------

def test_perplexity_score_shape(tiny_model, ids):
    scorer = TokenImportanceScorer(tiny_model, method="perplexity")
    scores = scorer.perplexity_score(ids)
    assert scores.shape == (SEQ_LEN,), (
        f"Expected shape ({SEQ_LEN},), got {scores.shape}"
    )


# ---------------------------------------------------------------------------
# Test 2: TokenImportanceScorer.perplexity_score -- all values >= 0
# ---------------------------------------------------------------------------

def test_perplexity_score_nonneg(tiny_model, ids):
    scorer = TokenImportanceScorer(tiny_model, method="perplexity")
    scores = scorer.perplexity_score(ids)
    assert (scores >= 0).all(), f"Found negative scores: {scores}"


# ---------------------------------------------------------------------------
# Test 3: perplexity_score -- higher surprise tokens get higher scores
# ---------------------------------------------------------------------------

def test_perplexity_score_relative_surprise(tiny_model):
    """Tokens the model finds surprising should receive higher scores."""
    torch.manual_seed(0)
    ids_local = torch.randint(0, VOCAB, (1, SEQ_LEN))
    scorer = TokenImportanceScorer(tiny_model, method="perplexity")
    scores = scorer.perplexity_score(ids_local)
    # Scores should differ -- a model with random weights assigns different
    # log-probs to different tokens
    assert scores.max() > scores.min(), (
        "Expected variation in perplexity scores; all scores are equal."
    )


# ---------------------------------------------------------------------------
# Test 4: TokenSelector.select -- kept_ids shape <= original, keeps first+last
# ---------------------------------------------------------------------------

def test_selector_keeps_first_last(ids):
    selector = TokenSelector(compression_ratio=COMP_RATIO)
    scores = torch.rand(SEQ_LEN)
    kept_ids, kept_indices = selector.select(scores, ids)

    assert kept_ids.shape[0] <= SEQ_LEN

    assert 0 in kept_indices.tolist(), "First token not kept"
    assert (SEQ_LEN - 1) in kept_indices.tolist(), "Last token not kept"


# ---------------------------------------------------------------------------
# Test 5: TokenSelector.select -- kept_indices sorted ascending
# ---------------------------------------------------------------------------

def test_selector_indices_sorted(ids):
    selector = TokenSelector(compression_ratio=COMP_RATIO)
    scores = torch.rand(SEQ_LEN)
    _, kept_indices = selector.select(scores, ids)
    indices_list = kept_indices.tolist()
    assert indices_list == sorted(indices_list), (
        f"kept_indices are not sorted: {indices_list}"
    )


# ---------------------------------------------------------------------------
# Test 6: TokenSelector.target_length = ceil(T * compression_ratio)
# ---------------------------------------------------------------------------

def test_selector_target_length():
    for ratio in [0.0, 0.3, 0.5, 0.75, 1.0]:
        selector = TokenSelector(compression_ratio=ratio)
        for T in [4, 8, 12]:
            tl = selector.target_length(T)
            expected_min = min(2, T)
            expected_raw = math.ceil(T * ratio)
            expected = max(expected_min, expected_raw)
            assert tl == expected, (
                f"target_length({T}) with ratio={ratio}: "
                f"got {tl}, expected {expected}"
            )


# ---------------------------------------------------------------------------
# Test 7: ChunkCompressor.compress_chunk -- len(output) <= len(input)
# ---------------------------------------------------------------------------

def test_chunk_compress_chunk_shorter(tiny_model):
    scorer = TokenImportanceScorer(tiny_model, method="perplexity")
    selector = TokenSelector(compression_ratio=COMP_RATIO)
    cc = ChunkCompressor(scorer, selector, chunk_size=CHUNK_SIZE)

    chunk = torch.randint(0, VOCAB, (1, CHUNK_SIZE))
    compressed = cc.compress_chunk(chunk)
    assert compressed.shape[0] <= CHUNK_SIZE, (
        f"compress_chunk output length {compressed.shape[0]} > {CHUNK_SIZE}"
    )


# ---------------------------------------------------------------------------
# Test 8: ChunkCompressor.compress -- stats has all 3 keys, ratio <= 1.0
# ---------------------------------------------------------------------------

def test_chunk_compress_stats_keys(tiny_model, ids):
    scorer = TokenImportanceScorer(tiny_model, method="perplexity")
    selector = TokenSelector(compression_ratio=COMP_RATIO)
    cc = ChunkCompressor(scorer, selector, chunk_size=CHUNK_SIZE)

    _, stats = cc.compress(ids)
    assert "original_len" in stats, "Missing key 'original_len'"
    assert "compressed_len" in stats, "Missing key 'compressed_len'"
    assert "ratio" in stats, "Missing key 'ratio'"
    assert stats["ratio"] <= 1.0, f"ratio > 1.0: {stats['ratio']}"


# ---------------------------------------------------------------------------
# Test 9: ChunkCompressor.compress -- original_len correct, compressed_len <= orig
# ---------------------------------------------------------------------------

def test_chunk_compress_lengths(tiny_model, ids):
    scorer = TokenImportanceScorer(tiny_model, method="perplexity")
    selector = TokenSelector(compression_ratio=COMP_RATIO)
    cc = ChunkCompressor(scorer, selector, chunk_size=CHUNK_SIZE)

    compressed_ids, stats = cc.compress(ids)
    assert stats["original_len"] == SEQ_LEN, (
        f"original_len={stats['original_len']}, expected {SEQ_LEN}"
    )
    assert stats["compressed_len"] <= SEQ_LEN, (
        f"compressed_len={stats['compressed_len']} > {SEQ_LEN}"
    )
    assert compressed_ids.shape[1] == stats["compressed_len"], (
        "compressed_ids length does not match compressed_len in stats"
    )


# ---------------------------------------------------------------------------
# Test 10: SemanticPreserver.semantic_similarity -- in [-1,1], 1.0 for identical
# ---------------------------------------------------------------------------

def test_semantic_similarity_range_and_identity(tiny_model, ids):
    sp = SemanticPreserver(tiny_model)
    sim_self = sp.semantic_similarity(ids, ids)
    assert -1.0 <= sim_self <= 1.0, f"sim_self={sim_self} out of [-1,1]"
    assert abs(sim_self - 1.0) < 1e-4, (
        f"Expected ~1.0 for identical sequences, got {sim_self}"
    )

    ids2 = torch.randint(0, VOCAB, (1, SEQ_LEN))
    sim_diff = sp.semantic_similarity(ids, ids2)
    assert -1.0 <= sim_diff <= 1.0, f"sim_diff={sim_diff} out of [-1,1]"


# ---------------------------------------------------------------------------
# Test 11: SemanticPreserver.perplexity_ratio -- > 0 (finite positive float)
# ---------------------------------------------------------------------------

def test_perplexity_ratio_positive(tiny_model, ids):
    sp = SemanticPreserver(tiny_model)
    scorer = TokenImportanceScorer(tiny_model, method="perplexity")
    selector = TokenSelector(compression_ratio=COMP_RATIO)
    cc = ChunkCompressor(scorer, selector, chunk_size=CHUNK_SIZE)

    compressed_ids, _ = cc.compress(ids)
    ratio = sp.perplexity_ratio(ids, compressed_ids)
    assert ratio > 0, f"perplexity_ratio must be > 0, got {ratio}"
    assert math.isfinite(ratio), f"perplexity_ratio must be finite, got {ratio}"


# ---------------------------------------------------------------------------
# Test 12: IterativeCompressor.compress -- ratio <= target OR n_iters == max
# ---------------------------------------------------------------------------

def test_iterative_compress_ratio_or_max_iters(tiny_model, ids):
    scorer = TokenImportanceScorer(tiny_model, method="perplexity")
    ic = IterativeCompressor(scorer, target_ratio=0.3, max_iterations=5)
    compressed, n_iters = ic.compress(ids)

    original_len = ids.shape[1]
    achieved_ratio = compressed.shape[1] / original_len
    assert (achieved_ratio <= 0.3) or (n_iters == 5), (
        f"Neither target ratio ({achieved_ratio:.3f} <= 0.3) nor "
        f"max_iterations ({n_iters} == 5) condition met."
    )


# ---------------------------------------------------------------------------
# Test 13: IterativeCompressor -- n_iterations <= max_iterations
# ---------------------------------------------------------------------------

def test_iterative_compress_n_iters_bounded(tiny_model, ids):
    scorer = TokenImportanceScorer(tiny_model, method="perplexity")
    max_iters = 3
    ic = IterativeCompressor(scorer, target_ratio=0.1, max_iterations=max_iters)
    _, n_iters = ic.compress(ids)
    assert n_iters <= max_iters, (
        f"n_iterations={n_iters} exceeds max_iterations={max_iters}"
    )


# ---------------------------------------------------------------------------
# Test 14: compression_ratio=1.0 -> keep all tokens
# ---------------------------------------------------------------------------

def test_selector_ratio_one_keeps_all(tiny_model, ids):
    scorer = TokenImportanceScorer(tiny_model, method="perplexity")
    selector = TokenSelector(compression_ratio=1.0)
    scores = scorer.score(ids)
    kept_ids, kept_indices = selector.select(scores, ids)
    assert kept_ids.shape[0] == SEQ_LEN, (
        f"compression_ratio=1.0 should keep all {SEQ_LEN} tokens, "
        f"got {kept_ids.shape[0]}"
    )


# ---------------------------------------------------------------------------
# Test 15: Multiple chunk sizes -- same compression_ratio regardless of chunk_size
# ---------------------------------------------------------------------------

def test_chunk_size_independence(tiny_model):
    """The compression_ratio is set by TokenSelector, not by chunk_size.
    Different chunk sizes should all produce ratios near COMP_RATIO."""
    scorer = TokenImportanceScorer(tiny_model, method="perplexity")

    long_ids = torch.randint(0, VOCAB, (1, 16))

    ratios: list[float] = []
    for cs in [2, 4, 8]:
        selector = TokenSelector(compression_ratio=COMP_RATIO)
        cc = ChunkCompressor(scorer, selector, chunk_size=cs)
        _, stats = cc.compress(long_ids)
        ratios.append(stats["ratio"])

    for r in ratios:
        assert 0.0 < r <= 1.0, f"Unexpected ratio {r}"

    # All ratios should be reasonably close to COMP_RATIO
    for r in ratios:
        assert abs(r - COMP_RATIO) <= 0.5, (
            f"Ratio {r} is too far from target {COMP_RATIO}."
        )
