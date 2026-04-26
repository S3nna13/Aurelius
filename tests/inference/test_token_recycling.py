"""Tests for src/inference/token_recycling.py.

Covers:
  1.  TokenRecyclingConfig defaults
  2.  RecyclingBuffer add / get_recent
  3.  Draft after "A B C A B" should suggest "C" following "A B"
  4.  draft() returns None on empty buffer
  5.  draft() returns correct shape after warm-up
  6.  update() correctly builds adjacency table
  7.  get_bigram_candidates returns results sorted by count (descending)
  8.  batch_draft returns correct shape (n_candidates, n_tokens)
  9.  build_recycling_draft_tree produces a dict with correct structure
  10. Repeated patterns increase candidate confidence (higher counts)
  11. buffer_size limits memory usage (old tokens evicted)
  12. Works with arbitrary integer token ids
"""

from __future__ import annotations

import torch

from src.inference.token_recycling import (
    RecyclingBuffer,
    TokenRecycler,
    TokenRecyclingConfig,
    build_recycling_draft_tree,
)

# ---------------------------------------------------------------------------
# Test 1 — TokenRecyclingConfig defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = TokenRecyclingConfig()
    assert cfg.max_draft_tokens == 5
    assert cfg.buffer_size == 512
    assert cfg.use_bigram is True
    assert cfg.min_count == 1


# ---------------------------------------------------------------------------
# Test 2 — RecyclingBuffer add and get_recent
# ---------------------------------------------------------------------------


def test_buffer_add_and_get_recent():
    buf = RecyclingBuffer(max_size=10)
    for tok in [1, 2, 3, 4, 5]:
        buf.add(tok)

    assert buf.get_recent(3) == [3, 4, 5]
    assert buf.get_recent(5) == [1, 2, 3, 4, 5]
    # Requesting more than available returns all
    assert buf.get_recent(100) == [1, 2, 3, 4, 5]
    assert len(buf) == 5


# ---------------------------------------------------------------------------
# Test 3 — After seeing "A B C A B", draft after "A B" should suggest "C"
# ---------------------------------------------------------------------------


def test_bigram_suggests_correct_continuation():
    # Token ids: A=10, B=20, C=30
    A, B, C = 10, 20, 30
    buf = RecyclingBuffer(max_size=128)
    for tok in [A, B, C, A, B]:
        buf.add(tok)

    # Bigram (A, B) should have C as top candidate
    candidates = buf.get_bigram_candidates(A, B, top_k=1)
    assert len(candidates) == 1
    next_tok, count = candidates[0]
    assert next_tok == C
    assert count >= 1


# ---------------------------------------------------------------------------
# Test 4 — draft() returns None on empty buffer
# ---------------------------------------------------------------------------


def test_draft_returns_none_on_empty_buffer():
    recycler = TokenRecycler(vocab_size=1000)
    context = torch.tensor([5, 10, 15], dtype=torch.long)
    result = recycler.draft(context, n_tokens=3)
    assert result is None


# ---------------------------------------------------------------------------
# Test 5 — draft() returns correct shape after warm-up
# ---------------------------------------------------------------------------


def test_draft_shape_after_warmup():
    recycler = TokenRecycler(vocab_size=1000, max_draft_tokens=5, buffer_size=256)
    # Warm up with a repeating pattern so transitions exist
    tokens = torch.tensor([1, 2, 3, 4, 5, 1, 2, 3, 4, 5], dtype=torch.long)
    recycler.update(tokens)

    context = torch.tensor([1, 2], dtype=torch.long)
    draft = recycler.draft(context, n_tokens=5)
    assert draft is not None
    assert draft.shape == (5,)
    assert draft.dtype == torch.long


# ---------------------------------------------------------------------------
# Test 6 — update() correctly builds adjacency table
# ---------------------------------------------------------------------------


def test_update_builds_adjacency_table():
    recycler = TokenRecycler(vocab_size=500)
    tokens = torch.tensor([7, 8, 9], dtype=torch.long)
    recycler.update(tokens)

    # After [7, 8, 9]: adjacency[7] should contain 8; adjacency[8] should contain 9
    assert 7 in recycler.adjacency
    assert recycler.adjacency[7][8] >= 1
    assert 8 in recycler.adjacency
    assert recycler.adjacency[8][9] >= 1


# ---------------------------------------------------------------------------
# Test 7 — get_bigram_candidates returns sorted by count (descending)
# ---------------------------------------------------------------------------


def test_bigram_candidates_sorted_by_count():
    buf = RecyclingBuffer(max_size=256)
    # Feed sequence: [1, 2, 3, 1, 2, 4, 1, 2, 3, 1, 2, 3]
    # Bigram (1, 2) is followed by: 3 three times, 4 once
    for seq in [[1, 2, 3], [1, 2, 4], [1, 2, 3], [1, 2, 3]]:
        for tok in seq:
            buf.add(tok)

    cands = buf.get_bigram_candidates(1, 2, top_k=5)
    assert len(cands) >= 2
    # First element should be (3, count>=3) since 3 follows (1,2) most
    assert cands[0][0] == 3
    assert cands[0][1] >= 3
    # Counts should be non-increasing
    counts = [c for _, c in cands]
    assert counts == sorted(counts, reverse=True)


# ---------------------------------------------------------------------------
# Test 8 — batch_draft returns correct shape (n_candidates, n_tokens)
# ---------------------------------------------------------------------------


def test_batch_draft_shape():
    recycler = TokenRecycler(vocab_size=1000, buffer_size=256)
    tokens = torch.tensor([1, 2, 3, 4, 5, 1, 2, 6, 1, 2, 7, 1, 2, 8], dtype=torch.long)
    recycler.update(tokens)

    context = torch.tensor([1, 2], dtype=torch.long)
    result = recycler.batch_draft(context, n_candidates=4, n_tokens=5)
    assert result.shape == (4, 5)
    assert result.dtype == torch.long


# ---------------------------------------------------------------------------
# Test 9 — build_recycling_draft_tree produces dict with correct structure
# ---------------------------------------------------------------------------


def test_build_recycling_draft_tree_structure():
    buf = RecyclingBuffer(max_size=128)
    for tok in [10, 20, 30, 10, 20, 40]:
        buf.add(tok)

    tree = build_recycling_draft_tree(buf, context=[10, 20], depth=2)
    # Result must be a dict
    assert isinstance(tree, dict)
    # The bigram (10, 20) was followed by 30 and 40 — at least one should appear
    assert len(tree) >= 1
    # Values must also be dicts (subtrees)
    for key, subtree in tree.items():
        assert isinstance(key, int)
        assert isinstance(subtree, dict)


# ---------------------------------------------------------------------------
# Test 10 — Repeated patterns increase candidate confidence
# ---------------------------------------------------------------------------


def test_repeated_patterns_increase_count():
    buf = RecyclingBuffer(max_size=256)

    # Feed bigram (5, 6) → 7 multiple times
    for _ in range(5):
        buf.add(5)
        buf.add(6)
        buf.add(7)

    cands = buf.get_bigram_candidates(5, 6, top_k=1)
    assert len(cands) == 1
    assert cands[0][0] == 7
    assert cands[0][1] >= 4  # should have been seen multiple times


# ---------------------------------------------------------------------------
# Test 11 — buffer_size limits memory usage (old tokens evicted)
# ---------------------------------------------------------------------------


def test_buffer_size_limits_memory():
    max_size = 10
    buf = RecyclingBuffer(max_size=max_size)

    # Add 20 tokens
    for tok in range(20):
        buf.add(tok)

    # Buffer should hold at most max_size tokens
    assert len(buf) == max_size
    # The recent 10 tokens should be [10..19]
    recent = buf.get_recent(max_size)
    assert recent == list(range(10, 20))


# ---------------------------------------------------------------------------
# Test 12 — Works with arbitrary integer token ids
# ---------------------------------------------------------------------------


def test_works_with_arbitrary_integer_token_ids():
    recycler = TokenRecycler(vocab_size=200_000)
    # Use large, sparse token ids typical in real vocabularies
    tokens = torch.tensor([50256, 12345, 99999, 50256, 12345, 99999], dtype=torch.long)
    recycler.update(tokens)

    context = torch.tensor([50256, 12345], dtype=torch.long)
    draft = recycler.draft(context, n_tokens=3)
    assert draft is not None
    assert draft.shape == (3,)
    # First draft token should be 99999 (most common continuation of bigram)
    assert int(draft[0].item()) == 99999
