"""Unit tests for src/longcontext.context_quality_scorer."""

from __future__ import annotations

import pytest

from src.longcontext.context_quality_scorer import (
    CompactionRecommendation,
    ContextQualityScorer,
)

SCORER = ContextQualityScorer()


# ---------------------------------------------------------------------------
# 1. score_chunk without query
# ---------------------------------------------------------------------------
def test_score_chunk_without_query():
    chunk = "This is a reasonably long chunk of text with some structure."
    score = SCORER.score_chunk(chunk)
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# 2. score_chunk with query overlap
# ---------------------------------------------------------------------------
def test_score_chunk_with_query_overlap():
    chunk = "The quick brown fox jumps over the lazy dog"
    query = "quick fox dog"
    with_query = SCORER.score_chunk(chunk, query)
    without_query = SCORER.score_chunk(chunk)
    assert with_query > without_query


def test_score_chunk_no_overlap():
    chunk = "alpha beta gamma"
    query = "delta epsilon"
    score = SCORER.score_chunk(chunk, query)
    assert score == pytest.approx(SCORER.score_chunk(chunk))


# ---------------------------------------------------------------------------
# 3. structure bonus
# ---------------------------------------------------------------------------
def test_structure_bonus_code_block():
    chunk = "Some text\n```python\nx = 1\n```\nMore text"
    plain = "Some text\npython\nx = 1\nMore text"
    assert SCORER.score_chunk(chunk) > SCORER.score_chunk(plain)


def test_structure_bonus_list():
    chunk = "- item one\n- item two\n- item three"
    plain = "item one item two item three"
    assert SCORER.score_chunk(chunk) > SCORER.score_chunk(plain)


def test_structure_bonus_table():
    chunk = "| col1 | col2 |\n|------|------|\n| a    | b    |"
    plain = "col1 col2 a b"
    assert SCORER.score_chunk(chunk) > SCORER.score_chunk(plain)


# ---------------------------------------------------------------------------
# 4. redundancy penalty
# ---------------------------------------------------------------------------
def test_redundancy_penalty_lowers_score():
    redundant = "repeat " * 50
    varied = " ".join(f"word{i}" for i in range(50))
    assert SCORER.score_chunk(redundant) < SCORER.score_chunk(varied)


# ---------------------------------------------------------------------------
# 5. score_chunks batch
# ---------------------------------------------------------------------------
def test_score_chunks_returns_same_length():
    chunks = ["hello world", "foo bar baz qux", "a longer piece of text here"]
    scores = SCORER.score_chunks(chunks)
    assert len(scores) == len(chunks)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_score_chunks_with_query():
    chunks = ["hello world", "foo bar baz qux"]
    scores = SCORER.score_chunks(chunks, query="foo")
    assert len(scores) == 2
    # second chunk contains "foo"
    assert scores[1] > scores[0]


# ---------------------------------------------------------------------------
# 6. rank_chunks
# ---------------------------------------------------------------------------
def test_rank_chunks_descending():
    chunks = ["short", "medium length chunk here", "a" * 3000]
    ranked = SCORER.rank_chunks(chunks)
    assert len(ranked) == 3
    for i in range(len(ranked) - 1):
        assert ranked[i][1] >= ranked[i + 1][1]
    assert sorted(idx for idx, _ in ranked) == [0, 1, 2]


def test_rank_chunks_with_query():
    chunks = ["apple banana", "cherry date", "banana elderberry"]
    ranked = SCORER.rank_chunks(chunks, query="banana")
    # The two banana chunks should outrank the non-banana chunk.
    indices = [idx for idx, _ in ranked]
    assert indices[0] in (0, 2)
    assert indices[-1] == 1


# ---------------------------------------------------------------------------
# 7. compaction recommendation
# ---------------------------------------------------------------------------
def test_recommend_partitioning():
    chunks = ["good content here with some length", "a", "b", "c"]
    rec = SCORER.recommend(chunks, threshold=0.2)
    assert isinstance(rec, CompactionRecommendation)
    assert set(rec.keep_indices) | set(rec.drop_indices) == set(range(len(chunks)))
    assert len(rec.keep_indices) + len(rec.drop_indices) == len(chunks)


def test_recommend_avg_score():
    chunks = ["hello world", "foo bar"]
    rec = SCORER.recommend(chunks)
    expected = sum(SCORER.score_chunks(chunks)) / len(chunks)
    assert rec.avg_score == pytest.approx(expected)


def test_recommend_empty_chunks():
    rec = SCORER.recommend([])
    assert rec.keep_indices == []
    assert rec.drop_indices == []
    assert rec.avg_score == 0.0


def test_recommend_threshold_zero_keeps_all():
    chunks = ["x", "y"]
    rec = SCORER.recommend(chunks, threshold=0.0)
    assert rec.keep_indices == [0, 1]
    assert rec.drop_indices == []


def test_recommend_threshold_one_drops_all():
    chunks = ["x", "y"]
    rec = SCORER.recommend(chunks, threshold=1.0)
    assert rec.keep_indices == []
    assert rec.drop_indices == [0, 1]


# ---------------------------------------------------------------------------
# 8. edge cases
# ---------------------------------------------------------------------------
def test_empty_chunk_scores_zero():
    assert SCORER.score_chunk("") == 0.0


def test_very_long_chunk_saturates():
    chunk = "word " * 10_000
    score = SCORER.score_chunk(chunk)
    assert 0.0 <= score <= 1.0
    # Density is capped at 1.0, so score should not explode.
    assert score <= 1.0


def test_whitespace_only_chunk():
    assert SCORER.score_chunk("   \n\t  ") == 0.0


# ---------------------------------------------------------------------------
# 9. validation — fail loud
# ---------------------------------------------------------------------------
def test_score_chunk_non_string_raises():
    with pytest.raises(TypeError):
        SCORER.score_chunk(123)  # type: ignore[arg-type]


def test_score_chunk_query_non_string_raises():
    with pytest.raises(TypeError):
        SCORER.score_chunk("ok", query=123)  # type: ignore[arg-type]


def test_score_chunks_non_list_raises():
    with pytest.raises(TypeError):
        SCORER.score_chunks("not a list")  # type: ignore[arg-type]


def test_score_chunks_non_string_element_raises():
    with pytest.raises(TypeError):
        SCORER.score_chunks(["ok", 123])  # type: ignore[list-item]


def test_rank_chunks_non_list_raises():
    with pytest.raises(TypeError):
        SCORER.rank_chunks({"a", "b"})  # type: ignore[arg-type]


def test_recommend_threshold_validation():
    with pytest.raises(ValueError):
        SCORER.recommend([], threshold=-0.1)
    with pytest.raises(ValueError):
        SCORER.recommend([], threshold=1.5)


def test_recommend_threshold_non_numeric_raises():
    with pytest.raises(TypeError):
        SCORER.recommend([], threshold="high")  # type: ignore[arg-type]


def test_recommend_non_list_raises():
    with pytest.raises(TypeError):
        SCORER.recommend("bad")  # type: ignore[arg-type]
