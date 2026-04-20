"""Tests for ngram_deduplication."""

from __future__ import annotations

import pytest

from src.data.ngram_deduplication import (
    NgramDeduplicationToolkit,
    cluster_duplicates,
    ngram_jaccard,
)


def test_identical_strings():
    assert ngram_jaccard("abcdef", "abcdef", n=3) == 1.0


def test_disjoint():
    assert ngram_jaccard("aaaaaa", "bbbbbb", n=3) < 0.2


def test_cluster_merges_near_dupes():
    docs = ["hello world today", "hello world today!", "goodbye"]
    groups = cluster_duplicates(docs, n=4, threshold=0.7)
    flat = [i for g in groups for i in g]
    assert sorted(flat) == [0, 1, 2]
    assert any(len(g) > 1 for g in groups)


def test_toolkit_delegates():
    g = NgramDeduplicationToolkit.cluster_duplicates(["alpha", "alpha"], n=3, threshold=1.0)
    assert len(g) == 1 and len(g[0]) == 2


def test_invalid_n():
    with pytest.raises(ValueError):
        ngram_jaccard("a", "b", n=0)


def test_type_errors():
    with pytest.raises(TypeError):
        ngram_jaccard("a", 1)  # type: ignore[arg-type]


def test_cluster_bad_threshold():
    with pytest.raises(ValueError):
        cluster_duplicates(["a"], threshold=1.5)


def test_cluster_bad_doc_type():
    with pytest.raises(TypeError):
        cluster_duplicates(["a", None])  # type: ignore[list-item]


def test_empty_strings():
    assert ngram_jaccard("", "", n=3) == 1.0
