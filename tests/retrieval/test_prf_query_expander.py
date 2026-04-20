"""Tests for prf_query_expander."""

from __future__ import annotations

import pytest

from src.retrieval.prf_query_expander import PRFQueryExpander


def test_expands_with_shared_rare_term():
    ex = PRFQueryExpander()
    q = "python sort"
    docs = ["python uses timsort algorithm", "java has sort too"]
    r = ex.expand(q, docs, num_terms=2)
    assert len(r.added_terms) <= 2
    assert r.query.startswith("python sort")


def test_empty_docs_zero_terms():
    r = PRFQueryExpander().expand("hello", [], num_terms=0)
    assert r.query == "hello"
    assert r.added_terms == ()


def test_empty_docs_positive_terms_raises():
    with pytest.raises(RuntimeError):
        PRFQueryExpander().expand("x", [], num_terms=1)


def test_excludes_query_terms():
    ex = PRFQueryExpander()
    r = ex.expand("alpha", ["alpha beta gamma delta beta"], num_terms=4)
    assert "alpha" not in r.added_terms


def test_invalid_num_terms():
    with pytest.raises(ValueError):
        PRFQueryExpander().expand("a", ["b"], num_terms=-1)


def test_num_terms_cap():
    with pytest.raises(ValueError):
        PRFQueryExpander().expand("a", ["b"], num_terms=500)


def test_type_errors():
    with pytest.raises(TypeError):
        PRFQueryExpander().expand(1, ["a"])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        PRFQueryExpander().expand("a", "not-list")  # type: ignore[arg-type]


def test_non_str_document_raises():
    with pytest.raises(TypeError):
        PRFQueryExpander().expand("q", ["ok", 3], num_terms=1)  # type: ignore[list-item]


def test_min_doc_freq_filters():
    ex = PRFQueryExpander(min_doc_freq=2)
    docs = ["bar alpha", "bar beta", "gamma only once"]
    r = ex.expand("seed", docs, num_terms=6)
    assert "bar" in r.added_terms


def test_adversarial_unicode():
    ex = PRFQueryExpander()
    r = ex.expand("café", ["café résumé naïve"], num_terms=3)
    assert isinstance(r.query, str)
