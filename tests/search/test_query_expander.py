"""Tests for src/search/query_expander.py — 20 tests."""

from __future__ import annotations

import pytest

from src.search.bm25_index import BM25Index
from src.search.query_expander import (
    _MAX_EXPANSIONS,
    _MAX_QUERY_LEN,
    _MAX_SYNONYMS,
    QueryExpander,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fresh_expander(**kwargs) -> QueryExpander:
    return QueryExpander(**kwargs)


# ---------------------------------------------------------------------------
# split_identifier
# ---------------------------------------------------------------------------


def test_split_identifier_camel_case():
    qe = fresh_expander()
    result = qe.split_identifier("camelCase")
    assert result == ["camel", "case"]


def test_split_identifier_snake_case():
    qe = fresh_expander()
    result = qe.split_identifier("snake_case_name")
    assert result == ["snake", "case", "name"]


def test_split_identifier_mixed():
    qe = fresh_expander()
    result = qe.split_identifier("camelCase_with_underscore")
    assert "camel" in result
    assert "case" in result
    assert "with" in result
    assert "underscore" in result


def test_split_identifier_all_caps_abbreviation():
    qe = fresh_expander()
    # "HTTPRequest" → ["http", "request"]
    result = qe.split_identifier("HTTPRequest")
    assert "http" in result
    assert "request" in result


# ---------------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------------


def test_tokenize_plain_words_lowercased():
    qe = fresh_expander()
    result = qe.tokenize("hello world")
    assert result == ["hello", "world"]


def test_tokenize_camel_case_split():
    qe = fresh_expander()
    result = qe.tokenize("myFunction")
    assert "my" in result
    assert "function" in result


# ---------------------------------------------------------------------------
# expand
# ---------------------------------------------------------------------------


def test_expand_original_always_in_variants():
    qe = fresh_expander()
    eq = qe.expand("search query")
    assert "search query" in eq.variants


def test_expand_joined_token_string_added():
    qe = fresh_expander()
    # camelCase query → tokens joined should appear as a variant
    eq = qe.expand("myFunc")
    assert "my func" in eq.variants


def test_expand_synonym_substitution_adds_variant():
    qe = QueryExpander(synonyms={"search": ["find", "lookup"]})
    eq = qe.expand("search query")
    variant_words = set(" ".join(eq.variants).split())
    assert "find" in variant_words or any("find" in v for v in eq.variants)


def test_expand_synonyms_applied_count():
    qe = QueryExpander(synonyms={"search": ["find", "lookup"]})
    eq = qe.expand("search query")
    assert eq.synonyms_applied >= 1


def test_expand_max_expansions_caps_variants():
    qe = QueryExpander(max_expansions=3)
    # Long query with many tokens
    eq = qe.expand("alpha beta gamma delta epsilon zeta eta theta")
    assert len(eq.variants) <= 3


def test_expand_query_too_long_raises():
    qe = fresh_expander()
    with pytest.raises(ValueError, match="query exceeds"):
        qe.expand("q" * (_MAX_QUERY_LEN + 1))


# ---------------------------------------------------------------------------
# add_synonyms
# ---------------------------------------------------------------------------


def test_add_synonyms_lookup_works():
    qe = fresh_expander()
    qe.add_synonyms({"function": ["method", "procedure"]})
    eq = qe.expand("function call")
    assert any("method" in v for v in eq.variants)


def test_add_synonyms_overflow_raises():
    qe = fresh_expander()
    # Fill up to the limit with many entries
    big_dict = {f"word{i}": [f"syn{i}"] for i in range(_MAX_SYNONYMS + 1)}
    with pytest.raises(ValueError, match="synonym count would exceed"):
        qe.add_synonyms(big_dict)


# ---------------------------------------------------------------------------
# max_expansions constructor guard
# ---------------------------------------------------------------------------


def test_max_expansions_too_large_raises():
    with pytest.raises(ValueError, match="max_expansions exceeds"):
        QueryExpander(max_expansions=_MAX_EXPANSIONS + 1)


# ---------------------------------------------------------------------------
# expand: no synonyms
# ---------------------------------------------------------------------------


def test_expand_no_synonyms_only_token_variants():
    qe = fresh_expander()
    eq = qe.expand("hello world")
    assert eq.synonyms_applied == 0
    # original + individual tokens
    assert "hello" in eq.variants or "world" in eq.variants


# ---------------------------------------------------------------------------
# Adversarial
# ---------------------------------------------------------------------------


def test_expand_empty_query():
    qe = fresh_expander()
    eq = qe.expand("")
    assert eq.tokens == []
    assert eq.original == ""
    assert eq.variants[0] == ""


def test_expand_numbers_only():
    qe = fresh_expander()
    eq = qe.expand("12345")
    assert "12345" in eq.tokens


def test_expand_special_chars_only_alphanum_extracted():
    qe = fresh_expander()
    eq = qe.expand("!@#$% hello &*()")
    # Only "hello" should be extracted as a token
    assert eq.tokens == ["hello"]


# ---------------------------------------------------------------------------
# Round-trip: expand then search in BM25
# ---------------------------------------------------------------------------


def test_expand_then_bm25_search():
    """Expand a camelCase query and verify a variant appears in BM25 results."""
    idx = BM25Index()
    idx.add("doc1", "my function does computation")
    idx.add("doc2", "unrelated text about dogs")

    qe = fresh_expander()
    eq = qe.expand("myFunction")
    # At least one variant should score doc1 higher than doc2
    found = False
    for variant in eq.variants:
        results = idx.query(variant, top_k=2)
        if results and results[0][0] == "doc1":
            found = True
            break
    assert found, f"No variant hit doc1; variants={eq.variants}"
