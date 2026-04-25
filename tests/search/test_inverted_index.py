"""Tests for src/search/inverted_index.py  (≥28 tests)."""

import pytest
from src.search.inverted_index import InvertedIndex, Posting, INVERTED_INDEX_REGISTRY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_index(*docs) -> InvertedIndex:
    """Build an InvertedIndex from (doc_id, tokens) pairs."""
    idx = InvertedIndex()
    for doc_id, tokens in docs:
        idx.add_document(doc_id, tokens)
    return idx


# ---------------------------------------------------------------------------
# REGISTRY
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registry_exists(self):
        assert INVERTED_INDEX_REGISTRY is not None

    def test_registry_default_key(self):
        assert "default" in INVERTED_INDEX_REGISTRY

    def test_registry_default_is_inverted_index_class(self):
        assert INVERTED_INDEX_REGISTRY["default"] is InvertedIndex


# ---------------------------------------------------------------------------
# Empty index
# ---------------------------------------------------------------------------

class TestEmptyIndex:
    def test_lookup_returns_empty_list_for_unknown_term(self):
        idx = InvertedIndex()
        assert idx.lookup("foo") == []

    def test_search_empty_query_returns_empty(self):
        idx = InvertedIndex()
        assert idx.search([]) == []

    def test_doc_count_empty(self):
        idx = InvertedIndex()
        assert idx.doc_count() == 0

    def test_vocab_size_empty(self):
        idx = InvertedIndex()
        assert idx.vocab_size() == 0

    def test_search_unknown_term_returns_empty(self):
        idx = InvertedIndex()
        assert idx.search(["ghost"]) == []


# ---------------------------------------------------------------------------
# Single document
# ---------------------------------------------------------------------------

class TestSingleDocument:
    def test_add_document_and_lookup(self):
        idx = _make_index((1, ["hello", "world"]))
        postings = idx.lookup("hello")
        assert len(postings) == 1
        assert postings[0].doc_id == 1

    def test_term_freq_single_occurrence(self):
        idx = _make_index((1, ["hello", "world"]))
        assert idx.lookup("hello")[0].term_freq == 1

    def test_term_freq_duplicate_token_increments(self):
        idx = _make_index((1, ["cat", "cat", "cat"]))
        assert idx.lookup("cat")[0].term_freq == 3

    def test_term_freq_two_distinct_tokens(self):
        idx = _make_index((1, ["a", "b", "a"]))
        postings_a = idx.lookup("a")
        assert postings_a[0].term_freq == 2
        postings_b = idx.lookup("b")
        assert postings_b[0].term_freq == 1

    def test_posting_is_frozen(self):
        p = Posting(doc_id=1, term_freq=2)
        with pytest.raises((AttributeError, TypeError)):
            p.doc_id = 99  # type: ignore[misc]

    def test_doc_count_single_doc(self):
        idx = _make_index((1, ["x"]))
        assert idx.doc_count() == 1

    def test_vocab_size_single_doc(self):
        idx = _make_index((1, ["x", "y", "z"]))
        assert idx.vocab_size() == 3

    def test_search_single_token_returns_doc(self):
        idx = _make_index((7, ["python"]))
        assert idx.search(["python"]) == [7]

    def test_lookup_unknown_after_add(self):
        idx = _make_index((1, ["alpha"]))
        assert idx.lookup("beta") == []


# ---------------------------------------------------------------------------
# Multiple documents
# ---------------------------------------------------------------------------

class TestMultipleDocuments:
    def test_lookup_multiple_docs_sorted_ascending(self):
        idx = _make_index((3, ["cat"]), (1, ["cat"]), (2, ["cat"]))
        doc_ids = [p.doc_id for p in idx.lookup("cat")]
        assert doc_ids == [1, 2, 3]

    def test_posting_sort_by_doc_id(self):
        idx = _make_index((10, ["x"]), (5, ["x"]), (1, ["x"]))
        postings = idx.lookup("x")
        assert [p.doc_id for p in postings] == [1, 5, 10]

    def test_and_search_intersection(self):
        idx = _make_index(
            (1, ["python", "fast"]),
            (2, ["python"]),
            (3, ["fast"]),
        )
        result = idx.search(["python", "fast"])
        assert result == [1]

    def test_and_search_no_intersection(self):
        idx = _make_index((1, ["alpha"]), (2, ["beta"]))
        assert idx.search(["alpha", "beta"]) == []

    def test_and_search_all_docs_match(self):
        idx = _make_index((1, ["x", "y"]), (2, ["x", "y"]))
        assert idx.search(["x", "y"]) == [1, 2]

    def test_doc_count_multiple_docs(self):
        idx = _make_index((1, ["a"]), (2, ["b"]), (3, ["c"]))
        assert idx.doc_count() == 3

    def test_doc_count_shared_terms(self):
        idx = _make_index((1, ["a"]), (2, ["a"]))
        assert idx.doc_count() == 2

    def test_vocab_size_shared_terms(self):
        idx = _make_index((1, ["a", "b"]), (2, ["b", "c"]))
        assert idx.vocab_size() == 3

    def test_term_freq_per_doc_is_independent(self):
        idx = _make_index((1, ["z", "z"]), (2, ["z"]))
        postings = {p.doc_id: p.term_freq for p in idx.lookup("z")}
        assert postings[1] == 2
        assert postings[2] == 1

    def test_search_single_token_multiple_matches_sorted(self):
        idx = _make_index((3, ["go"]), (1, ["go"]), (2, ["go"]))
        assert idx.search(["go"]) == [1, 2, 3]

    def test_search_unknown_token_in_multi_token_query(self):
        idx = _make_index((1, ["real"]))
        assert idx.search(["real", "ghost"]) == []
