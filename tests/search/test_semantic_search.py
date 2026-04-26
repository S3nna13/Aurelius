"""Tests for src/search/semantic_search.py (~45 tests)."""

from __future__ import annotations

from src.search.semantic_search import (
    SEMANTIC_SEARCH,
    SemanticDocument,
    SemanticSearch,
)

# ---------------------------------------------------------------------------
# SemanticDocument
# ---------------------------------------------------------------------------


class TestSemanticDocument:
    def test_doc_id_field(self):
        d = SemanticDocument(doc_id="d1", text="hello world")
        assert d.doc_id == "d1"

    def test_text_field(self):
        d = SemanticDocument(doc_id="d1", text="hello world")
        assert d.text == "hello world"

    def test_default_metadata(self):
        d = SemanticDocument(doc_id="d1", text="hello")
        assert d.metadata == {}

    def test_custom_metadata(self):
        d = SemanticDocument(doc_id="d1", text="hello", metadata={"key": "val"})
        assert d.metadata["key"] == "val"

    def test_metadata_not_shared(self):
        d1 = SemanticDocument(doc_id="a", text="x")
        d2 = SemanticDocument(doc_id="b", text="y")
        d1.metadata["k"] = 1
        assert "k" not in d2.metadata


# ---------------------------------------------------------------------------
# SemanticSearch
# ---------------------------------------------------------------------------


class TestSemanticSearchAdd:
    def test_add_increases_len(self):
        ss = SemanticSearch()
        assert len(ss) == 0
        ss.add("d1", "hello world")
        assert len(ss) == 1

    def test_add_multiple(self):
        ss = SemanticSearch()
        ss.add("d1", "hello")
        ss.add("d2", "world")
        assert len(ss) == 2

    def test_add_overwrites_existing(self):
        ss = SemanticSearch()
        ss.add("d1", "hello world")
        ss.add("d1", "goodbye world")
        assert len(ss) == 1

    def test_add_with_metadata(self):
        ss = SemanticSearch()
        ss.add("d1", "hello", metadata={"src": "test"})
        assert len(ss) == 1


class TestSemanticSearchQuery:
    def test_query_returns_list(self):
        ss = SemanticSearch()
        ss.add("d1", "python programming language")
        result = ss.query("python")
        assert isinstance(result, list)

    def test_query_returns_tuples(self):
        ss = SemanticSearch()
        ss.add("d1", "python programming language")
        result = ss.query("python")
        assert len(result) > 0
        assert isinstance(result[0], tuple)
        assert len(result[0]) == 2

    def test_query_finds_added_doc(self):
        ss = SemanticSearch()
        ss.add("d1", "neural networks and deep learning")
        result = ss.query("neural")
        doc_ids = [r[0] for r in result]
        assert "d1" in doc_ids

    def test_query_score_positive_for_match(self):
        ss = SemanticSearch()
        ss.add("d1", "transformers attention mechanism")
        result = ss.query("transformers")
        assert len(result) > 0
        assert result[0][1] > 0.0

    def test_query_score_zero_for_no_match(self):
        ss = SemanticSearch()
        ss.add("d1", "completely unrelated text here")
        result = ss.query("xyznonexistenttoken")
        if result:
            assert result[0][1] == 0.0

    def test_query_sorted_descending(self):
        ss = SemanticSearch()
        ss.add("d1", "machine learning models")
        ss.add("d2", "machine learning machine learning machine")
        result = ss.query("machine learning")
        scores = [r[1] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_query_top_k_limits(self):
        ss = SemanticSearch()
        for i in range(10):
            ss.add(f"d{i}", f"document number {i} with some text content")
        result = ss.query("document", top_k=3)
        assert len(result) <= 3

    def test_query_empty_index(self):
        ss = SemanticSearch()
        result = ss.query("python")
        assert result == []

    def test_query_multiple_tokens(self):
        ss = SemanticSearch()
        ss.add("d1", "language models transformers")
        ss.add("d2", "cat and dog pets")
        result = ss.query("language transformers")
        doc_ids = [r[0] for r in result]
        assert "d1" in doc_ids

    def test_query_doc_id_in_result(self):
        ss = SemanticSearch()
        ss.add("myid", "unique content here")
        result = ss.query("unique content")
        assert result[0][0] == "myid"


class TestSemanticSearchRemove:
    def test_remove_known_returns_true(self):
        ss = SemanticSearch()
        ss.add("d1", "hello world")
        assert ss.remove("d1") is True

    def test_remove_unknown_returns_false(self):
        ss = SemanticSearch()
        assert ss.remove("nonexistent") is False

    def test_remove_decreases_len(self):
        ss = SemanticSearch()
        ss.add("d1", "hello world")
        ss.add("d2", "foo bar")
        ss.remove("d1")
        assert len(ss) == 1

    def test_remove_doc_no_longer_searchable(self):
        ss = SemanticSearch()
        ss.add("d1", "unique_token_xyz")
        ss.remove("d1")
        result = ss.query("unique_token_xyz")
        doc_ids = [r[0] for r in result]
        assert "d1" not in doc_ids

    def test_remove_then_add_again(self):
        ss = SemanticSearch()
        ss.add("d1", "hello world")
        ss.remove("d1")
        ss.add("d1", "new content")
        assert len(ss) == 1

    def test_remove_updates_df(self):
        ss = SemanticSearch()
        ss.add("d1", "hello world")
        ss.add("d2", "hello python")
        ss.remove("d1")
        # "world" should no longer be in df
        assert ss._df.get("world", 0) == 0

    def test_remove_empty_index(self):
        ss = SemanticSearch()
        assert ss.remove("nonexistent") is False


class TestSemanticSearchLen:
    def test_initial_len_zero(self):
        ss = SemanticSearch()
        assert len(ss) == 0

    def test_len_after_adds(self):
        ss = SemanticSearch()
        ss.add("d1", "a")
        ss.add("d2", "b")
        ss.add("d3", "c")
        assert len(ss) == 3

    def test_len_after_remove(self):
        ss = SemanticSearch()
        ss.add("d1", "a")
        ss.remove("d1")
        assert len(ss) == 0


class TestComputeTFIDF:
    def test_tfidf_positive_for_present_token(self):
        ss = SemanticSearch()
        ss.add("d1", "python is great")
        score = ss.compute_tfidf("d1", "python")
        assert score > 0.0

    def test_tfidf_zero_for_absent_token(self):
        ss = SemanticSearch()
        ss.add("d1", "python is great")
        score = ss.compute_tfidf("d1", "java")
        assert score == 0.0

    def test_tfidf_zero_for_unknown_doc(self):
        ss = SemanticSearch()
        score = ss.compute_tfidf("nonexistent", "python")
        assert score == 0.0


class TestGlobalInstance:
    def test_semantic_search_exists(self):
        assert SEMANTIC_SEARCH is not None

    def test_semantic_search_is_instance(self):
        assert isinstance(SEMANTIC_SEARCH, SemanticSearch)
