"""Tests for dense_retriever.py (~50 tests)."""

from __future__ import annotations

import math
import pytest

from src.retrieval.dense_retriever import DenseDocument, DenseRetriever


# ---------------------------------------------------------------------------
# DenseDocument dataclass
# ---------------------------------------------------------------------------

class TestDenseDocument:
    def test_fields_present(self):
        doc = DenseDocument(doc_id="d1", text="hello", embedding=[0.1, 0.2])
        assert doc.doc_id == "d1"
        assert doc.text == "hello"
        assert doc.embedding == [0.1, 0.2]

    def test_metadata_default_empty_dict(self):
        doc = DenseDocument(doc_id="d1", text="hello", embedding=[])
        assert doc.metadata == {}

    def test_metadata_custom(self):
        meta = {"source": "wiki"}
        doc = DenseDocument(doc_id="d1", text="hello", embedding=[], metadata=meta)
        assert doc.metadata["source"] == "wiki"

    def test_equality(self):
        a = DenseDocument(doc_id="d1", text="t", embedding=[1.0])
        b = DenseDocument(doc_id="d1", text="t", embedding=[1.0])
        assert a == b

    def test_required_fields(self):
        with pytest.raises(TypeError):
            DenseDocument()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# DenseRetriever._embed_stub
# ---------------------------------------------------------------------------

class TestDenseRetrieverEmbedStub:
    def setup_method(self):
        self.retriever = DenseRetriever(embedding_dim=16)

    def test_returns_list(self):
        vec = self.retriever._embed_stub("hello world")
        assert isinstance(vec, list)

    def test_length_equals_embedding_dim(self):
        vec = self.retriever._embed_stub("hello world")
        assert len(vec) == 16

    def test_deterministic_same_text(self):
        v1 = self.retriever._embed_stub("same text")
        v2 = self.retriever._embed_stub("same text")
        assert v1 == v2

    def test_different_texts_different_embeddings(self):
        v1 = self.retriever._embed_stub("foo bar")
        v2 = self.retriever._embed_stub("baz qux")
        assert v1 != v2

    def test_short_text_padded_with_zeros(self):
        vec = self.retriever._embed_stub("one")
        # Only 1 word, rest should be 0.0
        assert vec[1:] == [0.0] * 15

    def test_values_in_range(self):
        vec = self.retriever._embed_stub("hello world foo bar")
        assert all(0.0 <= v <= 1.0 for v in vec)

    def test_empty_text_all_zeros(self):
        vec = self.retriever._embed_stub("")
        assert vec == [0.0] * 16

    def test_long_text_truncated(self):
        long_text = " ".join(["word"] * 100)
        vec = self.retriever._embed_stub(long_text)
        assert len(vec) == 16

    def test_custom_dim(self):
        r = DenseRetriever(embedding_dim=32)
        vec = r._embed_stub("hello world foo bar")
        assert len(vec) == 32


# ---------------------------------------------------------------------------
# DenseRetriever.add and __len__
# ---------------------------------------------------------------------------

class TestDenseRetrieverAdd:
    def setup_method(self):
        self.retriever = DenseRetriever(embedding_dim=16)

    def test_add_returns_dense_document(self):
        doc = self.retriever.add("d1", "hello world")
        assert isinstance(doc, DenseDocument)

    def test_add_increases_len(self):
        assert len(self.retriever) == 0
        self.retriever.add("d1", "hello")
        assert len(self.retriever) == 1

    def test_add_multiple_increases_len(self):
        self.retriever.add("d1", "doc one")
        self.retriever.add("d2", "doc two")
        assert len(self.retriever) == 2

    def test_add_stores_doc_id(self):
        doc = self.retriever.add("myid", "text")
        assert doc.doc_id == "myid"

    def test_add_stores_text(self):
        doc = self.retriever.add("d1", "my text here")
        assert doc.text == "my text here"

    def test_add_embeds_to_correct_dim(self):
        doc = self.retriever.add("d1", "hello world")
        assert len(doc.embedding) == 16

    def test_add_with_metadata(self):
        doc = self.retriever.add("d1", "text", metadata={"src": "test"})
        assert doc.metadata["src"] == "test"

    def test_add_without_metadata_empty(self):
        doc = self.retriever.add("d1", "text")
        assert doc.metadata == {}

    def test_overwrite_same_id(self):
        self.retriever.add("d1", "first")
        self.retriever.add("d1", "second")
        # len should still be 1 (overwrite)
        assert len(self.retriever) == 1


# ---------------------------------------------------------------------------
# DenseRetriever.cosine_similarity
# ---------------------------------------------------------------------------

class TestCosinesimilarity:
    def setup_method(self):
        self.retriever = DenseRetriever()

    def test_identical_vectors_approx_one(self):
        v = [1.0, 0.0, 0.5]
        sim = self.retriever.cosine_similarity(v, v)
        assert sim == pytest.approx(1.0, abs=1e-4)

    def test_orthogonal_vectors_approx_zero(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        sim = self.retriever.cosine_similarity(a, b)
        assert sim == pytest.approx(0.0, abs=1e-6)

    def test_zero_vector_no_crash(self):
        a = [0.0, 0.0]
        b = [1.0, 0.5]
        sim = self.retriever.cosine_similarity(a, b)
        assert isinstance(sim, float)
        assert sim == pytest.approx(0.0, abs=1e-4)

    def test_both_zero_no_crash(self):
        a = [0.0, 0.0]
        b = [0.0, 0.0]
        sim = self.retriever.cosine_similarity(a, b)
        assert isinstance(sim, float)

    def test_similar_vectors_high_score(self):
        a = [1.0, 1.0, 1.0]
        b = [0.9, 1.1, 1.0]
        sim = self.retriever.cosine_similarity(a, b)
        assert sim > 0.9

    def test_opposite_vectors_negative(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        sim = self.retriever.cosine_similarity(a, b)
        assert sim < 0.0

    def test_returns_float(self):
        sim = self.retriever.cosine_similarity([1.0], [1.0])
        assert isinstance(sim, float)


# ---------------------------------------------------------------------------
# DenseRetriever.search
# ---------------------------------------------------------------------------

class TestDenseRetrieverSearch:
    def setup_method(self):
        self.retriever = DenseRetriever(embedding_dim=16)
        self.retriever.add("d1", "machine learning algorithms")
        self.retriever.add("d2", "deep learning neural network")
        self.retriever.add("d3", "cat and dog pets")

    def test_search_returns_list(self):
        result = self.retriever.search("machine learning")
        assert isinstance(result, list)

    def test_search_returns_tuples(self):
        result = self.retriever.search("machine learning")
        assert all(isinstance(r, tuple) and len(r) == 2 for r in result)

    def test_search_top_k(self):
        result = self.retriever.search("machine learning", top_k=2)
        assert len(result) <= 2

    def test_search_sorted_desc(self):
        result = self.retriever.search("machine learning")
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_search_matching_doc_found(self):
        result = self.retriever.search("machine learning algorithms")
        doc_ids = [r[0] for r in result]
        assert "d1" in doc_ids

    def test_search_empty_store(self):
        r = DenseRetriever(embedding_dim=16)
        result = r.search("query")
        assert result == []

    def test_search_default_top_k_five(self):
        r = DenseRetriever(embedding_dim=16)
        for i in range(10):
            r.add(f"d{i}", f"document number {i}")
        result = r.search("document")
        assert len(result) <= 5

    def test_search_doc_ids_are_strings(self):
        result = self.retriever.search("learning")
        assert all(isinstance(r[0], str) for r in result)

    def test_search_scores_are_floats(self):
        result = self.retriever.search("learning")
        assert all(isinstance(r[1], float) for r in result)


# ---------------------------------------------------------------------------
# DenseRetriever.remove
# ---------------------------------------------------------------------------

class TestDenseRetrieverRemove:
    def setup_method(self):
        self.retriever = DenseRetriever(embedding_dim=16)
        self.retriever.add("d1", "hello world")
        self.retriever.add("d2", "foo bar")

    def test_remove_known_returns_true(self):
        assert self.retriever.remove("d1") is True

    def test_remove_unknown_returns_false(self):
        assert self.retriever.remove("nonexistent") is False

    def test_remove_decreases_len(self):
        before = len(self.retriever)
        self.retriever.remove("d1")
        assert len(self.retriever) == before - 1

    def test_remove_unknown_does_not_change_len(self):
        before = len(self.retriever)
        self.retriever.remove("nonexistent")
        assert len(self.retriever) == before

    def test_remove_doc_not_in_search_after(self):
        self.retriever.remove("d1")
        result = self.retriever.search("hello world")
        doc_ids = [r[0] for r in result]
        assert "d1" not in doc_ids

    def test_remove_twice_second_returns_false(self):
        self.retriever.remove("d1")
        assert self.retriever.remove("d1") is False
