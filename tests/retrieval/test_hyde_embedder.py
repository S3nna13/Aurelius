"""Tests for HypotheticalDocEmbedder (HyDE — arXiv 2212.10496)."""

from __future__ import annotations

import numpy as np
import pytest

from src.retrieval.hyde_embedder import HyDEConfig, HypotheticalDocEmbedder


@pytest.fixture()
def default_embedder() -> HypotheticalDocEmbedder:
    return HypotheticalDocEmbedder()


@pytest.fixture()
def custom_config() -> HyDEConfig:
    return HyDEConfig(n_hypothetical=3, embedding_dim=64, normalize=False)


class TestHyDEConfig:
    def test_defaults(self):
        cfg = HyDEConfig()
        assert cfg.n_hypothetical == 5
        assert cfg.embedding_dim == 768
        assert cfg.normalize is True

    def test_custom_values(self, custom_config):
        assert custom_config.n_hypothetical == 3
        assert custom_config.embedding_dim == 64
        assert custom_config.normalize is False


class TestGenerateHypotheticalDocs:
    def test_returns_correct_count(self, default_embedder):
        docs = default_embedder.generate_hypothetical_docs("transformers", 5)
        assert len(docs) == 5

    def test_returns_strings(self, default_embedder):
        docs = default_embedder.generate_hypothetical_docs("attention", 3)
        assert all(isinstance(d, str) for d in docs)

    def test_query_appears_in_docs(self, default_embedder):
        query = "neural networks"
        docs = default_embedder.generate_hypothetical_docs(query, 4)
        assert all(query in d for d in docs)

    def test_zero_docs(self, default_embedder):
        docs = default_embedder.generate_hypothetical_docs("test", 0)
        assert docs == []


class TestEmbed:
    def test_shape(self, default_embedder):
        vec = default_embedder.embed("hello world")
        assert vec.shape == (768,)

    def test_deterministic(self, default_embedder):
        v1 = default_embedder.embed("same text")
        v2 = default_embedder.embed("same text")
        np.testing.assert_array_equal(v1, v2)

    def test_different_texts_differ(self, default_embedder):
        v1 = default_embedder.embed("text A")
        v2 = default_embedder.embed("text B")
        assert not np.array_equal(v1, v2)

    def test_custom_dim(self, custom_config):
        embedder = HypotheticalDocEmbedder(custom_config)
        vec = embedder.embed("test")
        assert vec.shape == (64,)


class TestEmbedQuery:
    def test_shape(self, default_embedder):
        vec = default_embedder.embed_query("what is attention?")
        assert vec.shape == (768,)

    def test_l2_normalized_when_enabled(self, default_embedder):
        vec = default_embedder.embed_query("language model")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5

    def test_not_normalized_when_disabled(self, custom_config):
        embedder = HypotheticalDocEmbedder(custom_config)
        vec = embedder.embed_query("language model")
        norm = np.linalg.norm(vec)
        assert norm > 1e-5

    def test_deterministic(self, default_embedder):
        q = "reproducible query"
        v1 = default_embedder.embed_query(q)
        v2 = default_embedder.embed_query(q)
        np.testing.assert_array_equal(v1, v2)

    def test_returns_float_array(self, default_embedder):
        vec = default_embedder.embed_query("dtype check")
        assert vec.dtype in (np.float32, np.float64)


class TestBatchEmbedQueries:
    def test_shape(self, default_embedder):
        queries = ["first query", "second query", "third query"]
        result = default_embedder.batch_embed_queries(queries)
        assert result.shape == (3, 768)

    def test_single_query(self, default_embedder):
        result = default_embedder.batch_embed_queries(["solo"])
        assert result.shape == (1, 768)

    def test_each_row_normalized(self, default_embedder):
        queries = ["alpha", "beta", "gamma"]
        result = default_embedder.batch_embed_queries(queries)
        for row in result:
            assert abs(np.linalg.norm(row) - 1.0) < 1e-5

    def test_consistent_with_single_embed_query(self, default_embedder):
        q = "consistency check"
        batch = default_embedder.batch_embed_queries([q])
        single = default_embedder.embed_query(q)
        np.testing.assert_array_almost_equal(batch[0], single)
