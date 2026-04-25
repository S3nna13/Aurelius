"""Tests for src/search/dense_embedder.py (12+ tests)."""

from __future__ import annotations

import math

import numpy
import pytest

from src.search.dense_embedder import (
    DenseEmbedder,
    EmbedderConfig,
    DENSE_EMBEDDER,
)


class TestEmbedderConfig:
    def test_defaults(self):
        cfg = EmbedderConfig()
        assert cfg.dim == 768
        assert cfg.normalize is True
        assert cfg.pooling == "mean"

    def test_custom(self):
        cfg = EmbedderConfig(dim=128, normalize=False, pooling="cls")
        assert cfg.dim == 128
        assert cfg.normalize is False
        assert cfg.pooling == "cls"


class TestTokenize:
    def test_returns_list_of_ints(self):
        e = DenseEmbedder(EmbedderConfig(dim=32), vocab_size=100)
        ids = e.tokenize("hello world")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_in_vocab_range(self):
        e = DenseEmbedder(EmbedderConfig(dim=32), vocab_size=100)
        ids = e.tokenize("foo bar baz")
        assert all(0 <= i < 100 for i in ids)

    def test_empty_string_no_error(self):
        e = DenseEmbedder(EmbedderConfig(dim=32), vocab_size=100)
        ids = e.tokenize("")
        assert len(ids) >= 1


class TestEmbed:
    def test_shape(self):
        e = DenseEmbedder(EmbedderConfig(dim=64), vocab_size=100)
        v = e.embed("test text")
        assert v.shape == (64,)

    def test_returns_ndarray(self):
        e = DenseEmbedder(EmbedderConfig(dim=64), vocab_size=100)
        v = e.embed("hello world")
        assert isinstance(v, numpy.ndarray)

    def test_normalized_unit_norm(self):
        e = DenseEmbedder(EmbedderConfig(dim=64, normalize=True), vocab_size=100)
        v = e.embed("normalize me")
        norm = float(numpy.linalg.norm(v))
        assert abs(norm - 1.0) < 1e-5

    def test_unnormalized_not_forced_unit(self):
        e = DenseEmbedder(EmbedderConfig(dim=64, normalize=False), vocab_size=100)
        v = e.embed("some text here")
        assert v.shape == (64,)

    def test_pooling_cls(self):
        e = DenseEmbedder(EmbedderConfig(dim=32, pooling="cls"), vocab_size=100)
        v = e.embed("cls pooling test")
        assert v.shape == (32,)

    def test_pooling_max(self):
        e = DenseEmbedder(EmbedderConfig(dim=32, pooling="max"), vocab_size=100)
        v = e.embed("max pooling test")
        assert v.shape == (32,)


class TestEmbedBatch:
    def test_shape(self):
        e = DenseEmbedder(EmbedderConfig(dim=32), vocab_size=100)
        out = e.embed_batch(["foo", "bar", "baz"])
        assert out.shape == (3, 32)

    def test_single_text(self):
        e = DenseEmbedder(EmbedderConfig(dim=32), vocab_size=100)
        out = e.embed_batch(["only one"])
        assert out.shape == (1, 32)


class TestSimilarity:
    def test_returns_float(self):
        e = DenseEmbedder(EmbedderConfig(dim=32), vocab_size=100)
        s = e.similarity("hello world", "hello world")
        assert isinstance(s, float)

    def test_identical_texts_near_one(self):
        e = DenseEmbedder(EmbedderConfig(dim=64, normalize=True), vocab_size=500)
        s = e.similarity("the quick brown fox", "the quick brown fox")
        assert abs(s - 1.0) < 1e-5


class TestTopKSimilar:
    def test_returns_list(self):
        e = DenseEmbedder(EmbedderConfig(dim=32), vocab_size=100)
        corpus = ["doc one", "doc two", "doc three"]
        out = e.top_k_similar("query", corpus, k=2)
        assert isinstance(out, list)
        assert len(out) == 2

    def test_tuples_idx_score(self):
        e = DenseEmbedder(EmbedderConfig(dim=32), vocab_size=100)
        corpus = ["alpha", "beta", "gamma"]
        out = e.top_k_similar("alpha", corpus, k=1)
        idx, score = out[0]
        assert isinstance(idx, int)
        assert isinstance(score, float)

    def test_descending_order(self):
        e = DenseEmbedder(EmbedderConfig(dim=32, normalize=True), vocab_size=500)
        corpus = [f"document number {i}" for i in range(6)]
        out = e.top_k_similar("document", corpus, k=4)
        scores = [s for _, s in out]
        assert scores == sorted(scores, reverse=True)

    def test_empty_corpus(self):
        e = DenseEmbedder(EmbedderConfig(dim=32), vocab_size=100)
        out = e.top_k_similar("query", [], k=3)
        assert out == []

    def test_k_larger_than_corpus(self):
        e = DenseEmbedder(EmbedderConfig(dim=32), vocab_size=100)
        corpus = ["a", "b"]
        out = e.top_k_similar("query", corpus, k=10)
        assert len(out) == 2


class TestGlobalInstance:
    def test_instance_exists(self):
        assert DENSE_EMBEDDER is not None

    def test_instance_type(self):
        assert isinstance(DENSE_EMBEDDER, DenseEmbedder)
