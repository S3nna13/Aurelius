"""Tests for src/search/cross_encoder_reranker.py (15+ tests)."""

from __future__ import annotations

import torch

from src.search.cross_encoder_reranker import (
    CROSS_ENCODER_RERANKER,
    CrossEncoderConfig,
    CrossEncoderModel,
    CrossEncoderReranker,
    RankedResult,
)


def make_result(doc_id: str, text: str, score: float = 0.0, rank: int = 1) -> RankedResult:
    return RankedResult(doc_id=doc_id, text=text, score=score, rank=rank)


class TestCrossEncoderConfig:
    def test_defaults(self):
        cfg = CrossEncoderConfig()
        assert cfg.hidden_dim == 256
        assert cfg.n_layers == 2
        assert cfg.dropout == 0.1

    def test_custom_values(self):
        cfg = CrossEncoderConfig(hidden_dim=128, n_layers=4, dropout=0.2)
        assert cfg.hidden_dim == 128
        assert cfg.n_layers == 4
        assert cfg.dropout == 0.2


class TestRankedResult:
    def test_fields(self):
        r = RankedResult(doc_id="d1", text="hello", score=0.9, rank=1)
        assert r.doc_id == "d1"
        assert r.text == "hello"
        assert r.score == 0.9
        assert r.rank == 1


class TestCrossEncoderModel:
    def test_forward_shape(self):
        cfg = CrossEncoderConfig(hidden_dim=64, n_layers=1)
        model = CrossEncoderModel(cfg, vocab_size=100)
        ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        out = model(ids)
        assert out.shape == (1, 1)

    def test_forward_batch(self):
        cfg = CrossEncoderConfig(hidden_dim=32, n_layers=1)
        model = CrossEncoderModel(cfg, vocab_size=100)
        ids = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.long)
        out = model(ids)
        assert out.shape == (3, 1)

    def test_forward_single_token(self):
        cfg = CrossEncoderConfig(hidden_dim=32, n_layers=1)
        model = CrossEncoderModel(cfg, vocab_size=100)
        ids = torch.tensor([[7]], dtype=torch.long)
        out = model(ids)
        assert out.shape == (1, 1)


class TestCrossEncoderRerankerInit:
    def test_default_init(self):
        r = CrossEncoderReranker()
        assert isinstance(r.model, CrossEncoderModel)

    def test_custom_config(self):
        cfg = CrossEncoderConfig(hidden_dim=64, n_layers=1)
        r = CrossEncoderReranker(config=cfg)
        assert r.model.embedding.embedding_dim == 64

    def test_custom_model(self):
        cfg = CrossEncoderConfig(hidden_dim=32, n_layers=1)
        m = CrossEncoderModel(cfg, vocab_size=500)
        r = CrossEncoderReranker(model=m)
        assert r.model is m


class TestEncodePair:
    def test_returns_tensor(self):
        r = CrossEncoderReranker()
        t = r.encode_pair("hello", "world")
        assert isinstance(t, torch.Tensor)

    def test_shape_batch_one(self):
        r = CrossEncoderReranker()
        t = r.encode_pair("foo bar", "baz qux")
        assert t.shape[0] == 1

    def test_max_32_tokens(self):
        r = CrossEncoderReranker()
        long_text = " ".join([f"word{i}" for i in range(40)])
        t = r.encode_pair(long_text, "")
        assert t.shape[1] <= 32

    def test_empty_strings_no_error(self):
        r = CrossEncoderReranker()
        t = r.encode_pair("", "")
        assert t.shape == (1, 1)


class TestScore:
    def test_returns_float(self):
        r = CrossEncoderReranker()
        s = r.score("query text", "document text")
        assert isinstance(s, float)

    def test_score_finite(self):
        import math

        r = CrossEncoderReranker()
        s = r.score("neural networks", "deep learning transformers")
        assert math.isfinite(s)


class TestRerank:
    def test_returns_list(self):
        r = CrossEncoderReranker()
        results = [make_result("d1", "foo"), make_result("d2", "bar")]
        out = r.rerank("query", results)
        assert isinstance(out, list)

    def test_length_matches(self):
        r = CrossEncoderReranker()
        results = [make_result(f"d{i}", f"text {i}") for i in range(5)]
        out = r.rerank("query", results)
        assert len(out) == 5

    def test_top_k_limits(self):
        r = CrossEncoderReranker()
        results = [make_result(f"d{i}", f"text {i}") for i in range(6)]
        out = r.rerank("query", results, top_k=3)
        assert len(out) == 3

    def test_ranks_are_1indexed(self):
        r = CrossEncoderReranker()
        results = [make_result("d1", "foo"), make_result("d2", "bar")]
        out = r.rerank("query", results)
        assert out[0].rank == 1
        assert out[1].rank == 2

    def test_ranks_sequential(self):
        r = CrossEncoderReranker()
        results = [make_result(f"d{i}", f"text {i}") for i in range(4)]
        out = r.rerank("query", results)
        for i, item in enumerate(out):
            assert item.rank == i + 1

    def test_returns_ranked_result_instances(self):
        r = CrossEncoderReranker()
        results = [make_result("d1", "hello world")]
        out = r.rerank("hello", results)
        assert isinstance(out[0], RankedResult)

    def test_scores_descending(self):
        r = CrossEncoderReranker()
        results = [make_result(f"d{i}", f"doc {i}") for i in range(4)]
        out = r.rerank("doc", results)
        scores = [item.score for item in out]
        assert scores == sorted(scores, reverse=True)

    def test_empty_results(self):
        r = CrossEncoderReranker()
        out = r.rerank("query", [])
        assert out == []


class TestBatchRerank:
    def test_returns_list_of_lists(self):
        r = CrossEncoderReranker()
        queries = ["q1", "q2"]
        rpc = [
            [make_result("a", "foo"), make_result("b", "bar")],
            [make_result("c", "baz")],
        ]
        out = r.batch_rerank(queries, rpc)
        assert isinstance(out, list)
        assert len(out) == 2
        assert isinstance(out[0], list)

    def test_lengths_preserved(self):
        r = CrossEncoderReranker()
        queries = ["q1", "q2"]
        rpc = [
            [make_result(f"d{i}", f"t {i}") for i in range(3)],
            [make_result(f"e{i}", f"u {i}") for i in range(2)],
        ]
        out = r.batch_rerank(queries, rpc)
        assert len(out[0]) == 3
        assert len(out[1]) == 2


class TestGlobalInstance:
    def test_instance_exists(self):
        assert CROSS_ENCODER_RERANKER is not None

    def test_instance_type(self):
        assert isinstance(CROSS_ENCODER_RERANKER, CrossEncoderReranker)
