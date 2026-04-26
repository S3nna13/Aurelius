"""Tests for neural_reranker.py (~50 tests)."""

from __future__ import annotations

import pytest

from src.retrieval.neural_reranker import (
    CrossEncoderReranker,
    ListwiseReranker,
    RerankScore,
)

# ---------------------------------------------------------------------------
# RerankScore dataclass
# ---------------------------------------------------------------------------


class TestRerankScore:
    def test_fields_present(self):
        rs = RerankScore(query="q", document="d", score=0.5, rank=1)
        assert rs.query == "q"
        assert rs.document == "d"
        assert rs.score == 0.5
        assert rs.rank == 1

    def test_default_construction_all_fields_required(self):
        with pytest.raises(TypeError):
            RerankScore()  # type: ignore[call-arg]

    def test_score_can_be_zero(self):
        rs = RerankScore(query="q", document="d", score=0.0, rank=1)
        assert rs.score == 0.0

    def test_rank_int(self):
        rs = RerankScore(query="q", document="d", score=1.0, rank=3)
        assert isinstance(rs.rank, int)

    def test_equality(self):
        a = RerankScore(query="q", document="d", score=0.5, rank=1)
        b = RerankScore(query="q", document="d", score=0.5, rank=1)
        assert a == b


# ---------------------------------------------------------------------------
# CrossEncoderReranker._score_pair
# ---------------------------------------------------------------------------


class TestCrossEncoderRerankerScorePair:
    def setup_method(self):
        self.reranker = CrossEncoderReranker()

    def test_default_model_name(self):
        assert self.reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_custom_model_name(self):
        r = CrossEncoderReranker(model_name="my-model")
        assert r.model_name == "my-model"

    def test_overlapping_terms_positive(self):
        score = self.reranker._score_pair("machine learning", "machine learning is great")
        assert score > 0.0

    def test_no_overlap_zero(self):
        score = self.reranker._score_pair("apple orange", "cat dog fish")
        assert score == 0.0

    def test_full_overlap(self):
        score = self.reranker._score_pair("hello world", "hello world")
        assert score == pytest.approx(1.0)

    def test_partial_overlap(self):
        score = self.reranker._score_pair("the quick brown fox", "the slow fox")
        assert 0.0 < score < 1.0

    def test_case_insensitive(self):
        s1 = self.reranker._score_pair("Hello World", "hello world")
        s2 = self.reranker._score_pair("hello world", "hello world")
        assert s1 == pytest.approx(s2)

    def test_empty_doc(self):
        score = self.reranker._score_pair("hello world", "")
        assert score == 0.0

    def test_single_word_overlap(self):
        score = self.reranker._score_pair("hello", "hello world")
        assert score == pytest.approx(1.0)

    def test_returns_float(self):
        score = self.reranker._score_pair("test query", "test document")
        assert isinstance(score, float)


# ---------------------------------------------------------------------------
# CrossEncoderReranker.rerank
# ---------------------------------------------------------------------------


class TestCrossEncoderRerankerRerank:
    def setup_method(self):
        self.reranker = CrossEncoderReranker()
        self.query = "machine learning model"
        self.docs = [
            "deep learning model architecture",
            "cat and dog pets",
            "machine learning algorithms",
        ]

    def test_returns_list(self):
        result = self.reranker.rerank(self.query, self.docs)
        assert isinstance(result, list)

    def test_returns_rerank_scores(self):
        result = self.reranker.rerank(self.query, self.docs)
        assert all(isinstance(r, RerankScore) for r in result)

    def test_sorted_desc_by_score(self):
        result = self.reranker.rerank(self.query, self.docs)
        scores = [r.score for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_rank_starts_at_one(self):
        result = self.reranker.rerank(self.query, self.docs)
        assert result[0].rank == 1

    def test_rank_sequential(self):
        result = self.reranker.rerank(self.query, self.docs)
        for i, r in enumerate(result):
            assert r.rank == i + 1

    def test_rank_one_highest_score(self):
        result = self.reranker.rerank(self.query, self.docs)
        assert result[0].score >= result[-1].score

    def test_all_docs_returned_without_top_k(self):
        result = self.reranker.rerank(self.query, self.docs)
        assert len(result) == len(self.docs)

    def test_top_k_limits_results(self):
        result = self.reranker.rerank(self.query, self.docs, top_k=2)
        assert len(result) == 2

    def test_top_k_none_returns_all(self):
        result = self.reranker.rerank(self.query, self.docs, top_k=None)
        assert len(result) == len(self.docs)

    def test_top_k_larger_than_docs(self):
        result = self.reranker.rerank(self.query, self.docs, top_k=100)
        assert len(result) == len(self.docs)

    def test_top_k_one(self):
        result = self.reranker.rerank(self.query, self.docs, top_k=1)
        assert len(result) == 1
        assert result[0].rank == 1

    def test_empty_documents(self):
        result = self.reranker.rerank(self.query, [])
        assert result == []

    def test_query_stored_in_result(self):
        result = self.reranker.rerank(self.query, self.docs)
        assert all(r.query == self.query for r in result)

    def test_document_stored_in_result(self):
        result = self.reranker.rerank(self.query, self.docs)
        returned_docs = {r.document for r in result}
        assert returned_docs == set(self.docs)


# ---------------------------------------------------------------------------
# CrossEncoderReranker.batch_rerank
# ---------------------------------------------------------------------------


class TestCrossEncoderRerankerBatchRerank:
    def setup_method(self):
        self.reranker = CrossEncoderReranker()

    def test_batch_rerank_returns_list(self):
        result = self.reranker.batch_rerank("query", [["doc a", "doc b"], ["doc c"]])
        assert isinstance(result, list)

    def test_batch_rerank_flattens(self):
        batches = [["doc a", "doc b"], ["doc c", "doc d"]]
        result = self.reranker.batch_rerank("query", batches)
        assert len(result) == 4

    def test_batch_rerank_sorted_desc(self):
        batches = [["machine learning", "cats"], ["machine algorithms"]]
        result = self.reranker.batch_rerank("machine learning", batches)
        scores = [r.score for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_batch_rerank_empty_batches(self):
        result = self.reranker.batch_rerank("query", [])
        assert result == []

    def test_batch_rerank_single_batch(self):
        result = self.reranker.batch_rerank("query", [["doc a"]])
        assert len(result) == 1

    def test_batch_rerank_rank_one(self):
        batches = [["machine learning", "cats"], ["machine algorithms"]]
        result = self.reranker.batch_rerank("machine learning", batches)
        assert result[0].rank == 1


# ---------------------------------------------------------------------------
# ListwiseReranker.softmax_scores
# ---------------------------------------------------------------------------


class TestListwiseRerankerSoftmax:
    def setup_method(self):
        self.reranker = ListwiseReranker()

    def test_sums_to_one(self):
        scores = [1.0, 2.0, 3.0]
        result = self.reranker.softmax_scores(scores)
        assert sum(result) == pytest.approx(1.0, abs=1e-6)

    def test_all_positive(self):
        scores = [1.0, 2.0, 3.0]
        result = self.reranker.softmax_scores(scores)
        assert all(s > 0 for s in result)

    def test_higher_score_higher_softmax(self):
        scores = [1.0, 3.0]
        result = self.reranker.softmax_scores(scores)
        assert result[1] > result[0]

    def test_uniform_input(self):
        scores = [2.0, 2.0, 2.0]
        result = self.reranker.softmax_scores(scores)
        assert all(s == pytest.approx(1 / 3, abs=1e-6) for s in result)

    def test_empty_returns_empty(self):
        assert self.reranker.softmax_scores([]) == []

    def test_temperature_scaling_higher_more_uniform(self):
        scores = [1.0, 5.0]
        low_t = ListwiseReranker(temperature=0.1).softmax_scores(scores)
        high_t = ListwiseReranker(temperature=10.0).softmax_scores(scores)
        # With high temperature, distribution is more uniform
        assert abs(high_t[0] - high_t[1]) < abs(low_t[0] - low_t[1])

    def test_single_score_is_one(self):
        result = self.reranker.softmax_scores([5.0])
        assert result == pytest.approx([1.0])

    def test_returns_list_of_floats(self):
        result = self.reranker.softmax_scores([1.0, 2.0])
        assert isinstance(result, list)
        assert all(isinstance(s, float) for s in result)


# ---------------------------------------------------------------------------
# ListwiseReranker.rerank
# ---------------------------------------------------------------------------


class TestListwiseRerankerRerank:
    def setup_method(self):
        self.reranker = ListwiseReranker()

    def test_returns_list_of_tuples(self):
        docs = ["doc a", "doc b"]
        scores = [1.0, 2.0]
        result = self.reranker.rerank(docs, scores)
        assert isinstance(result, list)
        assert all(isinstance(r, tuple) for r in result)

    def test_sorted_desc(self):
        docs = ["low doc", "high doc"]
        scores = [1.0, 5.0]
        result = self.reranker.rerank(docs, scores)
        assert result[0][0] == "high doc"

    def test_all_docs_returned(self):
        docs = ["a", "b", "c"]
        scores = [1.0, 2.0, 3.0]
        result = self.reranker.rerank(docs, scores)
        assert len(result) == 3

    def test_scores_sum_to_one(self):
        docs = ["a", "b", "c"]
        scores = [1.0, 2.0, 3.0]
        result = self.reranker.rerank(docs, scores)
        assert sum(s for _, s in result) == pytest.approx(1.0, abs=1e-6)

    def test_empty(self):
        result = self.reranker.rerank([], [])
        assert result == []
