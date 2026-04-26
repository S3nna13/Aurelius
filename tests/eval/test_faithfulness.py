"""Tests for lightweight faithfulness metrics."""

import pytest
import torch

from src.eval.faithfulness import (
    coverage_score,
    embedding_alignment,
    faithfulness_report,
    hallucination_rate,
    ngrams,
    normalize_text,
)


def test_normalize_text_extracts_lowercase_tokens():
    assert normalize_text("Hello, WORLD! It's 2025.") == ["hello", "world", "it's", "2025"]


def test_ngrams_extracts_expected_pairs():
    grams = ngrams(["a", "b", "c"], 2)
    assert grams == [("a", "b"), ("b", "c")]


def test_coverage_score_is_one_for_supported_answer():
    score = coverage_score("the cat sat", "the cat sat on the mat")
    assert score == pytest.approx(1.0)


def test_hallucination_rate_increases_for_unsupported_text():
    score = hallucination_rate("the dog barked loudly", "the cat sat on the mat")
    assert score > 0.5


def test_embedding_alignment_is_one_for_identical_embeddings():
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    score = embedding_alignment(embeddings, embeddings)
    assert score.item() == pytest.approx(1.0)


def test_faithfulness_report_combines_metrics():
    report = faithfulness_report("the cat sat", "the cat sat on the mat")
    assert report.unigram_coverage == pytest.approx(1.0)
    assert report.hallucination == pytest.approx(0.0)
    assert report.lexical_f1 > 0.5


def test_faithfulness_report_uses_optional_embeddings():
    answer_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    source_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    report = faithfulness_report(
        "the cat",
        "the cat sat",
        answer_embeddings=answer_embeddings,
        source_embeddings=source_embeddings,
    )
    assert report.embedding_support is not None
    assert report.embedding_support.item() == pytest.approx(1.0)


def test_faithfulness_report_requires_both_embedding_inputs():
    with pytest.raises(ValueError):
        faithfulness_report("answer", "source", answer_embeddings=torch.randn(2, 4))


def test_embedding_alignment_rejects_bad_rank():
    with pytest.raises(ValueError):
        embedding_alignment(torch.randn(2, 3, 4), torch.randn(2, 4))


def test_ngrams_rejects_non_positive_n():
    with pytest.raises(ValueError):
        ngrams(["a"], 0)
