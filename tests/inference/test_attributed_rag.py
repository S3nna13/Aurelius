"""Tests for src/inference/attributed_rag.py — ALCE-style attributed RAG."""

from __future__ import annotations

import pytest

from src.inference.attributed_rag import (
    AttributionConfig,
    AttributedRAGPipeline,
    CitedDocument,
    CitationVerifier,
    compute_citation_recall,
    extract_citations_from_text,
    insert_citations,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_docs(n: int = 3) -> list[CitedDocument]:
    return [
        CitedDocument(
            doc_id=i + 1,
            text=f"Document number {i + 1} contains important information about topic {i + 1}.",
            title=f"Title {i + 1}",
            relevance_score=1.0 - i * 0.1,
        )
        for i in range(n)
    ]


def _mock_retriever(docs: list[CitedDocument]):
    def _retriever(query: str) -> list[CitedDocument]:
        return docs
    return _retriever


def _mock_generator(fixed_response: str):
    def _generator(prompt: str) -> str:
        return fixed_response
    return _generator


# ---------------------------------------------------------------------------
# 1. test_attribution_config_defaults
# ---------------------------------------------------------------------------

def test_attribution_config_defaults():
    config = AttributionConfig()
    assert config.citation_format == "bracket"
    assert config.max_docs == 5
    assert config.min_citation_overlap == 0.5
    assert config.require_citation is True


# ---------------------------------------------------------------------------
# 2. test_cited_document_format_bracket
# ---------------------------------------------------------------------------

def test_cited_document_format_bracket():
    doc = CitedDocument(doc_id=3, text="Some text.", title="MyDoc")
    assert doc.format_citation("bracket") == "[3]"


# ---------------------------------------------------------------------------
# 3. test_cited_document_format_superscript
# ---------------------------------------------------------------------------

def test_cited_document_format_superscript():
    doc = CitedDocument(doc_id=7, text="Some text.", title="MyDoc")
    assert doc.format_citation("superscript") == "^7"


# ---------------------------------------------------------------------------
# 4. test_insert_citations_adds_markers
# ---------------------------------------------------------------------------

def test_insert_citations_adds_markers():
    response = "The sky is blue. Water is wet."
    citations = [(1, "The sky is blue."), (2, "Water is wet.")]
    result = insert_citations(response, citations, format="bracket")
    assert "[1]" in result
    assert "[2]" in result


# ---------------------------------------------------------------------------
# 5. test_extract_citations_from_text_bracket
# ---------------------------------------------------------------------------

def test_extract_citations_from_text_bracket():
    text = "According to [1] and [2], this is true."
    ids = extract_citations_from_text(text)
    assert 1 in ids
    assert 2 in ids


# ---------------------------------------------------------------------------
# 6. test_extract_citations_sorted_unique
# ---------------------------------------------------------------------------

def test_extract_citations_sorted_unique():
    text = "See [3], [1], [3], ^2 for details."
    ids = extract_citations_from_text(text)
    # Should be sorted and unique
    assert ids == sorted(set(ids))
    assert len(ids) == len(set(ids))
    assert ids == [1, 2, 3]


# ---------------------------------------------------------------------------
# 7. test_compute_citation_recall_full_overlap
# ---------------------------------------------------------------------------

def test_compute_citation_recall_full_overlap():
    sentence = "The sky is blue."
    doc = CitedDocument(doc_id=1, text="The sky is blue.", title="")
    config = AttributionConfig(min_citation_overlap=0.5)
    recall = compute_citation_recall(sentence, [doc], config)
    assert recall == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 8. test_compute_citation_recall_no_overlap
# ---------------------------------------------------------------------------

def test_compute_citation_recall_no_overlap():
    sentence = "The sky is blue."
    doc = CitedDocument(doc_id=1, text="Quantum mechanics governs subatomic particles.", title="")
    config = AttributionConfig(min_citation_overlap=0.5)
    recall = compute_citation_recall(sentence, [doc], config)
    assert recall == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 9. test_attributed_rag_generate_keys
# ---------------------------------------------------------------------------

def test_attributed_rag_generate_keys():
    docs = _make_docs(3)
    config = AttributionConfig(max_docs=3)
    pipeline = AttributedRAGPipeline(
        retriever_fn=_mock_retriever(docs),
        generate_fn=_mock_generator("The answer is found in [1] and [2]."),
        config=config,
    )
    result = pipeline.generate_with_citations("What is topic 1?")
    assert "response" in result
    assert "cited_docs" in result
    assert "n_docs_retrieved" in result
    assert result["n_docs_retrieved"] == 3
    assert isinstance(result["cited_docs"], list)


# ---------------------------------------------------------------------------
# 10. test_citation_verifier_verify_sentence_identical
# ---------------------------------------------------------------------------

def test_citation_verifier_verify_sentence_identical():
    config = AttributionConfig()
    verifier = CitationVerifier(config)
    sentence = "the cat sat on the mat"
    doc = CitedDocument(doc_id=1, text="the cat sat on the mat", title="")
    score = verifier.verify_sentence(sentence, doc)
    assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 11. test_citation_verifier_verify_response_keys
# ---------------------------------------------------------------------------

def test_citation_verifier_verify_response_keys():
    config = AttributionConfig(min_citation_overlap=0.5)
    verifier = CitationVerifier(config)
    docs = _make_docs(2)
    response = "Document number 1 contains important information about topic 1."
    result = verifier.verify_response(response, docs)
    assert "supported_rate" in result
    assert "unsupported_sentences" in result
    assert isinstance(result["supported_rate"], float)
    assert isinstance(result["unsupported_sentences"], float)
