"""Attributed RAG: generation with inline citations and source attribution (ALCE-style)."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AttributionConfig:
    citation_format: str = "bracket"  # "bracket" [1] | "superscript" ^1 | "inline" (Source: ...)
    max_docs: int = 5
    min_citation_overlap: float = 0.5  # min word overlap to count as supported
    require_citation: bool = True


# ---------------------------------------------------------------------------
# Core data structure
# ---------------------------------------------------------------------------


@dataclass
class CitedDocument:
    doc_id: int
    text: str
    title: str = ""
    relevance_score: float = 0.0

    def format_citation(self, format: str) -> str:
        """Return a formatted citation marker for this document.

        Args:
            format: one of "bracket", "superscript", or "inline".
        """
        if format == "bracket":
            return f"[{self.doc_id}]"
        elif format == "superscript":
            return f"^{self.doc_id}"
        else:  # "inline"
            label = self.title if self.title else str(self.doc_id)
            return f"(Source: {label})"


# ---------------------------------------------------------------------------
# Citation insertion helpers
# ---------------------------------------------------------------------------


def insert_citations(
    response: str,
    citations: list[tuple[int, str]],
    format: str = "bracket",
) -> str:
    """Insert citation markers after sentences supported by documents.

    Args:
        response: the generated text.
        citations: list of (doc_id, sentence_span) pairs indicating which
                   sentence each doc supports.
        format: citation format — "bracket", "superscript", or "inline".

    Returns:
        Response string with inline citation markers inserted after each
        supported sentence.
    """
    # Build a mapping from sentence text -> list of doc_ids
    sentence_citations: dict[str, list[int]] = {}
    for doc_id, sentence_span in citations:
        key = sentence_span.strip()
        sentence_citations.setdefault(key, []).append(doc_id)

    # Split response into sentences (naive period-based split)
    # We reassemble after insertion.
    parts = re.split(r"(?<=[.!?])\s+", response)
    result_parts: list[str] = []

    for part in parts:
        stripped = part.strip()
        doc_ids = sentence_citations.get(stripped, [])
        if doc_ids:
            markers = "".join(
                CitedDocument(doc_id=d, text="").format_citation(format) for d in sorted(doc_ids)
            )
            result_parts.append(part + markers)
        else:
            result_parts.append(part)

    return " ".join(result_parts)


# ---------------------------------------------------------------------------
# Recall metric
# ---------------------------------------------------------------------------


def _word_overlap(a: str, b: str) -> float:
    """Compute word-level F1 overlap between two strings."""
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    precision = len(intersection) / len(tokens_b)
    recall = len(intersection) / len(tokens_a)
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_citation_recall(
    response: str,
    docs: list[CitedDocument],
    config: AttributionConfig,
) -> float:
    """Compute the fraction of response sentences supported by any document.

    A sentence is considered supported if at least one document has a word
    overlap >= config.min_citation_overlap with that sentence.

    Args:
        response: the generated text.
        docs: retrieved/cited documents.
        config: AttributionConfig controlling min_citation_overlap.

    Returns:
        Float in [0, 1] — fraction of sentences with supporting evidence.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", response) if s.strip()]
    if not sentences:
        return 0.0

    supported = 0
    for sentence in sentences:
        for doc in docs:
            if _word_overlap(sentence, doc.text) >= config.min_citation_overlap:
                supported += 1
                break

    return supported / len(sentences)


# ---------------------------------------------------------------------------
# Citation extraction
# ---------------------------------------------------------------------------


def extract_citations_from_text(text: str) -> list[int]:
    """Extract citation numbers from text containing [1], [2], ^1, ^2, etc.

    Args:
        text: text with embedded citation markers.

    Returns:
        Sorted list of unique doc_ids found in the text.
    """
    bracket_ids = re.findall(r"\[(\d+)\]", text)
    superscript_ids = re.findall(r"\^(\d+)", text)
    all_ids = {int(n) for n in bracket_ids + superscript_ids}
    return sorted(all_ids)


# ---------------------------------------------------------------------------
# End-to-end attributed RAG pipeline
# ---------------------------------------------------------------------------


class AttributedRAGPipeline:
    """End-to-end RAG pipeline with source attribution."""

    def __init__(
        self,
        retriever_fn: Callable[[str], list[CitedDocument]],
        generate_fn: Callable[[str], str],
        config: AttributionConfig,
    ) -> None:
        self.retriever_fn = retriever_fn
        self.generate_fn = generate_fn
        self.config = config

    def retrieve(self, query: str) -> list[CitedDocument]:
        """Call retriever_fn and return the top max_docs documents."""
        docs = self.retriever_fn(query)
        return docs[: self.config.max_docs]

    def format_prompt(self, query: str, docs: list[CitedDocument]) -> str:
        """Build a numbered-document prompt for attributed generation.

        Format:
            Documents:
            [1] <doc1 text>
            [2] <doc2 text>
            ...
            Question: <query>
            Answer with citations:
        """
        lines = ["Documents:"]
        for i, doc in enumerate(docs, start=1):
            lines.append(f"[{i}] {doc.text}")
        lines.append(f"Question: {query}")
        lines.append("Answer with citations:")
        return "\n".join(lines)

    def generate_with_citations(self, query: str) -> dict[str, str | list]:
        """Full pipeline: retrieve → format prompt → generate → extract citations.

        Returns:
            dict with keys:
                "response": generated text string
                "cited_docs": sorted list of cited doc_ids found in response
                "n_docs_retrieved": number of documents retrieved
        """
        docs = self.retrieve(query)
        prompt = self.format_prompt(query, docs)
        response = self.generate_fn(prompt)
        cited_docs = extract_citations_from_text(response)
        return {
            "response": response,
            "cited_docs": cited_docs,
            "n_docs_retrieved": len(docs),
        }

    def evaluate_attribution(self, query: str, gold_response: str) -> dict[str, float]:
        """Generate a response and evaluate citation recall vs a gold response.

        Args:
            query: the question to answer.
            gold_response: the reference/gold answer text.

        Returns:
            dict with keys:
                "citation_recall": fraction of gold-response sentences supported
                "response_length": character length of the generated response
        """
        result = self.generate_with_citations(query)
        response: str = result["response"]  # type: ignore[assignment]

        docs = self.retrieve(query)
        recall = compute_citation_recall(gold_response, docs, self.config)

        return {
            "citation_recall": recall,
            "response_length": float(len(response)),
        }


# ---------------------------------------------------------------------------
# Citation verifier
# ---------------------------------------------------------------------------


class CitationVerifier:
    """Verify that cited documents actually support the claimed sentences."""

    def __init__(self, config: AttributionConfig) -> None:
        self.config = config

    def verify_sentence(self, sentence: str, doc: CitedDocument) -> float:
        """Compute word-overlap F1 between sentence and doc.text.

        Returns:
            Float in [0, 1] representing overlap F1 score.
        """
        return _word_overlap(sentence, doc.text)

    def verify_response(self, response: str, docs: list[CitedDocument]) -> dict[str, float]:
        """Check each response sentence against all docs.

        A sentence is considered supported if at least one doc has
        overlap >= config.min_citation_overlap.

        Returns:
            dict with keys:
                "supported_rate": fraction of sentences that are supported
                "unsupported_sentences": count of unsupported sentences (as float)
        """
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", response) if s.strip()]
        if not sentences:
            return {"supported_rate": 0.0, "unsupported_sentences": 0.0}

        unsupported = 0
        for sentence in sentences:
            supported = any(
                self.verify_sentence(sentence, doc) >= self.config.min_citation_overlap
                for doc in docs
            )
            if not supported:
                unsupported += 1

        supported_count = len(sentences) - unsupported
        return {
            "supported_rate": supported_count / len(sentences),
            "unsupported_sentences": float(unsupported),
        }
