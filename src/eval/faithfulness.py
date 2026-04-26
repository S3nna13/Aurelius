"""Lightweight faithfulness metrics for answer/source pairs."""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def normalize_text(text: str) -> list[str]:
    """Lowercase and tokenize text into lightweight lexical units."""
    return _TOKEN_RE.findall(text.lower())


def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """Extract token n-grams."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if len(tokens) < n:
        return []
    return [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]


def coverage_score(answer: str, source: str, n: int = 1) -> float:
    """Measure how much of the answer is lexically grounded in the source."""
    answer_ngrams = ngrams(normalize_text(answer), n)
    source_ngrams = set(ngrams(normalize_text(source), n))
    if not answer_ngrams:
        return 1.0
    covered = sum(gram in source_ngrams for gram in answer_ngrams)
    return covered / len(answer_ngrams)


def hallucination_rate(answer: str, source: str, n: int = 1) -> float:
    """Fraction of answer n-grams not supported by the source."""
    return 1.0 - coverage_score(answer, source, n=n)


def embedding_alignment(
    answer_embeddings: torch.Tensor, source_embeddings: torch.Tensor
) -> torch.Tensor:
    """Compute max-cosine alignment between answer and source embeddings."""
    if answer_embeddings.dim() != 2 or source_embeddings.dim() != 2:
        raise ValueError("answer_embeddings and source_embeddings must be 2D")
    answer_norm = answer_embeddings / answer_embeddings.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    source_norm = source_embeddings / source_embeddings.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    similarities = answer_norm @ source_norm.transpose(0, 1)
    return similarities.max(dim=-1).values.mean()


@dataclass(frozen=True)
class FaithfulnessReport:
    unigram_coverage: float
    bigram_coverage: float
    hallucination: float
    lexical_f1: float
    embedding_support: torch.Tensor | None


def faithfulness_report(
    answer: str,
    source: str,
    answer_embeddings: torch.Tensor | None = None,
    source_embeddings: torch.Tensor | None = None,
) -> FaithfulnessReport:
    """Build a compact faithfulness report from lexical and embedding signals."""
    unigram = coverage_score(answer, source, n=1)
    bigram = coverage_score(answer, source, n=2)
    hallucination = 1.0 - unigram

    answer_tokens = set(normalize_text(answer))
    source_tokens = set(normalize_text(source))
    if not answer_tokens and not source_tokens:
        lexical_f1 = 1.0
    else:
        overlap = len(answer_tokens & source_tokens)
        precision = overlap / max(len(answer_tokens), 1)
        recall = overlap / max(len(source_tokens), 1)
        lexical_f1 = (
            0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        )

    embedding_support = None
    if answer_embeddings is not None or source_embeddings is not None:
        if answer_embeddings is None or source_embeddings is None:
            raise ValueError(
                "answer_embeddings and source_embeddings must either both be set or both be None"
            )
        embedding_support = embedding_alignment(answer_embeddings, source_embeddings)

    return FaithfulnessReport(
        unigram_coverage=unigram,
        bigram_coverage=bigram,
        hallucination=hallucination,
        lexical_f1=lexical_f1,
        embedding_support=embedding_support,
    )
