"""Data evaluation — measure quality, diversity, coverage, and bias of datasets.

Metrics computed per dataset:
  - Quality distribution (mean, median, std of quality scores)
  - Domain coverage (tokens per domain)
  - Diversity (unique n-grams, type-token ratio)
  - Safety (toxicity rate, PII rate, harmful content rate)
  - Difficulty distribution (easy/medium/hard ratios)
  - Benchmark contamination (n-gram overlap with eval sets)
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .scorer import QualityScorer, QualityScore
from .safety import SafetyFilter

logger = logging.getLogger(__name__)


@dataclass
class DatasetEvalResult:
    name: str
    samples: int
    tokens: int
    mean_quality: float = 0.0
    median_quality: float = 0.0
    std_quality: float = 0.0
    safety_rate: float = 1.0
    toxicity_rate: float = 0.0
    pii_rate: float = 0.0
    type_token_ratio: float = 0.0
    domain_breakdown: dict[str, int] = field(default_factory=dict)
    difficulty_breakdown: dict[str, int] = field(default_factory=dict)
    contamination_ratio: float = 0.0


class DataEvaluator:
    """Comprehensive dataset evaluation."""

    def __init__(self):
        self.scorer = QualityScorer()
        self.safety = SafetyFilter()

    def evaluate(self, texts: list[str], name: str = "unknown", domains: list[str] | None = None) -> DatasetEvalResult:
        domains = domains or ["web"] * len(texts)
        qualities = self.scorer.score_batch(texts, domains[0] if domains else "web")
        safety_results = self.safety.check_batch(texts)

        quality_scores = [q.overall for q in qualities]
        sorted_scores = sorted(quality_scores)

        n = len(texts)
        total_tokens = sum(len(t.split()) for t in texts)
        all_words = " ".join(texts).split()
        type_token = len(set(w.lower() for w in all_words)) / max(len(all_words), 1)

        safety_rate = sum(1 for r in safety_results if r.safe) / max(n, 1)
        toxicity = sum(1 for r in safety_results if "toxic_language" in r.flags) / max(n, 1)
        pii = sum(1 for r in safety_results if any(f.startswith("pii:") for f in r.flags)) / max(n, 1)

        domain_tokens: dict[str, int] = {}
        for i, t in enumerate(texts):
            domain = domains[i] if i < len(domains) else "web"
            domain_tokens[domain] = domain_tokens.get(domain, 0) + len(t.split())

        difficulties = {"easy": 0, "medium": 0, "hard": 0}
        for q in quality_scores:
            if q < 0.4:
                difficulties["easy"] += 1
            elif q < 0.7:
                difficulties["medium"] += 1
            else:
                difficulties["hard"] += 1

        return DatasetEvalResult(
            name=name, samples=n, tokens=total_tokens,
            mean_quality=round(sum(quality_scores) / max(n, 1), 4),
            median_quality=round(sorted_scores[n // 2] if n > 0 else 0, 4),
            std_quality=round(self._std(quality_scores), 4),
            safety_rate=round(safety_rate, 4),
            toxicity_rate=round(toxicity, 4),
            pii_rate=round(pii, 4),
            type_token_ratio=round(type_token, 4),
            domain_breakdown=domain_tokens,
            difficulty_breakdown=difficulties,
        )

    def compare(self, results: list[DatasetEvalResult]) -> dict[str, Any]:
        best = max(results, key=lambda r: r.mean_quality) if results else None
        worst = min(results, key=lambda r: r.mean_quality) if results else None
        return {
            "datasets": len(results),
            "total_samples": sum(r.samples for r in results),
            "total_tokens": sum(r.tokens for r in results),
            "best_dataset": best.name if best else None,
            "worst_dataset": worst.name if worst else None,
            "avg_quality": round(sum(r.mean_quality for r in results) / max(len(results), 1), 4),
        }

    def contamination_check(self, train_texts: list[str], eval_texts: list[str], ngram_size: int = 8) -> float:
        train_ngrams: set[int] = set()
        for t in train_texts:
            words = t.lower().split()
            for i in range(len(words) - ngram_size + 1):
                train_ngrams.add(hash(" ".join(words[i:i+ngram_size])))

        if not train_ngrams:
            return 0.0

        overlaps = 0
        total = 0
        for t in eval_texts:
            words = t.lower().split()
            for i in range(len(words) - ngram_size + 1):
                if hash(" ".join(words[i:i+ngram_size])) in train_ngrams:
                    overlaps += 1
                total += 1

        return overlaps / max(total, 1)

    @staticmethod
    def _std(values: list[float]) -> float:
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return math.sqrt(variance)
