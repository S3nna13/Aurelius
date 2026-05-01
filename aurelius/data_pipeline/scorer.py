"""Quality scoring system — multi-dimension document quality evaluation.

Each document is scored on:
  1. Perplexity score (lower = better, indicates natural language)
  2. Educational value (keyword/semantic density of informative content)
  3. Instruction following potential (presence of Q&A, instructions)
  4. Code quality (syntax validity, comment ratio, documentation)
  5. Mathematical correctness (formula density, step-by-step reasoning)
  6. Safety (toxicity, bias, harmful content scores)
  7. Diversity (n-gram uniqueness vs corpus)
  8. Coherence (discourse markers, topic consistency)

Composite score: weighted average of all dimensions.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class QualityScore:
    overall: float = 0.0
    perplexity: float = 0.0
    educational: float = 0.0
    instruction: float = 0.0
    code_quality: float = 0.0
    math_quality: float = 0.0
    safety: float = 1.0
    diversity: float = 0.0
    coherence: float = 0.0

    def __post_init__(self):
        self.overall = self.compute_overall()

    def compute_overall(self) -> float:
        weights = {
            "perplexity": 0.15, "educational": 0.20, "instruction": 0.20,
            "code_quality": 0.15, "math_quality": 0.10, "safety": 0.10,
            "diversity": 0.05, "coherence": 0.05,
        }
        total = sum(
            getattr(self, dim) * weight for dim, weight in weights.items()
        )
        return round(min(max(total, 0.0), 1.0), 4)


class QualityScorer:
    """Multi-dimension document quality scorer."""

    def __init__(self):
        self._code_comment_re = re.compile(r"#.*$|//.*$|<!--.*?-->", re.MULTILINE)
        self._code_keywords = {"def ", "class ", "import ", "function", "return", "if ", "for ", "while "}
        self._math_patterns = [r"\\[", r"\$.*?\$", r"\\begin{equation}", r"\\frac", r"\b\d+\.\d+\b"]
        self._instruction_patterns = [r"\bhow\b", r"\bwhat\b", r"\bexplain\b", r"\bstep\b", r"\bfollow\b", r"\bexample\b"]
        self._coherence_markers = [r"\btherefore\b", r"\bhowever\b", r"\bmoreover\b", r"\bconsequently\b", r"\bfirst\b", r"\bfinally\b"]

    def score(self, text: str, domain: str = "web") -> QualityScore:
        return QualityScore(
            perplexity=self._score_perplexity(text),
            educational=self._score_educational(text),
            instruction=self._score_instruction(text),
            code_quality=self._score_code(text) if domain == "code" else 0.0,
            math_quality=self._score_math(text) if domain in ("math", "science") else 0.0,
            safety=self._score_safety(text),
            diversity=self._score_diversity(text),
            coherence=self._score_coherence(text),
        )

    def score_batch(self, texts: list[str], domain: str = "web") -> list[QualityScore]:
        return [self.score(t, domain) for t in texts]

    def filter_top(self, texts: list[str], scores: list[QualityScore], threshold: float = 0.5) -> list[tuple[str, QualityScore]]:
        return [(t, s) for t, s in zip(texts, scores) if s.overall >= threshold]

    def _score_perplexity(self, text: str) -> float:
        words = text.split()
        if len(words) < 50:
            return 0.3
        unique_ratio = len(set(w.lower() for w in words)) / max(len(words), 1)
        return min(max(unique_ratio * 1.5, 0.0), 1.0)

    def _score_educational(self, text: str) -> float:
        informative_words = ["because", "therefore", "result", "analysis", "study", "data", "research", "evidence", "conclusion"]
        hits = sum(1 for w in informative_words if w in text.lower())
        return min(hits / 5.0, 1.0)

    def _score_instruction(self, text: str) -> float:
        hits = sum(1 for p in self._instruction_patterns if re.search(p, text.lower()))
        return min(hits / 4.0, 1.0)

    def _score_code(self, text: str) -> float:
        lines = text.split("\n")
        code_lines = sum(1 for l in lines if any(kw in l for kw in self._code_keywords))
        if len(lines) == 0:
            return 0.0
        comment_lines = len(self._code_comment_re.findall(text))
        has_docstring = '"""' in text or "'''" in text
        score = (code_lines / max(len(lines), 1)) * 0.5 + (comment_lines / max(len(lines), 1)) * 0.3
        if has_docstring:
            score += 0.2
        return min(score, 1.0)

    def _score_math(self, text: str) -> float:
        hits = sum(1 for p in self._math_patterns if re.search(p, text))
        numbers = len(re.findall(r"\b\d+\b", text))
        return min((hits * 0.15) + (min(numbers / 50, 1.0) * 0.4), 1.0)

    def _score_safety(self, text: str) -> float:
        toxic_patterns = [r"\b(hate|kill|die|attack|bomb)\b", r"\b(nsfw|explicit|porn)\b"]
        toxicity = sum(1 for p in toxic_patterns if re.search(p, text.lower()))
        return max(1.0 - (toxicity * 0.3), 0.0)

    def _score_diversity(self, text: str) -> float:
        words = text.lower().split()
        if len(words) < 10:
            return 0.5
        fourgrams = {" ".join(words[i:i+4]) for i in range(len(words) - 4)}
        return min(len(fourgrams) / max(len(words), 1) * 4, 1.0)

    def _score_coherence(self, text: str) -> float:
        hits = sum(1 for p in self._coherence_markers if re.search(p, text.lower()))
        paras = text.count("\n\n") + 1
        return min((hits / max(paras, 1)) * 0.5 + (min(paras / 10, 1.0) * 0.5), 1.0)
