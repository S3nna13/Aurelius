from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum


class QualityIssue(StrEnum):
    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"
    HIGH_REPETITION = "high_repetition"
    LOW_ENTROPY = "low_entropy"
    TOXIC_KEYWORDS = "toxic_keywords"
    ENCODING_ERROR = "encoding_error"


@dataclass
class QualityReport:
    text: str
    issues: list[QualityIssue]
    passed: bool
    stats: dict


@dataclass
class QualityConfig:
    min_chars: int = 10
    max_chars: int = 100_000
    repetition_threshold: float = 0.4
    entropy_threshold: float = 2.5
    toxic_keywords: list[str] = field(default_factory=list)


class DataQualityChecker:
    def __init__(self, config: QualityConfig | None = None) -> None:
        self.config = config if config is not None else QualityConfig()

    def _char_entropy(self, text: str) -> float:
        if not text:
            return 0.0
        counts = Counter(text)
        total = len(text)
        return -sum((c / total) * math.log2(c / total) for c in counts.values())

    def _repetition_ratio(self, text: str, n: int = 4) -> float:
        words = text.split()
        if len(words) < n:
            return 0.0
        ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
        total = len(ngrams)
        unique = len(set(ngrams))
        return (total - unique) / max(total, 1)

    def check(self, text: str) -> QualityReport:
        cfg = self.config
        issues: list[QualityIssue] = []

        try:
            roundtripped = text.encode("utf-8").decode("utf-8")
            if roundtripped != text:
                issues.append(QualityIssue.ENCODING_ERROR)
        except (UnicodeEncodeError, UnicodeDecodeError):
            issues.append(QualityIssue.ENCODING_ERROR)

        char_count = len(text)
        word_count = len(text.split())
        entropy = self._char_entropy(text)
        repetition_ratio = self._repetition_ratio(text)

        if char_count < cfg.min_chars:
            issues.append(QualityIssue.TOO_SHORT)
        if char_count > cfg.max_chars:
            issues.append(QualityIssue.TOO_LONG)
        if repetition_ratio > cfg.repetition_threshold:
            issues.append(QualityIssue.HIGH_REPETITION)
        if entropy < cfg.entropy_threshold:
            issues.append(QualityIssue.LOW_ENTROPY)

        lower = text.lower()
        for kw in cfg.toxic_keywords:
            if kw.lower() in lower:
                issues.append(QualityIssue.TOXIC_KEYWORDS)
                break

        return QualityReport(
            text=text,
            issues=issues,
            passed=len(issues) == 0,
            stats={
                "char_count": char_count,
                "word_count": word_count,
                "entropy": entropy,
                "repetition_ratio": repetition_ratio,
            },
        )

    def filter(self, texts: list[str]) -> list[str]:
        return [t for t in texts if self.check(t).passed]

    def batch_check(self, texts: list[str]) -> list[QualityReport]:
        return [self.check(t) for t in texts]

    def summary(self, reports: list[QualityReport]) -> dict:
        total = len(reports)
        passed = sum(1 for r in reports if r.passed)
        issue_counts: dict[str, int] = {}
        for r in reports:
            for issue in r.issues:
                issue_counts[issue.value] = issue_counts.get(issue.value, 0) + 1
        return {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "issue_counts": issue_counts,
        }
