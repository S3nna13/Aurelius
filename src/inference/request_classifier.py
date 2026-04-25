"""Request classifier: heuristic task-type and complexity detection for inference routing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Pattern


class TaskType(str, Enum):
    CHAT = "chat"
    CODE = "code"
    MATH = "math"
    SUMMARIZE = "summarize"
    TRANSLATE = "translate"
    REASONING = "reasoning"
    RETRIEVAL = "retrieval"
    UNKNOWN = "unknown"


class ComplexityTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class ClassificationResult:
    task_type: TaskType
    complexity: ComplexityTier
    confidence: float
    token_estimate: int


_PATTERNS: list[tuple[TaskType, Pattern[str]]] = [
    (TaskType.CODE, re.compile(
        r"\b(def |function |class |import |#include|var |const |let |=>|lambda|async def)\b",
        re.IGNORECASE,
    )),
    (TaskType.MATH, re.compile(
        r"\b(equation|solve|calculate|integral|derivative|matrix|modulo|factorial|==|!=|<=|>=)\b",
        re.IGNORECASE,
    )),
    (TaskType.SUMMARIZE, re.compile(
        r"\b(summarize|summary|tldr|tl;dr|brief|condense|shorten|abstract)\b",
        re.IGNORECASE,
    )),
    (TaskType.TRANSLATE, re.compile(
        r"\b(translate|translation|in (spanish|french|german|japanese|chinese|arabic|portuguese|italian|korean|russian))\b",
        re.IGNORECASE,
    )),
    (TaskType.REASONING, re.compile(
        r"\b(why|explain|reason|because|therefore|analyze|analysis|compare|evaluate|justify)\b",
        re.IGNORECASE,
    )),
    (TaskType.RETRIEVAL, re.compile(
        r"\b(find|search|lookup|retrieve|fetch|query|what is|who is|where is|when did)\b",
        re.IGNORECASE,
    )),
]

_LOW_WORD_MAX = 200
_HIGH_WORD_MIN = 800


def _word_count(text: str) -> int:
    return len(text.split())


def _complexity(word_count: int) -> ComplexityTier:
    if word_count < _LOW_WORD_MAX:
        return ComplexityTier.LOW
    if word_count <= _HIGH_WORD_MIN:
        return ComplexityTier.MEDIUM
    return ComplexityTier.HIGH


def _token_estimate(word_count: int) -> int:
    return round(word_count * 1.3)


class RequestClassifier:
    def classify(self, prompt: str) -> ClassificationResult:
        words = _word_count(prompt)
        complexity = _complexity(words)
        token_est = _token_estimate(words)

        scores: dict[TaskType, int] = {}
        for task_type, pattern in _PATTERNS:
            matches = len(pattern.findall(prompt))
            if matches:
                scores[task_type] = scores.get(task_type, 0) + matches

        if scores:
            best = max(scores, key=lambda t: scores[t])
            total = sum(scores.values())
            confidence = min(scores[best] / max(total, 1), 1.0)
        else:
            best = TaskType.CHAT
            confidence = 0.5

        return ClassificationResult(
            task_type=best,
            complexity=complexity,
            confidence=round(confidence, 4),
            token_estimate=token_est,
        )

    def batch_classify(self, prompts: list[str]) -> list[ClassificationResult]:
        return [self.classify(p) for p in prompts]
