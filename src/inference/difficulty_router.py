from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum

_MATH_KEYWORDS = {
    "math",
    "proof",
    "theorem",
    "lemma",
    "corollary",
    "integral",
    "derivative",
    "equation",
    "algebra",
    "calculus",
    "statistics",
    "probability",
    "matrix",
    "vector",
    "eigenvalue",
    "gradient",
}

_CODE_KEYWORDS = {
    "function",
    "algorithm",
    "implement",
    "code",
    "class",
    "debug",
    "compile",
    "runtime",
    "recursion",
    "complexity",
    "python",
    "javascript",
    "typescript",
    "sql",
    "api",
    "async",
    "concurrency",
}

_SIMPLE_KEYWORDS = {
    "hi",
    "hello",
    "thanks",
    "thank",
    "greet",
    "hey",
    "bye",
    "goodbye",
    "please",
    "ok",
    "okay",
    "yes",
    "no",
    "sure",
}


class RouterStrategy(StrEnum):
    HEURISTIC = "heuristic"
    MATRIX_FACTOR = "mf"
    HYBRID = "hybrid"


@dataclass
class RoutingDecision:
    strategy: RouterStrategy
    complexity_score: float
    routed_to: str
    threshold: float
    reason: str


class DifficultyRouter:
    """RouteLLM-inspired: route easy queries to weak model, hard to strong model."""

    def __init__(
        self,
        threshold: float = 0.5,
        strategy: RouterStrategy = RouterStrategy.HYBRID,
    ) -> None:
        self.threshold = threshold
        self.strategy = strategy

    def score_heuristic(self, prompt: str) -> float:
        words = prompt.split()
        word_count = len(words)
        score = min(word_count / 400.0, 1.0)

        lowered = {w.strip(".,!?;:\"'()[]{}").lower() for w in words}

        if lowered & _MATH_KEYWORDS:
            score += 0.2
        if lowered & _CODE_KEYWORDS:
            score += 0.1
        if lowered & _SIMPLE_KEYWORDS:
            score -= 0.1

        return max(0.0, min(1.0, score))

    def score_mf(self, prompt: str) -> float:
        digest = hashlib.sha256(prompt.encode()).hexdigest()
        return (int(digest, 16) % 1000) / 1000.0

    def route(self, prompt: str) -> RoutingDecision:
        if self.strategy == RouterStrategy.HEURISTIC:
            score = self.score_heuristic(prompt)
            reason = "heuristic only"
        elif self.strategy == RouterStrategy.MATRIX_FACTOR:
            score = self.score_mf(prompt)
            reason = "matrix-factor stub only"
        else:
            h = self.score_heuristic(prompt)
            mf = self.score_mf(prompt)
            score = 0.6 * h + 0.4 * mf
            reason = f"hybrid: heuristic={h:.3f}, mf={mf:.3f}"

        routed_to = "strong" if score > self.threshold else "weak"
        return RoutingDecision(
            strategy=self.strategy,
            complexity_score=score,
            routed_to=routed_to,
            threshold=self.threshold,
            reason=reason,
        )

    def batch_route(self, prompts: list[str]) -> list[RoutingDecision]:
        return [self.route(p) for p in prompts]
