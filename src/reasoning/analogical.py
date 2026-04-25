"""Analogical reasoning module — retrieve and adapt similar solved problems."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AnalogicalExample:
    problem: str
    solution: str
    domain: str = ""
    similarity: float = 0.0


@dataclass
class AnalogicalReasoner:
    """Solve problems by retrieving similar cases and adapting solutions."""

    examples: list[AnalogicalExample] = field(default_factory=list)

    def add_example(self, example: AnalogicalExample) -> None:
        self.examples.append(example)

    def retrieve(self, problem: str, top_k: int = 3) -> list[AnalogicalExample]:
        scored = []
        for ex in self.examples:
            score = self._score_similarity(problem, ex.problem)
            ex.similarity = score
            scored.append(ex)
        scored.sort(key=lambda x: x.similarity, reverse=True)
        return scored[:top_k]

    def solve_by_analogy(self, problem: str) -> str | None:
        matches = self.retrieve(problem, top_k=1)
        if not matches:
            return None
        return self._adapt(matches[0].solution, problem)

    def _score_similarity(self, a: str, b: str) -> float:
        set_a = set(a.lower().split())
        set_b = set(b.lower().split())
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a | set_b)

    def _adapt(self, solution: str, problem: str) -> str:
        return solution


ANALOGICAL_REASONER = AnalogicalReasoner()