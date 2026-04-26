"""Self-consistency: Wang et al. 2022 'Self-Consistency Improves Chain of Thought Reasoning'."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class SelfConsistencyConfig:
    n_samples: int = 10
    temperature: float = 0.7
    aggregation: str = "majority_vote"


@dataclass(frozen=True)
class ConsistencyResult:
    answer: str
    confidence: float
    n_samples: int
    vote_distribution: dict[str, int]


class SelfConsistency:
    def __init__(self, config: SelfConsistencyConfig | None = None) -> None:
        self.config = config or SelfConsistencyConfig()

    def extract_answer(self, text: str) -> str:
        for pattern in [
            r"####\s*(.+)",
            r"[Aa]nswer:\s*(.+)",
            r"=\s*([^\n]+)",
            r"[Tt]he answer is\s*(.+)",
        ]:
            m = re.search(pattern, text)
            if m:
                return m.group(1).strip()
        return text.strip().splitlines()[-1].strip() if text.strip() else ""

    def aggregate(self, samples: list[str]) -> ConsistencyResult:
        if self.config.aggregation == "weighted_vote":
            weights: dict[str, float] = {}
            for s in samples:
                m = re.match(r"WEIGHT:([\d.]+):(.*)", s)
                if m:
                    w = float(m.group(1))
                    ans = m.group(2).strip()
                else:
                    w = 1.0
                    ans = s.strip()
                weights[ans] = weights.get(ans, 0.0) + w

            best = max(weights, key=lambda a: weights[a])
            total = sum(weights.values())
            confidence = weights[best] / total if total else 0.0
            vote_dist = {a: int(round(w)) for a, w in weights.items()}
            return ConsistencyResult(
                answer=best,
                confidence=confidence,
                n_samples=len(samples),
                vote_distribution=vote_dist,
            )

        counts = Counter(samples)
        best, count = counts.most_common(1)[0]
        return ConsistencyResult(
            answer=best,
            confidence=count / len(samples),
            n_samples=len(samples),
            vote_distribution=dict(counts),
        )

    def run(self, question: str, generate_fn: Callable[[str], str]) -> ConsistencyResult:
        samples = [self.extract_answer(generate_fn(question)) for _ in range(self.config.n_samples)]
        return self.aggregate(samples)


SELF_CONSISTENCY_REGISTRY: dict[str, type[SelfConsistency]] = {"default": SelfConsistency}
