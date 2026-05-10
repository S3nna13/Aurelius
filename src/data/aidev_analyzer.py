"""AIDev — analyze 932K Agentic-PRs, review quality, security."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgenticPR:
    repo: str
    author: str
    agent: str
    merged: bool = False
    review_score: float = 0.0
    vulnerability_count: int = 0


class AIDevAnalyzer:
    def __init__(self):
        self.prs: list[AgenticPR] = []

    def load(self, data: list[dict]) -> list[AgenticPR]:
        self.prs = [AgenticPR(**{k: v for k, v in d.items() if k in AgenticPR.__annotations__}) for d in data]
        return self.prs

    def merge_rate(self) -> float:
        if not self.prs:
            return 0.0
        return sum(1 for p in self.prs if p.merged) / len(self.prs)

    def vuln_density(self) -> float:
        if not self.prs:
            return 0.0
        return sum(p.vulnerability_count for p in self.prs) / len(self.prs)
