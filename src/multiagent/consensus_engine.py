"""Consensus aggregation engine for multiagent outputs."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum


class ConsensusMethod(str, Enum):
    MAJORITY_VOTE = "majority_vote"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    BORDA_COUNT = "borda_count"
    PLURALITY = "plurality"


@dataclass
class ConsensusConfig:
    method: ConsensusMethod = ConsensusMethod.MAJORITY_VOTE
    min_agreement: float = 0.5


@dataclass(frozen=True)
class ConsensusResult:
    winner: str
    agreement: float
    method: str


class ConsensusEngine:
    def __init__(self, config: ConsensusConfig | None = None) -> None:
        self._config = config or ConsensusConfig()

    def aggregate(self, responses: list[str]) -> ConsensusResult:
        method = self._config.method
        n = len(responses)

        if method in (ConsensusMethod.MAJORITY_VOTE, ConsensusMethod.PLURALITY):
            counts = Counter(responses)
            winner, count = counts.most_common(1)[0]
            return ConsensusResult(winner=winner, agreement=count / n, method=method.value)

        if method == ConsensusMethod.CONFIDENCE_WEIGHTED:
            tally: dict[str, float] = {}
            for resp in responses:
                if resp.startswith("CONFIDENCE:"):
                    parts = resp.split(":", 2)
                    if len(parts) == 3:
                        try:
                            weight = float(parts[1])
                            answer = parts[2]
                            tally[answer] = tally.get(answer, 0.0) + weight
                            continue
                        except ValueError:
                            pass
                tally[resp] = tally.get(resp, 0.0) + 1.0
            winner = max(tally, key=lambda k: tally[k])
            total = sum(tally.values())
            agreement = tally[winner] / total if total else 0.0
            return ConsensusResult(winner=winner, agreement=agreement, method=method.value)

        if method == ConsensusMethod.BORDA_COUNT:
            scores: dict[str, float] = {}
            for rank, resp in enumerate(responses):
                score = len(responses) - 1 - rank
                scores[resp] = scores.get(resp, 0.0) + score
            winner = max(scores, key=lambda k: scores[k])
            counts2 = Counter(responses)
            agreement = counts2[winner] / n
            return ConsensusResult(winner=winner, agreement=agreement, method=method.value)

        raise ValueError(f"Unknown method: {method}")

    def reached_consensus(self, result: ConsensusResult) -> bool:
        return result.agreement >= self._config.min_agreement


CONSENSUS_REGISTRY: dict[str, type[ConsensusEngine]] = {"default": ConsensusEngine}
