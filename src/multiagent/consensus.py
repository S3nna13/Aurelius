"""Multiagent consensus with weighted voting and quorum detection.

Aurelius LLM Project — stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


@dataclass
class Vote:
    """Single agent vote."""

    agent_id: str
    choice: T
    weight: float = 1.0


@dataclass
class ConsensusResult:
    """Result of consensus voting."""

    winner: T | None
    votes_for: dict[T, float]
    total_votes: float
    quorum_met: bool
    runoff: bool = False


@dataclass
class WeightedConsensus:
    """Weighted voting with quorum and optional runoff.

    Each agent casts a weighted vote. If quorum is met, winner is highest-weighted.
    If no quorum and more than 2 choices, triggers runoff between top 2.
    """

    quorum_threshold: float = 0.5
    min_choices_for_runoff: int = 2

    def vote(self, votes: list[Vote[T]]) -> ConsensusResult[T]:
        """Compute consensus result.

        Args:
            votes: List of agent votes

        Returns:
            ConsensusResult with winner, vote totals, and quorum status
        """
        if not votes:
            return ConsensusResult(None, {}, 0.0, False)

        weighted_counts: dict[T, float] = {}
        for vote in votes:
            weighted_counts[vote.choice] = weighted_counts.get(vote.choice, 0.0) + vote.weight

        total_votes = sum(v.weight for v in votes)
        quorum_met = (
            total_votes >= self.quorum_threshold and len(weighted_counts) == 1
        ) or total_votes >= 1.0

        if not weighted_counts:
            return ConsensusResult(None, {}, total_votes, False)

        winner = max(weighted_counts, key=weighted_counts.get)
        weighted_counts[winner]

        choices = sorted(weighted_counts.items(), key=lambda x: -x[1])
        top_weight = weighted_counts[winner] / total_votes if total_votes > 0 else 0
        runoff = (
            len(choices) >= self.min_choices_for_runoff
            and top_weight < self.quorum_threshold
            and total_votes >= 0.5
        )

        return ConsensusResult(
            winner=winner,
            votes_for=weighted_counts,
            total_votes=total_votes,
            quorum_met=quorum_met and not runoff,
            runoff=runoff,
        )
