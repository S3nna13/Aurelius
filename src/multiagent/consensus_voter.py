"""Consensus voter for multi-agent decision making.

Supports majority, unanimous, and weighted voting.
Fail closed: ties default to rejection unless configured otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Vote(Enum):
    YES = "yes"
    NO = "no"
    ABSTAIN = "abstain"


class TieBreak(Enum):
    REJECT = "reject"
    ACCEPT = "accept"


@dataclass
class Ballot:
    agent_id: str
    vote: Vote
    weight: float = 1.0
    rationale: str = ""


@dataclass
class ConsensusResult:
    passed: bool
    yes_weight: float
    no_weight: float
    abstain_weight: float
    total_votes: int
    tie_broken: bool
    ballots: list[Ballot] = field(default_factory=list)


@dataclass
class ConsensusVoter:
    """Tallies weighted ballots and decides if consensus is reached."""

    threshold: float = 0.5
    require_unanimous: bool = False
    tie_break: TieBreak = TieBreak.REJECT

    def tally(self, ballots: list[Ballot]) -> ConsensusResult:
        """Tally ballots and return a ConsensusResult."""
        yes_weight = 0.0
        no_weight = 0.0
        abstain_weight = 0.0
        total_votes = 0

        for b in ballots:
            if b.vote == Vote.YES:
                yes_weight += b.weight
            elif b.vote == Vote.NO:
                no_weight += b.weight
            elif b.vote == Vote.ABSTAIN:
                abstain_weight += b.weight
            total_votes += 1

        effective_total = yes_weight + no_weight
        tie_broken = False

        if self.require_unanimous:
            passed = yes_weight > 0 and no_weight == 0 and abstain_weight == 0
        elif effective_total == 0:
            passed = False
        else:
            yes_ratio = yes_weight / effective_total
            no_ratio = no_weight / effective_total
            if yes_ratio == no_ratio:
                tie_broken = True
                passed = self.tie_break == TieBreak.ACCEPT
            else:
                passed = yes_ratio >= self.threshold

        return ConsensusResult(
            passed=passed,
            yes_weight=yes_weight,
            no_weight=no_weight,
            abstain_weight=abstain_weight,
            total_votes=total_votes,
            tie_broken=tie_broken,
            ballots=list(ballots),
        )

    def vote(
        self,
        agent_id: str,
        choice: Vote,
        weight: float = 1.0,
        rationale: str = "",
    ) -> Ballot:
        """Convenience factory for a single ballot."""
        return Ballot(agent_id=agent_id, vote=choice, weight=weight, rationale=rationale)


# Module-level registry
CONSENSUS_VOTER_REGISTRY: dict[str, ConsensusVoter] = {}
DEFAULT_CONSENSUS_VOTER = ConsensusVoter()
CONSENSUS_VOTER_REGISTRY["default"] = DEFAULT_CONSENSUS_VOTER
