"""Tests for consensus_voter — ballot tallying and tie breaking."""
from __future__ import annotations

import pytest

from src.multiagent.consensus_voter import (
    Ballot,
    ConsensusResult,
    ConsensusVoter,
    TieBreak,
    Vote,
    CONSENSUS_VOTER_REGISTRY,
    DEFAULT_CONSENSUS_VOTER,
)


# ---------------------------------------------------------------------------
# Majority threshold
# ---------------------------------------------------------------------------


def test_simple_majority_passes():
    cv = ConsensusVoter(threshold=0.5)
    ballots = [
        Ballot("a", Vote.YES, 1.0),
        Ballot("b", Vote.YES, 1.0),
        Ballot("c", Vote.NO, 1.0),
    ]
    result = cv.tally(ballots)
    assert result.passed is True
    assert result.yes_weight == 2.0
    assert result.no_weight == 1.0


def test_simple_majority_fails():
    cv = ConsensusVoter(threshold=0.5)
    ballots = [
        Ballot("a", Vote.YES, 1.0),
        Ballot("b", Vote.NO, 1.0),
        Ballot("c", Vote.NO, 1.0),
    ]
    result = cv.tally(ballots)
    assert result.passed is False


# ---------------------------------------------------------------------------
# Weighted voting
# ---------------------------------------------------------------------------


def test_weighted_voting():
    cv = ConsensusVoter(threshold=0.5)
    ballots = [
        Ballot("a", Vote.YES, 3.0),
        Ballot("b", Vote.NO, 2.0),
    ]
    result = cv.tally(ballots)
    assert result.passed is True
    assert result.yes_weight == 3.0


# ---------------------------------------------------------------------------
# Unanimous
# ---------------------------------------------------------------------------


def test_unanimous_passes():
    cv = ConsensusVoter(require_unanimous=True)
    ballots = [
        Ballot("a", Vote.YES, 1.0),
        Ballot("b", Vote.YES, 1.0),
    ]
    result = cv.tally(ballots)
    assert result.passed is True


def test_unanimous_fails_with_abstain():
    cv = ConsensusVoter(require_unanimous=True)
    ballots = [
        Ballot("a", Vote.YES, 1.0),
        Ballot("b", Vote.ABSTAIN, 1.0),
    ]
    result = cv.tally(ballots)
    assert result.passed is False


def test_unanimous_fails_with_no():
    cv = ConsensusVoter(require_unanimous=True)
    ballots = [
        Ballot("a", Vote.YES, 1.0),
        Ballot("b", Vote.NO, 1.0),
    ]
    result = cv.tally(ballots)
    assert result.passed is False


# ---------------------------------------------------------------------------
# Tie breaking
# ---------------------------------------------------------------------------


def test_tie_reject_by_default():
    cv = ConsensusVoter(threshold=0.5, tie_break=TieBreak.REJECT)
    ballots = [
        Ballot("a", Vote.YES, 1.0),
        Ballot("b", Vote.NO, 1.0),
    ]
    result = cv.tally(ballots)
    assert result.passed is False
    assert result.tie_broken is True


def test_tie_accept_when_configured():
    cv = ConsensusVoter(threshold=0.5, tie_break=TieBreak.ACCEPT)
    ballots = [
        Ballot("a", Vote.YES, 1.0),
        Ballot("b", Vote.NO, 1.0),
    ]
    result = cv.tally(ballots)
    assert result.passed is True
    assert result.tie_broken is True


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_ballots_fails():
    cv = ConsensusVoter()
    result = cv.tally([])
    assert result.passed is False
    assert result.total_votes == 0


def test_all_abstain_fails():
    cv = ConsensusVoter()
    ballots = [
        Ballot("a", Vote.ABSTAIN, 1.0),
        Ballot("b", Vote.ABSTAIN, 1.0),
    ]
    result = cv.tally(ballots)
    assert result.passed is False


# ---------------------------------------------------------------------------
# Convenience vote factory
# ---------------------------------------------------------------------------


def test_vote_factory():
    cv = ConsensusVoter()
    b = cv.vote("alice", Vote.YES, weight=2.0, rationale="looks good")
    assert b.agent_id == "alice"
    assert b.vote == Vote.YES
    assert b.weight == 2.0
    assert b.rationale == "looks good"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_contains_default():
    assert "default" in CONSENSUS_VOTER_REGISTRY
    assert isinstance(CONSENSUS_VOTER_REGISTRY["default"], ConsensusVoter)


def test_default_is_consensus_voter():
    assert isinstance(DEFAULT_CONSENSUS_VOTER, ConsensusVoter)
