"""Tests for src/alignment/constitutional_committee.py."""

from __future__ import annotations

import pytest

from src.alignment.constitutional_committee import (
    CommitteeConfig,
    CommitteeMember,
    CommitteeVote,
    ConstitutionalCommittee,
    create_default_committee,
    score_with_committee,
)

# ---------------------------------------------------------------------------
# Test 1: CommitteeConfig defaults
# ---------------------------------------------------------------------------


def test_committee_config_defaults():
    cfg = CommitteeConfig()
    assert cfg.aggregation == "weighted_mean"
    assert cfg.revision_threshold == 0.5
    assert cfg.n_members == 4
    assert cfg.use_default_constitution is True


# ---------------------------------------------------------------------------
# Test 2: create_default_committee returns 4-member committee
# ---------------------------------------------------------------------------


def test_create_default_committee_four_members():
    committee = create_default_committee()
    assert isinstance(committee, ConstitutionalCommittee)
    assert len(committee.members) == 4
    member_names = {m.name for m in committee.members}
    assert member_names == {"harm_avoidance", "helpfulness", "honesty", "fairness"}


# ---------------------------------------------------------------------------
# Test 3: evaluate returns one CommitteeVote per member
# ---------------------------------------------------------------------------


def test_evaluate_returns_vote_per_member():
    committee = create_default_committee()
    votes = committee.evaluate("This is a perfectly safe and helpful response.")
    assert len(votes) == 4
    assert all(isinstance(v, CommitteeVote) for v in votes)
    returned_names = {v.member_name for v in votes}
    assert returned_names == {"harm_avoidance", "helpfulness", "honesty", "fairness"}


# ---------------------------------------------------------------------------
# Test 4: All scores are in [0, 1]
# ---------------------------------------------------------------------------


def test_all_scores_in_unit_interval():
    committee = create_default_committee()
    responses = [
        "I can help you with that — here is a step-by-step guide.",
        "Kill everyone immediately with a bomb.",
        "",
        "A" * 2000,
    ]
    for resp in responses:
        votes = committee.evaluate(resp)
        for v in votes:
            assert 0.0 <= v.score <= 1.0, (
                f"Score {v.score} out of range for member '{v.member_name}' "
                f"on response: {resp[:50]!r}"
            )


# ---------------------------------------------------------------------------
# Test 5: aggregate_votes weighted_mean returns (weighted) mean
# ---------------------------------------------------------------------------


def test_aggregate_weighted_mean():
    members = [
        CommitteeMember(name="a", principle="p", weight=1.0),
        CommitteeMember(name="b", principle="p", weight=1.0),
    ]
    committee = ConstitutionalCommittee(members=members, aggregation="weighted_mean")
    votes = [
        CommitteeVote(member_name="a", score=0.8, critique="ok", revision_needed=False),
        CommitteeVote(member_name="b", score=0.6, critique="ok", revision_needed=False),
    ]
    final_score, needs_revision = committee.aggregate_votes(votes)
    assert abs(final_score - 0.7) < 1e-4
    assert needs_revision is False  # 0.7 >= 0.5


def test_aggregate_weighted_mean_with_weights():
    members = [
        CommitteeMember(name="a", principle="p", weight=2.0),
        CommitteeMember(name="b", principle="p", weight=1.0),
    ]
    committee = ConstitutionalCommittee(members=members, aggregation="weighted_mean")
    votes = [
        CommitteeVote(member_name="a", score=0.9, critique="ok", revision_needed=False),
        CommitteeVote(member_name="b", score=0.3, critique="bad", revision_needed=True),
    ]
    final_score, _ = committee.aggregate_votes(votes)
    expected = (0.9 * 2.0 + 0.3 * 1.0) / 3.0
    assert abs(final_score - expected) < 1e-4


# ---------------------------------------------------------------------------
# Test 6: aggregate_votes 'min' returns minimum score
# ---------------------------------------------------------------------------


def test_aggregate_min():
    members = [
        CommitteeMember(name="x", principle="p", weight=1.0),
        CommitteeMember(name="y", principle="p", weight=1.0),
        CommitteeMember(name="z", principle="p", weight=1.0),
    ]
    committee = ConstitutionalCommittee(members=members, aggregation="min")
    votes = [
        CommitteeVote(member_name="x", score=0.9, critique="", revision_needed=False),
        CommitteeVote(member_name="y", score=0.2, critique="", revision_needed=True),
        CommitteeVote(member_name="z", score=0.7, critique="", revision_needed=False),
    ]
    final_score, needs_revision = committee.aggregate_votes(votes)
    assert abs(final_score - 0.2) < 1e-4
    assert needs_revision is True  # 0.2 < 0.5


# ---------------------------------------------------------------------------
# Test 7: aggregate_votes majority_vote returns True if majority needs revision
# ---------------------------------------------------------------------------


def test_aggregate_majority_vote_needs_revision():
    members = [CommitteeMember(name=f"m{i}", principle="p", weight=1.0) for i in range(4)]
    committee = ConstitutionalCommittee(members=members, aggregation="majority_vote")

    # 3 out of 4 flag revision → majority → needs_revision = True
    votes_majority = [
        CommitteeVote(member_name="m0", score=0.3, critique="", revision_needed=True),
        CommitteeVote(member_name="m1", score=0.4, critique="", revision_needed=True),
        CommitteeVote(member_name="m2", score=0.4, critique="", revision_needed=True),
        CommitteeVote(member_name="m3", score=0.8, critique="", revision_needed=False),
    ]
    _, needs_revision = committee.aggregate_votes(votes_majority)
    assert needs_revision is True

    # 1 out of 4 flags revision → minority → needs_revision = False
    votes_minority = [
        CommitteeVote(member_name="m0", score=0.9, critique="", revision_needed=False),
        CommitteeVote(member_name="m1", score=0.8, critique="", revision_needed=False),
        CommitteeVote(member_name="m2", score=0.7, critique="", revision_needed=False),
        CommitteeVote(member_name="m3", score=0.3, critique="", revision_needed=True),
    ]
    _, needs_revision2 = committee.aggregate_votes(votes_minority)
    assert needs_revision2 is False


# ---------------------------------------------------------------------------
# Test 8: get_revision_instructions includes critiques from flagged members
# ---------------------------------------------------------------------------


def test_get_revision_instructions_contains_flagged_critiques():
    members = [
        CommitteeMember(name="harm_avoidance", principle="p", weight=1.0),
        CommitteeMember(name="helpfulness", principle="p", weight=1.0),
    ]
    committee = ConstitutionalCommittee(members=members)
    votes = [
        CommitteeVote(
            member_name="harm_avoidance",
            score=0.2,
            critique="Dangerous language detected.",
            revision_needed=True,
        ),
        CommitteeVote(
            member_name="helpfulness",
            score=0.8,
            critique="Response is sufficiently helpful.",
            revision_needed=False,
        ),
    ]
    instructions = committee.get_revision_instructions(votes)
    assert "harm_avoidance" in instructions
    assert "Dangerous language detected." in instructions
    # helpfulness was not flagged — should not appear
    assert "Response is sufficiently helpful." not in instructions


def test_get_revision_instructions_empty_when_no_flags():
    committee = create_default_committee()
    votes = [
        CommitteeVote(member_name=m.name, score=0.9, critique="Fine.", revision_needed=False)
        for m in committee.members
    ]
    assert committee.get_revision_instructions(votes) == ""


# ---------------------------------------------------------------------------
# Test 9: filter_dataset keeps only high-scoring examples
# ---------------------------------------------------------------------------


def test_filter_dataset_keeps_high_scores():
    committee = create_default_committee()
    examples = [
        {
            "response": (
                "I can help you with that. Here is a step-by-step explanation to guide you "
                "through the process clearly and safely. Consider the following approach."
            )
        },
        {"response": "kill bomb weapon harm suicide illegal"},
        {"response": ""},
    ]
    kept = committee.filter_dataset(examples, threshold=0.5)
    # Each kept example must have committee_score >= 0.5
    for ex in kept:
        assert ex["committee_score"] >= 0.5
    # The safe response should be included; the harmful one should be excluded
    assert any("step-by-step" in ex["response"] for ex in kept)


def test_filter_dataset_adds_committee_score_key():
    committee = create_default_committee()
    examples = [{"response": "Here is a helpful and safe explanation. You can try this approach."}]
    result = committee.filter_dataset(examples, threshold=0.0)
    assert len(result) == 1
    assert "committee_score" in result[0]
    assert isinstance(result[0]["committee_score"], float)


# ---------------------------------------------------------------------------
# Test 10: score_with_committee returns dict with all required keys
# ---------------------------------------------------------------------------


def test_score_with_committee_keys():
    committee = create_default_committee()
    result = score_with_committee(committee, "This is a helpful response.")
    assert "score" in result
    assert "needs_revision" in result
    assert "critiques" in result
    assert "member_scores" in result
    assert isinstance(result["score"], float)
    assert isinstance(result["needs_revision"], bool)
    assert isinstance(result["critiques"], list)
    assert isinstance(result["member_scores"], dict)
    assert len(result["critiques"]) == 4
    assert len(result["member_scores"]) == 4


# ---------------------------------------------------------------------------
# Test 11: Safe response scores higher than harmful response
# ---------------------------------------------------------------------------


def test_safe_response_scores_higher_than_harmful():
    committee = create_default_committee()

    safe = (
        "I can help you with that. Here is a clear, safe explanation with practical steps "
        "to guide you through the process. Please consider the following advice carefully."
    )
    harmful = "You should kill them, use a weapon, and commit fraud. Hack their system and steal everything."  # noqa: E501

    safe_result = score_with_committee(committee, safe)
    harmful_result = score_with_committee(committee, harmful)

    assert safe_result["score"] > harmful_result["score"], (
        f"Expected safe response ({safe_result['score']:.3f}) to score higher than "
        f"harmful response ({harmful_result['score']:.3f})."
    )


# ---------------------------------------------------------------------------
# Test 12: Empty committee raises ValueError
# ---------------------------------------------------------------------------


def test_empty_committee_raises_value_error():
    with pytest.raises(ValueError, match="at least one CommitteeMember"):
        ConstitutionalCommittee(members=[])
