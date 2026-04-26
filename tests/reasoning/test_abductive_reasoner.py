"""Tests for abductive_reasoner — abductive inference engine."""

from __future__ import annotations

from src.reasoning.abductive_reasoner import (
    AbductiveReasoner,
    Hypothesis,
    Observation,
    abduce,
)


class TestObservation:
    def test_observation_creation(self):
        obs = Observation("the ground is wet")
        assert obs.statement == "the ground is wet"


class TestHypothesis:
    def test_hypothesis_creation(self):
        h = Hypothesis("it rained", plausibility=0.8)
        assert h.explanation == "it rained"
        assert h.plausibility == 0.8

    def test_default_plausibility(self):
        h = Hypothesis("it rained")
        assert h.plausibility == 0.5


class TestAbduce:
    def test_simple_abduction(self):
        obs = Observation("floor is wet")
        rules = {
            "it rained": "floor is wet",
            "someone spilled": "floor is wet",
        }
        hyps = abduce(obs, rules)
        assert len(hyps) == 2
        assert all(isinstance(h, Hypothesis) for h in hyps)

    def test_no_match_returns_empty(self):
        obs = Observation("sky is blue")
        hyps = abduce(obs, {})
        assert hyps == []


class TestAbductiveReasoner:
    def test_add_and_rank_hypotheses(self):
        reasoner = AbductiveReasoner()
        reasoner.add_rule("it rained", "ground is wet")
        reasoner.add_rule("sprinklers", "ground is wet")
        hyps = reasoner.explain(Observation("ground is wet"))
        assert len(hyps) == 2
        assert hyps[0].plausibility >= hyps[1].plausibility

    def test_empty_rules_no_hypotheses(self):
        reasoner = AbductiveReasoner()
        hyps = reasoner.explain(Observation("something happened"))
        assert hyps == []

    def test_most_plausible_picks_best(self):
        reasoner = AbductiveReasoner()
        reasoner.add_rule("common cause", "effect", plausibility=0.9)
        reasoner.add_rule("rare cause", "effect", plausibility=0.1)
        best = reasoner.most_plausible(Observation("effect"))
        assert best is not None
        assert best.explanation == "common cause"

    def test_most_plausible_no_hypotheses(self):
        reasoner = AbductiveReasoner()
        assert reasoner.most_plausible(Observation("x")) is None

    def test_rules_count(self):
        reasoner = AbductiveReasoner()
        reasoner.add_rule("a", "b")
        reasoner.add_rule("c", "d")
        assert reasoner.rule_count() == 2
