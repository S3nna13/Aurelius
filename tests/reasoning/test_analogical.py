"""Tests for analogical reasoner."""

from __future__ import annotations

from src.reasoning.analogical import AnalogicalExample, AnalogicalReasoner


class TestAnalogicalReasoner:
    def test_add_and_retrieve(self):
        ar = AnalogicalReasoner()
        ar.add_example(AnalogicalExample("how to sort a list", "use sorted()"))
        ar.add_example(AnalogicalExample("how to reverse a string", "use [::-1]"))
        results = ar.retrieve("sorting a list", top_k=1)
        assert len(results) == 1
        assert "sort" in results[0].problem

    def test_retrieve_empty(self):
        ar = AnalogicalReasoner()
        results = ar.retrieve("anything")
        assert results == []

    def test_solve_by_analogy_no_match(self):
        ar = AnalogicalReasoner()
        result = ar.solve_by_analogy("unique problem")
        assert result is None

    def test_solve_by_analogy_with_match(self):
        ar = AnalogicalReasoner()
        ar.add_example(AnalogicalExample("how to parse json", "json.loads"))
        result = ar.solve_by_analogy("parsing json data")
        assert result is not None

    def test_score_similarity_perfect(self):
        ar = AnalogicalReasoner()
        score = ar._score_similarity("hello world", "hello world")
        assert score == 1.0

    def test_score_similarity_partial(self):
        ar = AnalogicalReasoner()
        score = ar._score_similarity("hello world", "hello there")
        assert 0.0 < score < 1.0
