"""Tests for analogy_engine — analogical reasoning engine."""
from __future__ import annotations

from src.reasoning.analogy_engine import AnalogyEngine, Analogy, Concept


class TestConcept:
    def test_concept_with_attributes(self):
        c = Concept("apple", attributes={"color": "red", "shape": "round", "size": "small"})
        assert c.name == "apple"
        assert c.attributes["color"] == "red"


class TestAnalogy:
    def test_analogy_creation(self):
        source = Concept("sun", {"hot": True, "bright": True})
        target = Concept("fire", {"hot": True, "bright": True})
        a = Analogy(source=source, target=target, score=0.9)
        assert a.score == 0.9
        assert a.source.name == "sun"


class TestAnalogyEngine:
    def test_add_and_find_analogies(self):
        engine = AnalogyEngine()
        engine.add_concept(Concept("bird", {"flies": True, "has_wings": True, "size": "small"}))
        engine.add_concept(Concept("airplane", {"flies": True, "has_wings": True, "size": "large"}))
        engine.add_concept(Concept("fish", {"swims": True, "has_fins": True}))
        query = Concept("unknown", {"flies": True, "has_wings": True, "size": "small"})
        analogies = engine.find_analogies(query, top_k=2)
        assert len(analogies) == 2
        assert analogies[0].target.name == "bird"

    def test_no_analogies_for_unrelated(self):
        engine = AnalogyEngine()
        engine.add_concept(Concept("fish", {"swims": True}))
        query = Concept("rock", {"hard": True, "heavy": True, "gray": True})
        analogies = engine.find_analogies(query)
        assert analogies == []

    def test_top_k_limits_results(self):
        engine = AnalogyEngine()
        for i in range(10):
            engine.add_concept(Concept(f"c{i}", {"a": True, "b": True}))
        query = Concept("q", {"a": True, "b": True, "c": True})
        analogies = engine.find_analogies(query, top_k=3)
        assert len(analogies) == 3

    def test_exact_match_perfect_score(self):
        engine = AnalogyEngine()
        engine.add_concept(Concept("original", {"x": 1, "y": 2, "z": 3}))
        query = Concept("other", {"x": 1, "y": 2, "z": 3})
        analogies = engine.find_analogies(query)
        assert len(analogies) == 1
        assert analogies[0].target.name == "original"
        assert analogies[0].score == 1.0

    def test_engine_clear(self):
        engine = AnalogyEngine()
        engine.add_concept(Concept("a", {"attr": 1}))
        engine.clear()
        assert len(engine.find_analogies(Concept("q", {"attr": 1}))) == 0
