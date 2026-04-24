"""
Tests for src/interpretability/concept_bottleneck.py

Covers: Concept, ConceptScore, ConceptBottleneck, and CONCEPT_BOTTLENECK_REGISTRY.
Minimum 28 tests.
"""

from __future__ import annotations

import math
import pytest

from src.interpretability.concept_bottleneck import (
    Concept,
    ConceptScore,
    ConceptBottleneck,
    CONCEPT_BOTTLENECK_REGISTRY,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_concept(cid: int = 0, name: str = "test", desc: str = "A test concept") -> Concept:
    return Concept(name=name, description=desc, concept_id=cid)


def _make_concepts(n: int) -> list[Concept]:
    return [Concept(name=f"c{i}", description=f"concept {i}", concept_id=i) for i in range(n)]


def _make_bottleneck(n_concepts: int = 3, threshold: float = 0.5) -> ConceptBottleneck:
    concepts = _make_concepts(n_concepts)
    return ConceptBottleneck(concepts=concepts, threshold=threshold)


def _ones_weights(dim: int) -> list[float]:
    return [1.0] * dim


def _zeros_weights(dim: int) -> list[float]:
    return [0.0] * dim


def _make_activations(dim: int, val: float = 0.1) -> list[float]:
    return [val] * dim


# ---------------------------------------------------------------------------
# 1. Concept dataclass — frozen
# ---------------------------------------------------------------------------

class TestConceptDataclass:
    def test_fields_stored(self):
        c = _make_concept(cid=5, name="color", desc="Detects color")
        assert c.concept_id == 5
        assert c.name == "color"
        assert c.description == "Detects color"

    def test_frozen_name(self):
        c = _make_concept()
        with pytest.raises((AttributeError, TypeError)):
            c.name = "changed"  # type: ignore[misc]

    def test_frozen_concept_id(self):
        c = _make_concept(cid=1)
        with pytest.raises((AttributeError, TypeError)):
            c.concept_id = 99  # type: ignore[misc]

    def test_frozen_description(self):
        c = _make_concept(desc="original")
        with pytest.raises((AttributeError, TypeError)):
            c.description = "changed"  # type: ignore[misc]

    def test_equality(self):
        c1 = Concept(name="x", description="d", concept_id=1)
        c2 = Concept(name="x", description="d", concept_id=1)
        assert c1 == c2

    def test_inequality_different_id(self):
        c1 = Concept(name="x", description="d", concept_id=1)
        c2 = Concept(name="x", description="d", concept_id=2)
        assert c1 != c2


# ---------------------------------------------------------------------------
# 2. ConceptScore dataclass — frozen
# ---------------------------------------------------------------------------

class TestConceptScoreDataclass:
    def test_fields_stored(self):
        c = _make_concept()
        cs = ConceptScore(concept=c, score=0.8, active=True)
        assert cs.concept == c
        assert cs.score == 0.8
        assert cs.active is True

    def test_frozen_score(self):
        c = _make_concept()
        cs = ConceptScore(concept=c, score=0.5, active=False)
        with pytest.raises((AttributeError, TypeError)):
            cs.score = 0.9  # type: ignore[misc]

    def test_frozen_active(self):
        c = _make_concept()
        cs = ConceptScore(concept=c, score=0.5, active=False)
        with pytest.raises((AttributeError, TypeError)):
            cs.active = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 3. ConceptBottleneck — constructor
# ---------------------------------------------------------------------------

class TestBottleneckConstructor:
    def test_concepts_stored(self):
        concepts = _make_concepts(4)
        cb = ConceptBottleneck(concepts=concepts)
        assert cb.concepts == concepts

    def test_default_threshold(self):
        cb = ConceptBottleneck(concepts=[])
        assert cb.threshold == 0.5

    def test_custom_threshold(self):
        cb = ConceptBottleneck(concepts=[], threshold=0.7)
        assert cb.threshold == 0.7


# ---------------------------------------------------------------------------
# 4. register_probe
# ---------------------------------------------------------------------------

class TestRegisterProbe:
    def test_register_probe_stores_weights(self):
        cb = _make_bottleneck()
        weights = [0.1, 0.2, 0.3]
        cb.register_probe(concept_id=0, weights=weights)
        # Verify by scoring with known activations
        score = cb.score_concept(0, [1.0, 0.0, 0.0])
        # dot product = 0.1*1 + 0.2*0 + 0.3*0 = 0.1 -> sigmoid(0.1) > 0.5
        assert 0.0 <= score <= 1.0

    def test_register_probe_replaces_existing(self):
        cb = _make_bottleneck()
        cb.register_probe(0, [10.0])
        cb.register_probe(0, [-10.0])
        score = cb.score_concept(0, [1.0])
        # sigmoid(-10) should be near 0
        assert score < 0.01

    def test_unregistered_concept_score_is_zero(self):
        cb = _make_bottleneck(n_concepts=3)
        # No probes registered
        score = cb.score_concept(0, [1.0, 2.0, 3.0])
        assert score == 0.0


# ---------------------------------------------------------------------------
# 5. score_concept
# ---------------------------------------------------------------------------

class TestScoreConcept:
    def test_score_in_zero_one(self):
        cb = _make_bottleneck()
        cb.register_probe(0, [1.0, -1.0])
        score = cb.score_concept(0, [2.0, 1.0])
        assert 0.0 <= score <= 1.0

    def test_score_high_positive_input_near_one(self):
        cb = _make_bottleneck()
        cb.register_probe(0, [1.0])
        score = cb.score_concept(0, [100.0])
        assert score > 0.99

    def test_score_high_negative_input_near_zero(self):
        cb = _make_bottleneck()
        cb.register_probe(0, [1.0])
        score = cb.score_concept(0, [-100.0])
        assert score < 0.01

    def test_score_zero_input_is_half(self):
        cb = _make_bottleneck()
        cb.register_probe(0, [0.0])
        score = cb.score_concept(0, [1.0])
        assert abs(score - 0.5) < 1e-9

    def test_score_monotone_in_activation(self):
        cb = _make_bottleneck()
        cb.register_probe(0, [1.0])
        scores = [cb.score_concept(0, [float(x)]) for x in range(-5, 6)]
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1]

    def test_score_extreme_clamping_does_not_crash(self):
        cb = _make_bottleneck()
        cb.register_probe(0, [1.0])
        score_pos = cb.score_concept(0, [1e10])
        score_neg = cb.score_concept(0, [-1e10])
        assert math.isfinite(score_pos)
        assert math.isfinite(score_neg)


# ---------------------------------------------------------------------------
# 6. predict
# ---------------------------------------------------------------------------

class TestPredict:
    def test_predict_returns_list(self):
        cb = _make_bottleneck(n_concepts=3)
        for i in range(3):
            cb.register_probe(i, [1.0])
        result = cb.predict([0.5])
        assert isinstance(result, list)

    def test_predict_length_equals_registered_probes(self):
        cb = _make_bottleneck(n_concepts=3)
        cb.register_probe(0, [1.0])
        cb.register_probe(1, [1.0])
        # Concept 2 has no probe
        result = cb.predict([0.5])
        assert len(result) == 2

    def test_predict_returns_concept_score_objects(self):
        cb = _make_bottleneck(n_concepts=2)
        cb.register_probe(0, [1.0])
        cb.register_probe(1, [1.0])
        result = cb.predict([1.0])
        assert all(isinstance(cs, ConceptScore) for cs in result)

    def test_predict_active_when_score_above_threshold(self):
        cb = ConceptBottleneck(concepts=_make_concepts(1), threshold=0.1)
        cb.register_probe(0, [1.0])
        result = cb.predict([5.0])  # sigmoid(5) > 0.99 > 0.1
        assert result[0].active is True

    def test_predict_inactive_when_score_below_threshold(self):
        cb = ConceptBottleneck(concepts=_make_concepts(1), threshold=0.99)
        cb.register_probe(0, [-10.0])
        result = cb.predict([1.0])  # sigmoid(-10) ~ 0 < 0.99
        assert result[0].active is False

    def test_predict_no_registered_probes_empty(self):
        cb = _make_bottleneck(n_concepts=3)
        result = cb.predict([1.0, 2.0])
        assert result == []


# ---------------------------------------------------------------------------
# 7. active_concepts
# ---------------------------------------------------------------------------

class TestActiveConcepts:
    def test_active_concepts_returns_list(self):
        cb = _make_bottleneck(n_concepts=2)
        cb.register_probe(0, [1.0])
        cb.register_probe(1, [1.0])
        result = cb.active_concepts([0.5])
        assert isinstance(result, list)

    def test_active_concepts_returns_only_active(self):
        concepts = _make_concepts(2)
        cb = ConceptBottleneck(concepts=concepts, threshold=0.5)
        # Concept 0: high score (active), concept 1: low score (inactive)
        cb.register_probe(0, [1.0])   # sigmoid(big) > 0.5
        cb.register_probe(1, [-1.0])  # sigmoid(-big) < 0.5
        active = cb.active_concepts([100.0])
        assert len(active) == 1
        assert active[0].concept_id == 0

    def test_active_concepts_threshold_effect_high(self):
        cb = ConceptBottleneck(concepts=_make_concepts(1), threshold=0.99)
        cb.register_probe(0, [1.0])
        # sigmoid(0.5) ~ 0.62, which is < 0.99
        active = cb.active_concepts([0.5])
        assert active == []

    def test_active_concepts_threshold_effect_low(self):
        cb = ConceptBottleneck(concepts=_make_concepts(1), threshold=0.01)
        cb.register_probe(0, [1.0])
        # sigmoid(0.5) ~ 0.62 > 0.01
        active = cb.active_concepts([0.5])
        assert len(active) == 1

    def test_active_concepts_returns_concept_objects(self):
        cb = _make_bottleneck(n_concepts=2)
        cb.register_probe(0, [10.0])
        cb.register_probe(1, [10.0])
        active = cb.active_concepts([1.0])
        assert all(isinstance(c, Concept) for c in active)

    def test_active_concepts_all_inactive(self):
        cb = _make_bottleneck(n_concepts=3)
        for i in range(3):
            cb.register_probe(i, [-100.0])
        active = cb.active_concepts([1.0])
        assert active == []


# ---------------------------------------------------------------------------
# 8. intervention
# ---------------------------------------------------------------------------

class TestIntervention:
    def test_intervention_returns_list(self):
        cb = _make_bottleneck()
        activations = [0.1, 0.2, 0.3]
        result = cb.intervention(0, 0.8, activations)
        assert isinstance(result, list)

    def test_intervention_returns_copy_same_values(self):
        cb = _make_bottleneck()
        activations = [0.1, 0.2, 0.3]
        result = cb.intervention(0, 1.0, activations)
        assert result == activations

    def test_intervention_does_not_mutate_original(self):
        cb = _make_bottleneck()
        activations = [0.1, 0.2, 0.3]
        original = list(activations)
        cb.intervention(0, 1.0, activations)
        assert activations == original

    def test_intervention_returns_independent_copy(self):
        cb = _make_bottleneck()
        activations = [0.1, 0.2, 0.3]
        result = cb.intervention(0, 1.0, activations)
        result[0] = 999.0
        assert activations[0] == 0.1

    def test_intervention_length_unchanged(self):
        cb = _make_bottleneck()
        activations = [0.5] * 10
        result = cb.intervention(0, 1.0, activations)
        assert len(result) == 10


# ---------------------------------------------------------------------------
# 9. REGISTRY
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registry_has_default_key(self):
        assert "default" in CONCEPT_BOTTLENECK_REGISTRY

    def test_registry_default_is_class(self):
        assert CONCEPT_BOTTLENECK_REGISTRY["default"] is ConceptBottleneck

    def test_registry_default_is_instantiable(self):
        cls = CONCEPT_BOTTLENECK_REGISTRY["default"]
        instance = cls(concepts=_make_concepts(2))
        assert isinstance(instance, ConceptBottleneck)
