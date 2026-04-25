"""Tests for src/agent/reflection_engine.py"""

from __future__ import annotations

import pytest

from src.agent.reflection_engine import (
    REFLECTION_ENGINE_REGISTRY,
    Reflection,
    ReflectionEngine,
    ReflectionType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_engine() -> ReflectionEngine:
    return ReflectionEngine()


ACTIONS_SHORT = ["action_a", "action_b"]
ACTIONS_LONG  = [f"action_{i}" for i in range(8)]
ACTIONS_10    = [f"step_{i}" for i in range(10)]


# ---------------------------------------------------------------------------
# Reflection dataclass – frozen / auto-id
# ---------------------------------------------------------------------------

class TestReflectionDataclass:
    def test_create_returns_reflection(self):
        r = Reflection.create(
            reflection_type=ReflectionType.SAFETY,
            input_summary="a; b",
            critique="ok",
            suggestions=["check"],
            confidence=0.5,
        )
        assert isinstance(r, Reflection)

    def test_auto_reflection_id_is_8_chars(self):
        r = Reflection.create(
            reflection_type=ReflectionType.SAFETY,
            input_summary="",
            critique="",
            suggestions=[],
            confidence=0.0,
        )
        assert len(r.reflection_id) == 8

    def test_explicit_reflection_id_preserved(self):
        r = Reflection.create(
            reflection_type=ReflectionType.SAFETY,
            input_summary="",
            critique="",
            suggestions=[],
            confidence=0.0,
            reflection_id="myid1234",
        )
        assert r.reflection_id == "myid1234"

    def test_frozen_reflection_cannot_be_mutated(self):
        r = Reflection.create(
            reflection_type=ReflectionType.SAFETY,
            input_summary="",
            critique="",
            suggestions=[],
            confidence=0.0,
        )
        with pytest.raises((AttributeError, TypeError)):
            r.critique = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ReflectionEngine – reflect – return type and fields
# ---------------------------------------------------------------------------

class TestReflectReturnType:
    def test_reflect_returns_reflection_instance(self):
        engine = make_engine()
        r = engine.reflect(ReflectionType.SELF_CRITIQUE, ACTIONS_SHORT)
        assert isinstance(r, Reflection)

    def test_reflect_stores_correct_type(self):
        engine = make_engine()
        r = engine.reflect(ReflectionType.EFFICIENCY, ACTIONS_SHORT)
        assert r.reflection_type is ReflectionType.EFFICIENCY

    def test_reflect_auto_id_8_chars(self):
        engine = make_engine()
        r = engine.reflect(ReflectionType.SAFETY, ACTIONS_SHORT)
        assert len(r.reflection_id) == 8


# ---------------------------------------------------------------------------
# ReflectionEngine – SELF_CRITIQUE
# ---------------------------------------------------------------------------

class TestSelfCritique:
    def test_self_critique_returns_reflection(self):
        engine = make_engine()
        r = engine.reflect(ReflectionType.SELF_CRITIQUE, ACTIONS_SHORT)
        assert r.reflection_type is ReflectionType.SELF_CRITIQUE

    def test_self_critique_many_actions_suggestion(self):
        engine = make_engine()
        r = engine.reflect(ReflectionType.SELF_CRITIQUE, [f"a{i}" for i in range(6)])
        assert "Consolidate duplicate steps" in r.suggestions

    def test_self_critique_few_actions_suggestion(self):
        engine = make_engine()
        r = engine.reflect(ReflectionType.SELF_CRITIQUE, ["a", "b"])
        assert "Continue current approach" in r.suggestions

    def test_self_critique_critique_contains_action_count(self):
        engine = make_engine()
        r = engine.reflect(ReflectionType.SELF_CRITIQUE, ["x", "y", "z"])
        assert "3" in r.critique


# ---------------------------------------------------------------------------
# ReflectionEngine – GOAL_ALIGNMENT
# ---------------------------------------------------------------------------

class TestGoalAlignment:
    def test_goal_alignment_includes_goal_in_critique(self):
        engine = make_engine()
        r = engine.reflect(ReflectionType.GOAL_ALIGNMENT, ACTIONS_SHORT, goal="maximize revenue")
        assert "maximize revenue" in r.critique

    def test_goal_alignment_goal_truncated_at_50(self):
        engine = make_engine()
        long_goal = "x" * 100
        r = engine.reflect(ReflectionType.GOAL_ALIGNMENT, ACTIONS_SHORT, goal=long_goal)
        # Critique must contain the 50-char slice, not full 100-char goal
        assert long_goal not in r.critique
        assert "x" * 50 in r.critique

    def test_goal_alignment_suggestion(self):
        engine = make_engine()
        r = engine.reflect(ReflectionType.GOAL_ALIGNMENT, ACTIONS_SHORT, goal="test")
        assert "Verify all actions contribute to goal" in r.suggestions

    def test_goal_alignment_action_count_in_critique(self):
        engine = make_engine()
        r = engine.reflect(ReflectionType.GOAL_ALIGNMENT, ["a", "b", "c"], goal="g")
        assert "3" in r.critique


# ---------------------------------------------------------------------------
# ReflectionEngine – EFFICIENCY
# ---------------------------------------------------------------------------

class TestEfficiency:
    def test_efficiency_critique_contains_step_count(self):
        engine = make_engine()
        r = engine.reflect(ReflectionType.EFFICIENCY, ["s1", "s2"])
        assert "2" in r.critique

    def test_efficiency_many_steps_reduce_suggestion(self):
        engine = make_engine()
        r = engine.reflect(ReflectionType.EFFICIENCY, ["a", "b", "c", "d"])
        assert "Reduce steps" in r.suggestions

    def test_efficiency_few_steps_optimal_suggestion(self):
        engine = make_engine()
        r = engine.reflect(ReflectionType.EFFICIENCY, ["a", "b"])
        assert "Optimal step count" in r.suggestions

    def test_efficiency_exactly_3_steps_optimal(self):
        engine = make_engine()
        r = engine.reflect(ReflectionType.EFFICIENCY, ["a", "b", "c"])
        assert "Optimal step count" in r.suggestions


# ---------------------------------------------------------------------------
# ReflectionEngine – SAFETY
# ---------------------------------------------------------------------------

class TestSafety:
    def test_safety_fixed_critique(self):
        engine = make_engine()
        r = engine.reflect(ReflectionType.SAFETY, ACTIONS_SHORT)
        assert r.critique == "Safety review of actions."

    def test_safety_suggestion(self):
        engine = make_engine()
        r = engine.reflect(ReflectionType.SAFETY, ACTIONS_SHORT)
        assert "Verify no destructive operations" in r.suggestions


# ---------------------------------------------------------------------------
# ReflectionEngine – COHERENCE
# ---------------------------------------------------------------------------

class TestCoherence:
    def test_coherence_action_count_in_critique(self):
        engine = make_engine()
        r = engine.reflect(ReflectionType.COHERENCE, ["a", "b", "c", "d"])
        assert "4" in r.critique

    def test_coherence_suggestion(self):
        engine = make_engine()
        r = engine.reflect(ReflectionType.COHERENCE, ACTIONS_SHORT)
        assert "Ensure logical flow" in r.suggestions


# ---------------------------------------------------------------------------
# ReflectionEngine – confidence & input_summary
# ---------------------------------------------------------------------------

class TestConfidenceAndSummary:
    def test_confidence_zero_for_empty_actions(self):
        engine = make_engine()
        r = engine.reflect(ReflectionType.SELF_CRITIQUE, [])
        assert r.confidence == 0.0

    def test_confidence_one_for_10_actions(self):
        engine = make_engine()
        r = engine.reflect(ReflectionType.SELF_CRITIQUE, ACTIONS_10)
        assert r.confidence == pytest.approx(1.0)

    def test_confidence_capped_at_one(self):
        engine = make_engine()
        r = engine.reflect(ReflectionType.SAFETY, [f"a{i}" for i in range(20)])
        assert r.confidence == pytest.approx(1.0)

    def test_confidence_partial_for_5_actions(self):
        engine = make_engine()
        r = engine.reflect(ReflectionType.SAFETY, [f"a{i}" for i in range(5)])
        assert r.confidence == pytest.approx(0.5)

    def test_input_summary_first_three_joined(self):
        engine = make_engine()
        r = engine.reflect(ReflectionType.SAFETY, ["x", "y", "z"])
        assert r.input_summary == "x; y; z"

    def test_input_summary_ellipsis_when_more_than_3(self):
        engine = make_engine()
        r = engine.reflect(ReflectionType.SAFETY, ["a", "b", "c", "d"])
        assert r.input_summary.endswith("...")

    def test_input_summary_empty_for_no_actions(self):
        engine = make_engine()
        r = engine.reflect(ReflectionType.COHERENCE, [])
        assert r.input_summary == ""


# ---------------------------------------------------------------------------
# ReflectionEngine – history & latest
# ---------------------------------------------------------------------------

class TestHistoryAndLatest:
    def test_history_starts_empty(self):
        engine = make_engine()
        assert engine.history() == []

    def test_history_grows_with_reflections(self):
        engine = make_engine()
        engine.reflect(ReflectionType.SAFETY, ACTIONS_SHORT)
        engine.reflect(ReflectionType.COHERENCE, ACTIONS_SHORT)
        assert len(engine.history()) == 2

    def test_history_returns_copy(self):
        engine = make_engine()
        engine.reflect(ReflectionType.SAFETY, ACTIONS_SHORT)
        h = engine.history()
        h.clear()
        assert len(engine.history()) == 1

    def test_latest_returns_most_recent(self):
        engine = make_engine()
        engine.reflect(ReflectionType.SAFETY, ACTIONS_SHORT)
        r2 = engine.reflect(ReflectionType.COHERENCE, ACTIONS_SHORT)
        assert engine.latest().reflection_id == r2.reflection_id

    def test_latest_none_when_empty(self):
        engine = make_engine()
        assert engine.latest() is None

    def test_latest_by_type_returns_most_recent_of_that_type(self):
        engine = make_engine()
        engine.reflect(ReflectionType.SAFETY, ["a"])
        r2 = engine.reflect(ReflectionType.SAFETY, ["b"])
        engine.reflect(ReflectionType.COHERENCE, ["c"])
        latest_safety = engine.latest(ReflectionType.SAFETY)
        assert latest_safety.reflection_id == r2.reflection_id

    def test_latest_by_type_none_when_no_match(self):
        engine = make_engine()
        engine.reflect(ReflectionType.SAFETY, ACTIONS_SHORT)
        assert engine.latest(ReflectionType.EFFICIENCY) is None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registry_contains_default(self):
        assert "default" in REFLECTION_ENGINE_REGISTRY

    def test_registry_default_is_reflection_engine_class(self):
        assert REFLECTION_ENGINE_REGISTRY["default"] is ReflectionEngine

    def test_registry_default_is_instantiable(self):
        cls = REFLECTION_ENGINE_REGISTRY["default"]
        obj = cls()
        assert isinstance(obj, ReflectionEngine)
