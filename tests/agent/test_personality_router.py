"""Tests for src/agent/personality_router.py."""

from __future__ import annotations

import pytest

from src.agent.personality_router import (
    PERSONALITY_KEYWORDS,
    PERSONALITY_ROUTER_REGISTRY,
    PersonalityRouter,
    PersonalityType,
    TaskAnalysis,
)


@pytest.fixture
def router() -> PersonalityRouter:
    return PersonalityRouter()


def test_registry_default_is_router_class():
    assert PERSONALITY_ROUTER_REGISTRY["default"] is PersonalityRouter


def test_personality_type_enum_values():
    assert PersonalityType.ARCHITECT.value == "architect"
    assert PersonalityType.DETECTIVE.value == "detective"
    assert PersonalityType.GUARDIAN.value == "guardian"
    assert PersonalityType.ANALYST.value == "analyst"
    assert PersonalityType.TEACHER.value == "teacher"


def test_each_personality_has_at_least_eight_keywords():
    for personality, words in PERSONALITY_KEYWORDS.items():
        assert len(words) >= 8, f"{personality} has only {len(words)} keywords"


def test_task_analysis_is_frozen():
    ta = TaskAnalysis(
        primary=PersonalityType.ARCHITECT,
        secondary=[],
        confidence=1.0,
        requires_collaboration=False,
    )
    with pytest.raises(Exception):
        ta.confidence = 0.5  # type: ignore[misc]


def test_architect_detected_for_design_task(router):
    analysis = router.analyze("Please design and implement a new module")
    assert analysis.primary == PersonalityType.ARCHITECT


def test_architect_detected_for_build_keyword(router):
    analysis = router.analyze("build a scaffold for the service")
    assert analysis.primary == PersonalityType.ARCHITECT


def test_detective_detected_for_debug_task(router):
    analysis = router.analyze("debug the crash and locate the bug")
    assert analysis.primary == PersonalityType.DETECTIVE


def test_guardian_detected_for_security_audit(router):
    analysis = router.analyze("perform a security audit for vulnerability findings")
    assert analysis.primary == PersonalityType.GUARDIAN


def test_analyst_detected_for_measure_task(router):
    analysis = router.analyze("analyze and measure the benchmark metric")
    assert analysis.primary == PersonalityType.ANALYST


def test_teacher_detected_for_explain_task(router):
    analysis = router.analyze("explain and document this for the tutorial guide")
    assert analysis.primary == PersonalityType.TEACHER


def test_unknown_task_has_zero_confidence(router):
    analysis = router.analyze("zzzz qqqq wxyz")
    assert analysis.confidence == 0.0


def test_unknown_task_has_no_secondary(router):
    analysis = router.analyze("zzzz qqqq wxyz")
    assert analysis.secondary == []


def test_unknown_task_not_require_collaboration(router):
    analysis = router.analyze("zzzz qqqq wxyz")
    assert analysis.requires_collaboration is False


def test_multi_personality_triggers_collaboration(router):
    # architect (design/build), detective (bug/fix), guardian (security)
    task = "design a fix for the bug and audit its security posture"
    analysis = router.analyze(task)
    assert analysis.requires_collaboration is True
    assert len(analysis.secondary) >= 2


def test_two_personalities_no_collaboration(router):
    # architect + detective only -> one secondary -> no collaboration
    task = "design a fix for the bug"
    analysis = router.analyze(task)
    assert len(analysis.secondary) == 1
    assert analysis.requires_collaboration is False


def test_confidence_is_fraction_of_total(router):
    task = "design build implement bug"
    analysis = router.analyze(task)
    assert 0.0 < analysis.confidence <= 1.0


def test_confidence_one_when_single_personality(router):
    analysis = router.analyze("design architecture pattern scaffold")
    assert analysis.confidence == 1.0


def test_route_returns_primary_first(router):
    task = "design a fix for the bug and audit its security"
    order = router.route(task)
    assert order[0] == router.analyze(task).primary


def test_route_contains_all_matched(router):
    task = "design a fix for the bug and audit its security"
    order = router.route(task)
    assert PersonalityType.ARCHITECT in order
    assert PersonalityType.DETECTIVE in order
    assert PersonalityType.GUARDIAN in order


def test_route_sorted_by_score_desc(router):
    # "debug bug fix error" gives detective the strongest score
    task = "debug bug fix error and also design something"
    order = router.route(task)
    assert order[0] == PersonalityType.DETECTIVE


def test_route_unknown_returns_single_default(router):
    order = router.route("zzzz qqqq")
    assert len(order) == 1  # only primary, no secondaries


def test_explain_mentions_primary(router):
    text = router.explain("design a new system")
    assert "architect" in text.lower()
    assert "Primary" in text


def test_explain_mentions_confidence(router):
    text = router.explain("design a new system")
    assert "Confidence" in text


def test_explain_mentions_collaboration_when_required(router):
    text = router.explain("design a fix for the bug and audit its security")
    assert "Collaboration: required" in text


def test_explain_mentions_no_collaboration_for_simple(router):
    text = router.explain("design a system")
    assert "Collaboration: not required" in text


def test_explain_lists_secondary(router):
    text = router.explain("design and debug the bug")
    assert "detective" in text.lower()


def test_explain_no_secondary_for_simple(router):
    text = router.explain("design architecture")
    assert "Secondary: none" in text


def test_analyze_case_insensitive(router):
    upper = router.analyze("DESIGN A NEW SYSTEM")
    lower = router.analyze("design a new system")
    assert upper.primary == lower.primary


def test_custom_keywords_overrides_defaults():
    custom = {p: ["foo"] for p in PersonalityType}
    r = PersonalityRouter(keywords=custom)
    assert r.analyze("no keyword here").confidence == 0.0


def test_analyze_empty_string(router):
    analysis = router.analyze("")
    assert analysis.confidence == 0.0
    assert analysis.secondary == []


def test_secondary_sorted_by_score(router):
    # architect: design, build, implement => 3
    # detective: debug, bug => 2
    # guardian: security => 1
    task = "design build implement debug bug security"
    analysis = router.analyze(task)
    assert analysis.primary == PersonalityType.ARCHITECT
    assert analysis.secondary[0] == PersonalityType.DETECTIVE
    assert analysis.secondary[1] == PersonalityType.GUARDIAN
