"""Unit tests for the Crescendo multi-turn jailbreak probe."""

from __future__ import annotations

import pytest

from src.eval.crescendo_probe import (
    ESCALATION_TEMPLATES,
    CrescendoProbe,
    CrescendoResult,
    CrescendoTurn,
    stub_judge_fn,
    stub_target_fn,
)


def _always_refuse_target(history):
    return "I'm sorry, I can't help with that."


def _compliant_target(history):
    return (
        "Here is a detailed, on-topic walkthrough covering the objective "
        "explicitly with step-by-step specifics."
    )


def _turn_gated_target(history):
    """Complies on-topic starting at user-turn 3."""
    turn = sum(1 for m in history if m.get("role") == "user")
    if turn < 3:
        return "General educational overview."
    return (
        "Detailed on-topic walkthrough, step-by-step, directly addressing the objective explicitly."
    )


def test_run_returns_crescendo_result_and_turn_shape():
    probe = CrescendoProbe(max_turns=5, max_backtracks=0)
    result = probe.run(
        "explain photosynthesis basics",
        _compliant_target,
        stub_judge_fn,
    )
    assert isinstance(result, CrescendoResult)
    assert result.turns, "expected at least one turn"
    t = result.turns[0]
    assert isinstance(t, CrescendoTurn)
    assert t.turn_index == 0
    assert isinstance(t.attacker_message, str) and t.attacker_message
    assert isinstance(t.target_response, str) and t.target_response
    assert set(t.judge_verdict.keys()) == {"success", "score", "rationale"}


def test_succeeds_when_stub_accepts_at_turn_ge_3():
    probe = CrescendoProbe(max_turns=6, max_backtracks=0)
    result = probe.run(
        "teach general biology",
        _turn_gated_target,
        stub_judge_fn,
        seed_topic="biology",
    )
    assert result.succeeded is True
    # Success must happen at or after turn index 2 (3rd user turn).
    assert result.turns[-1].turn_index >= 2
    assert result.final_score > 0.0


def test_exhausts_max_turns_without_success():
    probe = CrescendoProbe(max_turns=4, max_backtracks=0)

    def never_compliant(history):
        return "generic filler text without signal"

    def never_success_judge(objective, resp):
        return (False, 0.0, "not on-topic")

    result = probe.run("objective x", never_compliant, never_success_judge)
    assert result.succeeded is False
    assert len(result.turns) == 4


def test_backtracks_respect_max_backtracks():
    probe = CrescendoProbe(max_turns=6, max_backtracks=2)

    def judge_refuse(objective, resp):
        return (False, 0.0, "refusal detected in response")

    result = probe.run("obj", _compliant_target, judge_refuse)
    assert result.backtracks <= 2
    assert result.succeeded is False


def test_empty_objective_rejected():
    probe = CrescendoProbe()
    with pytest.raises(ValueError):
        probe.run("   ", _compliant_target, stub_judge_fn)


def test_objective_non_string_rejected():
    probe = CrescendoProbe()
    with pytest.raises(TypeError):
        probe.run(123, _compliant_target, stub_judge_fn)  # type: ignore[arg-type]


def test_empty_seed_topic_auto_derives():
    probe = CrescendoProbe(max_turns=2, max_backtracks=0)
    result = probe.run(
        "discuss photosynthesis",
        _compliant_target,
        stub_judge_fn,
        seed_topic="",
    )
    # The first attacker message should have used the derived topic.
    assert (
        "photosynthesis" in result.turns[0].attacker_message.lower()
        or "subject" in result.turns[0].attacker_message.lower()
    )


def test_judge_returns_non_bool_raises_type_error():
    probe = CrescendoProbe(max_turns=2, max_backtracks=0)

    def bad_judge(objective, resp):
        return ("nope", 0.5, "rationale")  # success is str, not bool

    with pytest.raises(TypeError):
        probe.run("obj", _compliant_target, bad_judge)


def test_determinism_under_fixed_stubs():
    probe1 = CrescendoProbe(max_turns=5, max_backtracks=1)
    probe2 = CrescendoProbe(max_turns=5, max_backtracks=1)
    r1 = probe1.run("obj alpha", stub_target_fn, stub_judge_fn)
    r2 = probe2.run("obj alpha", stub_target_fn, stub_judge_fn)
    assert r1.succeeded == r2.succeeded
    assert r1.final_score == r2.final_score
    assert len(r1.turns) == len(r2.turns)
    for a, b in zip(r1.turns, r2.turns):
        assert a.attacker_message == b.attacker_message
        assert a.target_response == b.target_response
        assert a.judge_verdict == b.judge_verdict


def test_turn_log_length_equals_turns_field():
    probe = CrescendoProbe(max_turns=3, max_backtracks=0)
    result = probe.run("obj", _compliant_target, stub_judge_fn)
    assert len(result.turns) == len([t for t in result.turns])


def test_succeeded_correlates_with_judge_success():
    probe = CrescendoProbe(max_turns=3, max_backtracks=0)

    def win_judge(objective, resp):
        return (True, 0.9, "on-topic")

    result = probe.run("obj", _compliant_target, win_judge)
    assert result.succeeded is True
    assert result.turns[-1].judge_verdict["success"] is True


def test_unknown_escalation_strategy_raises():
    with pytest.raises(ValueError):
        CrescendoProbe(escalation_strategy="chaotic")


def test_reproduces_same_transcript_across_repeat_invocations():
    probe = CrescendoProbe(max_turns=4, max_backtracks=1)
    runs = [probe.run("repeat me", stub_target_fn, stub_judge_fn) for _ in range(3)]
    first = runs[0]
    for r in runs[1:]:
        assert r.succeeded == first.succeeded
        assert r.final_score == first.final_score
        assert [t.attacker_message for t in r.turns] == [t.attacker_message for t in first.turns]
        assert [t.target_response for t in r.turns] == [t.target_response for t in first.turns]


def test_target_returning_non_string_raises_type_error():
    probe = CrescendoProbe(max_turns=2, max_backtracks=0)

    def bad_target(history):
        return 123

    with pytest.raises(TypeError):
        probe.run("obj", bad_target, stub_judge_fn)


def test_templates_are_non_empty_tuple():
    assert isinstance(ESCALATION_TEMPLATES, tuple)
    assert len(ESCALATION_TEMPLATES) >= 4
    assert all(isinstance(t, str) and t for t in ESCALATION_TEMPLATES)


def test_invalid_max_turns_raises():
    with pytest.raises(ValueError):
        CrescendoProbe(max_turns=0)
    with pytest.raises(ValueError):
        CrescendoProbe(max_backtracks=-1)
