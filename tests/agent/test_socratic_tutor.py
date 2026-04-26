"""Tests for src.agent.socratic_tutor."""

from __future__ import annotations

import pytest

from src.agent.socratic_tutor import (
    DEFAULT_SOCRATIC_TUTOR,
    SOCRATIC_TUTOR_REGISTRY,
    ConceptMastery,
    SocraticTutor,
    SocraticTutorError,
    TutorSession,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CONCEPTS = [
    {
        "concept_id": "c_add",
        "name": "Addition",
        "prerequisites": [],
        "hints": ["Think about combining quantities.", "Count them together."],
        "questions": [
            {
                "question": "What is 2 + 2?",
                "expected_answer_patterns": ["4", "four"],
                "hint_index": 0,
            },
            {
                "question": "What is 3 + 5?",
                "expected_answer_patterns": ["8", "eight"],
                "hint_index": 1,
            },
        ],
    },
    {
        "concept_id": "c_mul",
        "name": "Multiplication",
        "prerequisites": ["c_add"],
        "hints": ["Repeated addition."],
        "questions": [
            {
                "question": "What is 3 * 3?",
                "expected_answer_patterns": ["9", "nine"],
                "hint_index": 0,
            }
        ],
    },
]


@pytest.fixture
def tutor():
    return SocraticTutor(concepts=SAMPLE_CONCEPTS)


# ---------------------------------------------------------------------------
# Dataclass basics
# ---------------------------------------------------------------------------


def test_concept_mastery_defaults():
    m = ConceptMastery(concept_id="c1", name="Test")
    assert m.mastery_level == 0.0
    assert m.attempts == 0
    assert m.correct_streak == 0


def test_tutor_session_defaults():
    s = TutorSession(session_id="s1", user_id="u1")
    assert s.active_concept_id is None
    assert s.history == []


# ---------------------------------------------------------------------------
# Concept management
# ---------------------------------------------------------------------------


def test_add_concept(tutor):
    tutor.add_concept(
        {
            "concept_id": "c_sub",
            "name": "Subtraction",
            "prerequisites": [],
            "hints": [],
            "questions": [],
        }
    )
    assert "c_sub" in tutor._concepts


def test_add_concept_existing_user_inits_mastery(tutor):
    tutor.start_session("u1")
    tutor._ensure_user_mastery("u1")
    tutor.add_concept(
        {
            "concept_id": "c_new",
            "name": "New",
            "prerequisites": [],
            "hints": [],
            "questions": [],
        }
    )
    assert "c_new" in tutor._mastery["u1"]


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------


def test_start_session(tutor):
    session = tutor.start_session("u1")
    assert isinstance(session, TutorSession)
    assert session.user_id == "u1"
    assert session.session_id in tutor._sessions


# ---------------------------------------------------------------------------
# Question selection
# ---------------------------------------------------------------------------


def test_ask_question_auto_select(tutor):
    session = tutor.start_session("u1")
    result = tutor.ask_question(session.session_id)
    assert result["concept_id"] == "c_add"
    assert "question" in result


def test_ask_question_specific_concept(tutor):
    session = tutor.start_session("u1")
    result = tutor.ask_question(session.session_id, concept_id="c_add")
    assert result["concept_id"] == "c_add"


def test_ask_question_unknown_session(tutor):
    with pytest.raises(SocraticTutorError):
        tutor.ask_question("bad_session")


def test_ask_question_unknown_concept(tutor):
    session = tutor.start_session("u1")
    with pytest.raises(SocraticTutorError):
        tutor.ask_question(session.session_id, concept_id="nope")


def test_ask_question_prerequisites_not_met(tutor):
    session = tutor.start_session("u1")
    with pytest.raises(SocraticTutorError):
        tutor.ask_question(session.session_id, concept_id="c_mul")


def test_ask_question_mastered_concept(tutor):
    session = tutor.start_session("u1")
    # Answer question 0 correctly twice (it keeps being asked until streak >= 2)
    tutor.ask_question(session.session_id, concept_id="c_add")
    tutor.evaluate_answer(session.session_id, "4")
    tutor.ask_question(session.session_id, concept_id="c_add")
    tutor.evaluate_answer(session.session_id, "4")
    # Now question 1 is asked; answer it correctly twice
    tutor.ask_question(session.session_id, concept_id="c_add")
    tutor.evaluate_answer(session.session_id, "8")
    tutor.ask_question(session.session_id, concept_id="c_add")
    tutor.evaluate_answer(session.session_id, "8")
    result = tutor.ask_question(session.session_id, concept_id="c_add")
    assert result.get("mastered") is True


# ---------------------------------------------------------------------------
# Answer evaluation
# ---------------------------------------------------------------------------


def test_evaluate_correct_answer(tutor):
    session = tutor.start_session("u1")
    tutor.ask_question(session.session_id)
    result = tutor.evaluate_answer(session.session_id, "4")
    assert result["correct"] is True
    assert result["mastery_delta"] == 0.15
    assert result["next_hint"] is None


def test_evaluate_incorrect_answer(tutor):
    session = tutor.start_session("u1")
    tutor.ask_question(session.session_id)
    result = tutor.evaluate_answer(session.session_id, "99")
    assert result["correct"] is False
    assert result["mastery_delta"] == -0.05
    assert result["next_hint"] is not None


def test_evaluate_case_insensitive(tutor):
    session = tutor.start_session("u1")
    tutor.ask_question(session.session_id)
    result = tutor.evaluate_answer(session.session_id, "FOUR")
    assert result["correct"] is True


def test_evaluate_no_active_question(tutor):
    session = tutor.start_session("u1")
    with pytest.raises(SocraticTutorError):
        tutor.evaluate_answer(session.session_id, "4")


def test_evaluate_unknown_session(tutor):
    with pytest.raises(SocraticTutorError):
        tutor.evaluate_answer("bad", "4")


# ---------------------------------------------------------------------------
# Mastery tracking
# ---------------------------------------------------------------------------


def test_mastery_increases_on_correct(tutor):
    session = tutor.start_session("u1")
    tutor.ask_question(session.session_id)
    tutor.evaluate_answer(session.session_id, "4")
    mastery = tutor.get_mastery("u1", "c_add")
    assert mastery["mastery_level"] == 0.15
    assert mastery["attempts"] == 1


def test_mastery_decreases_on_incorrect(tutor):
    session = tutor.start_session("u1")
    tutor.ask_question(session.session_id)
    tutor.evaluate_answer(session.session_id, "wrong")
    mastery = tutor.get_mastery("u1", "c_add")
    assert mastery["mastery_level"] == 0.0  # floored at 0


def test_correct_streak_tracked(tutor):
    session = tutor.start_session("u1")
    tutor.ask_question(session.session_id)
    tutor.evaluate_answer(session.session_id, "4")
    mastery = tutor.get_mastery("u1", "c_add")
    assert mastery["correct_streak"] == 1


def test_streak_resets_on_incorrect(tutor):
    session = tutor.start_session("u1")
    tutor.ask_question(session.session_id)
    tutor.evaluate_answer(session.session_id, "4")
    tutor.ask_question(session.session_id)
    tutor.evaluate_answer(session.session_id, "wrong")
    mastery = tutor.get_mastery("u1", "c_add")
    assert mastery["correct_streak"] == 0


# ---------------------------------------------------------------------------
# Progress report
# ---------------------------------------------------------------------------


def test_progress_report_empty(tutor):
    report = tutor.get_progress_report("u1")
    assert report["concepts_started"] == 0
    assert report["concepts_mastered"] == 0
    assert report["avg_mastery"] == 0.0


def test_progress_report_with_attempts(tutor):
    session = tutor.start_session("u1")
    tutor.ask_question(session.session_id)
    tutor.evaluate_answer(session.session_id, "4")
    report = tutor.get_progress_report("u1")
    assert report["concepts_started"] == 1
    assert report["avg_mastery"] > 0.0


# ---------------------------------------------------------------------------
# Reset mastery
# ---------------------------------------------------------------------------


def test_reset_single_concept(tutor):
    session = tutor.start_session("u1")
    tutor.ask_question(session.session_id)
    tutor.evaluate_answer(session.session_id, "4")
    tutor.reset_mastery("u1", "c_add")
    mastery = tutor.get_mastery("u1", "c_add")
    assert mastery["mastery_level"] == 0.0
    assert mastery["attempts"] == 0


def test_reset_all_concepts(tutor):
    session = tutor.start_session("u1")
    tutor.ask_question(session.session_id)
    tutor.evaluate_answer(session.session_id, "4")
    tutor.reset_mastery("u1")
    for m in tutor.get_mastery("u1"):
        assert m["mastery_level"] == 0.0


def test_reset_unknown_concept(tutor):
    tutor._ensure_user_mastery("u1")
    with pytest.raises(SocraticTutorError):
        tutor.reset_mastery("u1", "nope")


# ---------------------------------------------------------------------------
# Singleton / registry
# ---------------------------------------------------------------------------


def test_default_singleton():
    assert isinstance(DEFAULT_SOCRATIC_TUTOR, SocraticTutor)


def test_registry_contains_default():
    assert "default" in SOCRATIC_TUTOR_REGISTRY
    assert SOCRATIC_TUTOR_REGISTRY["default"] is DEFAULT_SOCRATIC_TUTOR
