"""Unit tests for IFEval scoring harness."""

from __future__ import annotations

import pytest

from src.eval.ifeval_scorer import (
    IFEvalConstraint,
    IFEvalProblem,
    IFEvalResult,
    IFEvalScorer,
)


def _mk(problem_constraints):
    return IFEvalProblem(prompt="p", constraints=list(problem_constraints))


def test_length_words_pass_and_fail():
    s = IFEvalScorer()
    c = IFEvalConstraint("length_words", {"min": 3, "max": 5})
    assert s._check_one(c, "one two three") is True
    assert s._check_one(c, "one two") is False
    assert s._check_one(c, "a b c d e f") is False


def test_contains_keyword_case_insensitive_default():
    s = IFEvalScorer()
    c = IFEvalConstraint("contains_keyword", {"keyword": "Python"})
    assert s._check_one(c, "I use python every day.") is True
    assert s._check_one(c, "I use Rust.") is False


def test_avoids_keyword():
    s = IFEvalScorer()
    c = IFEvalConstraint("avoids_keyword", {"keyword": "foo"})
    assert s._check_one(c, "bar baz") is True
    assert s._check_one(c, "a FOO b") is False


def test_case_lower_upper_title():
    s = IFEvalScorer()
    lo = IFEvalConstraint("case", {"mode": "lower"})
    up = IFEvalConstraint("case", {"mode": "upper"})
    ti = IFEvalConstraint("case", {"mode": "title"})
    assert s._check_one(lo, "all lowercase words") is True
    assert s._check_one(lo, "Some Upper") is False
    assert s._check_one(up, "ALL CAPS HERE") is True
    assert s._check_one(up, "not all caps") is False
    assert s._check_one(ti, "All Title Case Words") is True
    assert s._check_one(ti, "Not title Case") is False


def test_json_format_valid_and_invalid():
    s = IFEvalScorer()
    c = IFEvalConstraint("json_format", {})
    assert s._check_one(c, '{"a": 1, "b": [2, 3]}') is True
    assert s._check_one(c, '```json\n{"ok": true}\n```') is True
    assert s._check_one(c, "not json {") is False
    assert s._check_one(c, "") is False


def test_start_with_and_end_with():
    s = IFEvalScorer()
    sw = IFEvalConstraint("start_with", {"phrase": "Dear"})
    ew = IFEvalConstraint("end_with", {"phrase": "regards."})
    assert s._check_one(sw, "Dear friend, hello") is True
    assert s._check_one(sw, "Hi dear") is False
    assert s._check_one(ew, "With warmest regards.") is True
    assert s._check_one(ew, "No ending") is False


def test_min_bullets_with_dash_marker():
    s = IFEvalScorer()
    c = IFEvalConstraint("min_bullets", {"n": 3, "marker": "- "})
    text = "Intro\n- one\n- two\n- three\nEnd"
    assert s._check_one(c, text) is True
    assert s._check_one(c, "- only one") is False


def test_placeholder_present_marker():
    s = IFEvalScorer()
    c = IFEvalConstraint("placeholder_present", {"marker": "[NAME]"})
    assert s._check_one(c, "Hello [NAME], welcome") is True
    assert s._check_one(c, "Hello there") is False


def test_quote_count_between_min_and_max():
    s = IFEvalScorer()
    c = IFEvalConstraint("quote_count", {"min": 2, "max": 3})
    assert s._check_one(c, 'He said "hi" and "bye"') is True
    assert s._check_one(c, 'Only "one"') is False
    assert s._check_one(c, '"a" "b" "c" "d"') is False


def test_max_punctuation():
    s = IFEvalScorer()
    c = IFEvalConstraint("max_punctuation", {"chars": "!?", "max": 2})
    assert s._check_one(c, "Hi! Is this ok?") is True
    assert s._check_one(c, "Wow! Really?! Yes!!") is False


def test_score_one_multiple_constraints():
    s = IFEvalScorer()
    prob = _mk(
        [
            IFEvalConstraint("length_words", {"min": 2}),
            IFEvalConstraint("contains_keyword", {"keyword": "hello"}),
        ]
    )
    r = s.score_one(prob, "hello world")
    assert isinstance(r, IFEvalResult)
    assert r.passed == [True, True]
    assert r.strict_pass is True
    assert r.per_constraint == [("length_words", True), ("contains_keyword", True)]


def test_strict_pass_requires_all():
    s = IFEvalScorer()
    prob = _mk(
        [
            IFEvalConstraint("length_words", {"min": 10}),  # fails
            IFEvalConstraint("contains_keyword", {"keyword": "hi"}),
        ]
    )
    r = s.score_one(prob, "hi world")
    assert r.strict_pass is False
    assert r.passed == [False, True]


def test_score_aggregates_three_problems():
    s = IFEvalScorer()
    p1 = _mk([IFEvalConstraint("contains_keyword", {"keyword": "a"})])
    p2 = _mk([IFEvalConstraint("contains_keyword", {"keyword": "b"})])
    p3 = _mk([IFEvalConstraint("contains_keyword", {"keyword": "c"})])
    out = s.score([p1, p2, p3], ["a here", "no match", "c yes"])
    assert out["n_problems"] == 3
    assert out["strict_accuracy"] == pytest.approx(2 / 3)
    assert out["loose_accuracy"] >= out["strict_accuracy"]
    assert "contains_keyword" in out["per_type_accuracy"]
    assert out["per_type_accuracy"]["contains_keyword"] == pytest.approx(2 / 3)


def test_unknown_constraint_type_raises():
    s = IFEvalScorer()
    prob = _mk([IFEvalConstraint("bogus_type", {})])
    with pytest.raises(ValueError):
        s.score_one(prob, "whatever")


def test_empty_responses_list_raises():
    s = IFEvalScorer()
    with pytest.raises(ValueError):
        s.score([], [])


def test_mismatched_lengths_raises():
    s = IFEvalScorer()
    with pytest.raises(ValueError):
        s.score([_mk([IFEvalConstraint("length_words", {})])], ["a", "b"])


def test_per_type_accuracy_correct_mixed():
    s = IFEvalScorer()
    # Two problems, each with two constraint types.
    p1 = _mk(
        [
            IFEvalConstraint("length_words", {"min": 1}),  # pass
            IFEvalConstraint("contains_keyword", {"keyword": "zzz"}),  # fail
        ]
    )
    p2 = _mk(
        [
            IFEvalConstraint("length_words", {"min": 100}),  # fail
            IFEvalConstraint("contains_keyword", {"keyword": "hi"}),  # pass
        ]
    )
    out = s.score([p1, p2], ["hi there", "hi there"])
    assert out["per_type_accuracy"]["length_words"] == pytest.approx(0.5)
    assert out["per_type_accuracy"]["contains_keyword"] == pytest.approx(0.5)


def test_determinism():
    s = IFEvalScorer()
    p = _mk(
        [
            IFEvalConstraint("length_words", {"min": 2, "max": 10}),
            IFEvalConstraint("start_with", {"phrase": "Hello"}),
            IFEvalConstraint("json_format", {}),
        ]
    )
    resp = "Hello world"
    a = s.score_one(p, resp)
    b = s.score_one(p, resp)
    c = s.score_one(p, resp)
    assert a.passed == b.passed == c.passed
    assert a.strict_pass == b.strict_pass == c.strict_pass
    agg1 = s.score([p, p], [resp, resp])
    agg2 = s.score([p, p], [resp, resp])
    assert agg1 == agg2


def test_length_sentences():
    s = IFEvalScorer()
    c = IFEvalConstraint("length_sentences", {"min": 2, "max": 3})
    assert s._check_one(c, "One. Two.") is True
    assert s._check_one(c, "One.") is False
    assert s._check_one(c, "A. B. C. D.") is False
