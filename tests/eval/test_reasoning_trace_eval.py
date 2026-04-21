"""
tests/eval/test_reasoning_trace_eval.py

Unit + integration tests for ReasoningTraceEval (Cycle 135-F).

Coverage
--------
 1.  test_config_defaults
 2.  test_parse_steps_count
 3.  test_parse_steps_reasoning_words
 4.  test_parse_steps_math
 5.  test_faithfulness_all_reasoning
 6.  test_faithfulness_none
 7.  test_length_efficiency_equal
 8.  test_length_efficiency_long_trace
 9.  test_redundancy_score_unique
10.  test_redundancy_score_repetitive
11.  test_step_consistency_overlap
12.  test_evaluate_result_type
13.  test_evaluate_overall_range
14.  test_aggregate_keys
15.  test_step_consistency_fewer_than_two_steps
16.  test_evaluate_batch_length_mismatch
Integration: test_integration_good_mediocre_bad
"""

import pytest

from src.eval.reasoning_trace_eval import (
    ReasoningTraceConfig,
    ReasoningTraceEval,
    StepAnalysis,
    TraceEvalResult,
)
from src.eval import BENCHMARK_REGISTRY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_evaluator(**kwargs) -> ReasoningTraceEval:
    return ReasoningTraceEval(ReasoningTraceConfig(**kwargs))


# ---------------------------------------------------------------------------
# 1. Config defaults
# ---------------------------------------------------------------------------

class TestConfigDefaults:
    def test_config_defaults(self):
        cfg = ReasoningTraceConfig()
        assert cfg.step_delimiter == "\n"
        assert cfg.answer_prefix == "Therefore"
        assert cfg.min_steps == 2
        assert cfg.max_length_ratio == 5.0
        assert cfg.redundancy_ngram == 4


# ---------------------------------------------------------------------------
# 2–4. parse_steps
# ---------------------------------------------------------------------------

class TestParseSteps:
    def test_parse_steps_count(self):
        """Correct number of non-empty steps from newline delimiter."""
        ev = _make_evaluator()
        trace = "Step one\nStep two\nStep three"
        steps = ev.parse_steps(trace)
        assert len(steps) == 3

    def test_parse_steps_reasoning_words(self):
        """Step containing 'because' sets has_reasoning_words=True."""
        ev = _make_evaluator()
        trace = "This is true because it follows from axiom A"
        steps = ev.parse_steps(trace)
        assert len(steps) == 1
        assert steps[0].has_reasoning_words is True

    def test_parse_steps_math(self):
        """Step containing '2+2=4' sets has_math_notation=True."""
        ev = _make_evaluator()
        trace = "We know that 2+2=4 by addition"
        steps = ev.parse_steps(trace)
        assert steps[0].has_math_notation is True

    def test_parse_steps_empty_lines_ignored(self):
        """Blank lines between steps are not counted as steps."""
        ev = _make_evaluator()
        trace = "Step A\n\nStep B\n\nStep C"
        steps = ev.parse_steps(trace)
        assert len(steps) == 3


# ---------------------------------------------------------------------------
# 5–6. faithfulness_score
# ---------------------------------------------------------------------------

class TestFaithfulnessScore:
    def test_faithfulness_all_reasoning(self):
        """All non-answer steps have reasoning words → score == 1.0."""
        ev = _make_evaluator()
        steps = [
            StepAnalysis("We conclude since A", 5, False, True, False),
            StepAnalysis("Thus B follows", 3, False, True, False),
        ]
        assert ev.faithfulness_score(steps) == pytest.approx(1.0)

    def test_faithfulness_none(self):
        """No non-answer steps have reasoning words → score == 0.0."""
        ev = _make_evaluator()
        steps = [
            StepAnalysis("We look at A", 4, False, False, False),
            StepAnalysis("We look at B", 4, False, False, False),
        ]
        assert ev.faithfulness_score(steps) == pytest.approx(0.0)

    def test_faithfulness_half(self):
        """Half of non-answer steps have reasoning words → score == 0.5."""
        ev = _make_evaluator()
        steps = [
            StepAnalysis("X because Y", 3, False, True, False),
            StepAnalysis("Z and W", 3, False, False, False),
        ]
        assert ev.faithfulness_score(steps) == pytest.approx(0.5)

    def test_faithfulness_excludes_answer_step(self):
        """Answer steps are excluded from the denominator."""
        ev = _make_evaluator()
        steps = [
            StepAnalysis("Therefore the answer is 42", 5, True, True, False),
        ]
        # Only answer steps → no non-answer steps → 0.0
        assert ev.faithfulness_score(steps) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 7–8. length_efficiency
# ---------------------------------------------------------------------------

class TestLengthEfficiency:
    def test_length_efficiency_equal(self):
        """When trace ≈ answer in length, efficiency should be ≈ 1.0."""
        ev = _make_evaluator(max_length_ratio=5.0)
        text = "hello world"
        score = ev.length_efficiency(text, text)
        # ratio == 1, max_length_ratio == 5 → efficiency = 5/1 → capped at 1
        assert score == pytest.approx(1.0)

    def test_length_efficiency_long_trace(self):
        """A trace 10× longer than the answer gets a score < 1.0."""
        ev = _make_evaluator(max_length_ratio=5.0)
        answer = "x" * 10
        trace = "x" * 100   # 10× longer
        score = ev.length_efficiency(trace, answer)
        # actual_ratio = 10; efficiency = 5/10 = 0.5
        assert score == pytest.approx(0.5)

    def test_length_efficiency_within_ratio(self):
        """Trace exactly max_length_ratio times longer → efficiency == 1.0."""
        ev = _make_evaluator(max_length_ratio=5.0)
        answer = "x" * 10
        trace = "x" * 50   # 5× longer (at the boundary)
        score = ev.length_efficiency(trace, answer)
        assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 9–10. redundancy_score
# ---------------------------------------------------------------------------

class TestRedundancyScore:
    def test_redundancy_score_unique(self):
        """All different words → high (≈1.0) redundancy score."""
        ev = _make_evaluator(redundancy_ngram=2)
        trace = "alpha beta gamma delta epsilon zeta eta theta"
        score = ev.redundancy_score(trace)
        assert score == pytest.approx(1.0)

    def test_redundancy_score_repetitive(self):
        """Repeating the same phrase many times → low redundancy score."""
        ev = _make_evaluator(redundancy_ngram=2)
        trace = " ".join(["foo bar"] * 20)
        score = ev.redundancy_score(trace)
        # Only one unique 2-gram but many total → score ≪ 1
        assert score < 0.2

    def test_redundancy_score_short_trace(self):
        """Trace shorter than ngram size → returns 1.0."""
        ev = _make_evaluator(redundancy_ngram=4)
        score = ev.redundancy_score("just two")
        assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 11. step_consistency
# ---------------------------------------------------------------------------

class TestStepConsistency:
    def test_step_consistency_overlap(self):
        """Steps sharing several words should yield consistency > 0."""
        ev = _make_evaluator()
        steps = [
            StepAnalysis("the cat sat on the mat", 6, False, False, False),
            StepAnalysis("the cat then moved away from the mat", 7, False, False, False),
        ]
        score = ev.step_consistency(steps)
        assert score > 0.0

    def test_step_consistency_no_overlap(self):
        """Completely disjoint steps → consistency == 0.0."""
        ev = _make_evaluator()
        steps = [
            StepAnalysis("alpha beta gamma", 3, False, False, False),
            StepAnalysis("delta epsilon zeta", 3, False, False, False),
        ]
        assert ev.step_consistency(steps) == pytest.approx(0.0)

    def test_step_consistency_fewer_than_two_steps(self):
        """Fewer than 2 steps → consistency == 0.0."""
        ev = _make_evaluator()
        steps = [StepAnalysis("only one step", 3, False, False, False)]
        assert ev.step_consistency(steps) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 12–13. evaluate — result type and range
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_evaluate_result_type(self):
        """evaluate() must return a TraceEvalResult instance."""
        ev = _make_evaluator()
        result = ev.evaluate("Step A\nStep B\nTherefore C", "C")
        assert isinstance(result, TraceEvalResult)

    def test_evaluate_overall_range(self):
        """overall score must be in [0, 1]."""
        ev = _make_evaluator()
        trace = "We know X because Y\nSince Y, Z follows\nTherefore Z"
        result = ev.evaluate(trace, "Z")
        assert 0.0 <= result.overall <= 1.0

    def test_evaluate_has_final_answer_true(self):
        """Trace containing answer_prefix → has_final_answer == True."""
        ev = _make_evaluator()
        result = ev.evaluate("Step one\nTherefore done", "done")
        assert result.has_final_answer is True

    def test_evaluate_has_final_answer_false(self):
        """Trace without answer_prefix → has_final_answer == False."""
        ev = _make_evaluator()
        result = ev.evaluate("Step one\nStep two", "done")
        assert result.has_final_answer is False

    def test_evaluate_batch_length_mismatch(self):
        """evaluate_batch raises ValueError when lengths differ."""
        ev = _make_evaluator()
        with pytest.raises(ValueError):
            ev.evaluate_batch(["trace1", "trace2"], ["answer1"])


# ---------------------------------------------------------------------------
# 14. aggregate keys
# ---------------------------------------------------------------------------

class TestAggregate:
    def test_aggregate_keys(self):
        """aggregate() must return all required keys."""
        ev = _make_evaluator()
        results = ev.evaluate_batch(
            ["Step A\nTherefore B", "Step X\nTherefore Y"],
            ["B", "Y"],
        )
        agg = ev.aggregate(results)
        required_keys = {
            "n_steps_mean",
            "faithfulness_mean",
            "efficiency_mean",
            "redundancy_mean",
            "consistency_mean",
            "overall_mean",
        }
        assert required_keys.issubset(agg.keys())

    def test_aggregate_empty(self):
        """aggregate([]) returns all-zero dict with correct keys."""
        ev = _make_evaluator()
        agg = ev.aggregate([])
        assert agg["overall_mean"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 15. BENCHMARK_REGISTRY
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registry_entry(self):
        """ReasoningTraceEval is registered in BENCHMARK_REGISTRY."""
        assert "reasoning_trace_eval" in BENCHMARK_REGISTRY
        assert BENCHMARK_REGISTRY["reasoning_trace_eval"] is ReasoningTraceEval


# ---------------------------------------------------------------------------
# Integration test: good > mediocre > bad
# ---------------------------------------------------------------------------

class TestIntegration:
    """
    Evaluate three traces of different quality and assert the ordering
    good_overall > mediocre_overall > bad_overall.
    """

    # Good trace: multiple steps with explicit reasoning, minimal redundancy,
    # answer-prefixed conclusion, moderate length relative to answer.
    GOOD_TRACE = (
        "We start by noting that X equals 2 because the definition says so.\n"
        "Since X equals 2, we can therefore conclude Y equals 4 by doubling.\n"
        "Thus the result follows since doubling 2 gives 4.\n"
        "Therefore the answer is 4."
    )
    GOOD_ANSWER = "The answer is 4."

    # Mediocre trace: some reasoning words, slightly repetitive, no clear
    # answer prefix.
    MEDIOCRE_TRACE = (
        "We look at the problem and note some facts.\n"
        "We look at the facts and note X may be relevant.\n"
        "We note that X implies Y in some contexts since Y follows from X.\n"
        "The result is probably 4."
    )
    MEDIOCRE_ANSWER = "The answer is 4."

    # Bad trace: no reasoning words, highly repetitive, no answer step,
    # extremely long relative to answer.
    BAD_TRACE = " ".join(["blah blah blah blah"] * 30)
    BAD_ANSWER = "4"

    def test_integration_good_mediocre_bad(self):
        ev = ReasoningTraceEval()
        good = ev.evaluate(self.GOOD_TRACE, self.GOOD_ANSWER)
        mediocre = ev.evaluate(self.MEDIOCRE_TRACE, self.MEDIOCRE_ANSWER)
        bad = ev.evaluate(self.BAD_TRACE, self.BAD_ANSWER)

        assert good.overall > mediocre.overall, (
            f"Expected good ({good.overall:.3f}) > mediocre ({mediocre.overall:.3f})"
        )
        assert mediocre.overall > bad.overall, (
            f"Expected mediocre ({mediocre.overall:.3f}) > bad ({bad.overall:.3f})"
        )
