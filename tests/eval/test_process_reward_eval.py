"""Unit tests for ProcessRewardEval (10–16 tests).

Tests cover:
  1. Config defaults
  2-4. step_accuracy (perfect / zero / partial)
  5-6. step_precision_recall_f1 (perfect / no positive predictions)
  7-8. final_answer_correlation (positive / no-variance)
  9-10. first_error_detection (found / missed)
  11. first_error_detection — no errors in any solution
  12-13. calibration_error (perfect / imperfect)
  14. evaluate() keys
  15. BENCHMARK_REGISTRY entry
"""

import pytest

from src.eval import BENCHMARK_REGISTRY
from src.eval.process_reward_eval import (
    PRMEvalConfig,
    ProcessRewardEval,
    SolutionEval,
    StepPrediction,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_step(
    idx: int,
    reward: float,
    gt: bool,
    threshold: float = 0.5,
) -> StepPrediction:
    return StepPrediction(
        step_idx=idx,
        predicted_reward=reward,
        is_correct_pred=reward > threshold,
        is_correct_gt=gt,
    )


def _perfect_solution(n: int = 3, reward: float = 0.9, gt: bool = True) -> SolutionEval:
    steps = [_make_step(i, reward, gt) for i in range(n)]
    return SolutionEval(
        problem_id="p0",
        steps=steps,
        final_answer_correct=gt,
        prm_final_score=reward,
    )


# ---------------------------------------------------------------------------
# Test 1: config defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = PRMEvalConfig()
    assert cfg.correct_threshold == 0.5
    assert cfg.iou_threshold == 0.5


# ---------------------------------------------------------------------------
# Tests 2–4: step_accuracy
# ---------------------------------------------------------------------------


def test_step_accuracy_perfect():
    """All predictions match ground truth → 1.0."""
    sol = _perfect_solution(n=4, reward=0.9, gt=True)
    evaluator = ProcessRewardEval()
    assert evaluator.step_accuracy([sol]) == 1.0


def test_step_accuracy_zero():
    """All predictions wrong → 0.0."""
    # reward=0.9 → is_correct_pred=True, gt=False → mismatch
    steps = [_make_step(i, 0.9, False) for i in range(4)]
    sol = SolutionEval("p1", steps, False, 0.9)
    evaluator = ProcessRewardEval()
    assert evaluator.step_accuracy([sol]) == 0.0


def test_step_accuracy_partial():
    """3 out of 5 steps match → 0.6."""
    # steps 0,1,2: reward=0.9, gt=True  → match
    # steps 3,4:   reward=0.9, gt=False → mismatch
    steps = [_make_step(i, 0.9, True) for i in range(3)] + [
        _make_step(i + 3, 0.9, False) for i in range(2)
    ]
    sol = SolutionEval("p2", steps, False, 0.5)
    evaluator = ProcessRewardEval()
    assert abs(evaluator.step_accuracy([sol]) - 0.6) < 1e-9


# ---------------------------------------------------------------------------
# Tests 5–6: step_precision_recall_f1
# ---------------------------------------------------------------------------


def test_precision_recall_f1_perfect():
    """All true positives → precision=recall=f1=1.0."""
    sol = _perfect_solution(n=4, reward=0.9, gt=True)
    evaluator = ProcessRewardEval()
    result = evaluator.step_precision_recall_f1([sol])
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(1.0)


def test_precision_recall_f1_empty_pos():
    """No positive predictions at all → precision=0, no crash."""
    # reward=0.1 → is_correct_pred=False
    steps = [_make_step(i, 0.1, True) for i in range(3)]
    sol = SolutionEval("p3", steps, True, 0.1)
    evaluator = ProcessRewardEval()
    result = evaluator.step_precision_recall_f1([sol])
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0


# ---------------------------------------------------------------------------
# Tests 7–8: final_answer_correlation
# ---------------------------------------------------------------------------


def test_final_answer_correlation_perfect():
    """High PRM score → correct final; low → incorrect. Expect positive corr."""
    solutions = [
        SolutionEval("q0", [], True, 0.9),
        SolutionEval("q1", [], True, 0.85),
        SolutionEval("q2", [], False, 0.1),
        SolutionEval("q3", [], False, 0.15),
    ]
    evaluator = ProcessRewardEval()
    corr = evaluator.final_answer_correlation(solutions)
    assert corr > 0.8


def test_final_answer_correlation_no_variance():
    """All PRM scores identical → correlation = 0.0 (no variance)."""
    solutions = [
        SolutionEval("q0", [], True, 0.5),
        SolutionEval("q1", [], False, 0.5),
        SolutionEval("q2", [], True, 0.5),
    ]
    evaluator = ProcessRewardEval()
    corr = evaluator.final_answer_correlation(solutions)
    assert corr == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests 9–11: first_error_detection
# ---------------------------------------------------------------------------


def test_first_error_detection_found():
    """First error flagged → detection_rate=1.0."""
    # Step 0: gt=True, pred=True (ok)
    # Step 1: gt=False, pred=False → first error, flagged
    steps = [
        _make_step(0, 0.9, True),
        _make_step(1, 0.1, False),
    ]
    sol = SolutionEval("r0", steps, False, 0.1)
    evaluator = ProcessRewardEval()
    result = evaluator.first_error_detection([sol])
    assert result["detection_rate"] == pytest.approx(1.0)
    assert result["mean_position"] == pytest.approx(1.0)


def test_first_error_detection_missed():
    """First error not flagged → detection_rate=0.0."""
    # Step 0: gt=True, pred=True
    # Step 1: gt=False, pred=True (missed — reward > threshold)
    steps = [
        _make_step(0, 0.9, True),
        _make_step(1, 0.9, False),  # pred=True, gt=False → missed
    ]
    sol = SolutionEval("r1", steps, False, 0.9)
    evaluator = ProcessRewardEval()
    result = evaluator.first_error_detection([sol])
    assert result["detection_rate"] == pytest.approx(0.0)


def test_first_error_no_errors():
    """All steps correct → no applicable solutions → detection_rate=None."""
    steps = [_make_step(i, 0.9, True) for i in range(3)]
    sol = SolutionEval("r2", steps, True, 0.9)
    evaluator = ProcessRewardEval()
    result = evaluator.first_error_detection([sol])
    assert result["detection_rate"] is None
    assert result["mean_position"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests 12–13: calibration_error
# ---------------------------------------------------------------------------


def test_calibration_perfect():
    """Predicted reward closely matches gt rates within each bin → ECE ~ 0."""
    # All steps in the 0.8–0.9 bin (bin 8) with gt=True (fraction=1.0).
    # mean_pred ≈ 0.85, fraction_gt = 1.0 → error = 0.15. This is not ~0,
    # so we test a configuration where mean_pred == fraction_gt.
    # Put all steps at reward=1.0 (bin 9) with gt=True → |1.0 - 1.0| = 0.
    steps = [_make_step(i, 1.0, True) for i in range(5)]
    sol = SolutionEval("c0", steps, True, 1.0)
    evaluator = ProcessRewardEval()
    ece = evaluator.calibration_error([sol])
    assert ece == pytest.approx(0.0, abs=1e-9)


def test_calibration_imperfect():
    """Predicted all 1.0 but some steps incorrect → ECE > 0."""
    # 3 steps with reward=1.0 but gt=False → mean_pred=1.0, fraction_gt=0.0
    steps = [_make_step(i, 1.0, False) for i in range(3)]
    sol = SolutionEval("c1", steps, False, 1.0)
    evaluator = ProcessRewardEval()
    ece = evaluator.calibration_error([sol])
    assert ece > 0.0


# ---------------------------------------------------------------------------
# Test 14: evaluate() returns all expected keys
# ---------------------------------------------------------------------------


def test_evaluate_keys():
    """evaluate() dict must contain all expected metric keys."""
    steps = [_make_step(i, 0.8, True) for i in range(3)]
    sol = SolutionEval("e0", steps, True, 0.8)
    evaluator = ProcessRewardEval()
    result = evaluator.evaluate([sol])
    expected_keys = {
        "step_accuracy",
        "precision",
        "recall",
        "f1",
        "final_answer_correlation",
        "detection_rate",
        "mean_position",
        "calibration_error",
    }
    assert expected_keys.issubset(result.keys())


# ---------------------------------------------------------------------------
# Test 15: registry
# ---------------------------------------------------------------------------


def test_registry():
    """BENCHMARK_REGISTRY['process_reward_eval'] is ProcessRewardEval."""
    assert BENCHMARK_REGISTRY["process_reward_eval"] is ProcessRewardEval
