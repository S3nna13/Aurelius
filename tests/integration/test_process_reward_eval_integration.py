"""Integration test for ProcessRewardEval.

Builds 5 SolutionEval instances with mixed step correctness and PRM scores,
calls evaluate(), and verifies all keys are present, all values are in valid
ranges, and the registry is correctly wired.
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
# Fixture: 5 solutions with mixed step correctness and PRM scores
# ---------------------------------------------------------------------------


def _build_solutions() -> list[SolutionEval]:
    threshold = 0.5

    def step(idx: int, reward: float, gt: bool) -> StepPrediction:
        return StepPrediction(
            step_idx=idx,
            predicted_reward=reward,
            is_correct_pred=reward > threshold,
            is_correct_gt=gt,
            step_text=f"Step {idx}",
        )

    solutions = [
        # Solution 0: all steps correct, high PRM
        SolutionEval(
            problem_id="prob_0",
            steps=[step(0, 0.9, True), step(1, 0.85, True), step(2, 0.92, True)],
            final_answer_correct=True,
            prm_final_score=0.85,
        ),
        # Solution 1: first step wrong, PRM catches it
        SolutionEval(
            problem_id="prob_1",
            steps=[step(0, 0.3, False), step(1, 0.4, False), step(2, 0.8, True)],
            final_answer_correct=False,
            prm_final_score=0.3,
        ),
        # Solution 2: mixed, error not caught at step 1
        SolutionEval(
            problem_id="prob_2",
            steps=[step(0, 0.9, True), step(1, 0.7, False), step(2, 0.6, True)],
            final_answer_correct=False,
            prm_final_score=0.6,
        ),
        # Solution 3: all steps incorrect, low PRM
        SolutionEval(
            problem_id="prob_3",
            steps=[step(0, 0.1, False), step(1, 0.2, False), step(2, 0.15, False)],
            final_answer_correct=False,
            prm_final_score=0.1,
        ),
        # Solution 4: high PRM, correct final
        SolutionEval(
            problem_id="prob_4",
            steps=[
                step(0, 0.95, True),
                step(1, 0.88, True),
                step(2, 0.91, True),
                step(3, 0.93, True),
            ],
            final_answer_correct=True,
            prm_final_score=0.88,
        ),
    ]
    return solutions


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


def test_process_reward_eval_full_pipeline():
    """End-to-end: build solutions, evaluate, check all keys and value ranges."""
    solutions = _build_solutions()
    evaluator = ProcessRewardEval(config=PRMEvalConfig(correct_threshold=0.5))

    result = evaluator.evaluate(solutions)

    # All expected keys must be present
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
    assert expected_keys.issubset(result.keys()), f"Missing keys: {expected_keys - result.keys()}"

    # step_accuracy in [0, 1]
    assert 0.0 <= result["step_accuracy"] <= 1.0

    # precision, recall, f1 in [0, 1]
    assert 0.0 <= result["precision"] <= 1.0
    assert 0.0 <= result["recall"] <= 1.0
    assert 0.0 <= result["f1"] <= 1.0

    # final_answer_correlation in [-1, 1]
    corr = result["final_answer_correlation"]
    assert -1.0 <= corr <= 1.0

    # detection_rate: either None (no errors anywhere) or in [0, 1]
    dr = result["detection_rate"]
    if dr is not None:
        assert 0.0 <= dr <= 1.0

    # mean_position >= 0
    assert result["mean_position"] >= 0.0

    # calibration_error >= 0
    assert result["calibration_error"] >= 0.0

    # Sanity: step_accuracy > 0 (we have some matching steps)
    assert result["step_accuracy"] > 0.0

    # Sanity: there are errors in solutions, so detection_rate is not None
    assert dr is not None

    # Registry is correctly wired
    assert BENCHMARK_REGISTRY["process_reward_eval"] is ProcessRewardEval

    # Can instantiate from registry
    cls = BENCHMARK_REGISTRY["process_reward_eval"]
    instance = cls()
    assert isinstance(instance, ProcessRewardEval)

    # Re-running evaluate on the same solutions is idempotent
    result2 = instance.evaluate(solutions)
    assert result2["step_accuracy"] == pytest.approx(result["step_accuracy"])
    assert result2["f1"] == pytest.approx(result["f1"])
    assert result2["calibration_error"] == pytest.approx(result["calibration_error"])
