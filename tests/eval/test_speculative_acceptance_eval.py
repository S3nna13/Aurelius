"""
test_speculative_acceptance_eval.py — Unit + integration tests for
SpeculativeAcceptanceEval (Cycle 137-F).

No python built-in eval() is used anywhere in this file.

Test inventory (14 unit + 1 integration = 15 total):
  1.  test_config_defaults
  2.  test_acceptance_rate_all_accepted
  3.  test_acceptance_rate_none_accepted
  4.  test_acceptance_rate_partial
  5.  test_mean_accepted_with_bonus
  6.  test_mean_accepted_without_bonus
  7.  test_theoretical_speedup_perfect
  8.  test_theoretical_speedup_zero
  9.  test_theoretical_speedup_half
  10. test_theoretical_speedup_invalid_alpha
  11. test_per_position_acceptance_decreases
  12. test_per_position_acceptance_empty
  13. test_evaluate_result_type
  14. test_evaluate_total_tokens
  15. test_aggregate_keys
  16. test_acceptance_curve_monotone
  INT. test_integration_10_steps
"""

from __future__ import annotations

import math

import pytest

from src.eval import BENCHMARK_REGISTRY
from src.eval.speculative_acceptance_eval import (
    SpecAcceptConfig,
    SpecEvalResult,
    SpecStep,
    SpeculativeAcceptanceEval,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_eval(max_draft_len: int = 8) -> SpeculativeAcceptanceEval:
    return SpeculativeAcceptanceEval(SpecAcceptConfig(max_draft_len=max_draft_len))


def _step(draft: list, accepted: list, bonus=None) -> SpecStep:
    return SpecStep(draft_tokens=draft, accepted_tokens=accepted, bonus_token=bonus)


# ---------------------------------------------------------------------------
# 1. Config defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = SpecAcceptConfig()
    assert cfg.max_draft_len == 8
    assert cfg.n_trials == 1000


# ---------------------------------------------------------------------------
# 2. Acceptance rate — all accepted
# ---------------------------------------------------------------------------


def test_acceptance_rate_all_accepted():
    evaluator = _make_eval()
    steps = [
        _step([1, 2, 3], [1, 2, 3]),
        _step([4, 5], [4, 5]),
    ]
    rate = evaluator.compute_acceptance_rate(steps)
    assert math.isclose(rate, 1.0), f"Expected 1.0, got {rate}"


# ---------------------------------------------------------------------------
# 3. Acceptance rate — none accepted
# ---------------------------------------------------------------------------


def test_acceptance_rate_none_accepted():
    evaluator = _make_eval()
    steps = [
        _step([1, 2, 3], []),
        _step([4, 5], []),
    ]
    rate = evaluator.compute_acceptance_rate(steps)
    assert math.isclose(rate, 0.0), f"Expected 0.0, got {rate}"


# ---------------------------------------------------------------------------
# 4. Acceptance rate — partial
# ---------------------------------------------------------------------------


def test_acceptance_rate_partial():
    evaluator = _make_eval()
    # 3 draft tokens, 1 accepted  →  rate = 1/3 ... twice = 2/6 = 1/3
    steps = [
        _step([10, 20, 30], [10]),
        _step([40, 50, 60], [40]),
    ]
    rate = evaluator.compute_acceptance_rate(steps)
    assert math.isclose(rate, 1 / 3, rel_tol=1e-9), f"Expected 1/3, got {rate}"


# ---------------------------------------------------------------------------
# 5. Mean accepted per step — bonus counted
# ---------------------------------------------------------------------------


def test_mean_accepted_with_bonus():
    evaluator = _make_eval()
    # 2 accepted + bonus  →  3 tokens per step, both steps identical
    steps = [
        _step([1, 2, 3], [1, 2], bonus=99),
        _step([4, 5, 6], [4, 5], bonus=100),
    ]
    mean = evaluator.mean_accepted_per_step(steps)
    assert math.isclose(mean, 3.0), f"Expected 3.0, got {mean}"


# ---------------------------------------------------------------------------
# 6. Mean accepted per step — no bonus
# ---------------------------------------------------------------------------


def test_mean_accepted_without_bonus():
    evaluator = _make_eval()
    steps = [
        _step([1, 2, 3], [1, 2]),  # 2 accepted, no bonus -> 2
        _step([4, 5], [4]),  # 1 accepted, no bonus -> 1
    ]
    mean = evaluator.mean_accepted_per_step(steps)
    # (2 + 1) / 2 = 1.5
    assert math.isclose(mean, 1.5), f"Expected 1.5, got {mean}"


# ---------------------------------------------------------------------------
# 7. Theoretical speedup — perfect acceptance (alpha=1.0)
# ---------------------------------------------------------------------------


def test_theoretical_speedup_perfect():
    evaluator = _make_eval(max_draft_len=7)
    # alpha=1.0 -> speedup = K+1 = 8
    speedup = evaluator.theoretical_speedup(alpha=1.0, k=7)
    assert math.isclose(speedup, 8.0), f"Expected 8.0, got {speedup}"


# ---------------------------------------------------------------------------
# 8. Theoretical speedup — zero acceptance (alpha=0.0)
# ---------------------------------------------------------------------------


def test_theoretical_speedup_zero():
    evaluator = _make_eval()
    # alpha=0 -> (1 - 0) / (1 - 0) = 1.0  (only bonus token ever emitted)
    speedup = evaluator.theoretical_speedup(alpha=0.0, k=8)
    assert math.isclose(speedup, 1.0), f"Expected 1.0, got {speedup}"


# ---------------------------------------------------------------------------
# 9. Theoretical speedup — alpha=0.5
# ---------------------------------------------------------------------------


def test_theoretical_speedup_half():
    evaluator = _make_eval()
    k = 4
    alpha = 0.5
    # (1 - 0.5^5) / (1 - 0.5) = (1 - 1/32) / 0.5 = (31/32) / 0.5 = 31/16
    expected = (1 - 0.5**5) / 0.5
    speedup = evaluator.theoretical_speedup(alpha=alpha, k=k)
    assert math.isclose(speedup, expected, rel_tol=1e-9), f"Expected {expected}, got {speedup}"
    assert speedup < k + 1, "alpha<1 speedup must be less than K+1"
    assert speedup > 1.0, "alpha>0 speedup must exceed 1.0"


# ---------------------------------------------------------------------------
# 10. Invalid alpha raises ValueError
# ---------------------------------------------------------------------------


def test_theoretical_speedup_invalid_alpha():
    evaluator = _make_eval()
    with pytest.raises(ValueError):
        evaluator.theoretical_speedup(alpha=1.5, k=4)
    with pytest.raises(ValueError):
        evaluator.theoretical_speedup(alpha=-0.1, k=4)


# ---------------------------------------------------------------------------
# 11. Per-position acceptance — decreasing along draft sequence
# ---------------------------------------------------------------------------


def test_per_position_acceptance_decreases():
    evaluator = _make_eval(max_draft_len=4)
    # Position 0 always accepted, position 1 half the time,
    # position 2 never, position 3 never.
    steps = [
        _step([10, 20, 30, 40], [10, 20]),  # pos 0,1 accepted
        _step([11, 21, 31, 41], [11]),  # pos 0 accepted
        _step([12, 22, 32, 42], [12, 22]),  # pos 0,1 accepted
        _step([13, 23, 33, 43], [13]),  # pos 0 accepted
    ]
    rates = evaluator.per_position_acceptance(steps)
    assert len(rates) == 4
    # pos 0: 4/4=1.0, pos 1: 2/4=0.5, pos 2: 0.0, pos 3: 0.0
    assert math.isclose(rates[0], 1.0)
    assert math.isclose(rates[1], 0.5)
    assert math.isclose(rates[2], 0.0)
    assert math.isclose(rates[3], 0.0)
    # Must be non-increasing
    for i in range(len(rates) - 1):
        assert rates[i] >= rates[i + 1], f"Rates not non-increasing at position {i}: {rates}"


# ---------------------------------------------------------------------------
# 12. Per-position acceptance — empty steps list
# ---------------------------------------------------------------------------


def test_per_position_acceptance_empty():
    evaluator = _make_eval()
    rates = evaluator.per_position_acceptance([])
    assert rates == []


# ---------------------------------------------------------------------------
# 13. evaluate() returns SpecEvalResult
# ---------------------------------------------------------------------------


def test_evaluate_result_type():
    evaluator = _make_eval()
    steps = [_step([1, 2], [1], bonus=9)]
    result = evaluator.evaluate(steps)
    assert isinstance(result, SpecEvalResult)
    assert result.n_steps == 1


# ---------------------------------------------------------------------------
# 14. evaluate() — total_generated equals accepted + bonus count
# ---------------------------------------------------------------------------


def test_evaluate_total_tokens():
    evaluator = _make_eval()
    steps = [
        _step([1, 2, 3], [1, 2], bonus=99),  # 2 accepted + 1 bonus = 3
        _step([4, 5, 6], [4], bonus=None),  # 1 accepted + 0 bonus = 1
        _step([7, 8], [], bonus=77),  # 0 accepted + 1 bonus = 1
    ]
    result = evaluator.evaluate(steps)
    assert result.total_draft_tokens == 8
    assert result.total_accepted_tokens == 3
    assert result.total_generated_tokens == 5  # 3 + 2 bonuses
    assert result.n_steps == 3


# ---------------------------------------------------------------------------
# 15. aggregate() — correct keys present
# ---------------------------------------------------------------------------


def test_aggregate_keys():
    evaluator = _make_eval()
    steps = [_step([1, 2], [1], bonus=9)]
    results = [evaluator.evaluate(steps), evaluator.evaluate(steps)]
    summary = evaluator.aggregate(results)
    expected_keys = {
        "acceptance_rate_mean",
        "speedup_mean",
        "efficiency_mean",
        "mean_accepted_per_step_mean",
        "n_steps_mean",
    }
    assert expected_keys == set(summary.keys()), (
        f"Missing keys: {expected_keys - set(summary.keys())}"
    )


# ---------------------------------------------------------------------------
# 16. acceptance_curve() — higher alpha yields higher speedup (monotone)
# ---------------------------------------------------------------------------


def test_acceptance_curve_monotone():
    evaluator = _make_eval()
    k = 6
    alphas = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    curve = evaluator.acceptance_curve(alphas, k=k)
    speedups = [curve[a] for a in alphas]
    for i in range(len(speedups) - 1):
        assert speedups[i] <= speedups[i + 1], (
            f"Curve not monotone at alpha={alphas[i]}: {speedups}"
        )
    # Boundary checks
    assert math.isclose(speedups[0], 1.0)  # alpha=0 -> speedup=1
    assert math.isclose(speedups[-1], k + 1)  # alpha=1 -> speedup=K+1


# ---------------------------------------------------------------------------
# Integration test — 10 steps with K=4, mixed acceptance patterns
# ---------------------------------------------------------------------------


def test_integration_10_steps():
    """
    End-to-end integration: 10 steps with K=4, varying acceptance depth.
    Verifies that all SpecEvalResult fields are in their valid ranges and
    that the aggregate summary contains the expected keys.
    """
    K = 4
    evaluator = SpeculativeAcceptanceEval(SpecAcceptConfig(max_draft_len=K))

    # Build 10 steps with varied acceptance patterns
    steps = [
        # full acceptance with bonus
        _step([1, 2, 3, 4], [1, 2, 3, 4], bonus=5),
        # three accepted, no bonus
        _step([6, 7, 8, 9], [6, 7, 8]),
        # two accepted with bonus
        _step([10, 11, 12, 13], [10, 11], bonus=14),
        # one accepted, no bonus
        _step([15, 16, 17, 18], [15]),
        # none accepted, bonus only
        _step([19, 20, 21, 22], [], bonus=23),
        # full acceptance, no bonus
        _step([24, 25, 26, 27], [24, 25, 26, 27]),
        # two accepted with bonus
        _step([28, 29, 30, 31], [28, 29], bonus=32),
        # three accepted with bonus
        _step([33, 34, 35, 36], [33, 34, 35], bonus=37),
        # one accepted with bonus
        _step([38, 39, 40, 41], [38], bonus=42),
        # zero accepted, no bonus
        _step([43, 44, 45, 46], []),
    ]

    result = evaluator.evaluate(steps)

    # Structural checks
    assert isinstance(result, SpecEvalResult)
    assert result.n_steps == 10

    # Total token accounting
    assert result.total_draft_tokens == 40  # 10 steps x 4 tokens
    expected_accepted = 4 + 3 + 2 + 1 + 0 + 4 + 2 + 3 + 1 + 0  # = 20
    assert result.total_accepted_tokens == expected_accepted
    bonus_count = sum(1 for s in steps if s.bonus_token is not None)  # 6 bonuses
    assert result.total_generated_tokens == expected_accepted + bonus_count

    # Acceptance rate in [0, 1]
    assert 0.0 <= result.acceptance_rate <= 1.0
    # 20 accepted / 40 draft = 0.5
    assert math.isclose(result.acceptance_rate, 0.5)

    # draft_efficiency equals acceptance_rate
    assert math.isclose(result.draft_efficiency, result.acceptance_rate)

    # Mean per step in [0, K+1]
    assert 0.0 <= result.mean_accepted_per_step <= K + 1

    # Theoretical speedup based on alpha=0.5, k=4
    expected_speedup = (1 - 0.5**5) / (1 - 0.5)
    assert math.isclose(result.theoretical_speedup, expected_speedup, rel_tol=1e-9)
    assert result.theoretical_speedup > 1.0

    # Per-position acceptance — should be non-increasing for this dataset
    rates = evaluator.per_position_acceptance(steps)
    assert len(rates) == K
    for i in range(len(rates) - 1):
        assert rates[i] >= rates[i + 1], f"Per-position not decreasing: {rates}"

    # Aggregate over two identical results
    summary = evaluator.aggregate([result, result])
    required_keys = {
        "acceptance_rate_mean",
        "speedup_mean",
        "efficiency_mean",
        "mean_accepted_per_step_mean",
        "n_steps_mean",
    }
    assert required_keys.issubset(set(summary.keys()))
    assert math.isclose(summary["acceptance_rate_mean"], result.acceptance_rate)

    # Registry check
    assert BENCHMARK_REGISTRY["speculative_acceptance"] is SpeculativeAcceptanceEval
