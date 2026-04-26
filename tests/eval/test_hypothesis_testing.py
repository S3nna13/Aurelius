"""Tests for statistical hypothesis testing module."""

import pytest

from src.eval.hypothesis_testing import (
    HypothesisTestConfig,
    ModelComparisonSuite,
    TestResult,
    bootstrap_confidence_interval,
    # Legacy imports still work
    cohens_d,
    compute_power,
    mcnemar_test,
    paired_t_test,
    wilcoxon_signed_rank,
)


# ---------------------------------------------------------------------------
# 1. test_config_defaults
# ---------------------------------------------------------------------------
def test_config_defaults():
    cfg = HypothesisTestConfig()
    assert cfg.alpha == 0.05
    assert cfg.n_bootstrap == 1000
    assert cfg.alternative == "two-sided"
    assert cfg.seed == 42


# ---------------------------------------------------------------------------
# 2. test_paired_t_test_returns_result
# ---------------------------------------------------------------------------
def test_paired_t_test_returns_result():
    scores_a = [0.6, 0.7, 0.8, 0.55, 0.65]
    scores_b = [0.5, 0.6, 0.7, 0.45, 0.55]
    result = paired_t_test(scores_a, scores_b)
    assert isinstance(result, TestResult)
    assert hasattr(result, "statistic")
    assert hasattr(result, "p_value")
    assert hasattr(result, "reject_null")
    assert hasattr(result, "effect_size")
    assert hasattr(result, "confidence_interval")
    assert hasattr(result, "test_name")
    assert isinstance(result.confidence_interval, tuple)
    assert len(result.confidence_interval) == 2


# ---------------------------------------------------------------------------
# 3. test_paired_t_test_identical
# ---------------------------------------------------------------------------
def test_paired_t_test_identical():
    data = [0.5, 0.6, 0.7, 0.8, 0.9]
    result = paired_t_test(data, data)
    assert result.reject_null is False
    assert result.p_value >= 0.05


# ---------------------------------------------------------------------------
# 4. test_paired_t_test_significant
# ---------------------------------------------------------------------------
def test_paired_t_test_significant():
    scores_a = [1.0] * 50
    scores_b = [0.0] * 50
    result = paired_t_test(scores_a, scores_b)
    assert result.reject_null is True
    assert result.p_value < 0.05


# ---------------------------------------------------------------------------
# 5. test_paired_t_test_effect_size
# ---------------------------------------------------------------------------
def test_paired_t_test_effect_size():
    scores_a = [1.0] * 50
    scores_b = [0.0] * 50
    result = paired_t_test(scores_a, scores_b)
    assert abs(result.effect_size) > 1.0


# ---------------------------------------------------------------------------
# 6. test_wilcoxon_returns_result
# ---------------------------------------------------------------------------
def test_wilcoxon_returns_result():
    scores_a = [0.6, 0.7, 0.8, 0.55, 0.65]
    scores_b = [0.5, 0.6, 0.7, 0.45, 0.55]
    result = wilcoxon_signed_rank(scores_a, scores_b)
    assert isinstance(result, TestResult)
    assert hasattr(result, "statistic")
    assert hasattr(result, "p_value")
    assert hasattr(result, "reject_null")
    assert hasattr(result, "effect_size")
    assert hasattr(result, "confidence_interval")
    assert hasattr(result, "test_name")


# ---------------------------------------------------------------------------
# 7. test_wilcoxon_identical
# ---------------------------------------------------------------------------
def test_wilcoxon_identical():
    data = [0.5, 0.6, 0.7, 0.8, 0.9]
    result = wilcoxon_signed_rank(data, data)
    assert result.reject_null is False
    assert result.p_value >= 0.05


# ---------------------------------------------------------------------------
# 8. test_wilcoxon_significant
# ---------------------------------------------------------------------------
def test_wilcoxon_significant():
    scores_a = [1.0] * 50
    scores_b = [0.0] * 50
    result = wilcoxon_signed_rank(scores_a, scores_b)
    assert result.reject_null is True
    assert result.p_value < 0.05


# ---------------------------------------------------------------------------
# 9. test_mcnemar_returns_result
# ---------------------------------------------------------------------------
def test_mcnemar_returns_result():
    correct_a = [True, False, True, True, False]
    correct_b = [False, True, True, False, True]
    result = mcnemar_test(correct_a, correct_b)
    assert isinstance(result, TestResult)
    assert hasattr(result, "statistic")
    assert hasattr(result, "p_value")
    assert hasattr(result, "reject_null")
    assert hasattr(result, "effect_size")
    assert hasattr(result, "confidence_interval")
    assert hasattr(result, "test_name")
    assert result.test_name == "mcnemar_test"


# ---------------------------------------------------------------------------
# 10. test_mcnemar_symmetric
# ---------------------------------------------------------------------------
def test_mcnemar_symmetric():
    # Equal b and c → should fail to reject (symmetric, no advantage)
    correct_a = [True, False, True, False, True, False]
    correct_b = [False, True, False, True, False, True]
    result = mcnemar_test(correct_a, correct_b)
    assert result.reject_null is False


# ---------------------------------------------------------------------------
# 11. test_bootstrap_ci_contains_zero
# ---------------------------------------------------------------------------
def test_bootstrap_ci_contains_zero():
    data = [0.5] * 30
    lo, hi = bootstrap_confidence_interval(data, data, n_bootstrap=1000, alpha=0.05, seed=42)
    assert lo <= 0.0 <= hi


# ---------------------------------------------------------------------------
# 12. test_bootstrap_ci_excludes_zero
# ---------------------------------------------------------------------------
def test_bootstrap_ci_excludes_zero():
    scores_a = [1.0] * 50
    scores_b = [0.0] * 50
    lo, hi = bootstrap_confidence_interval(
        scores_a, scores_b, n_bootstrap=1000, alpha=0.05, seed=42
    )
    # Mean diff is 1.0; CI should be entirely above 0
    assert lo > 0.0


# ---------------------------------------------------------------------------
# 13. test_cohens_d_zero
# ---------------------------------------------------------------------------
def test_cohens_d_zero():
    data = [0.5, 0.6, 0.7, 0.8, 0.9]
    d = cohens_d(data, data)
    assert d == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 14. test_cohens_d_large
# ---------------------------------------------------------------------------
def test_cohens_d_large():
    scores_a = [1.0] * 50
    scores_b = [0.0] * 50
    d = cohens_d(scores_a, scores_b)
    assert abs(d) > 1.0


# ---------------------------------------------------------------------------
# 15. test_compute_power_increases_with_n
# ---------------------------------------------------------------------------
def test_compute_power_increases_with_n():
    effect = 0.5
    power_small = compute_power(effect, n=10)
    power_large = compute_power(effect, n=100)
    assert power_large > power_small


# ---------------------------------------------------------------------------
# 16. test_suite_compare_keys
# ---------------------------------------------------------------------------
def test_suite_compare_keys():
    cfg = HypothesisTestConfig(alpha=0.05)
    suite = ModelComparisonSuite(cfg=cfg)
    scores_a = [0.6, 0.7, 0.8, 0.55, 0.65, 0.72]
    scores_b = [0.5, 0.6, 0.7, 0.45, 0.55, 0.62]
    results = suite.compare(scores_a, scores_b)
    assert "t_test" in results
    assert "wilcoxon" in results
    assert isinstance(results["t_test"], TestResult)
    assert isinstance(results["wilcoxon"], TestResult)
