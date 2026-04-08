"""Statistical hypothesis testing for model comparison (pure Python/PyTorch, no scipy)."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class TestResult:
    statistic: float
    p_value: float
    reject_null: bool
    alpha: float = 0.05

    @property
    def significant(self) -> bool:
        return self.reject_null


def bootstrap_test(
    scores_a: list[float],
    scores_b: list[float],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> TestResult:
    """Bootstrap significance test for difference in means."""
    rng = random.Random(seed)
    n_a = len(scores_a)
    n_b = len(scores_b)
    observed_diff = sum(scores_a) / n_a - sum(scores_b) / n_b

    pooled = scores_a + scores_b

    count_ge = 0
    for _ in range(n_bootstrap):
        sample = [rng.choice(pooled) for _ in range(n_a + n_b)]
        resample_a = sample[:n_a]
        resample_b = sample[n_a:]
        diff = sum(resample_a) / n_a - sum(resample_b) / n_b
        if diff >= observed_diff:
            count_ge += 1

    p_value = count_ge / n_bootstrap
    return TestResult(
        statistic=observed_diff,
        p_value=p_value,
        reject_null=p_value < alpha,
        alpha=alpha,
    )


def _normal_cdf(z: float) -> float:
    """Approximate normal CDF using math.erfc."""
    return 0.5 * math.erfc(-z / math.sqrt(2))


def mcnemar_test(
    correct_a: list[bool],
    correct_b: list[bool],
    alpha: float = 0.05,
) -> TestResult:
    """McNemar's test for paired binary outcomes."""
    n01 = sum(1 for a, b in zip(correct_a, correct_b) if not a and b)
    n10 = sum(1 for a, b in zip(correct_a, correct_b) if a and not b)

    total = n01 + n10
    if total == 0:
        return TestResult(statistic=0.0, p_value=1.0, reject_null=False, alpha=alpha)

    if total >= 25:
        # Normal approximation
        z = (n01 - n10) / math.sqrt(total)
        statistic = z
        # Two-tailed p-value
        p_value = 2 * _normal_cdf(-abs(z))
    else:
        # Chi-squared with continuity correction
        statistic = (abs(n01 - n10) - 1) ** 2 / total
        # p-value from chi-squared df=1 via normal approximation:
        # chi2 ~ z^2, so z = sqrt(chi2), two-tailed
        z = math.sqrt(statistic)
        p_value = 2 * _normal_cdf(-z)

    return TestResult(
        statistic=statistic,
        p_value=p_value,
        reject_null=p_value < alpha,
        alpha=alpha,
    )


def permutation_test(
    scores_a: list[float],
    scores_b: list[float],
    n_permutations: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> TestResult:
    """Permutation test for difference in means."""
    rng = random.Random(seed)
    n_a = len(scores_a)
    n_b = len(scores_b)
    observed_stat = sum(scores_a) / n_a - sum(scores_b) / n_b

    pooled = scores_a + scores_b
    count_ge = 0
    for _ in range(n_permutations):
        shuffled = pooled[:]
        rng.shuffle(shuffled)
        perm_a = shuffled[:n_a]
        perm_b = shuffled[n_a:]
        diff = sum(perm_a) / n_a - sum(perm_b) / n_b
        if diff >= abs(observed_stat):
            count_ge += 1

    p_value = count_ge / n_permutations
    return TestResult(
        statistic=observed_stat,
        p_value=p_value,
        reject_null=p_value < alpha,
        alpha=alpha,
    )


def _t_cdf_approx(t: float, df: int) -> float:
    """Approximate two-tailed p-value for t-distribution."""
    if df >= 30:
        # Normal approximation
        return 2 * _normal_cdf(-abs(t))
    # Simple lookup for small df using normal approximation with slight correction
    # For small n we still use the normal approx (conservative)
    return 2 * _normal_cdf(-abs(t))


def paired_t_test(
    scores_a: list[float],
    scores_b: list[float],
    alpha: float = 0.05,
) -> TestResult:
    """Paired t-test (no scipy)."""
    n = len(scores_a)
    diffs = [a - b for a, b in zip(scores_a, scores_b)]
    mean_diff = sum(diffs) / n
    var_diff = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
    std_diff = math.sqrt(var_diff)

    if std_diff == 0.0:
        # No variation — cannot reject null; p_value = 1.0
        return TestResult(statistic=0.0, p_value=1.0, reject_null=False, alpha=alpha)

    t_stat = mean_diff / (std_diff / math.sqrt(n))
    p_value = _t_cdf_approx(t_stat, df=n - 1)
    return TestResult(
        statistic=t_stat,
        p_value=p_value,
        reject_null=p_value < alpha,
        alpha=alpha,
    )


def wilcoxon_signed_rank_test(
    scores_a: list[float],
    scores_b: list[float],
    alpha: float = 0.05,
) -> TestResult:
    """Non-parametric Wilcoxon signed-rank test."""
    diffs = [a - b for a, b in zip(scores_a, scores_b)]
    # Remove zero differences
    non_zero = [(i, d) for i, d in enumerate(diffs) if d != 0.0]

    if not non_zero:
        return TestResult(statistic=0.0, p_value=1.0, reject_null=False, alpha=alpha)

    n = len(non_zero)
    # Sort by absolute value
    sorted_by_abs = sorted(non_zero, key=lambda x: abs(x[1]))

    # Assign ranks (1-based), handle ties with average rank
    ranks: list[float] = []
    i = 0
    while i < n:
        j = i
        abs_val = abs(sorted_by_abs[i][1])
        while j < n and abs(sorted_by_abs[j][1]) == abs_val:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for _ in range(j - i):
            ranks.append(avg_rank)
        i = j

    w_plus = sum(r for (_, d), r in zip(sorted_by_abs, ranks) if d > 0)
    w_minus = sum(r for (_, d), r in zip(sorted_by_abs, ranks) if d < 0)
    w_stat = min(w_plus, w_minus)

    # Normal approximation
    mean_w = n * (n + 1) / 4.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0
    z = (w_stat - mean_w) / math.sqrt(var_w)
    p_value = 2 * _normal_cdf(z)  # z is typically negative for W = min

    return TestResult(
        statistic=w_stat,
        p_value=p_value,
        reject_null=p_value < alpha,
        alpha=alpha,
    )


class ModelComparisonSuite:
    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha

    def compare(
        self,
        scores_a: list[float],
        scores_b: list[float],
        correct_a: list[bool] | None = None,
        correct_b: list[bool] | None = None,
    ) -> dict[str, TestResult]:
        results: dict[str, TestResult] = {}
        results["bootstrap"] = bootstrap_test(scores_a, scores_b, alpha=self.alpha)
        results["permutation"] = permutation_test(scores_a, scores_b, alpha=self.alpha)
        results["t_test"] = paired_t_test(scores_a, scores_b, alpha=self.alpha)
        if correct_a is not None and correct_b is not None:
            results["mcnemar"] = mcnemar_test(correct_a, correct_b, alpha=self.alpha)
        return results

    def summary(self, results: dict[str, TestResult]) -> dict[str, dict]:
        return {
            name: {"p_value": r.p_value, "significant": r.significant}
            for name, r in results.items()
        }
