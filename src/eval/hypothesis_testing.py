"""Statistical hypothesis testing for model comparison (pure Python/PyTorch, no scipy)."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class HypothesisTestConfig:
    alpha: float = 0.05              # significance level
    n_bootstrap: int = 1000          # bootstrap samples
    alternative: str = "two-sided"   # "two-sided" | "greater" | "less"
    seed: int = 42


@dataclass
class TestResult:
    statistic: float
    p_value: float
    reject_null: bool                 # True if p < alpha
    effect_size: float = 0.0
    confidence_interval: tuple[float, float] = (0.0, 0.0)
    test_name: str = ""
    # Legacy field kept for backward-compat
    alpha: float = 0.05

    @property
    def significant(self) -> bool:
        return self.reject_null


# Prevent pytest from trying to collect this helper dataclass as a test class.
TestResult.__test__ = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normal_cdf(z: float) -> float:
    """Standard normal CDF via math.erf."""
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def _normal_cdf_erfc(z: float) -> float:
    """Approximate normal CDF using math.erfc (alternative)."""
    return 0.5 * math.erfc(-z / math.sqrt(2))


def _t_cdf_approx(t: float, df: int) -> float:
    """Approximate two-tailed p-value for t-distribution (normal approx)."""
    return 2 * _normal_cdf(-abs(t))


def _normal_ppf(p: float) -> float:
    """Inverse normal CDF (percent-point function) via binary search."""
    if p <= 0:
        return -math.inf
    if p >= 1:
        return math.inf
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if _normal_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ---------------------------------------------------------------------------
# Core statistical functions
# ---------------------------------------------------------------------------

def cohens_d(scores_a: list[float], scores_b: list[float]) -> float:
    """Cohen's d effect size: (mean_a - mean_b) / pooled_std."""
    n_a = len(scores_a)
    n_b = len(scores_b)
    mean_a = sum(scores_a) / n_a
    mean_b = sum(scores_b) / n_b
    var_a = sum((x - mean_a) ** 2 for x in scores_a) / (n_a - 1) if n_a > 1 else 0.0
    var_b = sum((x - mean_b) ** 2 for x in scores_b) / (n_b - 1) if n_b > 1 else 0.0
    pooled_std = math.sqrt((var_a + var_b) / 2)
    if pooled_std == 0.0:
        # Both distributions are constant; if means differ, effect is infinite
        if mean_a == mean_b:
            return 0.0
        return math.copysign(1e6, mean_a - mean_b)
    return (mean_a - mean_b) / pooled_std


def bootstrap_confidence_interval(
    scores_a: list[float],
    scores_b: list[float],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap CI for mean difference (scores_a - scores_b).

    Sample pairs with replacement, compute mean diff each time.
    Return (alpha/2, 1-alpha/2) percentiles.
    """
    rng = random.Random(seed)
    n = len(scores_a)
    diffs = [a - b for a, b in zip(scores_a, scores_b)]
    boot_means: list[float] = []
    for _ in range(n_bootstrap):
        sample = [rng.choice(diffs) for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    lo_idx = int(math.floor((alpha / 2) * n_bootstrap))
    hi_idx = int(math.floor((1 - alpha / 2) * n_bootstrap)) - 1
    lo_idx = max(0, min(lo_idx, n_bootstrap - 1))
    hi_idx = max(0, min(hi_idx, n_bootstrap - 1))
    return (boot_means[lo_idx], boot_means[hi_idx])


def compute_power(
    effect_size: float,
    n: int,
    alpha: float = 0.05,
) -> float:
    """Approximate statistical power using normal approximation.

    z_alpha = Phi_inv(1 - alpha/2), z_beta = effect_size * sqrt(n) - z_alpha
    power = Phi(z_beta)
    """
    z_alpha = _normal_ppf(1 - alpha / 2)
    z_beta = effect_size * math.sqrt(n) - z_alpha
    return _normal_cdf(z_beta)


def paired_t_test(
    scores_a: list[float],
    scores_b: list[float],
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> TestResult:
    """Paired t-test for model A vs model B.

    Differences d = scores_a - scores_b.
    t = mean(d) / (std(d) / sqrt(n))
    p_value: use normal approximation (2*(1-Phi(|t|))).
    effect_size: Cohen's d = mean(d) / std(d)
    """
    n = len(scores_a)
    diffs = [a - b for a, b in zip(scores_a, scores_b)]
    mean_diff = sum(diffs) / n
    var_diff = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
    std_diff = math.sqrt(var_diff)

    if std_diff == 0.0:
        # All differences are identical — if nonzero, the result is perfectly clear
        ci = bootstrap_confidence_interval(scores_a, scores_b, alpha=alpha)
        if mean_diff == 0.0:
            return TestResult(
                statistic=0.0,
                p_value=1.0,
                reject_null=False,
                effect_size=0.0,
                confidence_interval=ci,
                test_name="paired_t_test",
                alpha=alpha,
            )
        # Nonzero constant difference: t → ±∞, p → 0, effect → ∞ (cap at large value)
        sign = 1.0 if mean_diff > 0 else -1.0
        large_effect = sign * 1e6
        return TestResult(
            statistic=sign * math.inf,
            p_value=0.0,
            reject_null=True,
            effect_size=large_effect,
            confidence_interval=ci,
            test_name="paired_t_test",
            alpha=alpha,
        )

    t_stat = mean_diff / (std_diff / math.sqrt(n))
    effect = mean_diff / std_diff  # Cohen's d for paired differences

    if alternative == "two-sided":
        p_value = 2 * _normal_cdf(-abs(t_stat))
    elif alternative == "greater":
        p_value = 1 - _normal_cdf(t_stat)
    else:  # "less"
        p_value = _normal_cdf(t_stat)

    ci = bootstrap_confidence_interval(scores_a, scores_b, alpha=alpha)
    return TestResult(
        statistic=t_stat,
        p_value=p_value,
        reject_null=p_value < alpha,
        effect_size=effect,
        confidence_interval=ci,
        test_name="paired_t_test",
        alpha=alpha,
    )


def wilcoxon_signed_rank(
    scores_a: list[float],
    scores_b: list[float],
    alpha: float = 0.05,
) -> TestResult:
    """Non-parametric Wilcoxon signed-rank test.

    1. Compute differences, exclude zeros
    2. Rank by absolute value
    3. W+ = sum of ranks where difference > 0
    4. Expected W under null = n*(n+1)/4
    5. Std(W) = sqrt(n*(n+1)*(2n+1)/24)
    6. Z = (W+ - E[W]) / Std(W), p-value from normal approximation
    effect_size = W+ / (n*(n+1)/2) (rank biserial correlation)
    """
    diffs = [a - b for a, b in zip(scores_a, scores_b)]
    non_zero = [(i, d) for i, d in enumerate(diffs) if d != 0.0]

    if not non_zero:
        ci = bootstrap_confidence_interval(scores_a, scores_b, alpha=alpha)
        return TestResult(
            statistic=0.0,
            p_value=1.0,
            reject_null=False,
            effect_size=0.0,
            confidence_interval=ci,
            test_name="wilcoxon_signed_rank",
            alpha=alpha,
        )

    n = len(non_zero)
    sorted_by_abs = sorted(non_zero, key=lambda x: abs(x[1]))

    # Assign ranks with average for ties
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
    mean_w = n * (n + 1) / 4.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0
    z = (w_plus - mean_w) / math.sqrt(var_w)

    p_value = 2 * _normal_cdf(-abs(z))

    max_w = n * (n + 1) / 2.0
    effect = w_plus / max_w if max_w > 0 else 0.0

    ci = bootstrap_confidence_interval(scores_a, scores_b, alpha=alpha)
    return TestResult(
        statistic=w_plus,
        p_value=p_value,
        reject_null=p_value < alpha,
        effect_size=effect,
        confidence_interval=ci,
        test_name="wilcoxon_signed_rank",
        alpha=alpha,
    )


# Backward-compat alias
def wilcoxon_signed_rank_test(
    scores_a: list[float],
    scores_b: list[float],
    alpha: float = 0.05,
) -> TestResult:
    """Alias for wilcoxon_signed_rank (backward compatibility)."""
    return wilcoxon_signed_rank(scores_a, scores_b, alpha=alpha)


def mcnemar_test(
    correct_a: list[bool],
    correct_b: list[bool],
    alpha: float = 0.05,
) -> TestResult:
    """McNemar's test for comparing binary outcomes.

    b = count where A correct, B wrong
    c = count where A wrong, B correct
    chi2 = (|b - c| - 1)^2 / (b + c) (with continuity correction)
    p-value: p = exp(-chi2/2)
    effect_size = (b - c) / n
    """
    n = len(correct_a)
    b = sum(1 for a, bb in zip(correct_a, correct_b) if a and not bb)   # A correct, B wrong
    c = sum(1 for a, bb in zip(correct_a, correct_b) if not a and bb)   # A wrong, B correct

    total = b + c
    if total == 0:
        return TestResult(
            statistic=0.0,
            p_value=1.0,
            reject_null=False,
            effect_size=0.0,
            confidence_interval=(0.0, 0.0),
            test_name="mcnemar_test",
            alpha=alpha,
        )

    chi2 = (abs(b - c) - 1) ** 2 / total
    p_value = math.exp(-chi2 / 2)
    effect = (b - c) / n if n > 0 else 0.0

    return TestResult(
        statistic=chi2,
        p_value=p_value,
        reject_null=p_value < alpha,
        effect_size=effect,
        confidence_interval=(0.0, 0.0),
        test_name="mcnemar_test",
        alpha=alpha,
    )


# ---------------------------------------------------------------------------
# Bootstrap significance test (legacy, kept for backward compat)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Permutation test (legacy, kept for backward compat)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Model comparison suite
# ---------------------------------------------------------------------------

class ModelComparisonSuite:
    """Run multiple hypothesis tests and aggregate results."""

    def __init__(self, cfg: HypothesisTestConfig | None = None, alpha: float = 0.05) -> None:
        if cfg is not None:
            self.cfg = cfg
        else:
            self.cfg = HypothesisTestConfig(alpha=alpha)
        # Legacy attribute
        self.alpha = self.cfg.alpha

    def compare(
        self,
        scores_a: list[float],
        scores_b: list[float],
        name_a: str = "model_a",
        name_b: str = "model_b",
        # Legacy params
        correct_a: list[bool] | None = None,
        correct_b: list[bool] | None = None,
    ) -> dict[str, TestResult]:
        """Run paired_t_test and wilcoxon_signed_rank. Return {"t_test": ..., "wilcoxon": ...}"""
        results: dict[str, TestResult] = {}
        results["t_test"] = paired_t_test(
            scores_a, scores_b, alpha=self.cfg.alpha, alternative=self.cfg.alternative
        )
        results["wilcoxon"] = wilcoxon_signed_rank(
            scores_a, scores_b, alpha=self.cfg.alpha
        )
        # Legacy support: include bootstrap and permutation when old-style call
        if correct_a is not None and correct_b is not None:
            results["mcnemar"] = mcnemar_test(correct_a, correct_b, alpha=self.cfg.alpha)
        return results

    def compare_binary(
        self,
        correct_a: list[bool],
        correct_b: list[bool],
    ) -> TestResult:
        """Run McNemar's test."""
        return mcnemar_test(correct_a, correct_b, alpha=self.cfg.alpha)

    def summary(self, results: dict[str, TestResult]) -> dict[str, float]:
        """Summarize: {"n_significant", "min_p_value", "max_effect_size"}"""
        n_significant = sum(1 for r in results.values() if r.reject_null)
        min_p = min(r.p_value for r in results.values()) if results else float("inf")
        max_effect = max(abs(r.effect_size) for r in results.values()) if results else 0.0
        return {
            "n_significant": float(n_significant),
            "min_p_value": min_p,
            "max_effect_size": max_effect,
        }
