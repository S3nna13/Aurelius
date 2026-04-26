"""Pass@k Evaluator — unbiased estimator for code evaluation.

Implements the Chen et al. 2021 ("Evaluating Large Language Models Trained on
Code") unbiased pass@k metric using the numerically stable product formula:

    pass@k = 1 - prod_{i=0}^{k-1} (n - c - i) / (n - i)

where n = total samples per problem, c = correct samples, k = samples to consider.

Cycle 136-F.
"""

from __future__ import annotations

from dataclasses import dataclass

# Registry import — additive, leaves all prior entries untouched.
from src.eval import BENCHMARK_REGISTRY

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PassAtKConfig:
    """Configuration for the Pass@k evaluator."""

    k_values: list[int]
    """List of k values to evaluate, e.g. [1, 5, 10, 100]."""

    n_samples: int = 200
    """Default total samples per problem (n)."""


@dataclass
class ProblemResult:
    """Per-problem sampling result."""

    problem_id: str
    """Unique identifier for the problem."""

    n_samples: int
    """Total number of attempts (n)."""

    n_correct: int
    """Number of passing attempts (c)."""


@dataclass
class PassAtKResult:
    """Aggregated pass@k result for a single k value."""

    k: int
    """The k value this result was computed for."""

    pass_at_k: float
    """Mean pass@k across all problems, in [0, 1]."""

    n_problems: int
    """Number of problems used in the computation."""


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class PassAtKEvaluator:
    """Unbiased pass@k estimator (Chen et al. 2021).

    Parameters
    ----------
    config:
        A :class:`PassAtKConfig` specifying which k values to evaluate and the
        default total-sample count.
    """

    def __init__(self, config: PassAtKConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Core single-problem estimator
    # ------------------------------------------------------------------

    def estimate_single(self, n: int, c: int, k: int) -> float:
        """Return the unbiased pass@k estimate for a single problem.

        Parameters
        ----------
        n:
            Total samples generated for the problem.
        c:
            Number of correct (passing) samples.
        k:
            Number of samples to consider for the estimate.

        Returns
        -------
        float
            Estimated probability in [0, 1].
        """
        # Edge cases
        if n < k:
            # Cannot draw k distinct samples from n — treat as 1 if any correct
            return 1.0 if c > 0 else 0.0
        if c == 0:
            return 0.0
        if n - c < k:
            # Impossible to draw k samples all from the (n-c) incorrect pool
            return 1.0

        # Numerically stable product formula:
        #   pass@k = 1 - prod_{i=0}^{k-1} (n - c - i) / (n - i)
        result = 1.0
        for i in range(k):
            result *= (n - c - i) / (n - i)
        return 1.0 - result

    # ------------------------------------------------------------------
    # Batch estimator
    # ------------------------------------------------------------------

    def estimate_batch(self, results: list[ProblemResult], k: int) -> float:
        """Return the mean pass@k across all problems.

        Parameters
        ----------
        results:
            Per-problem results.
        k:
            The k value to estimate.

        Returns
        -------
        float
            Mean pass@k, or 0.0 for an empty list.
        """
        if not results:
            return 0.0
        total = sum(self.estimate_single(r.n_samples, r.n_correct, k) for r in results)
        return total / len(results)

    # ------------------------------------------------------------------
    # Full evaluation
    # ------------------------------------------------------------------

    def evaluate(self, results: list[ProblemResult]) -> list[PassAtKResult]:
        """Compute pass@k for every k in ``config.k_values``.

        Parameters
        ----------
        results:
            Per-problem results.

        Returns
        -------
        list[PassAtKResult]
            One entry per k value in :attr:`PassAtKConfig.k_values`.
        """
        out: list[PassAtKResult] = []
        for k in self.config.k_values:
            val = self.estimate_batch(results, k)
            out.append(PassAtKResult(k=k, pass_at_k=val, n_problems=len(results)))
        return out

    # ------------------------------------------------------------------
    # Summary dict
    # ------------------------------------------------------------------

    def summary(self, results: list[ProblemResult]) -> dict[str, float]:
        """Return a ``{"pass@k": value}`` dict for all configured k values.

        Parameters
        ----------
        results:
            Per-problem results.

        Returns
        -------
        dict[str, float]
            Keys are ``"pass@1"``, ``"pass@5"``, etc.
        """
        return {f"pass@{r.k}": r.pass_at_k for r in self.evaluate(results)}

    # ------------------------------------------------------------------
    # Difficulty distribution
    # ------------------------------------------------------------------

    def difficulty_distribution(self, results: list[ProblemResult]) -> dict[str, int]:
        """Bucket problems by solve-rate into four difficulty categories.

        Categories
        ----------
        unsolved
            ``c == 0``
        easy
            ``0 < c/n < 0.5``
        medium
            ``0.5 <= c/n < 1.0``
        all_correct
            ``c == n``

        Parameters
        ----------
        results:
            Per-problem results.

        Returns
        -------
        dict[str, int]
            Counts for each category; sum equals len(results).
        """
        counts: dict[str, int] = {
            "unsolved": 0,
            "easy": 0,
            "medium": 0,
            "all_correct": 0,
        }
        for r in results:
            n, c = r.n_samples, r.n_correct
            if c == 0:
                counts["unsolved"] += 1
            elif c == n:
                counts["all_correct"] += 1
            elif c / n < 0.5:
                counts["easy"] += 1
            else:
                counts["medium"] += 1
        return counts


# ---------------------------------------------------------------------------
# Registry registration
# ---------------------------------------------------------------------------

BENCHMARK_REGISTRY["pass_at_k"] = PassAtKEvaluator
