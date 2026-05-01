"""
ELO / Bradley-Terry rating system for pairwise model comparison.

Implements the mathematical foundation used by Chatbot Arena / LMSYS rankings:
  - Standard ELO with K-factor updates
  - Bradley-Terry MLE via multiplicative iterative algorithm
  - Bootstrap confidence intervals on ELO ratings
  - TrueSkill-lite Bayesian skill tracker (Gaussian mean + variance)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class ComparisonResult:
    model_a: str
    model_b: str
    winner: str  # "model_a", "model_b", or "tie"
    prompt_id: str | None = None


# ---------------------------------------------------------------------------
# Standard ELO
# ---------------------------------------------------------------------------


class ELORating:
    """Standard ELO rating system with configurable K-factor."""

    def __init__(self, k: float = 32.0, base_rating: float = 1000.0) -> None:
        self.k = k
        self.base_rating = base_rating
        self._ratings: dict[str, float] = {}

    def _get_rating(self, model: str) -> float:
        return self._ratings.setdefault(model, self.base_rating)

    def expected_win_prob(self, model_a: str, model_b: str) -> float:
        """P(A beats B) = 1 / (1 + 10^((Rb - Ra) / 400))."""
        ra = self._get_rating(model_a)
        rb = self._get_rating(model_b)
        return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))

    def update(self, result: ComparisonResult) -> None:
        """Apply one ELO update from a comparison result."""
        ra = self._get_rating(result.model_a)
        rb = self._get_rating(result.model_b)

        ea = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
        eb = 1.0 - ea

        if result.winner == "model_a":
            sa, sb = 1.0, 0.0
        elif result.winner == "model_b":
            sa, sb = 0.0, 1.0
        else:  # tie
            sa, sb = 0.5, 0.5

        self._ratings[result.model_a] = ra + self.k * (sa - ea)
        self._ratings[result.model_b] = rb + self.k * (sb - eb)

    def get_ratings(self) -> dict[str, float]:
        return dict(self._ratings)


# ---------------------------------------------------------------------------
# Bradley-Terry MLE
# ---------------------------------------------------------------------------


class BradleyTerry:
    """
    Bradley-Terry model fitted by the multiplicative iterative algorithm.

    Strengths are kept in linear scale internally; log-scale values are
    returned by fit() for comparability with ELO ratings.
    """

    def __init__(self, n_iter: int = 1000, tol: float = 1e-6) -> None:
        self.n_iter = n_iter
        self.tol = tol
        self._strengths: dict[str, float] = {}
        self._models: list[str] = []

    def fit(self, results: list[ComparisonResult]) -> dict[str, float]:
        """
        Fit strengths from pairwise win counts using the standard
        multiplicative update:

            p_i  <-  W_i / sum_{j != i} (n_ij / (p_i + p_j))

        Ties award 0.5 win credit to each model.

        Returns dict[model_name -> log(strength)].
        """
        # Collect models and build win / game count tables
        models_set: set[str] = set()
        for r in results:
            models_set.add(r.model_a)
            models_set.add(r.model_b)
        self._models = sorted(models_set)
        n = len(self._models)
        idx = {m: i for i, m in enumerate(self._models)}

        # wins[i] = total wins (including 0.5 for ties) by model i
        # games[i][j] = number of times i and j met (i != j)
        wins = [0.0] * n
        games = [[0.0] * n for _ in range(n)]

        for r in results:
            i, j = idx[r.model_a], idx[r.model_b]
            if r.winner == "model_a":
                wins[i] += 1.0
            elif r.winner == "model_b":
                wins[j] += 1.0
            else:  # tie
                wins[i] += 0.5
                wins[j] += 0.5
            games[i][j] += 1.0
            games[j][i] += 1.0

        # Initialise strengths uniformly
        p = [1.0] * n

        for _ in range(self.n_iter):
            p_new = [0.0] * n
            for i in range(n):
                denom = 0.0
                for j in range(n):
                    if i != j and games[i][j] > 0:
                        denom += games[i][j] / (p[i] + p[j])
                if denom == 0.0 or wins[i] == 0.0:
                    p_new[i] = p[i]
                else:
                    p_new[i] = wins[i] / denom

            # Normalise so that sum = n (keeps values from drifting)
            total = sum(p_new)
            if total > 0:
                p_new = [v * n / total for v in p_new]

            # Check convergence
            max_delta = max(abs(p_new[i] - p[i]) for i in range(n))
            p = p_new
            if max_delta < self.tol:
                break

        self._strengths = {self._models[i]: p[i] for i in range(n)}
        # Return log-scale strengths (log(p))
        return {m: math.log(max(v, 1e-12)) for m, v in self._strengths.items()}

    def predict_win_prob(self, model_a: str, model_b: str) -> float:
        """P(A beats B) = p_a / (p_a + p_b) in Bradley-Terry model."""
        pa = self._strengths.get(model_a, 1.0)
        pb = self._strengths.get(model_b, 1.0)
        return pa / (pa + pb)


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


def bootstrap_elo_ci(
    results: list[ComparisonResult],
    n_bootstrap: int = 200,
    confidence: float = 0.95,
) -> dict[str, tuple[float, float]]:
    """
    Resample with replacement over the results list, refit ELO ratings for
    each bootstrap sample, and return confidence intervals.

    Returns {model: (lower_ci, upper_ci)}.
    """
    if not results:
        return {}

    rng = random.Random(42)  # noqa: S311
    n = len(results)

    # Collect all model names up front so every bootstrap sample includes them
    all_models: set[str] = set()
    for r in results:
        all_models.add(r.model_a)
        all_models.add(r.model_b)

    bootstrap_ratings: dict[str, list[float]] = {m: [] for m in all_models}

    for _ in range(n_bootstrap):
        sample = [rng.choice(results) for _ in range(n)]
        elo = ELORating()
        for r in sample:
            elo.update(r)
        ratings = elo.get_ratings()
        for m in all_models:
            bootstrap_ratings[m].append(ratings.get(m, elo.base_rating))

    alpha = 1.0 - confidence
    lo_pct = alpha / 2.0
    hi_pct = 1.0 - alpha / 2.0

    ci: dict[str, tuple[float, float]] = {}
    for model, vals in bootstrap_ratings.items():
        vals_sorted = sorted(vals)
        lo_idx = int(math.floor(lo_pct * len(vals_sorted)))
        hi_idx = min(int(math.ceil(hi_pct * len(vals_sorted))), len(vals_sorted) - 1)
        ci[model] = (vals_sorted[lo_idx], vals_sorted[hi_idx])

    return ci


# ---------------------------------------------------------------------------
# TrueSkill-lite Bayesian Skill Tracker
# ---------------------------------------------------------------------------


class BayesianSkillTracker:
    """
    TrueSkill-lite: tracks a Gaussian belief (mu, sigma^2) per model.

    Update rule (simplified Thurstone / TrueSkill-style):
      - The performance difference D = perf_a - perf_b is a Gaussian with
        mean mu_a - mu_b and variance sigma_a^2 + sigma_b^2 + beta^2.
      - We compute the v/w correction factors from the truncated normal and
        update mu and sigma for each player.

    Conservative score for leaderboard = mu - 3 * sigma.
    """

    def __init__(self, mu0: float = 25.0, sigma0: float = 8.33) -> None:
        self.mu0 = mu0
        self.sigma0 = sigma0
        # beta controls performance variability; set to sigma0 / 2 by convention
        self.beta = sigma0 / 2.0
        self._mu: dict[str, float] = {}
        self._sigma: dict[str, float] = {}

    def _get(self, model: str) -> tuple[float, float]:
        mu = self._mu.setdefault(model, self.mu0)
        sigma = self._sigma.setdefault(model, self.sigma0)
        return mu, sigma

    @staticmethod
    def _norm_pdf(x: float) -> float:
        return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

    @staticmethod
    def _norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def _v_win(self, t: float) -> float:
        """Mill's ratio correction for winner."""
        cdf = self._norm_cdf(t)
        if cdf < 1e-12:
            return -t
        return self._norm_pdf(t) / cdf

    def _w_win(self, t: float) -> float:
        """Variance correction for winner."""
        v = self._v_win(t)
        return v * (v + t)

    def _v_draw(self, t: float, eps: float) -> float:
        """Mill's ratio correction for draw."""
        num = self._norm_pdf(-eps - t) - self._norm_pdf(eps - t)
        den = self._norm_cdf(eps - t) - self._norm_cdf(-eps - t)
        if abs(den) < 1e-12:
            return 0.0
        return num / den

    def _w_draw(self, t: float, eps: float) -> float:
        """Variance correction for draw."""
        num = (eps - t) * self._norm_pdf(eps - t) + (eps + t) * self._norm_pdf(eps + t)
        den = self._norm_cdf(eps - t) - self._norm_cdf(-eps - t)
        if abs(den) < 1e-12:
            return 0.0
        return num / den

    def update(self, result: ComparisonResult) -> None:
        mu_a, sigma_a = self._get(result.model_a)
        mu_b, sigma_b = self._get(result.model_b)

        c = math.sqrt(sigma_a**2 + sigma_b**2 + 2.0 * self.beta**2)
        eps = 0.0  # draw margin (simplified: no draw zone)
        t = (mu_a - mu_b) / c

        if result.winner == "model_a":
            v = self._v_win(t)
            w = self._w_win(t)
            sign_a, sign_b = 1.0, -1.0
        elif result.winner == "model_b":
            # Flip perspective: treat b as winner
            t_b = (mu_b - mu_a) / c
            v = self._v_win(t_b)
            w = self._w_win(t_b)
            sign_a, sign_b = -1.0, 1.0
        else:  # tie
            v = self._v_draw(t, eps)
            w = self._w_draw(t, eps)
            sign_a, sign_b = 1.0, -1.0  # symmetric for ties

        delta_a = (sigma_a**2 / c) * sign_a * v
        delta_b = (sigma_b**2 / c) * sign_b * v

        gamma_a = (sigma_a**2 / c**2) * w
        gamma_b = (sigma_b**2 / c**2) * w

        self._mu[result.model_a] = mu_a + delta_a
        self._mu[result.model_b] = mu_b + delta_b

        new_var_a = sigma_a**2 * (1.0 - gamma_a)
        new_var_b = sigma_b**2 * (1.0 - gamma_b)

        # Clamp variance to remain positive
        self._sigma[result.model_a] = math.sqrt(max(new_var_a, 1e-6))
        self._sigma[result.model_b] = math.sqrt(max(new_var_b, 1e-6))

    def get_skills(self) -> dict[str, tuple[float, float]]:
        """Return {model: (mu, sigma)}."""
        models = set(self._mu) | set(self._sigma)
        return {m: (self._mu.get(m, self.mu0), self._sigma.get(m, self.sigma0)) for m in models}

    def leaderboard(self) -> list[tuple[str, float]]:
        """Return list of (model, conservative_score) sorted descending.

        Conservative score = mu - 3 * sigma (lower-confidence bound).
        """
        scores = [(m, mu - 3.0 * sigma) for m, (mu, sigma) in self.get_skills().items()]
        return sorted(scores, key=lambda x: x[1], reverse=True)
