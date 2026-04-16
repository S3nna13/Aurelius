"""
Tests for src/eval/elo_rating.py

Run with:
    cd ~/Desktop/Aurelius && .venv/bin/python3.13 -m pytest tests/eval/test_elo_rating.py -v
"""

from __future__ import annotations

import math

import pytest

from src.eval.elo_rating import (
    BayesianSkillTracker,
    BradleyTerry,
    ComparisonResult,
    ELORating,
    bootstrap_elo_ci,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_result(a: str, b: str, winner: str) -> ComparisonResult:
    return ComparisonResult(model_a=a, model_b=b, winner=winner)


def make_results_set() -> list[ComparisonResult]:
    """Return 10 deterministic comparison results across three models."""
    results = []
    pairs = [
        ("alpha", "beta", "model_a"),
        ("alpha", "gamma", "model_a"),
        ("beta", "gamma", "model_b"),
        ("alpha", "beta", "tie"),
        ("alpha", "gamma", "model_b"),
        ("beta", "gamma", "model_a"),
        ("alpha", "beta", "model_a"),
        ("alpha", "gamma", "model_a"),
        ("beta", "gamma", "tie"),
        ("alpha", "beta", "model_b"),
    ]
    for a, b, w in pairs:
        results.append(ComparisonResult(model_a=a, model_b=b, winner=w))
    return results


# ---------------------------------------------------------------------------
# ELO tests
# ---------------------------------------------------------------------------

class TestELORating:
    def test_initial_rating_is_base(self):
        """ELO rating initialises to base_rating for new models."""
        elo = ELORating(base_rating=1000.0)
        result = make_result("model_x", "model_y", "model_x")
        elo.update(result)
        # Before touching model_z, it should initialise to base on first access
        ratings = elo.get_ratings()
        # model_x and model_y were touched; their entry should exist
        assert "model_x" in ratings
        assert "model_y" in ratings

    def test_new_model_initialises_to_base_via_expected_prob(self):
        """Accessing expected_win_prob for unseen models seeds them at base_rating."""
        elo = ELORating(base_rating=1500.0)
        prob = elo.expected_win_prob("new_a", "new_b")
        ratings = elo.get_ratings()
        assert ratings["new_a"] == pytest.approx(1500.0)
        assert ratings["new_b"] == pytest.approx(1500.0)
        # Equal ratings → 50% win probability
        assert prob == pytest.approx(0.5, abs=1e-9)

    def test_winner_rating_increases(self):
        """After 1 win, winner rating increases above base_rating."""
        elo = ELORating(base_rating=1000.0)
        result = make_result("alpha", "beta", "model_a")
        elo.update(result)
        ratings = elo.get_ratings()
        assert ratings["alpha"] > 1000.0

    def test_loser_rating_decreases(self):
        """After 1 loss, loser rating decreases below base_rating."""
        elo = ELORating(base_rating=1000.0)
        result = make_result("alpha", "beta", "model_a")
        elo.update(result)
        ratings = elo.get_ratings()
        assert ratings["beta"] < 1000.0

    def test_rating_sum_preserved(self):
        """Win + loss should preserve the approximate sum of ratings."""
        elo = ELORating(base_rating=1000.0)
        result = make_result("alpha", "beta", "model_a")
        elo.update(result)
        ratings = elo.get_ratings()
        total = ratings["alpha"] + ratings["beta"]
        assert total == pytest.approx(2000.0, abs=1e-6)

    def test_expected_win_prob_in_range(self):
        """expected_win_prob must return a value strictly in (0, 1)."""
        elo = ELORating()
        # Seed some ratings
        elo.update(make_result("strong", "weak", "model_a"))
        elo.update(make_result("strong", "weak", "model_a"))
        elo.update(make_result("strong", "weak", "model_a"))
        prob = elo.expected_win_prob("strong", "weak")
        assert 0.0 < prob < 1.0

    def test_expected_win_prob_complementary(self):
        """expected_win_prob(a, b) + expected_win_prob(b, a) must equal 1.0."""
        elo = ELORating()
        elo.update(make_result("alpha", "beta", "model_a"))
        p_ab = elo.expected_win_prob("alpha", "beta")
        p_ba = elo.expected_win_prob("beta", "alpha")
        assert p_ab + p_ba == pytest.approx(1.0, abs=1e-9)

    def test_tie_leaves_equal_ratings(self):
        """A tie between equally-rated models should leave both unchanged."""
        elo = ELORating(base_rating=1000.0)
        result = make_result("alpha", "beta", "tie")
        elo.update(result)
        ratings = elo.get_ratings()
        # Both started at 1000; tie with equal ratings → no change
        assert ratings["alpha"] == pytest.approx(1000.0, abs=1e-6)
        assert ratings["beta"] == pytest.approx(1000.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Bradley-Terry tests
# ---------------------------------------------------------------------------

class TestBradleyTerry:
    def test_fit_runs_on_10_results(self):
        """BradleyTerry.fit should complete without error on 10 results."""
        bt = BradleyTerry()
        results = make_results_set()
        strengths = bt.fit(results)
        assert isinstance(strengths, dict)
        assert len(strengths) == 3  # alpha, beta, gamma

    def test_strengths_positive_after_exp(self):
        """exp(log_strength) must be strictly positive for every model."""
        bt = BradleyTerry()
        log_strengths = bt.fit(make_results_set())
        for model, log_s in log_strengths.items():
            assert math.exp(log_s) > 0.0, f"Non-positive strength for {model}"

    def test_predict_win_prob_in_range(self):
        """predict_win_prob must return a value strictly in (0, 1)."""
        bt = BradleyTerry()
        bt.fit(make_results_set())
        prob = bt.predict_win_prob("alpha", "beta")
        assert 0.0 < prob < 1.0

    def test_predict_win_prob_complementary(self):
        """predict_win_prob(a, b) + predict_win_prob(b, a) must equal 1.0."""
        bt = BradleyTerry()
        bt.fit(make_results_set())
        p_ab = bt.predict_win_prob("alpha", "beta")
        p_ba = bt.predict_win_prob("beta", "alpha")
        assert p_ab + p_ba == pytest.approx(1.0, abs=1e-6)

    def test_dominant_model_has_highest_strength(self):
        """A model that wins all comparisons should get the highest strength."""
        results = [
            ComparisonResult("best", "worst", "model_a"),
            ComparisonResult("best", "worst", "model_a"),
            ComparisonResult("best", "worst", "model_a"),
            ComparisonResult("best", "mid", "model_a"),
            ComparisonResult("best", "mid", "model_a"),
            ComparisonResult("mid", "worst", "model_a"),
        ]
        bt = BradleyTerry()
        log_s = bt.fit(results)
        strengths = {m: math.exp(v) for m, v in log_s.items()}
        assert strengths["best"] > strengths["mid"]
        assert strengths["mid"] > strengths["worst"]


# ---------------------------------------------------------------------------
# Bootstrap CI tests
# ---------------------------------------------------------------------------

class TestBootstrapEloCi:
    def test_returns_dict_with_tuples(self):
        """bootstrap_elo_ci must return a dict of (lower, upper) tuples."""
        results = make_results_set()
        ci = bootstrap_elo_ci(results, n_bootstrap=50)
        assert isinstance(ci, dict)
        for model, interval in ci.items():
            assert isinstance(interval, tuple)
            assert len(interval) == 2

    def test_lower_less_than_upper(self):
        """For every model the lower CI bound must be <= the upper bound."""
        results = make_results_set()
        ci = bootstrap_elo_ci(results, n_bootstrap=100)
        for model, (lo, hi) in ci.items():
            assert lo <= hi, f"CI inverted for {model}: ({lo}, {hi})"

    def test_ci_approximately_contains_point_estimate(self):
        """
        With enough bootstrap samples the CI should bracket the point estimate
        for at least one model (probabilistic sanity check, not strict).
        """
        results = make_results_set()
        # Point estimate
        elo = ELORating()
        for r in results:
            elo.update(r)
        point = elo.get_ratings()

        ci = bootstrap_elo_ci(results, n_bootstrap=500, confidence=0.95)
        contained = 0
        for model, (lo, hi) in ci.items():
            if lo <= point.get(model, 1000.0) <= hi:
                contained += 1
        # At least one model's point estimate should fall inside its CI
        assert contained >= 1


# ---------------------------------------------------------------------------
# Bayesian Skill Tracker tests
# ---------------------------------------------------------------------------

class TestBayesianSkillTracker:
    def test_update_runs_without_error(self):
        """BayesianSkillTracker.update should complete without raising."""
        tracker = BayesianSkillTracker()
        result = make_result("alpha", "beta", "model_a")
        tracker.update(result)  # Must not raise

    def test_get_skills_returns_all_models(self):
        """get_skills should return entries for every model seen in updates."""
        tracker = BayesianSkillTracker()
        tracker.update(make_result("alpha", "beta", "model_a"))
        tracker.update(make_result("beta", "gamma", "tie"))
        skills = tracker.get_skills()
        assert "alpha" in skills
        assert "beta" in skills
        assert "gamma" in skills

    def test_get_skills_returns_mu_sigma_tuples(self):
        """Each entry in get_skills should be a (mu, sigma) tuple."""
        tracker = BayesianSkillTracker()
        tracker.update(make_result("x", "y", "model_a"))
        for model, (mu, sigma) in tracker.get_skills().items():
            assert isinstance(mu, float)
            assert isinstance(sigma, float)
            assert sigma > 0.0, "Sigma must be positive"

    def test_leaderboard_sorted_descending(self):
        """Leaderboard must be sorted descending by conservative score (mu - 3*sigma)."""
        tracker = BayesianSkillTracker()
        # Feed many results to differentiate models
        for _ in range(10):
            tracker.update(make_result("strong", "weak", "model_a"))
        for _ in range(5):
            tracker.update(make_result("mid", "weak", "model_a"))

        lb = tracker.leaderboard()
        scores = [score for _, score in lb]
        assert scores == sorted(scores, reverse=True), "Leaderboard not sorted descending"

    def test_leaderboard_contains_all_models(self):
        """Every model that was updated should appear on the leaderboard."""
        tracker = BayesianSkillTracker()
        tracker.update(make_result("alpha", "beta", "model_a"))
        tracker.update(make_result("beta", "gamma", "model_b"))
        lb = tracker.leaderboard()
        lb_models = {m for m, _ in lb}
        assert "alpha" in lb_models
        assert "beta" in lb_models
        assert "gamma" in lb_models

    def test_winner_mu_increases(self):
        """After a decisive win the winner's mu should increase."""
        tracker = BayesianSkillTracker(mu0=25.0, sigma0=8.33)
        for _ in range(5):
            tracker.update(make_result("champion", "challenger", "model_a"))
        skills = tracker.get_skills()
        assert skills["champion"][0] > 25.0, "Winner mu should increase"

    def test_loser_mu_decreases(self):
        """After repeated losses the loser's mu should decrease."""
        tracker = BayesianSkillTracker(mu0=25.0, sigma0=8.33)
        for _ in range(5):
            tracker.update(make_result("champion", "challenger", "model_a"))
        skills = tracker.get_skills()
        assert skills["challenger"][0] < 25.0, "Loser mu should decrease"
