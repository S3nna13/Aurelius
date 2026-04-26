"""Tests for reward_rank_correlation module (~15 tests)."""

from __future__ import annotations

import torch
from aurelius.eval.reward_rank_correlation import (
    KendallTau,
    RankCorrelationMetrics,
    RewardRankEvaluator,
    SpearmanCorrelation,
)

# ---------------------------------------------------------------------------
# SpearmanCorrelation tests
# ---------------------------------------------------------------------------


class TestSpearmanCorrelation:
    def setup_method(self):
        self.spearman = SpearmanCorrelation()

    def test_perfect_positive_correlation(self):
        """Identical vectors should yield rho = 1.0."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        rho = self.spearman(x, x.clone())
        assert abs(rho - 1.0) < 1e-6

    def test_perfect_negative_correlation(self):
        """Reversed vector should yield rho = -1.0."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        y = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
        rho = self.spearman(x, y)
        assert abs(rho - (-1.0)) < 1e-6

    def test_random_vectors_loose_bound(self):
        """Independent random vectors should have |rho| < 0.5 (loose check)."""
        torch.manual_seed(42)
        x = torch.randn(100)
        y = torch.randn(100)
        rho = self.spearman(x, y)
        assert abs(rho) < 0.5

    def test_output_in_range(self):
        """Output must be in [-1, 1]."""
        torch.manual_seed(7)
        x = torch.randn(20)
        y = torch.randn(20)
        rho = self.spearman(x, y)
        assert -1.0 <= rho <= 1.0

    def test_explicit_sequence(self):
        """x = y = [1, 2, 3, 4, 5] should yield rho = 1.0."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        rho = self.spearman(x, x.clone())
        assert abs(rho - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# KendallTau tests
# ---------------------------------------------------------------------------


class TestKendallTau:
    def setup_method(self):
        self.kendall = KendallTau()

    def test_perfect_agreement(self):
        """Identical ordering → tau = 1.0."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        tau = self.kendall(x, x.clone())
        assert abs(tau - 1.0) < 1e-6

    def test_perfect_disagreement(self):
        """Completely reversed ordering → tau = -1.0."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        y = torch.tensor([4.0, 3.0, 2.0, 1.0])
        tau = self.kendall(x, y)
        assert abs(tau - (-1.0)) < 1e-6

    def test_output_in_range(self):
        """Output must be in [-1, 1]."""
        torch.manual_seed(99)
        x = torch.randn(15)
        y = torch.randn(15)
        tau = self.kendall(x, y)
        assert -1.0 <= tau <= 1.0

    def test_all_concordant_gives_tau_one(self):
        """All concordant pairs → tau = 1.0."""
        x = torch.tensor([1.0, 3.0, 5.0])
        y = torch.tensor([2.0, 4.0, 6.0])
        tau = self.kendall(x, y)
        assert abs(tau - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# RankCorrelationMetrics tests
# ---------------------------------------------------------------------------


class TestRankCorrelationMetrics:
    def test_dataclass_fields_exist(self):
        """Dataclass must have all required fields."""
        m = RankCorrelationMetrics(
            spearman_rho=0.9,
            kendall_tau=0.8,
            n_pairs=10,
            concordant=9,
            discordant=1,
        )
        assert hasattr(m, "spearman_rho")
        assert hasattr(m, "kendall_tau")
        assert hasattr(m, "n_pairs")
        assert hasattr(m, "concordant")
        assert hasattr(m, "discordant")


# ---------------------------------------------------------------------------
# RewardRankEvaluator tests
# ---------------------------------------------------------------------------


class TestRewardRankEvaluator:
    def setup_method(self):
        self.evaluator = RewardRankEvaluator(
            spearman=SpearmanCorrelation(),
            kendall=KendallTau(),
        )

    def test_evaluate_returns_metrics(self):
        """evaluate() must return a RankCorrelationMetrics instance."""
        scores = torch.tensor([3.0, 1.0, 4.0, 1.5])
        human = torch.tensor([2.0, 1.0, 4.0, 3.0])
        result = self.evaluator.evaluate(scores, human)
        assert isinstance(result, RankCorrelationMetrics)

    def test_evaluate_identical_rankings(self):
        """Identical reward and human rankings → rho = 1, tau = 1."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = self.evaluator.evaluate(x, x.clone())
        assert abs(result.spearman_rho - 1.0) < 1e-5
        assert abs(result.kendall_tau - 1.0) < 1e-5

    def test_evaluate_pairwise_all_correct(self):
        """All preferred scores > rejected → accuracy = 1.0."""
        reward_scores = torch.tensor([0.1, 0.5, 0.9, 0.3, 0.8])
        preferred = torch.tensor([2, 4, 1])  # scores: 0.9, 0.8, 0.5
        rejected = torch.tensor([0, 3, 0])  # scores: 0.1, 0.3, 0.1
        acc = self.evaluator.evaluate_pairwise(reward_scores, preferred, rejected)
        assert abs(acc - 1.0) < 1e-6

    def test_evaluate_pairwise_all_wrong(self):
        """All rejected scores > preferred → accuracy = 0.0."""
        reward_scores = torch.tensor([0.9, 0.1, 0.8, 0.2])
        preferred = torch.tensor([1, 3])  # scores: 0.1, 0.2
        rejected = torch.tensor([0, 2])  # scores: 0.9, 0.8
        acc = self.evaluator.evaluate_pairwise(reward_scores, preferred, rejected)
        assert abs(acc - 0.0) < 1e-6

    def test_evaluate_pairwise_output_in_range(self):
        """evaluate_pairwise output must be in [0, 1]."""
        torch.manual_seed(13)
        reward_scores = torch.randn(10)
        preferred = torch.randint(0, 10, (5,))
        rejected = torch.randint(0, 10, (5,))
        acc = self.evaluator.evaluate_pairwise(reward_scores, preferred, rejected)
        assert 0.0 <= acc <= 1.0
