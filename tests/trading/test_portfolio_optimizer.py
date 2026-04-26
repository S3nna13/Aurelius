"""Tests for portfolio_optimizer — mean-variance portfolio optimization."""

from __future__ import annotations

import torch

from src.trading.portfolio_optimizer import PortfolioOptimizer, optimal_weights


class TestOptimalWeights:
    def test_two_assets_equal_return(self):
        returns = torch.tensor([[0.01, 0.02], [0.015, 0.025]])
        cov = torch.tensor([[0.1, 0.05], [0.05, 0.2]])
        w = optimal_weights(returns.mean(dim=0), cov)
        assert w.shape == (2,)
        assert abs(w.sum().item() - 1.0) < 1e-6

    def test_single_asset(self):
        w = optimal_weights(torch.tensor([0.01]), torch.tensor([[0.1]]))
        assert abs(w[0].item() - 1.0) < 1e-6


class TestPortfolioOptimizer:
    def test_optimize_returns_weights(self):
        prices = torch.randn(100, 5)
        opt = PortfolioOptimizer()
        w = opt.optimize(prices)
        assert w.shape == (5,)
        assert abs(w.sum().item() - 1.0) < 0.01

    def test_sharpe_ratio_is_calculated(self):
        prices = torch.cumsum(torch.randn(100, 3) * 0.01, dim=0) + 100
        opt = PortfolioOptimizer()
        w = opt.optimize(prices)
        sr = opt.sharpe(prices @ w)
        assert isinstance(sr, float)

    def test_equal_weight_fallback(self):
        opt = PortfolioOptimizer()
        w = opt.optimize(torch.randn(10, 1))
        assert abs(w[0].item() - 1.0) < 0.01
