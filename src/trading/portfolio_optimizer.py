from __future__ import annotations

import torch


def optimal_weights(mean_returns: torch.Tensor, cov_matrix: torch.Tensor) -> torch.Tensor:
    n = len(mean_returns)
    if n == 1:
        return torch.ones(1)
    inv_cov = torch.inverse(cov_matrix)
    ones = torch.ones(n)
    w = inv_cov @ ones / (ones @ inv_cov @ ones)
    return w / w.sum()


class PortfolioOptimizer:
    def optimize(self, prices: torch.Tensor) -> torch.Tensor:
        returns = prices[1:] / prices[:-1] - 1
        mean_r = returns.mean(dim=0)
        cov = returns.T.cov()
        try:
            w = optimal_weights(mean_r, cov)
            return w / w.sum()
        except Exception:
            return torch.ones(prices.shape[1]) / prices.shape[1]

    def sharpe(self, portfolio_returns: torch.Tensor) -> float:
        return (
            (portfolio_returns.mean() / portfolio_returns.std()).item()
            if portfolio_returns.std() > 0
            else 0.0
        )


PORTFOLIO_OPTIMIZER = PortfolioOptimizer()
