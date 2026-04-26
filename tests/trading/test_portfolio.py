from __future__ import annotations

import unittest

from src.trading.portfolio import (
    PORTFOLIO_REGISTRY,
    Portfolio,
    list_portfolios,
    register_portfolio,
)


class TestPortfolio(unittest.TestCase):
    def setUp(self):
        self.portfolio = Portfolio(name="test", initial_cash=100000.0)

    def test_initial_state(self):
        assert self.portfolio.cash == 100000.0
        assert len(self.portfolio.positions) == 0
        assert len(self.portfolio.trades) == 0
        assert self.portfolio.total_equity() == 100000.0

    def test_buy_reduces_cash(self):
        self.portfolio.buy("AAPL", 10, 150.0)
        assert self.portfolio.cash == 100000.0 - 1500.0
        assert len(self.portfolio.positions) == 1
        assert self.portfolio.positions["AAPL"].quantity == 10

    def test_buy_insufficient_cash_raises(self):
        with self.assertRaises(ValueError):
            self.portfolio.buy("AAPL", 100000, 10.0)

    def test_buy_same_symbol_averages_price(self):
        self.portfolio.buy("AAPL", 10, 100.0)
        self.portfolio.buy("AAPL", 10, 200.0)
        pos = self.portfolio.positions["AAPL"]
        assert pos.quantity == 20
        assert pos.avg_price == 150.0
        assert pos.cost_basis == 3000.0

    def test_sell_reduces_position(self):
        self.portfolio.buy("AAPL", 10, 100.0)
        trade = self.portfolio.sell("AAPL", 5, 150.0)
        pos = self.portfolio.positions["AAPL"]
        assert pos.quantity == 5
        assert trade.pnl > 0  # Sold at profit

    def test_sell_entire_position_removes_it(self):
        self.portfolio.buy("AAPL", 10, 100.0)
        self.portfolio.sell("AAPL", 10, 150.0)
        assert "AAPL" not in self.portfolio.positions

    def test_sell_insufficient_quantity_raises(self):
        self.portfolio.buy("AAPL", 5, 100.0)
        with self.assertRaises(ValueError):
            self.portfolio.sell("AAPL", 10, 150.0)

    def test_sell_nonexistent_symbol_raises(self):
        with self.assertRaises(ValueError):
            self.portfolio.sell("NONEXISTENT", 1, 100.0)

    def test_update_market_price_updates_unrealized_pnl(self):
        self.portfolio.buy("AAPL", 10, 100.0)
        self.portfolio.update_market_price("AAPL", 150.0)
        pos = self.portfolio.positions["AAPL"]
        assert pos.market_price == 150.0
        assert pos.unrealized_pnl == 500.0

    def test_total_equity_includes_positions(self):
        self.portfolio.buy("AAPL", 10, 100.0)
        self.portfolio.update_market_price("AAPL", 150.0)
        equity = self.portfolio.total_equity()
        assert equity == 100000.0 - 1000.0 + 1500.0

    def test_metrics_after_trades(self):
        self.portfolio.buy("AAPL", 10, 100.0)
        self.portfolio.sell("AAPL", 10, 150.0)
        metrics = self.portfolio.metrics()
        assert metrics.num_trades == 2
        assert metrics.realized_pnl > 0

    def test_win_rate(self):
        self.portfolio.buy("AAPL", 10, 100.0)
        self.portfolio.sell("AAPL", 10, 150.0)
        self.portfolio.buy("MSFT", 5, 200.0)
        self.portfolio.sell("MSFT", 5, 190.0)
        metrics = self.portfolio.metrics()
        assert metrics.num_trades == 4
        assert metrics.num_wins == 1
        assert metrics.num_losses == 1
        assert metrics.win_rate == 25.0

    def test_metrics_with_no_trades(self):
        metrics = self.portfolio.metrics()
        assert metrics.num_trades == 0
        assert metrics.win_rate == 0.0
        assert metrics.sharpe_ratio == 0.0
        assert metrics.max_drawdown == 0.0


class TestRegistry(unittest.TestCase):
    def test_register_and_list(self):
        p = Portfolio(name="reg_test", initial_cash=50000.0)
        register_portfolio("reg_test", p)
        assert "reg_test" in list_portfolios()
        assert PORTFOLIO_REGISTRY["reg_test"] is p
