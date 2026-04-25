from __future__ import annotations

import unittest
from datetime import datetime, timezone

from src.trading.backtester import (
    Backtester,
    BacktestConfig,
    BacktestResult,
    TradeRecord,
    BACKTESTER_REGISTRY,
    register_strategy,
    list_strategies,
)
from src.trading.market_data import OHLCVData
from src.trading.portfolio import OrderSide
from src.trading.trading_agent import TradingSignal


class _BuyAndHoldStrategy:
    name = "buy_and_hold"

    def on_bar(self, agent, bar):
        if not agent.portfolio.positions:
            agent.execute_signal(bar.symbol, TradingSignal.BUY, bar.close)

    def on_start(self, portfolio):
        pass

    def on_end(self, portfolio):
        return {}


class _SimpleMAStrategy:
    name = "simple_ma"

    def __init__(self):
        self.prices: list[float] = []

    def on_bar(self, agent, bar):
        self.prices.append(bar.close)
        if len(self.prices) < 3:
            agent.execute_signal(bar.symbol, TradingSignal.BUY, bar.close)
            return
        sma_short = sum(self.prices[-3:]) / 3
        sma_long = sum(self.prices) / len(self.prices)
        current_price = bar.close
        portfolio = agent.portfolio
        has_position = any(abs(p.quantity) > 0 for p in portfolio.positions.values())
        if sma_short > sma_long and not has_position:
            agent.execute_signal(bar.symbol, TradingSignal.BUY, current_price)
        elif sma_short < sma_long and has_position:
            pos = next(p for p in portfolio.positions.values() if abs(p.quantity) > 0)
            portfolio.sell(pos.symbol, abs(pos.quantity), current_price)

    def on_start(self, portfolio):
        self.prices = []

    def on_end(self, portfolio):
        return {}


def _make_bars(symbol: str, count: int = 10, start_price: float = 100.0) -> list[OHLCVData]:
    bars = []
    price = start_price
    for i in range(count):
        price += (i % 3 - 1) * 2
        bar = OHLCVData(
            timestamp=datetime(2024, 1, 1 + i, tzinfo=timezone.utc),
            open=price - 1,
            high=price + 2,
            low=price - 2,
            close=price,
            volume=1000,
            symbol=symbol,
        )
        bars.append(bar)
    return bars


class TestBacktestConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = BacktestConfig()
        assert cfg.initial_cash == 100000.0
        assert cfg.symbols == ["AAPL"]
        assert cfg.commission_pct == 0.001
        assert cfg.slippage_pct == 0.001


class TestBacktester(unittest.TestCase):
    def setUp(self):
        self.backtester = Backtester()

    def test_run_buy_and_hold(self):
        data = {"AAPL": _make_bars("AAPL", 20)}
        strategy = _BuyAndHoldStrategy()
        result = self.backtester.run(strategy, data)
        assert isinstance(result, BacktestResult)
        assert result.num_trades > 0
        assert result.metrics.total_value > 0

    def test_run_sma_strategy(self):
        data = {"AAPL": _make_bars("AAPL", 30)}
        strategy = _SimpleMAStrategy()
        result = self.backtester.run(strategy, data)
        assert isinstance(result, BacktestResult)
        assert result.num_trades >= 0

    def test_metrics_are_populated(self):
        data = {"AAPL": _make_bars("AAPL", 20)}
        strategy = _BuyAndHoldStrategy()
        result = self.backtester.run(strategy, data)
        assert result.metrics is not None
        assert result.sharpe is not None
        assert result.max_drawdown is not None

    def test_multiple_symbols(self):
        config = BacktestConfig(symbols=["AAPL", "MSFT"])
        bt = Backtester(config)
        data = {
            "AAPL": _make_bars("AAPL", 10),
            "MSFT": _make_bars("MSFT", 10, start_price=200),
        }
        strategy = _BuyAndHoldStrategy()
        result = bt.run(strategy, data)
        assert result.num_trades >= 0

    def test_sharpe_computation(self):
        returns = [0.01, 0.02, -0.01, 0.015, 0.005, -0.005]
        sharpe = self.backtester._compute_sharpe(returns)
        assert isinstance(sharpe, float)

    def test_sortino_computation(self):
        returns = [0.01, 0.02, -0.01, 0.015, 0.005, -0.005]
        sortino = self.backtester._compute_sortino(returns)
        assert isinstance(sortino, float)

    def test_max_drawdown(self):
        curve = [100000, 110000, 105000, 95000, 98000, 90000, 92000]
        dd = self.backtester._compute_max_drawdown(curve)
        assert dd > 0

    def test_trade_records_have_commissions(self):
        data = {"AAPL": _make_bars("AAPL", 10)}
        strategy = _BuyAndHoldStrategy()
        result = self.backtester.run(strategy, data)
        if result.trades:
            assert all(t.commission > 0 for t in result.trades)


class TestStrategyRegistry(unittest.TestCase):
    def test_register_and_list(self):
        register_strategy("test_strat", _BuyAndHoldStrategy)
        assert "test_strat" in list_strategies()
        assert BACKTESTER_REGISTRY["test_strat"] == _BuyAndHoldStrategy
