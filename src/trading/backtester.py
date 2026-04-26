from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from src.trading.market_data import OHLCVData
from src.trading.portfolio import OrderSide, Portfolio, PortfolioMetrics
from src.trading.trading_agent import TradingAgent

BACKTESTER_REGISTRY: dict[str, type[Strategy]] = {}


def register_strategy(name: str, cls: type[Strategy]) -> None:
    BACKTESTER_REGISTRY[name] = cls


def list_strategies() -> list[str]:
    return list(BACKTESTER_REGISTRY)


class Strategy(Protocol):
    name: str

    def on_bar(self, agent: TradingAgent, bar: OHLCVData) -> None: ...

    def on_start(self, portfolio: Portfolio) -> None: ...

    def on_end(self, portfolio: Portfolio) -> dict[str, float]: ...


@dataclass
class BacktestConfig:
    initial_cash: float = 100000.0
    commission_pct: float = 0.001
    slippage_pct: float = 0.001
    start_date: str = ""
    end_date: str = ""
    symbols: list[str] = field(default_factory=lambda: ["AAPL"])


@dataclass
class TradeRecord:
    timestamp: datetime
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float
    pnl: float = 0.0


@dataclass
class BacktestResult:
    config: BacktestConfig
    metrics: PortfolioMetrics
    trades: list[TradeRecord]
    equity_curve: list[float]
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    num_trades: int = 0
    calmar_ratio: float = 0.0


class Backtester:
    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()

    def run(
        self,
        strategy: Strategy,
        data: dict[str, list[OHLCVData]],
    ) -> BacktestResult:
        portfolio = Portfolio(name="backtest", initial_cash=self.config.initial_cash)
        agent = TradingAgent(portfolio)
        trades: list[TradeRecord] = []
        equity_curve: list[float] = [self.config.initial_cash]

        strategy.on_start(portfolio)

        for symbol in self.config.symbols:
            bars = data.get(symbol, [])
            for bar in bars:
                strategy.on_bar(agent, bar)
                portfolio.update_market_price(symbol, bar.close)
                equity_curve.append(portfolio.total_equity())

        for symbol in self.config.symbols:
            pos = portfolio.positions.get(symbol)
            if pos is not None and abs(pos.quantity) > 0:
                last_bar = data.get(symbol, [])
                if last_bar:
                    price = last_bar[-1].close
                    if pos.quantity > 0:
                        portfolio.sell(symbol, abs(pos.quantity), price)
                    else:
                        portfolio.buy(symbol, abs(pos.quantity), price)

        strategy.on_end(portfolio)
        metrics = portfolio.metrics()

        for t in portfolio.trades:
            commission = t.quantity * t.price * self.config.commission_pct
            trades.append(
                TradeRecord(
                    timestamp=t.timestamp,
                    symbol=t.symbol,
                    side=t.side,
                    quantity=t.quantity,
                    price=t.price,
                    commission=commission,
                    pnl=t.pnl - commission,
                )
            )

        returns = self._compute_returns(equity_curve)
        sharpe = self._compute_sharpe(returns)
        sortino = self._compute_sortino(returns)
        max_dd = self._compute_max_drawdown(equity_curve)

        sum(1 for t in trades if t.pnl > 0)
        sum(1 for t in trades if t.pnl < 0)
        total_wins = sum(t.pnl for t in trades if t.pnl > 0)
        total_losses = abs(sum(t.pnl for t in trades if t.pnl < 0))
        profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")

        total_return = metrics.total_return
        annualized_return = total_return * (252 / max(len(trades), 1)) if trades else 0
        calmar = annualized_return / max_dd if max_dd > 0 else 0

        return BacktestResult(
            config=self.config,
            metrics=metrics,
            trades=trades,
            equity_curve=equity_curve,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe=sharpe,
            sortino=sortino,
            max_drawdown=max_dd,
            win_rate=metrics.win_rate,
            profit_factor=profit_factor,
            num_trades=len(trades),
            calmar_ratio=calmar,
        )

    def _compute_returns(self, curve: list[float]) -> list[float]:
        if len(curve) < 2:
            return []
        returns = []
        for i in range(1, len(curve)):
            prev = curve[i - 1]
            curr = curve[i]
            if prev != 0:
                returns.append((curr - prev) / prev)
        return returns

    def _compute_sharpe(self, returns: list[float], risk_free: float = 0.0) -> float:
        if len(returns) < 2:
            return 0.0
        avg_ret = statistics.fmean(returns)
        std_ret = statistics.stdev(returns)
        if std_ret == 0:
            return 0.0
        return (avg_ret - risk_free) / std_ret * math.sqrt(252)

    def _compute_sortino(self, returns: list[float], risk_free: float = 0.0) -> float:
        if len(returns) < 2:
            return 0.0
        downside = [r - risk_free for r in returns if r < risk_free]
        if not downside:
            return 0.0
        avg_ret = statistics.fmean(returns)
        downside_std = statistics.stdev(downside) if len(downside) > 1 else abs(downside[0])
        if downside_std == 0:
            return 0.0
        return (avg_ret - risk_free) / downside_std * math.sqrt(252)

    def _compute_max_drawdown(self, curve: list[float]) -> float:
        if len(curve) < 2:
            return 0.0
        peak = curve[0]
        max_dd = 0.0
        for v in curve:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak != 0 else 0
            if dd > max_dd:
                max_dd = dd
        return max_dd * 100
