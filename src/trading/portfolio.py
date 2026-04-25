from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

PORTFOLIO_REGISTRY: dict[str, Portfolio] = {}


def register_portfolio(name: str, portfolio: Portfolio) -> None:
    PORTFOLIO_REGISTRY[name] = portfolio


def list_portfolios() -> list[str]:
    return list(PORTFOLIO_REGISTRY)


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Trade:
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    trade_id: str = ""
    commission: float = 0.0
    pnl: float = 0.0


@dataclass
class Position:
    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    cost_basis: float = 0.0
    market_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    return_pct: float = 0.0

    @property
    def market_value(self) -> float:
        return self.quantity * self.market_price

    @property
    def is_long(self) -> bool:
        return self.quantity >= 0

    @property
    def is_short(self) -> bool:
        return self.quantity < 0


@dataclass
class PortfolioMetrics:
    total_value: float = 0.0
    cash: float = 0.0
    invested: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_return: float = 0.0
    return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    num_trades: int = 0
    num_wins: int = 0
    num_losses: int = 0


class Portfolio:
    def __init__(self, name: str = "default", initial_cash: float = 100000.0):
        self.name = name
        self.cash = initial_cash
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []
        self._equity_curve: list[float] = [initial_cash]
        self._timestamps: list[datetime] = [datetime.now(timezone.utc)]

    def buy(self, symbol: str, quantity: float, price: float) -> Trade:
        cost = quantity * price
        if cost > self.cash:
            raise ValueError(f"Insufficient cash: need {cost:.2f}, have {self.cash:.2f}")
        self.cash -= cost
        pos = self.positions.get(symbol)
        if pos is None:
            new_qty = quantity
            new_avg = price
            new_cost = cost
        else:
            total_qty = pos.quantity + quantity
            total_cost = pos.cost_basis + cost
            new_qty = total_qty
            new_avg = total_cost / total_qty if total_qty != 0 else 0
            new_cost = total_cost
        self.positions[symbol] = Position(
            symbol=symbol,
            quantity=new_qty,
            avg_price=new_avg,
            cost_basis=new_cost,
            market_price=price,
        )
        trade = Trade(
            symbol=symbol,
            side=OrderSide.BUY,
            quantity=quantity,
            price=price,
            trade_id=f"buy-{symbol}-{len(self.trades)}",
        )
        self.trades.append(trade)
        self._update_equity()
        return trade

    def sell(self, symbol: str, quantity: float, price: float) -> Trade:
        pos = self.positions.get(symbol)
        if pos is None or abs(pos.quantity) < quantity:
            raise ValueError(f"Insufficient position: have {pos.quantity if pos else 0}, want {quantity}")
        proceeds = quantity * price
        cost_of_sold = quantity * pos.avg_price
        realized = proceeds - cost_of_sold
        self.cash += proceeds
        remaining = pos.quantity - quantity
        if abs(remaining) < 1e-10:
            del self.positions[symbol]
        else:
            new_cost = pos.cost_basis - cost_of_sold
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=remaining,
                avg_price=pos.avg_price,
                cost_basis=new_cost,
                market_price=price,
            )
        trade = Trade(
            symbol=symbol,
            side=OrderSide.SELL,
            quantity=quantity,
            price=price,
            pnl=realized,
            trade_id=f"sell-{symbol}-{len(self.trades)}",
        )
        self.trades.append(trade)
        self._update_equity()
        return trade

    def update_market_price(self, symbol: str, price: float) -> None:
        pos = self.positions.get(symbol)
        if pos is not None:
            pos.market_price = price
            pos.unrealized_pnl = pos.quantity * (price - pos.avg_price)
            pos.return_pct = ((price - pos.avg_price) / pos.avg_price * 100) if pos.avg_price != 0 else 0.0

    def total_equity(self) -> float:
        pos_value = sum(p.market_value for p in self.positions.values())
        return self.cash + pos_value

    def metrics(self) -> PortfolioMetrics:
        total_value = self.total_equity()
        invested = sum(p.cost_basis for p in self.positions.values())
        unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        realized = sum(t.pnl for t in self.trades)
        total_ret = unrealized + realized
        ret_pct = (total_ret / (total_value - total_ret) * 100) if (total_value - total_ret) != 0 else 0.0
        wins = sum(1 for t in self.trades if t.pnl > 0)
        losses = sum(1 for t in self.trades if t.pnl < 0)
        num_trades = len(self.trades)
        win_rate = (wins / num_trades * 100) if num_trades > 0 else 0.0
        sharpe = self._compute_sharpe()
        max_dd = self._compute_max_drawdown()
        return PortfolioMetrics(
            total_value=total_value,
            cash=self.cash,
            invested=invested,
            unrealized_pnl=unrealized,
            realized_pnl=realized,
            total_return=total_ret,
            return_pct=ret_pct,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            num_trades=num_trades,
            num_wins=wins,
            num_losses=losses,
        )

    def _update_equity(self) -> None:
        self._equity_curve.append(self.total_equity())
        self._timestamps.append(datetime.now(timezone.utc))

    def _compute_sharpe(self, risk_free: float = 0.0) -> float:
        if len(self._equity_curve) < 2:
            return 0.0
        returns = []
        for i in range(1, len(self._equity_curve)):
            prev = self._equity_curve[i - 1]
            curr = self._equity_curve[i]
            if prev != 0:
                returns.append((curr - prev) / prev)
        if len(returns) < 2:
            return 0.0
        avg_ret = statistics.fmean(returns)
        std_ret = statistics.stdev(returns)
        if std_ret == 0:
            return 0.0
        return (avg_ret - risk_free) / std_ret * math.sqrt(252)

    def _compute_max_drawdown(self) -> float:
        if len(self._equity_curve) < 2:
            return 0.0
        peak = self._equity_curve[0]
        max_dd = 0.0
        for v in self._equity_curve:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak != 0 else 0
            if dd > max_dd:
                max_dd = dd
        return max_dd * 100
