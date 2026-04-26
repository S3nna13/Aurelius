from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum

from src.trading.portfolio import OrderSide, OrderStatus, Portfolio

AGENT_REGISTRY: dict[str, type[TradingAgent]] = {}


def register_agent(name: str, cls: type[TradingAgent]) -> None:
    AGENT_REGISTRY[name] = cls


def list_agents() -> list[str]:
    return list(AGENT_REGISTRY)


class TradingSignal(Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@dataclass
class Order:
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float = 0.0
    stop_price: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    order_id: str = ""
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    filled_at: datetime | None = None

    @property
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED

    @property
    def remaining(self) -> float:
        return self.quantity - self.filled_quantity


@dataclass
class AgentConfig:
    name: str = "default_agent"
    max_position_size: float = 10000.0
    max_positions: int = 10
    stop_loss_pct: float = 5.0
    take_profit_pct: float = 20.0
    cooldown_seconds: float = 60.0


class TradingAgent:
    def __init__(self, portfolio: Portfolio, config: AgentConfig | None = None):
        self.portfolio = portfolio
        self.config = config or AgentConfig()
        self.orders: list[Order] = []
        self._last_trade_time: dict[str, float] = {}
        self._active = True

    @property
    def is_active(self) -> bool:
        return self._active

    def activate(self) -> None:
        self._active = True

    def deactivate(self) -> None:
        self._active = False

    def generate_signal(
        self,
        symbol: str,
        price: float,
        indicators: dict[str, float] | None = None,
    ) -> TradingSignal:
        if not self._active:
            return TradingSignal.HOLD
        indicators = indicators or {}
        rsi = indicators.get("rsi", 50)
        sma_short = indicators.get("sma_short")
        sma_long = indicators.get("sma_long")
        indicators.get("volatility", 0)

        if self._in_cooldown(symbol):
            return TradingSignal.HOLD

        if rsi is not None:
            if rsi < 30:
                return TradingSignal.STRONG_BUY
            elif rsi < 40:
                return TradingSignal.BUY
            elif rsi > 70:
                return TradingSignal.STRONG_SELL
            elif rsi > 60:
                return TradingSignal.SELL

        if sma_short is not None and sma_long is not None:
            if sma_short > sma_long and price > sma_short:
                return TradingSignal.BUY
            elif sma_short < sma_long and price < sma_short:
                return TradingSignal.SELL

        return TradingSignal.HOLD

    def execute_signal(self, symbol: str, signal: TradingSignal, price: float) -> Order | None:
        if signal == TradingSignal.HOLD:
            return None
        if not self._can_trade(symbol, price):
            return None

        side = (
            OrderSide.BUY
            if signal in (TradingSignal.BUY, TradingSignal.STRONG_BUY)
            else OrderSide.SELL
        )
        qty = self._calculate_position_size(symbol, price, side)
        if qty <= 0:
            return None

        order = Order(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=qty,
            price=price,
        )
        self.orders.append(order)
        self._last_trade_time[symbol] = time.time()

        if side == OrderSide.BUY:
            self.portfolio.buy(symbol, qty, price)
        else:
            self.portfolio.sell(symbol, qty, price)

        order.status = OrderStatus.FILLED
        order.filled_quantity = qty
        order.avg_fill_price = price
        order.filled_at = datetime.now(UTC)
        return order

    def _in_cooldown(self, symbol: str) -> bool:
        last = self._last_trade_time.get(symbol, 0.0)
        return (time.time() - last) < self.config.cooldown_seconds

    def _can_trade(self, symbol: str, price: float) -> bool:
        if not self._active:
            return False
        current_positions = len(
            [p for p in self.portfolio.positions.values() if abs(p.quantity) > 0]
        )
        if (
            current_positions >= self.config.max_positions
            and symbol not in self.portfolio.positions
        ):
            return False
        return True

    def _calculate_position_size(self, symbol: str, price: float, side: OrderSide) -> float:
        max_cost = self.config.max_position_size
        if side == OrderSide.BUY:
            max_cost = min(max_cost, self.portfolio.cash)
        qty = max_cost / price if price > 0 else 0
        return math.floor(qty)

    def apply_stop_loss(self, symbol: str, price: float) -> bool:
        pos = self.portfolio.positions.get(symbol)
        if pos is None or pos.quantity == 0:
            return False
        loss_pct = (price - pos.avg_price) / pos.avg_price * 100 if pos.avg_price != 0 else 0
        if loss_pct <= -self.config.stop_loss_pct:
            if pos.quantity > 0:
                self.portfolio.sell(symbol, abs(pos.quantity), price)
            else:
                self.portfolio.buy(symbol, abs(pos.quantity), price)
            return True
        return False

    def apply_take_profit(self, symbol: str, price: float) -> bool:
        pos = self.portfolio.positions.get(symbol)
        if pos is None or pos.quantity == 0:
            return False
        gain_pct = (price - pos.avg_price) / pos.avg_price * 100 if pos.avg_price != 0 else 0
        if gain_pct >= self.config.take_profit_pct:
            if pos.quantity > 0:
                self.portfolio.sell(symbol, abs(pos.quantity), price)
            else:
                self.portfolio.buy(symbol, abs(pos.quantity), price)
            return True
        return False
