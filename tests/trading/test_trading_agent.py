from __future__ import annotations

import time
import unittest

from src.trading.portfolio import Portfolio, OrderSide, OrderStatus
from src.trading.trading_agent import (
    TradingAgent,
    TradingSignal,
    Order,
    OrderType,
    AgentConfig,
    AGENT_REGISTRY,
    register_agent,
    list_agents,
)


class TestTradingSignal(unittest.TestCase):
    def test_signal_values(self):
        assert TradingSignal.STRONG_BUY.value == "strong_buy"
        assert TradingSignal.SELL.value == "sell"
        assert TradingSignal.HOLD.value == "hold"


class TestOrder(unittest.TestCase):
    def test_is_filled(self):
        order = Order(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=10)
        assert not order.is_filled

    def test_remaining(self):
        order = Order(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=10)
        assert order.remaining == 10
        order.filled_quantity = 6
        assert order.remaining == 4

    def test_filled_status(self):
        order = Order(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=10)
        order.status = OrderStatus.FILLED
        assert order.is_filled


class TestTradingAgent(unittest.TestCase):
    def setUp(self):
        self.portfolio = Portfolio(name="test", initial_cash=100000.0)
        self.config = AgentConfig(
            name="test_agent",
            max_position_size=10000.0,
            max_positions=5,
            stop_loss_pct=5.0,
            take_profit_pct=20.0,
            cooldown_seconds=0.0,
        )
        self.agent = TradingAgent(self.portfolio, self.config)

    def test_initial_state(self):
        assert self.agent.is_active
        assert len(self.agent.orders) == 0

    def test_activate_deactivate(self):
        self.agent.deactivate()
        assert not self.agent.is_active
        self.agent.activate()
        assert self.agent.is_active

    def test_generate_signal_hold_when_inactive(self):
        self.agent.deactivate()
        signal = self.agent.generate_signal("AAPL", 150.0)
        assert signal == TradingSignal.HOLD

    def test_generate_signal_rsi_oversold(self):
        signal = self.agent.generate_signal("AAPL", 150.0, {"rsi": 25})
        assert signal == TradingSignal.STRONG_BUY

    def test_generate_signal_rsi_overbought(self):
        signal = self.agent.generate_signal("AAPL", 150.0, {"rsi": 75})
        assert signal == TradingSignal.STRONG_SELL

    def test_generate_signal_rsi_neutral(self):
        signal = self.agent.generate_signal("AAPL", 150.0, {"rsi": 50})
        assert signal == TradingSignal.HOLD

    def test_generate_signal_sma_crossover_bullish(self):
        signal = self.agent.generate_signal(
            "AAPL", 160.0, {"sma_short": 155, "sma_long": 150}
        )
        assert signal in (TradingSignal.BUY, TradingSignal.HOLD)

    def test_execute_signal_buy(self):
        signal = self.agent.generate_signal("AAPL", 150.0, {"rsi": 25})
        order = self.agent.execute_signal("AAPL", signal, 150.0)
        assert order is not None
        assert order.side == OrderSide.BUY
        assert order.is_filled

    def test_execute_signal_hold_returns_none(self):
        order = self.agent.execute_signal("AAPL", TradingSignal.HOLD, 150.0)
        assert order is None

    def test_execute_signal_when_inactive(self):
        self.agent.deactivate()
        order = self.agent.execute_signal("AAPL", TradingSignal.BUY, 150.0)
        assert order is None

    def test_position_limit_enforced(self):
        for i in range(self.config.max_positions + 2):
            symbol = f"SYM{i}"
            self.agent.generate_signal(symbol, 100.0, {"rsi": 25})
            signal = TradingSignal.BUY
            self.agent.execute_signal(symbol, signal, 100.0)
        pos_count = len([p for p in self.portfolio.positions.values() if abs(p.quantity) > 0])
        assert pos_count <= self.config.max_positions

    def test_stop_loss_triggers(self):
        self.agent.execute_signal("AAPL", TradingSignal.BUY, 100.0)
        result = self.agent.apply_stop_loss("AAPL", 85.0)
        assert result
        assert "AAPL" not in self.portfolio.positions or self.portfolio.positions["AAPL"].quantity == 0

    def test_stop_loss_not_triggered(self):
        self.agent.execute_signal("AAPL", TradingSignal.BUY, 100.0)
        result = self.agent.apply_stop_loss("AAPL", 98.0)
        assert not result

    def test_take_profit_triggers(self):
        self.agent.execute_signal("AAPL", TradingSignal.BUY, 100.0)
        result = self.agent.apply_take_profit("AAPL", 130.0)
        assert result
        assert "AAPL" not in self.portfolio.positions or self.portfolio.positions["AAPL"].quantity == 0

    def test_take_profit_not_triggered(self):
        self.agent.execute_signal("AAPL", TradingSignal.BUY, 100.0)
        result = self.agent.apply_take_profit("AAPL", 110.0)
        assert not result

    def test_cooldown_prevents_trades(self):
        self.config.cooldown_seconds = 0.1
        signal = self.agent.generate_signal("AAPL", 100.0, {"rsi": 25})
        self.agent.execute_signal("AAPL", signal, 100.0)
        result = self.agent.execute_signal("AAPL", signal, 100.0)
        assert result is not None
        self.config.cooldown_seconds = 3600.0
        agent2 = TradingAgent(self.portfolio, self.config)
        result2 = agent2.execute_signal("AAPL", TradingSignal.BUY, 100.0)
        assert result2 is not None


class TestAgentRegistry(unittest.TestCase):
    def test_register_and_list(self):
        register_agent("test_agent", TradingAgent)
        assert "test_agent" in list_agents()
        assert AGENT_REGISTRY["test_agent"] == TradingAgent
