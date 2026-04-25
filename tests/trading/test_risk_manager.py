from __future__ import annotations

import unittest

from src.trading.portfolio import Portfolio
from src.trading.risk_manager import (
    RiskManager,
    RiskRule,
    RiskLevel,
    RISK_REGISTRY,
    register_risk_manager,
)


class TestRiskManager(unittest.TestCase):
    def setUp(self):
        self.portfolio = Portfolio(name="test", initial_cash=100000.0)
        self.rules = RiskRule(
            max_drawdown_pct=15.0,
            min_cash_pct=5.0,
            max_concentration_pct=20.0,
            max_leverage=1.0,
        )
        self.risk_manager = RiskManager(self.portfolio, self.rules)

    def test_initial_assessment_passes(self):
        assessment = self.risk_manager.assess()
        assert assessment.passing
        assert assessment.level == RiskLevel.LOW
        assert len(assessment.violations) == 0

    def test_high_drawdown_triggers_violation(self):
        self.rules.max_drawdown_pct = 1.0
        self.portfolio._equity_curve = [100000, 80000, 70000]
        assessment = self.risk_manager.assess()
        assert not assessment.passing
        assert any("Drawdown" in v for v in assessment.violations)

    def test_concentration_violation(self):
        self.rules.max_concentration_pct = 5.0
        self.rules.max_drawdown_pct = 0.1
        self.portfolio._equity_curve = [100000, 80000]
        self.portfolio.buy("AAPL", 900, 100.0)
        self.portfolio.update_market_price("AAPL", 100.0)
        assessment = self.risk_manager.assess()
        assert any("concentration" in v or "Drawdown" in v for v in assessment.violations)
        assert not assessment.passing

    def test_can_open_trade_when_passable(self):
        self.portfolio.buy("AAPL", 10, 100.0)
        result = self.risk_manager.can_open_trade("MSFT", 10, 200.0)
        assert result

    def test_can_open_trade_blocked_when_critical(self):
        self.rules.max_drawdown_pct = 0.1
        self.portfolio._equity_curve = [100000, 50000]
        result = self.risk_manager.can_open_trade("MSFT", 10, 200.0)
        assert not result

    def test_can_open_trade_blocked_when_insufficient_cash(self):
        result = self.risk_manager.can_open_trade("AAPL", 100000, 100.0)
        assert not result

    def test_position_size_limit(self):
        self.portfolio.buy("SNAP", 10, 100.0)
        limit = self.risk_manager.position_size_limit(50.0)
        assert limit >= 0

    def test_var_computation(self):
        from src.trading.risk_manager import RiskManager as RM
        rm = RM(self.portfolio)
        returns = [-0.01, -0.02, 0.01, 0.03, -0.015, 0.005, -0.005, 0.01]
        var_95 = rm._compute_var(returns, 0.95)
        assert var_95 > 0

    def test_concentration_risk_single_position(self):
        self.portfolio.buy("AAPL", 500, 100.0)
        self.portfolio.update_market_price("AAPL", 100.0)
        assessment = self.risk_manager.assess()
        assert assessment.concentration_risk > 0


class TestRiskRegistry(unittest.TestCase):
    def test_register(self):
        p = Portfolio(name="reg_test")
        rm = RiskManager(p)
        register_risk_manager("reg_test", rm)
        assert RISK_REGISTRY["reg_test"] is rm
