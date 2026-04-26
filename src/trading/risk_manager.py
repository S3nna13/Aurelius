from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from src.trading.portfolio import Portfolio

RISK_REGISTRY: dict[str, RiskManager] = {}


def register_risk_manager(name: str, mgr: RiskManager) -> None:
    RISK_REGISTRY[name] = mgr


class RiskLevel(Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskRule:
    name: str = "default"
    description: str = ""
    max_position_size_pct: float = 10.0
    max_sector_exposure_pct: float = 30.0
    max_leverage: float = 1.0
    max_drawdown_pct: float = 15.0
    min_cash_pct: float = 5.0
    max_concentration_pct: float = 20.0
    var_confidence: float = 0.95
    enabled: bool = True


@dataclass
class RiskAssessment:
    level: RiskLevel = RiskLevel.LOW
    score: float = 0.0
    violations: list[str] = field(default_factory=list)
    var_95: float = 0.0
    var_99: float = 0.0
    concentration_risk: float = 0.0
    drawdown_risk: float = 0.0
    leverage_ratio: float = 1.0
    cash_ratio: float = 1.0
    passing: bool = True


class RiskManager:
    def __init__(self, portfolio: Portfolio, rules: RiskRule | None = None):
        self.portfolio = portfolio
        self.rules = rules or RiskRule()

    def assess(self) -> RiskAssessment:
        metrics = self.portfolio.metrics()
        violations: list[str] = []
        score = 0.0

        drawdown_pct = metrics.max_drawdown
        if drawdown_pct > self.rules.max_drawdown_pct:
            violations.append(
                f"Drawdown {drawdown_pct:.1f}% exceeds limit {self.rules.max_drawdown_pct}%"
            )
            score += 30

        cash_pct = (metrics.cash / metrics.total_value * 100) if metrics.total_value > 0 else 100
        if cash_pct < self.rules.min_cash_pct:
            violations.append(f"Cash {cash_pct:.1f}% below minimum {self.rules.min_cash_pct}%")
            score += 15

        for symbol, pos in self.portfolio.positions.items():
            pos_pct = (
                (pos.market_value / metrics.total_value * 100) if metrics.total_value > 0 else 0
            )
            if pos_pct > self.rules.max_concentration_pct:
                violations.append(
                    f"{symbol} concentration {pos_pct:.1f}% exceeds {self.rules.max_concentration_pct}%"  # noqa: E501
                )
                score += 20

        returns = self._compute_daily_returns()
        var_95 = self._compute_var(returns, 0.95) if returns else 0.0
        var_99 = self._compute_var(returns, 0.99) if returns else 0.0

        leverage = metrics.invested / metrics.total_value if metrics.total_value > 0 else 0
        if leverage > self.rules.max_leverage:
            violations.append(f"Leverage {leverage:.2f}x exceeds {self.rules.max_leverage}x")
            score += 25

        if score >= 50:
            level = RiskLevel.CRITICAL
        elif score >= 30:
            level = RiskLevel.HIGH
        elif score >= 15:
            level = RiskLevel.MODERATE
        else:
            level = RiskLevel.LOW

        return RiskAssessment(
            level=level,
            score=score,
            violations=violations,
            var_95=var_95,
            var_99=var_99,
            concentration_risk=self._compute_concentration(),
            drawdown_risk=drawdown_pct,
            leverage_ratio=leverage,
            cash_ratio=cash_pct / 100,
            passing=level not in (RiskLevel.HIGH, RiskLevel.CRITICAL),
        )

    def can_open_trade(self, symbol: str, quantity: float, price: float) -> bool:
        assessment = self.assess()
        if not assessment.passing:
            return False
        cost = quantity * price
        if cost > self.portfolio.cash:
            return False
        pos_pct = (
            (cost / self.portfolio.total_equity() * 100)
            if self.portfolio.total_equity() > 0
            else 100
        )
        if pos_pct > self.rules.max_position_size_pct:
            return False
        return True

    def position_size_limit(self, price: float) -> float:
        max_cost = self.portfolio.total_equity() * (self.rules.max_position_size_pct / 100)
        return max_cost / price if price > 0 else 0

    def _compute_daily_returns(self) -> list[float]:
        if hasattr(self.portfolio, "_equity_curve"):
            curve = self.portfolio._equity_curve  # noqa: SLF001
            if len(curve) < 2:
                return []
            returns = []
            for i in range(1, len(curve)):
                prev = curve[i - 1]
                curr = curve[i]
                if prev != 0:
                    returns.append((curr - prev) / prev)
            return returns
        return []

    def _compute_var(self, returns: list[float], confidence: float) -> float:
        if not returns:
            return 0.0
        sorted_rets = sorted(returns)
        idx = int((1 - confidence) * len(sorted_rets))
        idx = max(0, min(idx, len(sorted_rets) - 1))
        return abs(sorted_rets[idx]) * 100

    def _compute_concentration(self) -> float:
        if not self.portfolio.positions:
            return 0.0
        values = [p.market_value for p in self.portfolio.positions.values()]
        total = sum(values)
        if total == 0:
            return 0.0
        weights = [v / total for v in values]
        hhi = sum(w**2 for w in weights)
        normalized = (hhi - 1 / len(values)) / (1 - 1 / len(values)) if len(values) > 1 else 1.0
        return normalized * 100
