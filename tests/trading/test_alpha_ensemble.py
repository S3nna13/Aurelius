from __future__ import annotations

from src.trading.alpha_ensemble import (
    ALPHA_ENSEMBLE,
    AlphaEnsemble,
    AlphaSignal,
    SignalDirection,
    VolatilityGate,
)


class TestAlphaSignal:
    def test_weighted_score_buy(self):
        signal = AlphaSignal(direction=SignalDirection.BUY, confidence=0.8)
        assert signal.weighted_score() == 0.8

    def test_weighted_score_sell(self):
        signal = AlphaSignal(direction=SignalDirection.SELL, confidence=0.6)
        assert signal.weighted_score() == -0.6

    def test_weighted_score_hold(self):
        signal = AlphaSignal(direction=SignalDirection.HOLD, confidence=0.0)
        assert signal.weighted_score() == 0.0


class TestAlphaEnsemble:
    def test_aggregate_empty(self):
        ensemble = AlphaEnsemble()
        result = ensemble.aggregate()
        assert result.direction == SignalDirection.HOLD

    def test_aggregate_buy_consensus(self):
        ensemble = AlphaEnsemble()
        ensemble.add_signal(
            AlphaSignal(direction=SignalDirection.BUY, confidence=0.8, source="tech")
        )
        ensemble.add_signal(AlphaSignal(direction=SignalDirection.BUY, confidence=0.7, source="ml"))
        result = ensemble.aggregate()
        assert result.direction == SignalDirection.STRONG_BUY

    def test_aggregate_mixed(self):
        ensemble = AlphaEnsemble()
        ensemble.add_signal(
            AlphaSignal(direction=SignalDirection.BUY, confidence=0.5, source="tech")
        )
        ensemble.add_signal(
            AlphaSignal(direction=SignalDirection.SELL, confidence=0.4, source="ml")
        )
        result = ensemble.aggregate()
        assert result.direction == SignalDirection.HOLD

    def test_aggregate_strong_sell(self):
        ensemble = AlphaEnsemble()
        ensemble.add_signal(
            AlphaSignal(direction=SignalDirection.SELL, confidence=0.9, source="tech")
        )
        ensemble.add_signal(
            AlphaSignal(direction=SignalDirection.SELL, confidence=0.8, source="ml")
        )
        result = ensemble.aggregate()
        assert result.direction == SignalDirection.STRONG_SELL

    def test_add_signal_increases_list(self):
        ensemble = AlphaEnsemble()
        ensemble.add_signal(AlphaSignal(direction=SignalDirection.BUY, confidence=0.5))
        ensemble.add_signal(AlphaSignal(direction=SignalDirection.HOLD, confidence=0.0))
        assert len(ensemble.signals) == 2

    def test_clear_removes_signals(self):
        ensemble = AlphaEnsemble()
        ensemble.add_signal(AlphaSignal(direction=SignalDirection.BUY, confidence=0.5))
        ensemble.clear()
        assert len(ensemble.signals) == 0

    def test_module_instance(self):
        assert ALPHA_ENSEMBLE is not None


class TestVolatilityGate:
    def test_suppress_on_high_volatility(self):
        gate = VolatilityGate(max_volatility=0.05)
        returns = [0.1, -0.08, 0.12, -0.09, 0.11]
        assert gate.should_suppress(returns) is True

    def test_no_suppress_on_low_volatility(self):
        gate = VolatilityGate(max_volatility=0.05)
        returns = [0.01, -0.005, 0.02, -0.01, 0.015]
        assert gate.should_suppress(returns) is False

    def test_no_suppress_with_few_samples(self):
        gate = VolatilityGate()
        assert gate.should_suppress([0.1]) is False
