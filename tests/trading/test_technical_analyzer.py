from __future__ import annotations

import pytest

from src.trading.technical_analyzer import TECHNICAL_ANALYZER, TechnicalIndicators


class TestTechnicalAnalyzer:
    def test_compute_indicators_returns_indicator_object(self):
        prices = [float(i) for i in range(100)]
        result = TECHNICAL_ANALYZER.compute_indicators(prices)
        assert isinstance(result, TechnicalIndicators)

    def test_rsi_oversold(self):
        prices = [100.0 - i for i in range(30)]
        result = TECHNICAL_ANALYZER.compute_indicators(prices)
        assert result.rsi < 30

    def test_rsi_overbought(self):
        prices = [100.0 + i for i in range(30)]
        result = TECHNICAL_ANALYZER.compute_indicators(prices)
        assert result.rsi > 70

    def test_macd_calculation(self):
        prices = [float(i) for i in range(30)]
        result = TECHNICAL_ANALYZER.compute_indicators(prices)
        assert result.macd_line != 0.0
        assert result.macd_signal != 0.0

    def test_bollinger_bands(self):
        prices = [100.0] * 25 + [110.0, 90.0, 105.0, 95.0, 102.0]
        result = TECHNICAL_ANALYZER.compute_indicators(prices)
        assert result.upper_band > result.middle_band
        assert result.lower_band < result.middle_band

    def test_sma_50(self):
        prices = [float(i) for i in range(60)]
        result = TECHNICAL_ANALYZER.compute_indicators(prices)
        assert result.sma_50 > 0

    def test_ema_12_vs_ema_26(self):
        prices = [float(i) for i in range(30)]
        result = TECHNICAL_ANALYZER.compute_indicators(prices)
        assert result.ema_12 != 0.0
        assert result.ema_26 != 0.0

    def test_signal_strength_positive(self):
        ind = TechnicalIndicators(rsi=25.0, macd_histogram=1.0, sma_50=110.0, sma_200=100.0)
        assert ind.signal_strength > 0

    def test_signal_strength_negative(self):
        ind = TechnicalIndicators(rsi=75.0, macd_histogram=-1.0, sma_50=100.0, sma_200=110.0)
        assert ind.signal_strength < 0

    def test_signal_strength_neutral(self):
        ind = TechnicalIndicators()
        assert ind.signal_strength == 0.0

    def test_sma_with_insufficient_data(self):
        result = TECHNICAL_ANALYZER.compute_indicators([1.0, 2.0])
        assert result.sma_50 == 0.0

    def test_rsi_extreme_rise(self):
        prices = [100.0 + i * 2 for i in range(30)]
        result = TECHNICAL_ANALYZER.compute_indicators(prices)
        assert result.rsi == 100.0
