from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class TechnicalIndicators:
    rsi: float = 50.0
    macd_line: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    upper_band: float = 0.0
    middle_band: float = 0.0
    lower_band: float = 0.0
    sma_50: float = 0.0
    sma_200: float = 0.0
    ema_12: float = 0.0
    ema_26: float = 0.0

    @property
    def signal_strength(self) -> float:
        score = 0.0
        if self.rsi < 30:
            score += 1.0
        elif self.rsi > 70:
            score -= 1.0
        if self.macd_histogram > 0:
            score += 0.5
        elif self.macd_histogram < 0:
            score -= 0.5
        if self.sma_50 > self.sma_200:
            score += 0.5
        elif self.sma_50 < self.sma_200:
            score -= 0.5
        return max(-2.0, min(2.0, score))


class TechnicalAnalyzer:
    def compute_indicators(self, prices: Sequence[float]) -> TechnicalIndicators:
        if len(prices) < 26:
            return TechnicalIndicators()
        rsi = self._rsi(prices, 14)
        macd_line, macd_signal = self._macd(prices)
        macd_histogram = macd_line - macd_signal
        upper, middle, lower = self._bollinger(prices, 20)
        sma_50 = self._sma(prices, min(50, len(prices)))
        sma_200 = self._sma(prices, min(200, len(prices)))
        ema_12 = self._ema(prices, 12)
        ema_26 = self._ema(prices, 26)
        return TechnicalIndicators(
            rsi=rsi, macd_line=macd_line, macd_signal=macd_signal,
            macd_histogram=macd_histogram, upper_band=upper,
            middle_band=middle, lower_band=lower, sma_50=sma_50,
            sma_200=sma_200, ema_12=ema_12, ema_26=ema_26,
        )

    def _sma(self, prices: Sequence[float], period: int) -> float:
        if len(prices) < period or period < 1:
            return 0.0
        return sum(prices[-period:]) / period

    def _ema(self, prices: Sequence[float], period: int) -> float:
        if len(prices) < period or period < 1:
            return 0.0
        multiplier = 2.0 / (period + 1)
        ema = sum(prices[:period]) / period
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema
        return ema

    def _rsi(self, prices: Sequence[float], period: int) -> float:
        if len(prices) < period + 1:
            return 50.0
        gains, losses = 0.0, 0.0
        for i in range(len(prices) - period, len(prices)):
            change = prices[i] - prices[i - 1]
            if change >= 0:
                gains += change
            else:
                losses -= change
        avg_gain = gains / period
        avg_loss = losses / period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def _macd(self, prices: Sequence[float]) -> tuple[float, float]:
        ema_12 = self._ema(prices, 12)
        ema_26 = self._ema(prices, 26)
        macd_line = ema_12 - ema_26
        signal = self._ema([macd_line] * 9, 9) if len(prices) >= 26 else 0.0
        return macd_line, signal

    def _bollinger(self, prices: Sequence[float], period: int) -> tuple[float, float, float]:
        if len(prices) < period:
            return 0.0, 0.0, 0.0
        recent = prices[-period:]
        middle = sum(recent) / period
        variance = sum((p - middle) ** 2 for p in recent) / period
        std = math.sqrt(variance)
        return middle + 2 * std, middle, middle - 2 * std


TECHNICAL_ANALYZER = TechnicalAnalyzer()
