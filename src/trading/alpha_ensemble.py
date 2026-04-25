from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class SignalDirection(Enum):
    STRONG_BUY = 2
    BUY = 1
    HOLD = 0
    SELL = -1
    STRONG_SELL = -2


@dataclass
class AlphaSignal:
    direction: SignalDirection = SignalDirection.HOLD
    confidence: float = 0.0
    source: str = ""
    metadata: dict = field(default_factory=dict)

    def weighted_score(self) -> float:
        return self.direction.value * self.confidence


SignalProvider = Callable[[], AlphaSignal]


@dataclass
class AlphaEnsemble:
    signals: list[AlphaSignal] = field(default_factory=list)

    def add_signal(self, signal: AlphaSignal) -> None:
        self.signals.append(signal)

    def aggregate(self) -> AlphaSignal:
        if not self.signals:
            return AlphaSignal(direction=SignalDirection.HOLD, confidence=0.0)
        total = sum(s.direction.value * s.confidence for s in self.signals)
        total_weight = sum(abs(s.confidence) for s in self.signals)
        if total_weight == 0:
            return AlphaSignal(direction=SignalDirection.HOLD, confidence=0.0)
        avg = total / total_weight
        confidence = min(1.0, abs(avg))
        if avg > 0.5:
            direction = SignalDirection.STRONG_BUY
        elif avg > 0.15:
            direction = SignalDirection.BUY
        elif avg < -0.5:
            direction = SignalDirection.STRONG_SELL
        elif avg < -0.15:
            direction = SignalDirection.SELL
        else:
            direction = SignalDirection.HOLD
        return AlphaSignal(direction=direction, confidence=confidence)

    def clear(self) -> None:
        self.signals.clear()


@dataclass
class VolatilityGate:
    max_volatility: float = 0.05

    def should_suppress(self, recent_returns: list[float]) -> bool:
        if len(recent_returns) < 5:
            return False
        mean = sum(recent_returns) / len(recent_returns)
        variance = sum((r - mean) ** 2 for r in recent_returns) / len(recent_returns)
        std = math.sqrt(variance)
        return std > self.max_volatility


ALPHA_ENSEMBLE = AlphaEnsemble()
