from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum


class SentimentLabel(Enum):
    BULLISH = 1
    NEUTRAL = 0
    BEARISH = -1


class SentimentSource(Enum):
    NEWS = "news"
    SOCIAL = "social"
    ANALYST = "analyst"
    INTERNAL = "internal"


@dataclass
class SentimentResult:
    label: SentimentLabel
    score: float
    source: SentimentSource


_POSITIVE_WORDS = {
    "great",
    "good",
    "excellent",
    "positive",
    "up",
    "beat",
    "growth",
    "profit",
    "gain",
    "bullish",
    "strong",
    "outperform",
}
_NEGATIVE_WORDS = {
    "terrible",
    "bad",
    "poor",
    "negative",
    "down",
    "loss",
    "decline",
    "bankrupt",
    "bearish",
    "weak",
    "underperform",
    "drop",
}


def analyze_sentiment(text: str, source: SentimentSource) -> SentimentResult:
    words = set(text.lower().split())
    pos = len(words & _POSITIVE_WORDS)
    neg = len(words & _NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return SentimentResult(SentimentLabel.NEUTRAL, 0.0, source)
    score = (pos - neg) / total
    if score > 0.2:
        label = SentimentLabel.BULLISH
    elif score < -0.2:
        label = SentimentLabel.BEARISH
    else:
        label = SentimentLabel.NEUTRAL
    return SentimentResult(label=label, score=score, source=source)


def combine_sources(results: Sequence[SentimentResult]) -> SentimentResult:
    if not results:
        return SentimentResult(SentimentLabel.NEUTRAL, 0.0, SentimentSource.INTERNAL)
    avg = sum(r.label.value * r.score for r in results) / max(sum(abs(r.score) for r in results), 1)
    if avg > 0.2:
        label = SentimentLabel.BULLISH
    elif avg < -0.2:
        label = SentimentLabel.BEARISH
    else:
        label = SentimentLabel.NEUTRAL
    return SentimentResult(label=label, score=avg, source=SentimentSource.INTERNAL)


class MarketSentiment:
    def __init__(self) -> None:
        self._signals: dict[str, list[SentimentResult]] = {}

    def add_signal(self, ticker: str, result: SentimentResult) -> None:
        if ticker not in self._signals:
            self._signals[ticker] = []
        self._signals[ticker].append(result)

    def get_combined(self, ticker: str) -> SentimentResult | None:
        results = self._signals.get(ticker)
        if not results:
            return None
        return combine_sources(results)

    def clear(self) -> None:
        self._signals.clear()


MARKET_SENTIMENT = MarketSentiment()
