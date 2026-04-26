"""Tests for market_sentiment — news/social media sentiment scoring."""
from __future__ import annotations

from src.trading.market_sentiment import (
    MarketSentiment,
    SentimentResult,
    SentimentSource,
    SentimentLabel,
    analyze_sentiment,
    combine_sources,
)


class TestSentimentLabel:
    def test_label_values(self):
        assert SentimentLabel.BULLISH.value > 0
        assert SentimentLabel.BEARISH.value < 0
        assert SentimentLabel.NEUTRAL.value == 0


class TestSentimentResult:
    def test_result_creation(self):
        r = SentimentResult(label=SentimentLabel.BULLISH, score=0.8, source=SentimentSource.NEWS)
        assert r.score == 0.8
        assert r.label == SentimentLabel.BULLISH


class TestAnalyzeSentiment:
    def test_positive_text(self):
        r = analyze_sentiment("great earnings beat expectations", SentimentSource.NEWS)
        assert r.label in (SentimentLabel.BULLISH, SentimentLabel.NEUTRAL)

    def test_negative_text(self):
        r = analyze_sentiment("terrible loss company bankrupt", SentimentSource.SOCIAL)
        assert r.label in (SentimentLabel.BEARISH, SentimentLabel.NEUTRAL)

    def test_empty_text(self):
        r = analyze_sentiment("", SentimentSource.NEWS)
        assert r.label == SentimentLabel.NEUTRAL

    def test_mixed_text(self):
        r = analyze_sentiment("good earnings but high expenses", SentimentSource.NEWS)
        assert isinstance(r, SentimentResult)


class TestCombineSources:
    def test_combine_multiple(self):
        results = [
            SentimentResult(SentimentLabel.BULLISH, 0.8, SentimentSource.NEWS),
            SentimentResult(SentimentLabel.BULLISH, 0.6, SentimentSource.SOCIAL),
        ]
        combined = combine_sources(results)
        assert combined.label == SentimentLabel.BULLISH

    def test_mixed_sources_averages(self):
        results = [
            SentimentResult(SentimentLabel.BULLISH, 0.9, SentimentSource.NEWS),
            SentimentResult(SentimentLabel.BEARISH, 0.8, SentimentSource.SOCIAL),
        ]
        combined = combine_sources(results)
        assert combined.label == SentimentLabel.NEUTRAL


class TestMarketSentiment:
    def test_add_and_get(self):
        ms = MarketSentiment()
        ms.add_signal("AAPL", SentimentResult(SentimentLabel.BULLISH, 0.7, SentimentSource.NEWS))
        ms.add_signal("AAPL", SentimentResult(SentimentLabel.BEARISH, 0.6, SentimentSource.SOCIAL))
        result = ms.get_combined("AAPL")
        assert result is not None
        assert result.label == SentimentLabel.NEUTRAL

    def test_ticker_not_found(self):
        ms = MarketSentiment()
        assert ms.get_combined("UNKNOWN") is None

    def test_clear(self):
        ms = MarketSentiment()
        ms.add_signal("AAPL", SentimentResult(SentimentLabel.BULLISH, 0.5, SentimentSource.NEWS))
        ms.clear()
        assert ms.get_combined("AAPL") is None
