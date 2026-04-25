from __future__ import annotations

import unittest
from datetime import datetime, timezone

from src.trading.market_data import (
    MarketDataProvider,
    OHLCVData,
    SymbolInfo,
    BarSize,
    MARKET_DATA_REGISTRY,
    register_provider,
    list_providers,
    get_provider,
)


def _make_bar(close: float, timestamp: datetime | None = None) -> OHLCVData:
    return OHLCVData(
        timestamp=timestamp or datetime.now(timezone.utc),
        open=close - 1,
        high=close + 2,
        low=close - 2,
        close=close,
        volume=1000,
        symbol="TEST",
    )


class TestOHLCVData(unittest.TestCase):
    def test_spread(self):
        bar = OHLCVData(
            timestamp=datetime.now(timezone.utc),
            open=100, high=110, low=90, close=105, volume=1000,
        )
        assert bar.spread == 20

    def test_change(self):
        bar = OHLCVData(
            timestamp=datetime.now(timezone.utc),
            open=100, high=110, low=90, close=105, volume=1000,
        )
        assert bar.change == 5

    def test_change_pct_positive(self):
        bar = OHLCVData(
            timestamp=datetime.now(timezone.utc),
            open=100, high=110, low=90, close=110, volume=1000,
        )
        assert bar.change_pct == 10.0

    def test_change_pct_zero_open(self):
        bar = OHLCVData(
            timestamp=datetime.now(timezone.utc),
            open=0, high=10, low=0, close=5, volume=1000,
        )
        assert bar.change_pct == 0.0


class TestMarketDataProvider(unittest.TestCase):
    def setUp(self):
        self.provider = MarketDataProvider(name="test")
        self.bars = [
            _make_bar(100),
            _make_bar(102),
            _make_bar(101),
            _make_bar(105),
            _make_bar(108),
        ]
        self.provider.cache("TEST", self.bars)

    def test_get_cached_returns_cached_data(self):
        result = self.provider.get_cached("TEST")
        assert len(result) == 5

    def test_latest_returns_last_bar(self):
        result = self.provider.latest("TEST")
        assert result is not None
        assert result.close == 108

    def test_latest_no_data_returns_none(self):
        result = self.provider.latest("NONEXISTENT")
        assert result is None

    def test_sma_computes_correctly(self):
        result = self.provider.sma("TEST", 3)
        assert result is not None
        assert round(result, 2) == 104.67

    def test_sma_not_enough_data_returns_none(self):
        result = self.provider.sma("TEST", 100)
        assert result is None

    def test_ema_computes(self):
        result = self.provider.ema("TEST", 3)
        assert result is not None
        assert result > 0

    def test_rsi_overbought(self):
        bars = [_make_bar(100 + i) for i in range(20)]
        up_provider = MarketDataProvider(name="up")
        up_provider.cache("TEST", bars)
        result = up_provider.rsi("TEST", 14)
        assert result is not None
        assert result > 70

    def test_rsi_oversold(self):
        bars = [_make_bar(100 - i) for i in range(20)]
        down_provider = MarketDataProvider(name="down")
        down_provider.cache("TEST", bars)
        result = down_provider.rsi("TEST", 14)
        assert result is not None
        assert result < 30

    def test_rsi_not_enough_data(self):
        result = self.provider.rsi("TEST", 100)
        assert result is None

    def test_volatility_computes(self):
        result = self.provider.volatility("TEST", 3)
        assert result is not None
        assert result >= 0

    def test_volatility_not_enough_data(self):
        result = self.provider.volatility("TEST", 100)
        assert result is None

    def test_fetch_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.provider.fetch("TEST")


class TestRegistry(unittest.TestCase):
    def test_register_and_list(self):
        class TestProvider(MarketDataProvider):
            pass

        register_provider("test_provider", TestProvider)
        assert "test_provider" in list_providers()
        assert get_provider("test_provider") == TestProvider

    def test_get_provider_nonexistent(self):
        assert get_provider("nonexistent") is None
