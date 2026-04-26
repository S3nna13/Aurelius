from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

MARKET_DATA_REGISTRY: dict[str, type[MarketDataProvider]] = {}


def register_provider(name: str, cls: type[MarketDataProvider]) -> None:
    MARKET_DATA_REGISTRY[name] = cls


def list_providers() -> list[str]:
    return list(MARKET_DATA_REGISTRY)


def get_provider(name: str) -> type[MarketDataProvider] | None:
    return MARKET_DATA_REGISTRY.get(name)


class BarSize(Enum):
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"


@dataclass
class SymbolInfo:
    symbol: str
    name: str = ""
    exchange: str = ""
    currency: str = "USD"
    asset_type: str = "stock"
    min_tick: float = 0.01
    lot_size: int = 1


@dataclass
class OHLCVData:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str = ""
    bar_size: BarSize = BarSize.DAY_1

    @property
    def spread(self) -> float:
        return self.high - self.low

    @property
    def change(self) -> float:
        return self.close - self.open

    @property
    def change_pct(self) -> float:
        if self.open == 0:
            return 0.0
        return (self.close - self.open) / self.open * 100.0


class MarketDataProvider:
    def __init__(self, name: str = "base"):
        self.name = name
        self._cache: dict[str, list[OHLCVData]] = {}

    def fetch(
        self,
        symbol: str,
        bar_size: BarSize = BarSize.DAY_1,
        limit: int = 100,
    ) -> list[OHLCVData]:
        raise NotImplementedError

    def get_cached(self, symbol: str) -> list[OHLCVData]:
        return self._cache.get(symbol, [])

    def cache(self, symbol: str, data: list[OHLCVData]) -> None:
        self._cache[symbol] = data

    def latest(self, symbol: str) -> OHLCVData | None:
        cached = self.get_cached(symbol)
        if not cached:
            return None
        return cached[-1]

    def sma(self, symbol: str, period: int = 20) -> float | None:
        cached = self.get_cached(symbol)
        if len(cached) < period:
            return None
        recent = cached[-period:]
        return statistics.fmean(o.close for o in recent)

    def ema(self, symbol: str, period: int = 20) -> float | None:
        cached = self.get_cached(symbol)
        if len(cached) < period:
            return None
        multiplier = 2.0 / (period + 1)
        values = [o.close for o in cached[-period:]]
        ema_val = values[0]
        for v in values[1:]:
            ema_val = (v - ema_val) * multiplier + ema_val
        return ema_val

    def rsi(self, symbol: str, period: int = 14) -> float | None:
        cached = self.get_cached(symbol)
        if len(cached) < period + 1:
            return None
        closes = [o.close for o in cached[-(period + 1) :]]
        gains = []
        losses = []
        for i in range(1, len(closes)):
            diff = closes[i] - closes[i - 1]
            if diff >= 0:
                gains.append(diff)
                losses.append(0.0)
            else:
                gains.append(0.0)
                losses.append(abs(diff))
        avg_gain = statistics.fmean(gains)
        avg_loss = statistics.fmean(losses)
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def volatility(self, symbol: str, period: int = 20) -> float | None:
        cached = self.get_cached(symbol)
        if len(cached) < period:
            return None
        returns = []
        for i in range(1, period):
            prev = cached[-(i + 1)].close
            curr = cached[-i].close
            if prev != 0:
                returns.append((curr - prev) / prev)
        if len(returns) < 2:
            return None
        return statistics.stdev(returns)
