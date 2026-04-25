"""Anomaly detector: statistical anomaly detection on time-series metrics."""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from enum import Enum


class AnomalyMethod(str, Enum):
    ZSCORE = "zscore"
    IQR = "iqr"
    ROLLING_STDDEV = "rolling_stddev"
    THRESHOLD = "threshold"


@dataclass(frozen=True)
class Anomaly:
    metric_name: str
    timestamp: float
    value: float
    method: AnomalyMethod
    score: float
    is_anomaly: bool


@dataclass
class AnomalyDetectorConfig:
    zscore_threshold: float = 3.0
    iqr_multiplier: float = 1.5
    rolling_window: int = 20
    static_threshold: dict[str, tuple[float, float]] = field(default_factory=dict)


class AnomalyDetector:
    def __init__(self, config: AnomalyDetectorConfig | None = None) -> None:
        self.config = config or AnomalyDetectorConfig()

    # ------------------------------------------------------------------
    # z-score
    # ------------------------------------------------------------------
    def detect_zscore(self, name: str, values: list[float]) -> list[Anomaly]:
        if not values:
            return []
        n = len(values)
        mean = math.fsum(values) / n
        if n < 2:
            stdev = 0.0
        else:
            stdev = statistics.pstdev(values)
        threshold = self.config.zscore_threshold
        results: list[Anomaly] = []
        for i, v in enumerate(values):
            if stdev == 0.0:
                z = 0.0
            else:
                z = (v - mean) / stdev
            results.append(
                Anomaly(
                    metric_name=name,
                    timestamp=float(i),
                    value=v,
                    method=AnomalyMethod.ZSCORE,
                    score=abs(z),
                    is_anomaly=abs(z) > threshold,
                )
            )
        return results

    # ------------------------------------------------------------------
    # IQR
    # ------------------------------------------------------------------
    def detect_iqr(self, name: str, values: list[float]) -> list[Anomaly]:
        if not values:
            return []
        n = len(values)
        sorted_vals = sorted(values)
        if n < 4:
            q1 = sorted_vals[0]
            q3 = sorted_vals[-1]
        else:
            q1 = _percentile(sorted_vals, 0.25)
            q3 = _percentile(sorted_vals, 0.75)
        iqr = q3 - q1
        low = q1 - self.config.iqr_multiplier * iqr
        high = q3 + self.config.iqr_multiplier * iqr
        results: list[Anomaly] = []
        for i, v in enumerate(values):
            if v < low:
                distance = low - v
                score = distance / iqr if iqr > 0 else 0.0
                is_anom = True
            elif v > high:
                distance = v - high
                score = distance / iqr if iqr > 0 else 0.0
                is_anom = True
            else:
                score = 0.0
                is_anom = False
            results.append(
                Anomaly(
                    metric_name=name,
                    timestamp=float(i),
                    value=v,
                    method=AnomalyMethod.IQR,
                    score=score,
                    is_anomaly=is_anom,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Rolling
    # ------------------------------------------------------------------
    def detect_rolling(
        self,
        name: str,
        values: list[float],
        timestamps: list[float] | None = None,
    ) -> list[Anomaly]:
        if not values:
            return []
        w = self.config.rolling_window
        results: list[Anomaly] = []
        for i, v in enumerate(values):
            ts = timestamps[i] if timestamps is not None and i < len(timestamps) else float(i)
            start = max(0, i - w)
            window = values[start:i]  # look-back only; current point is the candidate
            if len(window) < 2:
                results.append(
                    Anomaly(
                        metric_name=name,
                        timestamp=ts,
                        value=v,
                        method=AnomalyMethod.ROLLING_STDDEV,
                        score=0.0,
                        is_anomaly=False,
                    )
                )
                continue
            rm = math.fsum(window) / len(window)
            rs = statistics.pstdev(window)
            if rs == 0.0:
                score = 0.0
                is_anom = False
            else:
                score = abs(v - rm) / rs
                is_anom = score > 3.0
            results.append(
                Anomaly(
                    metric_name=name,
                    timestamp=ts,
                    value=v,
                    method=AnomalyMethod.ROLLING_STDDEV,
                    score=score,
                    is_anomaly=is_anom,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Threshold
    # ------------------------------------------------------------------
    def detect_threshold(
        self, name: str, values: list[float], low: float, high: float
    ) -> list[Anomaly]:
        results: list[Anomaly] = []
        for i, v in enumerate(values):
            if v < low:
                score = low - v
                is_anom = True
            elif v > high:
                score = v - high
                is_anom = True
            else:
                score = 0.0
                is_anom = False
            results.append(
                Anomaly(
                    metric_name=name,
                    timestamp=float(i),
                    value=v,
                    method=AnomalyMethod.THRESHOLD,
                    score=score,
                    is_anomaly=is_anom,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------
    def detect(
        self,
        name: str,
        values: list[float],
        method: AnomalyMethod = AnomalyMethod.ZSCORE,
        **kwargs,
    ) -> list[Anomaly]:
        if method == AnomalyMethod.ZSCORE:
            return self.detect_zscore(name, values)
        if method == AnomalyMethod.IQR:
            return self.detect_iqr(name, values)
        if method == AnomalyMethod.ROLLING_STDDEV:
            return self.detect_rolling(name, values, kwargs.get("timestamps"))
        if method == AnomalyMethod.THRESHOLD:
            if "low" in kwargs and "high" in kwargs:
                low = kwargs["low"]
                high = kwargs["high"]
            elif name in self.config.static_threshold:
                low, high = self.config.static_threshold[name]
            else:
                low, high = float("-inf"), float("inf")
            return self.detect_threshold(name, values, low, high)
        raise ValueError(f"Unknown method: {method}")

    def summary(self, anomalies: list[Anomaly]) -> dict:
        if not anomalies:
            return {"count": 0, "anomaly_rate": 0.0, "worst_score": 0.0}
        flagged = [a for a in anomalies if a.is_anomaly]
        worst = max((a.score for a in anomalies), default=0.0)
        return {
            "count": len(flagged),
            "anomaly_rate": (len(flagged) / len(anomalies)) * 100.0,
            "worst_score": worst,
        }


def _percentile(sorted_vals: list[float], p: float) -> float:
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_vals[0]
    idx = (n - 1) * p
    lo = int(idx)
    hi = lo + 1
    if hi >= n:
        return sorted_vals[-1]
    frac = idx - lo
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


ANOMALY_DETECTOR_REGISTRY = {"default": AnomalyDetector}
