"""Rate abuse detector for Aurelius security surface.

Detects BURST, SUSTAINED, DISTRIBUTED, and CREDENTIAL_STUFFING abuse patterns
from request log records.  Pure stdlib — no third-party dependencies.
"""

from __future__ import annotations

import enum
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RequestRecord:
    """Single request log entry."""

    client_id: str
    timestamp: float
    endpoint: str
    bytes_sent: int = 0


class AbusePattern(enum.Enum):
    BURST = "BURST"
    SUSTAINED = "SUSTAINED"
    DISTRIBUTED = "DISTRIBUTED"
    CREDENTIAL_STUFFING = "CREDENTIAL_STUFFING"


@dataclass(frozen=True)
class AbuseAlert:
    """Alert raised when an abuse pattern is detected."""

    client_id: str
    pattern: AbusePattern
    severity: str
    request_count: int
    window_s: float
    detail: str


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class RateAbuseDetector:
    """Detects rate-based abuse patterns in request logs."""

    def __init__(
        self,
        burst_threshold: int = 100,
        burst_window_s: float = 10.0,
        sustained_threshold: int = 1000,
        sustained_window_s: float = 60.0,
    ) -> None:
        self.burst_threshold = burst_threshold
        self.burst_window_s = burst_window_s
        self.sustained_threshold = sustained_threshold
        self.sustained_window_s = sustained_window_s

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(self, records: List[RequestRecord]) -> List[AbuseAlert]:
        """Analyse *records* and return deduplicated abuse alerts."""

        alerts: List[AbuseAlert] = []
        seen: set = set()  # (client_id, pattern) dedup key

        # Group records by client_id
        by_client: dict = defaultdict(list)
        for rec in records:
            by_client[rec.client_id].append(rec.timestamp)

        for client_id, timestamps in by_client.items():
            sorted_ts = sorted(timestamps)

            # --- BURST detection (sliding window) ---
            burst_count = self._max_in_sliding_window(sorted_ts, self.burst_window_s)
            key = (client_id, AbusePattern.BURST)
            if burst_count > self.burst_threshold and key not in seen:
                seen.add(key)
                alerts.append(
                    AbuseAlert(
                        client_id=client_id,
                        pattern=AbusePattern.BURST,
                        severity="high",
                        request_count=burst_count,
                        window_s=self.burst_window_s,
                        detail=(
                            f"{burst_count} requests in {self.burst_window_s}s window "
                            f"(threshold {self.burst_threshold})"
                        ),
                    )
                )

            # --- SUSTAINED detection (total window) ---
            if sorted_ts:
                total_window = sorted_ts[-1] - sorted_ts[0]
                check_window = max(total_window, self.sustained_window_s)
                sustained_in_window = self._count_in_window(sorted_ts, check_window)
                key = (client_id, AbusePattern.SUSTAINED)
                if len(sorted_ts) > self.sustained_threshold and key not in seen:
                    seen.add(key)
                    alerts.append(
                        AbuseAlert(
                            client_id=client_id,
                            pattern=AbusePattern.SUSTAINED,
                            severity="medium",
                            request_count=len(sorted_ts),
                            window_s=self.sustained_window_s,
                            detail=(
                                f"{len(sorted_ts)} requests over {self.sustained_window_s}s "
                                f"(threshold {self.sustained_threshold})"
                            ),
                        )
                    )

        # --- DISTRIBUTED detection ---
        heavy_clients = [
            cid for cid, ts in by_client.items() if len(ts) > 50
        ]
        dist_key = ("*distributed*", AbusePattern.DISTRIBUTED)
        if len(heavy_clients) > 10 and dist_key not in seen:
            seen.add(dist_key)
            alerts.append(
                AbuseAlert(
                    client_id="*distributed*",
                    pattern=AbusePattern.DISTRIBUTED,
                    severity="high",
                    request_count=len(heavy_clients),
                    window_s=self.sustained_window_s,
                    detail=(
                        f"{len(heavy_clients)} unique clients each sent >50 requests "
                        f"(threshold 10 clients)"
                    ),
                )
            )

        return alerts

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self, alerts: List[AbuseAlert]) -> dict:
        """Return aggregated summary of *alerts*."""
        by_pattern: dict = {}
        high_severity = 0
        for alert in alerts:
            name = alert.pattern.value
            by_pattern[name] = by_pattern.get(name, 0) + 1
            if alert.severity == "high":
                high_severity += 1
        return {
            "total_alerts": len(alerts),
            "by_pattern": by_pattern,
            "high_severity": high_severity,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _max_in_sliding_window(sorted_ts: list, window_s: float) -> int:
        """Return the maximum number of events that fall within any *window_s* interval."""
        if not sorted_ts:
            return 0
        max_count = 0
        left = 0
        for right in range(len(sorted_ts)):
            while sorted_ts[right] - sorted_ts[left] > window_s:
                left += 1
            max_count = max(max_count, right - left + 1)
        return max_count

    @staticmethod
    def _count_in_window(sorted_ts: list, window_s: float) -> int:
        """Count events within the last *window_s* seconds of *sorted_ts*."""
        if not sorted_ts:
            return 0
        cutoff = sorted_ts[-1] - window_s
        count = sum(1 for t in sorted_ts if t >= cutoff)
        return count


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

RATE_ABUSE_DETECTOR_REGISTRY: dict = {
    "default": RateAbuseDetector,
}
