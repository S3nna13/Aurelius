"""Statistical log anomaly detector for application-layer SIEM-like monitoring.

Streams structured log records (dicts) and emits anomalies across several
dimensions:

- Volume spikes: requests/minute > rolling_mean + k*std (Welford streaming)
- Rare tokens: previously unseen ``source_ip`` or ``user_agent`` appearing
  more than ``rare_threshold`` times inside the current window
- High-entropy URLs: Shannon entropy of the path component exceeding a
  configured threshold (suggesting generated/obfuscated URLs)
- Off-hours activity: records whose hour-of-day lies outside the trained
  activity profile (populated via :meth:`train_hours`)
- Failed-auth clusters: 5+ authentication failures from the same source
  within ``auth_window_seconds``

This detector is complementary to
``src.security.network_intrusion_detector`` which operates at the
network-packet level; here we reason purely about application log fields.

Pure stdlib (``math``, ``collections``, ``statistics``, ``time``, ``warnings``).
"""

from __future__ import annotations

import math
import warnings
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Optional, Set, Tuple


@dataclass
class LogAnomaly:
    """A single anomaly detected from a structured log record."""

    timestamp: float
    log_record: dict
    anomaly_type: str
    severity: str
    score: float
    reason: str


def _shannon_entropy(text: str) -> float:
    """Return Shannon entropy (base 2) of ``text``.

    Empty strings return ``0.0``.
    """

    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy -= p * math.log2(p)
    return entropy


def _score_to_severity(score: float) -> str:
    """Bucket a normalised score in [0, +inf) into a severity label."""

    if score >= 8.0:
        return "critical"
    if score >= 5.0:
        return "high"
    if score >= 2.5:
        return "medium"
    return "low"


class _Welford:
    """Welford streaming mean/variance (matches loss_variance_monitor pattern)."""

    __slots__ = ("count", "mean", "_m2")

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self._m2 = 0.0

    def update(self, x: float) -> None:
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self._m2 += delta * delta2

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        return self._m2 / (self.count - 1)

    @property
    def stddev(self) -> float:
        return math.sqrt(self.variance)


class LogAnomalyDetector:
    """Streaming statistical detector for structured application log records.

    Parameters
    ----------
    window_size_seconds:
        Rolling window used for rare-token detection and to age out records
        when tracking per-minute volume. Default ``300`` seconds.
    volume_sigma:
        ``k`` multiplier on the running stddev for volume spike alerting.
    entropy_threshold:
        Shannon-entropy (bits) cutoff above which a URL path is flagged as
        high-entropy.
    auth_fail_threshold:
        Number of authentication failures from a single source inside
        ``auth_window_seconds`` that triggers an ``auth_cluster`` anomaly.
    auth_window_seconds:
        Sliding window (seconds) for the auth-cluster heuristic.
    rare_threshold:
        A brand-new ``source_ip`` or ``user_agent`` that appears more than
        this many times inside ``window_size_seconds`` triggers a rare-token
        anomaly. Default ``3``.
    """

    # ------------------------------------------------------------------
    # Construction & configuration

    def __init__(
        self,
        window_size_seconds: int = 300,
        volume_sigma: float = 3.0,
        entropy_threshold: float = 4.5,
        auth_fail_threshold: int = 5,
        auth_window_seconds: int = 60,
        rare_threshold: int = 3,
    ) -> None:
        if window_size_seconds <= 0:
            raise ValueError("window_size_seconds must be positive")
        if auth_window_seconds <= 0:
            raise ValueError("auth_window_seconds must be positive")
        if auth_fail_threshold <= 0:
            raise ValueError("auth_fail_threshold must be positive")

        self.window_size_seconds = int(window_size_seconds)
        self.volume_sigma = float(volume_sigma)
        self.entropy_threshold = float(entropy_threshold)
        self.auth_fail_threshold = int(auth_fail_threshold)
        self.auth_window_seconds = int(auth_window_seconds)
        self.rare_threshold = int(rare_threshold)

        # Per-minute request volume stats
        self._volume = _Welford()
        # (minute_bucket -> count) — we keep the most recent and flush older
        self._minute_counts: Dict[int, int] = defaultdict(int)
        self._current_minute: Optional[int] = None

        # Rolling window of (timestamp, source_ip, user_agent)
        self._window: Deque[Tuple[float, Optional[str], Optional[str]]] = deque()
        self._window_ip_counts: Counter = Counter()
        self._window_ua_counts: Counter = Counter()

        # Set of tokens observed before the rolling window started tracking
        # them (so "rare new token" is well defined).
        self._known_ips: Set[str] = set()
        self._known_uas: Set[str] = set()

        # Auth failure sliding deques keyed by source_ip
        self._auth_fails: Dict[str, Deque[float]] = defaultdict(deque)
        self._auth_cluster_fired: Dict[str, float] = {}

        # Trained hours-of-activity (set of ints 0..23). Empty => disabled.
        self._trained_hours: Set[int] = set()

        # Anomaly queue — cleared by detect_anomalies()
        self._pending: List[LogAnomaly] = []

        # Counters for .stats()
        self._observed = 0
        self._skipped = 0
        self._anomaly_counts: Counter = Counter()
        self._severity_counts: Counter = Counter()

    # ------------------------------------------------------------------
    # Training

    def train_hours(self, hours: Iterable[int]) -> None:
        """Seed the detector with known-good hours-of-activity.

        Hours should be integers in ``0..23``. If the trained set is empty
        the off-hours check is disabled.
        """

        cleaned: Set[int] = set()
        for h in hours:
            try:
                hi = int(h)
            except (TypeError, ValueError):
                continue
            if 0 <= hi <= 23:
                cleaned.add(hi)
        self._trained_hours = cleaned

    # ------------------------------------------------------------------
    # Streaming observation

    def observe(self, log_record: dict) -> None:
        """Ingest a single structured log record."""

        if not isinstance(log_record, dict):
            warnings.warn(
                "LogAnomalyDetector.observe: record is not a dict, skipping",
                RuntimeWarning,
                stacklevel=2,
            )
            self._skipped += 1
            return

        ts = log_record.get("timestamp")
        if not isinstance(ts, (int, float)) or math.isnan(float(ts)):
            warnings.warn(
                "LogAnomalyDetector.observe: record missing numeric 'timestamp', skipping",
                RuntimeWarning,
                stacklevel=2,
            )
            self._skipped += 1
            return
        ts = float(ts)

        self._observed += 1

        source_ip = log_record.get("source_ip")
        if source_ip is not None and not isinstance(source_ip, str):
            source_ip = str(source_ip)
        user_agent = log_record.get("user_agent")
        if user_agent is not None and not isinstance(user_agent, str):
            user_agent = str(user_agent)
        path = log_record.get("path", "")
        if not isinstance(path, str):
            path = str(path)
        status = log_record.get("status")
        event = log_record.get("event")

        # --- Volume tracking ----------------------------------------------
        minute_bucket = int(ts // 60)
        if self._current_minute is None:
            self._current_minute = minute_bucket
        if minute_bucket != self._current_minute:
            # Flush the previous minute into Welford stats
            prev = self._minute_counts.pop(self._current_minute, 0)
            if prev > 0:
                # Compare prev against running stats BEFORE updating, so the
                # first anomalous minute can flag against history.
                self._maybe_flag_volume(prev, self._current_minute)
                self._volume.update(float(prev))
            self._current_minute = minute_bucket
        self._minute_counts[minute_bucket] += 1

        # --- Rolling window maintenance -----------------------------------
        self._window.append((ts, source_ip, user_agent))
        if source_ip is not None:
            self._window_ip_counts[source_ip] += 1
        if user_agent is not None:
            self._window_ua_counts[user_agent] += 1
        self._evict_window(ts)

        # --- Rare-token detection -----------------------------------------
        if source_ip is not None and source_ip not in self._known_ips:
            count = self._window_ip_counts[source_ip]
            if count > self.rare_threshold:
                self._emit(
                    LogAnomaly(
                        timestamp=ts,
                        log_record=log_record,
                        anomaly_type="rare_token",
                        severity=_score_to_severity(float(count)),
                        score=float(count),
                        reason=(
                            f"new source_ip {source_ip!r} seen {count} times "
                            f"in last {self.window_size_seconds}s"
                        ),
                    )
                )
                # Promote to known once we've alerted to avoid spamming.
                self._known_ips.add(source_ip)
        if user_agent is not None and user_agent not in self._known_uas:
            count = self._window_ua_counts[user_agent]
            if count > self.rare_threshold:
                self._emit(
                    LogAnomaly(
                        timestamp=ts,
                        log_record=log_record,
                        anomaly_type="rare_token",
                        severity=_score_to_severity(float(count)),
                        score=float(count),
                        reason=(
                            f"new user_agent {user_agent!r} seen {count} times "
                            f"in last {self.window_size_seconds}s"
                        ),
                    )
                )
                self._known_uas.add(user_agent)

        # --- High-entropy URL ---------------------------------------------
        if path:
            # Entropy of the path portion excluding the leading slash
            target = path.lstrip("/") or path
            if len(target) >= 12:
                entropy = _shannon_entropy(target)
                if entropy >= self.entropy_threshold:
                    score = entropy
                    self._emit(
                        LogAnomaly(
                            timestamp=ts,
                            log_record=log_record,
                            anomaly_type="high_entropy",
                            severity=_score_to_severity(entropy),
                            score=float(entropy),
                            reason=(
                                f"path entropy {entropy:.2f} bits exceeds "
                                f"threshold {self.entropy_threshold}"
                            ),
                        )
                    )
                    _ = score  # quiet linters

        # --- Off-hours ----------------------------------------------------
        if self._trained_hours:
            hour = int((ts // 3600) % 24)
            if hour not in self._trained_hours:
                self._emit(
                    LogAnomaly(
                        timestamp=ts,
                        log_record=log_record,
                        anomaly_type="off_hours",
                        severity="medium",
                        score=3.0,
                        reason=(
                            f"activity at hour {hour} outside trained hours "
                            f"{sorted(self._trained_hours)}"
                        ),
                    )
                )

        # --- Failed-auth cluster ------------------------------------------
        is_auth_fail = False
        if event == "auth_fail":
            is_auth_fail = True
        elif isinstance(status, int) and status in (401, 403):
            is_auth_fail = True
        if is_auth_fail and source_ip is not None:
            dq = self._auth_fails[source_ip]
            dq.append(ts)
            cutoff = ts - self.auth_window_seconds
            while dq and dq[0] < cutoff:
                dq.popleft()
            if len(dq) >= self.auth_fail_threshold:
                last_fire = self._auth_cluster_fired.get(source_ip, -math.inf)
                if ts - last_fire >= self.auth_window_seconds:
                    n = len(dq)
                    self._emit(
                        LogAnomaly(
                            timestamp=ts,
                            log_record=log_record,
                            anomaly_type="auth_cluster",
                            severity=_score_to_severity(float(n)),
                            score=float(n),
                            reason=(
                                f"{n} auth failures from {source_ip!r} in "
                                f"<= {self.auth_window_seconds}s"
                            ),
                        )
                    )
                    self._auth_cluster_fired[source_ip] = ts

    # ------------------------------------------------------------------
    # Internal helpers

    def _maybe_flag_volume(self, count: int, minute_bucket: int) -> None:
        # Need at least 2 observations before stddev is meaningful.
        if self._volume.count < 2:
            return
        mean = self._volume.mean
        sd = self._volume.stddev
        if sd <= 0.0:
            return
        threshold = mean + self.volume_sigma * sd
        if count > threshold:
            score = (count - mean) / sd if sd > 0 else float(count)
            self._emit(
                LogAnomaly(
                    timestamp=float(minute_bucket * 60),
                    log_record={
                        "minute_bucket": minute_bucket,
                        "count": count,
                        "mean": mean,
                        "stddev": sd,
                    },
                    anomaly_type="volume_spike",
                    severity=_score_to_severity(float(score)),
                    score=float(score),
                    reason=(
                        f"volume {count}/min exceeds mean {mean:.1f} + "
                        f"{self.volume_sigma}*std {sd:.1f}"
                    ),
                )
            )

    def _evict_window(self, now: float) -> None:
        cutoff = now - self.window_size_seconds
        while self._window and self._window[0][0] < cutoff:
            _old_ts, old_ip, old_ua = self._window.popleft()
            if old_ip is not None:
                self._window_ip_counts[old_ip] -= 1
                if self._window_ip_counts[old_ip] <= 0:
                    del self._window_ip_counts[old_ip]
            if old_ua is not None:
                self._window_ua_counts[old_ua] -= 1
                if self._window_ua_counts[old_ua] <= 0:
                    del self._window_ua_counts[old_ua]

    def _emit(self, anomaly: LogAnomaly) -> None:
        self._pending.append(anomaly)
        self._anomaly_counts[anomaly.anomaly_type] += 1
        self._severity_counts[anomaly.severity] += 1

    # ------------------------------------------------------------------
    # Public consumption API

    def detect_anomalies(self) -> List[LogAnomaly]:
        """Return anomalies accumulated since the last call and clear the queue.

        Note: this also finalises the current minute into the volume stats so
        that a burst within the latest partial minute still gets a chance to
        alert at call time.
        """

        # Finalise the current minute bucket without consuming it — we peek
        # at the count to emit a potential volume alert, then leave the
        # counter in place so subsequent observations continue accumulating.
        if self._current_minute is not None:
            count = self._minute_counts.get(self._current_minute, 0)
            if count > 0:
                self._maybe_flag_volume(count, self._current_minute)
        out = self._pending
        self._pending = []
        return out

    def stats(self) -> dict:
        """Return a snapshot of detector internals."""

        return {
            "observed": self._observed,
            "skipped": self._skipped,
            "pending_anomalies": len(self._pending),
            "anomaly_counts": dict(self._anomaly_counts),
            "severity_counts": dict(self._severity_counts),
            "volume": {
                "minutes_tracked": self._volume.count,
                "mean": self._volume.mean,
                "stddev": self._volume.stddev,
            },
            "window": {
                "size_seconds": self.window_size_seconds,
                "current_size": len(self._window),
                "unique_ips": len(self._window_ip_counts),
                "unique_user_agents": len(self._window_ua_counts),
            },
            "trained_hours": sorted(self._trained_hours),
        }

    def reset(self) -> None:
        """Clear all streaming state but preserve configuration & training."""

        self._volume = _Welford()
        self._minute_counts.clear()
        self._current_minute = None
        self._window.clear()
        self._window_ip_counts.clear()
        self._window_ua_counts.clear()
        self._known_ips.clear()
        self._known_uas.clear()
        self._auth_fails.clear()
        self._auth_cluster_fired.clear()
        self._pending.clear()
        self._observed = 0
        self._skipped = 0
        self._anomaly_counts.clear()
        self._severity_counts.clear()


__all__ = ["LogAnomaly", "LogAnomalyDetector"]
