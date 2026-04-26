"""LLM output scanner for the Aurelius research platform.

Detects PII leakage, toxic content patterns, and anomalous output length in
model outputs using pure statistical and rule-based analysis.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# PII regex patterns (module-level constants)
# ---------------------------------------------------------------------------

_RE_EMAIL = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    re.IGNORECASE,
)

_RE_US_PHONE = re.compile(
    r"(?<!\d)(?:\+?1[\s\-.]?)?"
    r"(?:\(?\d{3}\)?[\s\-.]?)"
    r"\d{3}[\s\-.]?\d{4}"
    r"(?!\d)",
)

_RE_SSN = re.compile(
    r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)",
)

_RE_CREDIT_CARD = re.compile(
    r"(?<!\d)\d{4}[\s\-]\d{4}[\s\-]\d{4}[\s\-]\d{4}(?!\d)",
)

_RE_IPV4 = re.compile(
    r"(?<!\d)"
    r"(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)"
    r"(?!\d)",
)

# Map pattern name to compiled regex
_PII_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("email", _RE_EMAIL),
    ("us_phone", _RE_US_PHONE),
    ("ssn", _RE_SSN),
    ("credit_card", _RE_CREDIT_CARD),
    ("ipv4", _RE_IPV4),
]

# ---------------------------------------------------------------------------
# Toxicity token-id ranges (deterministic proxy, no real toxic words)
# ---------------------------------------------------------------------------
# Ranges are intentionally arbitrary synthetic ranges used as a test proxy.
# Each entry: (low_inclusive, high_inclusive, severity_weight)
_TOXIC_RANGES: list[tuple[int, int, float]] = [
    (1, 20, 0.5),  # suspicious range – lower severity
    (100, 110, 1.0),  # toxic range – full severity
]


# ---------------------------------------------------------------------------
# ScanResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class ScanResult:
    """Result of scanning a single model output.

    Attributes:
        has_pii: True when at least one PII pattern was detected.
        has_toxic: True when the toxicity score exceeds the configured threshold.
        anomalous_length: True when the output length exceeds the configured maximum.
        pii_types: List of PII category names that were found.
        toxic_score: Float in [0, 1] representing estimated toxicity.
        length: Number of characters (text scan) or tokens (token scan).
    """

    has_pii: bool
    has_toxic: bool
    anomalous_length: bool
    pii_types: list[str] = field(default_factory=list)
    toxic_score: float = 0.0
    length: int = 0


# ---------------------------------------------------------------------------
# OutputScanner
# ---------------------------------------------------------------------------


class OutputScanner:
    """Scans LLM outputs for PII leakage, toxic content, and anomalous length.

    Args:
        max_expected_length: Maximum expected output length in characters or
            tokens. Outputs exceeding this value set anomalous_length=True.
        toxic_threshold: Toxicity score above which has_toxic is set to True.
            Must be in [0, 1].
    """

    def __init__(
        self,
        max_expected_length: int = 512,
        toxic_threshold: float = 0.3,
    ) -> None:
        self.max_expected_length = max_expected_length
        self.toxic_threshold = toxic_threshold

    # ------------------------------------------------------------------
    # PII detection
    # ------------------------------------------------------------------

    def detect_pii_text(self, text: str) -> tuple[bool, list[str]]:
        """Detect PII patterns in a raw string.

        Args:
            text: The text to scan.

        Returns:
            A tuple (found, pii_types) where found is True when any PII is
            detected and pii_types is the list of PII category names found.
        """
        found_types: list[str] = []
        for name, pattern in _PII_PATTERNS:
            if pattern.search(text):
                found_types.append(name)
        return (len(found_types) > 0, found_types)

    def detect_pii(
        self,
        tokens: list[int],
        token_to_str: Callable[[int], str] | None = None,
    ) -> tuple[bool, list[str]]:
        """Detect PII patterns in a token sequence.

        Converts tokens to a string first, then applies regex scanning.

        Args:
            tokens: List of integer token ids.
            token_to_str: Optional callable that maps a single token id to its
                string representation. When None a simple decimal fallback is
                used: each integer is converted to its decimal string and the
                results are joined with spaces.

        Returns:
            A tuple (found, pii_types) identical to detect_pii_text.
        """
        if token_to_str is None:
            text = " ".join(str(t) for t in tokens)
        else:
            text = "".join(token_to_str(t) for t in tokens)
        return self.detect_pii_text(text)

    # ------------------------------------------------------------------
    # Toxicity scoring
    # ------------------------------------------------------------------

    def toxic_score(self, token_ids: list[int]) -> float:
        """Compute a heuristic toxicity score for a sequence of token ids.

        Tokens that fall within pre-defined synthetic ranges are counted as
        toxic or suspicious. The score is the weighted fraction of such tokens
        clamped to [0, 1].

        Args:
            token_ids: List of integer token ids.

        Returns:
            Float in [0, 1]. Returns 0.0 for empty sequences.
        """
        if not token_ids:
            return 0.0

        total_weight = 0.0
        for tid in token_ids:
            for lo, hi, weight in _TOXIC_RANGES:
                if lo <= tid <= hi:
                    total_weight += weight
                    break  # count each token at most once

        max_possible = len(token_ids) * max(w for _, _, w in _TOXIC_RANGES)
        if max_possible == 0.0:
            return 0.0

        score = total_weight / max_possible
        return float(min(score, 1.0))

    # ------------------------------------------------------------------
    # Unified scan methods
    # ------------------------------------------------------------------

    def scan_text(
        self,
        text: str,
        expected_max_length: int | None = None,
    ) -> ScanResult:
        """Run all checks on a raw text string.

        Args:
            text: The output string to scan.
            expected_max_length: Override for the maximum expected length. Falls
                back to self.max_expected_length when None.

        Returns:
            A ScanResult populated with results from all checks.
        """
        max_len = (
            expected_max_length if expected_max_length is not None else self.max_expected_length
        )
        length = len(text)

        has_pii, pii_types = self.detect_pii_text(text)

        # Represent text as token ids via ordinal values for toxicity scoring
        token_ids = [ord(c) for c in text]
        t_score = self.toxic_score(token_ids)

        return ScanResult(
            has_pii=has_pii,
            has_toxic=t_score > self.toxic_threshold,
            anomalous_length=length > max_len,
            pii_types=pii_types,
            toxic_score=t_score,
            length=length,
        )

    def scan_tokens(
        self,
        token_ids: list[int],
        token_to_str: Callable[[int], str] | None = None,
    ) -> ScanResult:
        """Run all checks on a token id sequence.

        Args:
            token_ids: List of integer token ids representing the model output.
            token_to_str: Optional callable to convert token ids to strings for
                PII detection. See detect_pii for details.

        Returns:
            A ScanResult populated with results from all checks.
        """
        length = len(token_ids)
        has_pii, pii_types = self.detect_pii(token_ids, token_to_str)
        t_score = self.toxic_score(token_ids)

        return ScanResult(
            has_pii=has_pii,
            has_toxic=t_score > self.toxic_threshold,
            anomalous_length=length > self.max_expected_length,
            pii_types=pii_types,
            toxic_score=t_score,
            length=length,
        )
