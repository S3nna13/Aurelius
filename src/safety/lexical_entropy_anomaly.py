"""Lexical entropy anomaly scoring for untrusted user text (stdlib only).

High **type-token ratio** / Shannon entropy over a whitespace token stream can
indicate machine-generated noise, stuffing attacks, or low-signal spam; very low
entropy indicates pathological repetition.  This is a coarse heuristic — not a
replacement for logits-based detectors — but it fits the
``SAFETY_FILTER_REGISTRY`` contract (string in, structured score out) without
torch.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass


_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class LexicalEntropyReport:
    """Aggregate anomaly signal for one text turn."""

    score: float
    is_anomaly: bool
    normalized_entropy: float
    token_count: int


class LexicalEntropyAnomalyDetector:
    """Flag unusually flat or repetitive natural-language-ish strings."""

    def __init__(
        self,
        *,
        low_threshold: float = 0.15,
        high_type_token_ratio: float | None = 0.97,
        min_tokens_for_ratio: int = 32,
        anomaly_score: float = 0.75,
    ) -> None:
        if not 0.0 <= low_threshold <= 1.0:
            raise ValueError("low_threshold must be in [0,1]")
        if high_type_token_ratio is not None and not 0.0 < high_type_token_ratio <= 1.0:
            raise ValueError("high_type_token_ratio must be in (0,1] or None")
        if min_tokens_for_ratio < 2:
            raise ValueError("min_tokens_for_ratio must be >= 2")
        if not 0.0 <= anomaly_score <= 1.0:
            raise ValueError("anomaly_score must be in [0,1]")
        self._low = low_threshold
        self._high_ttr = high_type_token_ratio
        self._min_tok = min_tokens_for_ratio
        self._anomaly_score = anomaly_score

    def score(self, text: str | bytes) -> LexicalEntropyReport:
        """Return a report; never raises on malformed bytes (decodes replace)."""
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")
        if not isinstance(text, str):
            raise TypeError("text must be str or bytes")
        norm = unicodedata.normalize("NFKC", text)
        norm = _WS.sub(" ", norm).strip()
        toks = [t for t in norm.split(" ") if t]
        if len(toks) < 2:
            return LexicalEntropyReport(
                score=0.0,
                is_anomaly=False,
                normalized_entropy=0.0,
                token_count=len(toks),
            )
        counts: dict[str, int] = {}
        for t in toks:
            counts[t] = counts.get(t, 0) + 1
        total = len(toks)
        probs = (c / total for c in counts.values())
        h = -sum(p * math.log(p + 1e-30) for p in probs)
        v = len(counts)
        if v <= 1:
            h_norm = 0.0
        else:
            h_norm = float(h / (math.log(v) + 1e-12))

        ttr = v / total
        high_signal = (
            self._high_ttr is not None
            and total >= self._min_tok
            and ttr >= self._high_ttr
        )
        single_type_spam = v == 1 and total >= 12
        low_signal = (h_norm <= self._low and v > 1) or single_type_spam

        if low_signal or high_signal:
            return LexicalEntropyReport(
                score=self._anomaly_score,
                is_anomaly=True,
                normalized_entropy=h_norm,
                token_count=total,
            )
        return LexicalEntropyReport(
            score=0.1,
            is_anomaly=False,
            normalized_entropy=h_norm,
            token_count=total,
        )


__all__ = ["LexicalEntropyAnomalyDetector", "LexicalEntropyReport"]
