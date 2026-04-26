"""Multi-category content filtering with confidence thresholds, ensemble scoring, and audit logging."""  # noqa: E501

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Config & Decision dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FilterConfig:
    categories: list[str] = field(
        default_factory=lambda: ["harmful", "explicit", "hate", "spam", "pii"]
    )
    thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "harmful": 0.7,
            "explicit": 0.8,
            "hate": 0.7,
            "spam": 0.5,
            "pii": 0.9,
        }
    )
    ensemble_method: str = "max"  # "max" | "mean" | "weighted"
    n_ensemble: int = 3
    log_decisions: bool = True
    appeal_enabled: bool = False


@dataclass
class FilterDecision:
    text: str
    is_safe: bool
    triggered_categories: list[str]
    scores: dict[str, float]
    confidence: float
    timestamp: float


# ---------------------------------------------------------------------------
# CategoryScorer
# ---------------------------------------------------------------------------


class CategoryScorer(nn.Module):
    """Byte-level embedding + linear classifier for harmful categories."""

    def __init__(self, d_model: int, categories: list[str]) -> None:
        super().__init__()
        self.categories = categories
        self.embed = nn.EmbeddingBag(256, d_model, mode="mean")
        self.classifier = nn.Linear(d_model, len(categories))

    def forward(self, text: str) -> dict[str, float]:  # type: ignore[override]
        byte_ids = torch.tensor(list(text.encode("utf-8")), dtype=torch.long).unsqueeze(
            0
        )  # (1, seq_len)
        emb = self.embed(byte_ids)  # (1, d_model)
        logits = self.classifier(emb)  # (1, n_categories)
        scores = torch.sigmoid(logits).squeeze(0)  # (n_categories,)
        return {cat: scores[i].item() for i, cat in enumerate(self.categories)}


# ---------------------------------------------------------------------------
# EnsembleFilter
# ---------------------------------------------------------------------------


class EnsembleFilter:
    """Multiple CategoryScorers vote on safety via an ensemble method."""

    def __init__(self, config: FilterConfig, d_model: int = 32) -> None:
        self.config = config
        self.scorers = nn.ModuleList(
            [CategoryScorer(d_model, config.categories) for _ in range(config.n_ensemble)]
        )

    def score(self, text: str) -> dict[str, float]:
        """Return aggregated {category: score} across all ensemble members."""
        all_scores: list[dict[str, float]] = [scorer(text) for scorer in self.scorers]

        method = self.config.ensemble_method
        result: dict[str, float] = {}

        for cat in self.config.categories:
            values = [s[cat] for s in all_scores]
            if method == "max":
                result[cat] = max(values)
            elif method == "mean":
                result[cat] = sum(values) / len(values)
            elif method == "weighted":
                # uniform weights — equal to mean unless subclassed
                n = len(values)
                weights = [1.0 / n] * n
                result[cat] = sum(w * v for w, v in zip(weights, values))
            else:
                raise ValueError(f"Unknown ensemble_method: {method!r}")

        return result


# ---------------------------------------------------------------------------
# AuditLog
# ---------------------------------------------------------------------------


class AuditLog:
    """Stores FilterDecisions for auditing and statistics."""

    def __init__(self, max_entries: int = 10_000) -> None:
        self.max_entries = max_entries
        self._log: list[FilterDecision] = []

    def add(self, decision: FilterDecision) -> None:
        if len(self._log) >= self.max_entries:
            self._log.pop(0)
        self._log.append(decision)

    def get_recent(self, n: int = 100) -> list[FilterDecision]:
        return self._log[-n:]

    def statistics(self) -> dict:
        n = len(self._log)
        if n == 0:
            return {"n_decisions": 0, "safe_rate": 0.0, "category_rates": {}}

        safe_count = sum(1 for d in self._log if d.is_safe)
        cat_rates: dict[str, float] = {}

        # Collect all known categories from logged decisions
        all_cats: set[str] = set()
        for d in self._log:
            all_cats.update(d.scores.keys())

        for cat in all_cats:
            triggered = sum(1 for d in self._log if cat in d.triggered_categories)
            cat_rates[cat] = triggered / n

        return {
            "n_decisions": n,
            "safe_rate": safe_count / n,
            "category_rates": cat_rates,
        }

    def clear(self) -> None:
        self._log.clear()


# ---------------------------------------------------------------------------
# ContentFilter
# ---------------------------------------------------------------------------


class ContentFilter:
    """Main content filtering interface."""

    def __init__(self, config: FilterConfig, d_model: int = 32) -> None:
        self.config = config
        self.ensemble = EnsembleFilter(config, d_model)
        self.audit_log = AuditLog()

    def filter(self, text: str) -> FilterDecision:  # noqa: A003
        scores = self.ensemble.score(text)
        thresholds = self.config.thresholds

        triggered = [
            cat
            for cat in self.config.categories
            if scores.get(cat, 0.0) >= thresholds.get(cat, 0.5)
        ]
        is_safe = len(triggered) == 0

        if triggered:
            confidence = 1.0 - max(abs(scores[cat] - thresholds[cat]) for cat in triggered)
        else:
            confidence = 1.0

        decision = FilterDecision(
            text=text,
            is_safe=is_safe,
            triggered_categories=triggered,
            scores=scores,
            confidence=confidence,
            timestamp=time.time(),
        )

        if self.config.log_decisions:
            self.audit_log.add(decision)

        return decision

    def filter_batch(self, texts: list[str]) -> list[FilterDecision]:
        return [self.filter(t) for t in texts]

    def update_threshold(self, category: str, new_threshold: float) -> None:
        self.config.thresholds[category] = new_threshold

    def get_stats(self) -> dict:
        return self.audit_log.statistics()


# ---------------------------------------------------------------------------
# PII detection (regex-based, no model)
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CC_RE = re.compile(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b")


def detect_pii(text: str) -> list[str]:
    """Return list of PII type names found in *text* using regex patterns."""
    found: list[str] = []
    if _EMAIL_RE.search(text):
        found.append("email")
    if _PHONE_RE.search(text):
        found.append("phone")
    if _SSN_RE.search(text):
        found.append("ssn")
    if _CC_RE.search(text):
        found.append("credit_card")
    return found


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_filter_metrics(
    decisions: list[FilterDecision], ground_truth: list[bool]
) -> dict[str, float]:
    """Compute precision, recall, F1, accuracy vs ground truth.

    ground_truth[i] = True  →  text[i] is harmful
    predicted_harmful        = not decision.is_safe
    """
    if len(decisions) != len(ground_truth):
        raise ValueError("decisions and ground_truth must have the same length")

    tp = fp = fn = tn = 0
    for decision, actual_harmful in zip(decisions, ground_truth):
        predicted_harmful = not decision.is_safe
        if predicted_harmful and actual_harmful:
            tp += 1
        elif predicted_harmful and not actual_harmful:
            fp += 1
        elif not predicted_harmful and actual_harmful:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(decisions) if decisions else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}
