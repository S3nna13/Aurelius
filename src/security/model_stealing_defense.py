"""Model-stealing defense module for the Aurelius LLM research platform.

Complements :mod:`src.security.model_extraction` (which implements offensive
knockoff-nets style attacks) by providing defensive countermeasures against
model extraction. References:

* Orekondy, T., Schiele, B., & Fritz, M. (2019). "Knockoff Nets: Stealing
  Functionality of Black-Box Models." CVPR 2019.
* Tramèr, F., Zhang, F., Juels, A., Reiter, M. K., & Ristenpart, T. (2016).
  "Stealing Machine Learning Models via Prediction APIs." USENIX Security.

The defenses implemented here are:

* A query-pattern detector that maintains per-client audit trails of prompt
  hashes, timestamps, entropies and token counts.
* An output-perturbation mechanism that injects small Gaussian noise into
  logits to degrade the quality of distillation-based clones.
* A rate-limit enforcer keyed on per-client query cadence.
* A threat analyzer that emits signals for high-entropy uniform queries,
  query-rate spikes and output-probing patterns characteristic of extraction.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import torch


@dataclass
class QueryAuditEntry:
    """Per-query audit record used by the model-stealing defense.

    Attributes:
        client_id: Opaque identifier of the querying client.
        prompt_hash: SHA-256 hex digest of the prompt text.
        timestamp: Unix epoch seconds at which the query was recorded.
        entropy: Shannon entropy (in bits) of the prompt's character
            distribution.
        n_tokens: Whitespace-token count of the prompt.
    """

    client_id: str
    prompt_hash: str
    timestamp: float
    entropy: float
    n_tokens: int


@dataclass
class StealingThreatReport:
    """Aggregated threat assessment for a single client.

    Attributes:
        client_id: Client the report applies to.
        threat_level: One of ``"low"``, ``"medium"``, ``"high"``, ``"critical"``.
        signals: Human-readable list of triggered detection signals.
        total_queries: Number of audit entries held for the client.
        suggested_action: Recommended operator action, always populated.
    """

    client_id: str
    threat_level: str
    signals: List[str] = field(default_factory=list)
    total_queries: int = 0
    suggested_action: str = ""


def _shannon_entropy_bits(text: str) -> float:
    """Compute Shannon entropy (bits) of the character distribution of ``text``."""
    if not text:
        return 0.0
    counts: Dict[str, int] = {}
    for ch in text:
        counts[ch] = counts.get(ch, 0) + 1
    n = float(len(text))
    entropy = 0.0
    for c in counts.values():
        p = c / n
        entropy -= p * math.log2(p)
    return entropy


class ModelStealingDefense:
    """Defensive module against black-box model extraction attacks.

    Maintains per-client audit trails and produces threat reports. Offers
    output perturbation and rate-limit enforcement hooks.

    Attributes:
        entropy_threshold: Minimum average character entropy above which a
            stream of queries is considered suspiciously diverse / uniform
            (a hallmark of distillation query sets).
        query_rate_threshold_per_minute: Per-client query rate (queries in
            the trailing 60 s) above which rate limiting is triggered.
        diversity_window: Number of most-recent audit entries considered when
            evaluating diversity / uniformity signals.
    """

    _PROBE_TOKENS = (
        "logit",
        "logits",
        "probability",
        "probabilities",
        "softmax",
        "distribution",
        "top-k",
        "top_k",
        "topk",
        "confidence",
    )

    def __init__(
        self,
        entropy_threshold: float = 5.0,
        query_rate_threshold_per_minute: int = 120,
        diversity_window: int = 64,
    ) -> None:
        if entropy_threshold < 0:
            raise ValueError("entropy_threshold must be non-negative")
        if query_rate_threshold_per_minute <= 0:
            raise ValueError("query_rate_threshold_per_minute must be positive")
        if diversity_window <= 0:
            raise ValueError("diversity_window must be positive")

        self.entropy_threshold = float(entropy_threshold)
        self.query_rate_threshold_per_minute = int(query_rate_threshold_per_minute)
        self.diversity_window = int(diversity_window)

        self._audit: Dict[str, Deque[QueryAuditEntry]] = defaultdict(
            lambda: deque(maxlen=max(1024, 4 * self.diversity_window))
        )
        self._probe_hits: Dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _validate_client_id(client_id: str) -> None:
        if not isinstance(client_id, str) or not client_id:
            raise ValueError("client_id must be a non-empty string")

    # -------------------------------------------------------------- recording
    def record_query(self, client_id: str, prompt: str) -> QueryAuditEntry:
        """Record a query in the per-client audit trail.

        Args:
            client_id: Non-empty client identifier.
            prompt: Prompt text submitted by the client.

        Returns:
            The ``QueryAuditEntry`` that was appended to the audit trail.
        """
        self._validate_client_id(client_id)
        prompt = prompt if isinstance(prompt, str) else str(prompt)

        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        entry = QueryAuditEntry(
            client_id=client_id,
            prompt_hash=digest,
            timestamp=time.time(),
            entropy=_shannon_entropy_bits(prompt),
            n_tokens=len(prompt.split()),
        )
        self._audit[client_id].append(entry)

        lowered = prompt.lower()
        if any(tok in lowered for tok in self._PROBE_TOKENS):
            self._probe_hits[client_id] += 1

        return entry

    # ----------------------------------------------------------------- query
    def _recent_rate_per_minute(self, client_id: str) -> float:
        entries = self._audit.get(client_id)
        if not entries:
            return 0.0
        now = time.time()
        recent = [e for e in entries if now - e.timestamp <= 60.0]
        return float(len(recent))

    def should_rate_limit(self, client_id: str) -> bool:
        """Return ``True`` when the client has exceeded the configured rate."""
        self._validate_client_id(client_id)
        return self._recent_rate_per_minute(client_id) > self.query_rate_threshold_per_minute

    # --------------------------------------------------------------- analysis
    def analyze(self, client_id: str) -> StealingThreatReport:
        """Produce a :class:`StealingThreatReport` for ``client_id``."""
        self._validate_client_id(client_id)
        entries = list(self._audit.get(client_id, []))
        total = len(entries)

        signals: List[str] = []

        # Rate spike signal.
        rate = self._recent_rate_per_minute(client_id)
        if rate > self.query_rate_threshold_per_minute:
            signals.append(
                f"rate_limit:{int(rate)}qpm>{self.query_rate_threshold_per_minute}"
            )

        # High-entropy / uniformly diverse query stream.
        window = entries[-self.diversity_window :]
        if len(window) >= max(4, self.diversity_window // 4):
            avg_entropy = sum(e.entropy for e in window) / len(window)
            unique_hashes = {e.prompt_hash for e in window}
            diversity = len(unique_hashes) / len(window)
            if avg_entropy >= self.entropy_threshold and diversity >= 0.9:
                signals.append(
                    f"extraction:high_entropy_uniform(avg={avg_entropy:.2f},div={diversity:.2f})"
                )

        # Output-probing pattern signal.
        probe_hits = self._probe_hits.get(client_id, 0)
        if probe_hits >= 3 or (total > 0 and probe_hits / max(total, 1) >= 0.25):
            signals.append(f"output_probing:{probe_hits}")

        # Threat-level aggregation.
        n = len(signals)
        if n == 0:
            level = "low"
            action = "monitor"
        elif n == 1:
            level = "medium"
            action = "increase_logging"
        elif n == 2:
            level = "high"
            action = "throttle_and_add_output_noise"
        else:
            level = "critical"
            action = "block_client_and_alert_operator"

        return StealingThreatReport(
            client_id=client_id,
            threat_level=level,
            signals=signals,
            total_queries=total,
            suggested_action=action,
        )

    # ----------------------------------------------------------------- noise
    def add_output_noise(
        self,
        logits: torch.Tensor,
        noise_std: float = 0.01,
    ) -> torch.Tensor:
        """Perturb ``logits`` with zero-mean Gaussian noise.

        Args:
            logits: Output logits tensor. Not modified in place.
            noise_std: Standard deviation of the perturbation. ``0`` returns
                a copy of the input unchanged.

        Returns:
            A new tensor of the same shape and dtype as ``logits``.
        """
        if not isinstance(logits, torch.Tensor):
            raise TypeError("logits must be a torch.Tensor")
        if noise_std < 0:
            raise ValueError("noise_std must be non-negative")
        if noise_std == 0:
            return logits.clone()
        noise = torch.randn_like(logits) * float(noise_std)
        return logits + noise

    # ----------------------------------------------------------------- reset
    def reset(self, client_id: Optional[str] = None) -> None:
        """Reset state for a single client or all clients when ``None``."""
        if client_id is None:
            self._audit.clear()
            self._probe_hits.clear()
            return
        self._validate_client_id(client_id)
        self._audit.pop(client_id, None)
        self._probe_hits.pop(client_id, None)
