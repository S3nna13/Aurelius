"""Message content filter for inter-agent traffic.

Scans payloads for PII, toxic patterns, and policy violations.
Fail closed: ambiguous or oversized inputs default to BLOCK.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class FilterVerdict(Enum):
    ALLOW = "allow"
    BLOCK = "block"
    SANITIZE = "sanitize"


@dataclass
class FilterMatch:
    category: str
    matched_text: str
    position: int
    rule_name: str


@dataclass
class FilterResult:
    verdict: FilterVerdict
    matches: list[FilterMatch] = field(default_factory=list)
    sanitized: str = ""
    confidence: float = 1.0


@dataclass
class FilterConfig:
    max_payload_length: int = 50_000
    block_on_oversized: bool = True
    pii_rules_enabled: bool = True
    toxicity_rules_enabled: bool = True
    custom_blocklist: list[str] = field(default_factory=list)


class MessageContentFilter:
    """Stateful filter with configurable rule sets.

    All rules run in sequence; the most restrictive verdict wins.
    """

    # Simple regex patterns — sufficient for agent-agent traffic audit
    _PII_PATTERNS: dict[str, re.Pattern] = {
        "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+", re.IGNORECASE),
        "phone": re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    }

    _TOXIC_TOKENS: set[str] = {
        "kill", "die", "attack", "bomb", "shoot", "murder",
        "rape", "terrorist", "hostage", "weapon",
    }

    def __init__(self, config: FilterConfig | None = None) -> None:
        self._config = config or FilterConfig()
        self._blocklist: set[str] = set(
            w.lower() for w in self._config.custom_blocklist
        )

    def scan(self, text: str) -> FilterResult:
        """Scan *text* and return a FilterResult.

        Oversized payloads default to BLOCK (fail closed).
        """
        if len(text) > self._config.max_payload_length:
            if self._config.block_on_oversized:
                return FilterResult(
                    verdict=FilterVerdict.BLOCK,
                    matches=[
                        FilterMatch(
                            category="oversized",
                            matched_text=f"length={len(text)}",
                            position=0,
                            rule_name="max_payload_length",
                        )
                    ],
                )

        matches: list[FilterMatch] = []

        # PII scan
        if self._config.pii_rules_enabled:
            for name, pattern in self._PII_PATTERNS.items():
                for m in pattern.finditer(text):
                    matches.append(
                        FilterMatch(
                            category="pii",
                            matched_text=m.group(0),
                            position=m.start(),
                            rule_name=name,
                        )
                    )

        # Toxicity scan (simple lexical)
        if self._config.toxicity_rules_enabled:
            words = re.findall(r"\b\w+\b", text.lower())
            for idx, word in enumerate(words):
                if word in self._TOXIC_TOKENS or word in self._blocklist:
                    # approximate character position
                    pos = text.lower().find(word)
                    matches.append(
                        FilterMatch(
                            category="toxicity",
                            matched_text=word,
                            position=pos if pos >= 0 else 0,
                            rule_name="lexical_blocklist",
                        )
                    )
            # Also check whole-phrase blocklist entries
            text_lower = text.lower()
            for phrase in self._blocklist:
                if " " in phrase and phrase in text_lower:
                    pos = text_lower.find(phrase)
                    matches.append(
                        FilterMatch(
                            category="toxicity",
                            matched_text=phrase,
                            position=pos,
                            rule_name="phrase_blocklist",
                        )
                    )

        if matches:
            # If any PII is found, recommend SANITIZE; BLOCK for toxicity
            has_pii = any(m.category == "pii" for m in matches)
            has_toxic = any(m.category == "toxicity" for m in matches)
            if has_toxic:
                return FilterResult(
                    verdict=FilterVerdict.BLOCK,
                    matches=matches,
                )
            if has_pii:
                sanitized = self._redact_pii(text)
                return FilterResult(
                    verdict=FilterVerdict.SANITIZE,
                    matches=matches,
                    sanitized=sanitized,
                )

        return FilterResult(verdict=FilterVerdict.ALLOW)

    def _redact_pii(self, text: str) -> str:
        """Replace detected PII with [REDACTED]."""
        result = text
        for pattern in self._PII_PATTERNS.values():
            result = pattern.sub("[REDACTED]", result)
        return result

    def check(self, text: str) -> bool:
        """Convenience: return True only if verdict is ALLOW."""
        return self.scan(text).verdict == FilterVerdict.ALLOW


# Module-level registry
MESSAGE_FILTER_REGISTRY: dict[str, MessageContentFilter] = {}
DEFAULT_MESSAGE_FILTER = MessageContentFilter()
MESSAGE_FILTER_REGISTRY["default"] = DEFAULT_MESSAGE_FILTER
