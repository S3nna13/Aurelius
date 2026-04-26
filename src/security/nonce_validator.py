"""Nonce + timestamp validator for anti-replay protection.

Trail of Bits: validate before trusting, reject stale messages.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class NonceValidator:
    """Validate request nonces to prevent replay attacks."""

    max_age_seconds: float = 60.0
    _seen: set[str] = field(default_factory=set, repr=False)

    def validate(self, nonce: str, timestamp: float) -> bool:
        now = time.time()
        age = abs(now - timestamp)
        if age > self.max_age_seconds:
            return False
        if nonce in self._seen:
            return False
        self._seen.add(nonce)
        if len(self._seen) > 10000:
            self._seen.clear()
        return True

    def is_valid(self, nonce: str, timestamp: float) -> tuple[bool, str]:
        now = time.time()
        age = abs(now - timestamp)
        if age > self.max_age_seconds:
            return False, f"timestamp expired (age={age:.1f}s)"
        if nonce in self._seen:
            return False, "nonce already used"
        self._seen.add(nonce)
        return True, "ok"


NONCE_VALIDATOR = NonceValidator()
