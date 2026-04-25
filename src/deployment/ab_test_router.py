"""A/B test request router with deterministic, hash-based variant assignment."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass
class Variant:
    name: str
    weight: float
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Assignment:
    request_id: str
    variant_name: str
    bucket: int


class ABTestRouter:
    """Routes incoming requests to variants deterministically via SHA-256 hashing.

    Weights are normalised to [0, 100) on construction and after any mutation.
    Bucket is derived from the first 8 hex characters of the SHA-256 digest of
    the request_id, taken modulo 100.

    Security: Finding AUR-SEC-2026-0001; CWE-327 (weak hash).
    """

    def __init__(self, variants: list[Variant]) -> None:
        if not variants:
            raise ValueError("At least one variant must be provided.")
        total = sum(v.weight for v in variants)
        if total <= 0:
            raise ValueError("Sum of variant weights must be greater than 0.")
        self._variants: list[Variant] = [
            Variant(name=v.name, weight=v.weight, metadata=dict(v.metadata))
            for v in variants
        ]
        self._normalize()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize(self) -> None:
        """Recompute normalised weights so they sum to exactly 100."""
        raw_total = sum(v.weight for v in self._variants)
        if raw_total <= 0:
            raise ValueError("Sum of variant weights must be greater than 0.")
        scale = 100.0 / raw_total
        for v in self._variants:
            v.weight = v.weight * scale

    @staticmethod
    def _bucket(request_id: str) -> int:
        digest = hashlib.sha256(request_id.encode()).hexdigest()
        return int(digest[:8], 16) % 100

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assign(self, request_id: str) -> Assignment:
        """Deterministically assign *request_id* to a variant."""
        bucket = self._bucket(request_id)
        cumulative = 0.0
        for v in self._variants:
            cumulative += v.weight
            if bucket < cumulative:
                return Assignment(
                    request_id=request_id,
                    variant_name=v.name,
                    bucket=bucket,
                )
        # Floating-point edge case: assign to last variant
        last = self._variants[-1]
        return Assignment(request_id=request_id, variant_name=last.name, bucket=bucket)

    def assignment_stats(self, assignments: list[Assignment]) -> dict:
        by_variant: dict[str, int] = {v.name: 0 for v in self._variants}
        for a in assignments:
            by_variant[a.variant_name] = by_variant.get(a.variant_name, 0) + 1
        return {"total": len(assignments), "by_variant": by_variant}

    def add_variant(self, variant: Variant) -> None:
        """Append a new variant and re-normalise weights."""
        self._variants.append(
            Variant(name=variant.name, weight=variant.weight, metadata=dict(variant.metadata))
        )
        self._normalize()

    def remove_variant(self, name: str) -> bool:
        """Remove a variant by name.  Returns True if found and removed."""
        for i, v in enumerate(self._variants):
            if v.name == name:
                self._variants.pop(i)
                if self._variants:
                    self._normalize()
                return True
        return False


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

AB_TEST_ROUTER_REGISTRY: dict[str, type[ABTestRouter]] = {
    "default": ABTestRouter,
}
