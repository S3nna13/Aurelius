"""Protocol version and capability negotiation for Aurelius peers.

Two endpoints exchange their supported :class:`ProtocolVersion` lists and the
set of capability names they offer.  :class:`ProtocolNegotiator` finds the
highest mutually supported version and checks that all *required* capabilities
are present.

Pure stdlib only.  No external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

# ---------------------------------------------------------------------------
# ProtocolVersion
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProtocolVersion:
    major: int
    minor: int
    patch: int = 0

    # Comparison helpers via tuple representation
    def _as_tuple(self) -> tuple:
        return (self.major, self.minor, self.patch)

    def __str__(self) -> str:  # noqa: D105
        return f"{self.major}.{self.minor}.{self.patch}"

    def __le__(self, other: ProtocolVersion) -> bool:  # noqa: D105
        return self._as_tuple() <= other._as_tuple()

    def __ge__(self, other: ProtocolVersion) -> bool:  # noqa: D105
        return self._as_tuple() >= other._as_tuple()

    def __lt__(self, other: ProtocolVersion) -> bool:  # noqa: D105
        return self._as_tuple() < other._as_tuple()

    def __gt__(self, other: ProtocolVersion) -> bool:  # noqa: D105
        return self._as_tuple() > other._as_tuple()


# ---------------------------------------------------------------------------
# ProtocolCapability
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProtocolCapability:
    name: str
    required: bool = True
    version_min: str = "1.0.0"


# ---------------------------------------------------------------------------
# NegotiationOutcome
# ---------------------------------------------------------------------------


class NegotiationOutcome(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    DOWNGRADED = "downgraded"


# ---------------------------------------------------------------------------
# Negotiator
# ---------------------------------------------------------------------------


class ProtocolNegotiator:
    """Negotiates a shared protocol version and checks capability coverage.

    Usage::

        neg = ProtocolNegotiator(
            supported_versions=[ProtocolVersion(1, 0), ProtocolVersion(2, 0)],
            capabilities=[ProtocolCapability("streaming", required=True)],
        )
        result = neg.negotiate(
            client_versions=[ProtocolVersion(1, 0)],
            client_capabilities=["streaming"],
        )
    """

    def __init__(
        self,
        supported_versions: list[ProtocolVersion],
        capabilities: list[ProtocolCapability],
    ) -> None:
        self.supported_versions: list[ProtocolVersion] = supported_versions
        self.capabilities: list[ProtocolCapability] = capabilities

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def negotiate(
        self,
        client_versions: list[ProtocolVersion],
        client_capabilities: list[str],
    ) -> dict:
        """Find the best shared version and verify capabilities.

        Returns a dict with keys:
          ``outcome``         – :attr:`NegotiationOutcome.value` string
          ``version``         – negotiated version string, or ``None``
          ``missing_required``– required caps the client did not offer
          ``optional_missing``– optional caps the client did not offer
        """
        common_version = self.highest_common(self.supported_versions, client_versions)

        client_cap_set = set(client_capabilities)
        missing_required: list[str] = []
        optional_missing: list[str] = []

        for cap in self.capabilities:
            if cap.name not in client_cap_set:
                if cap.required:
                    missing_required.append(cap.name)
                else:
                    optional_missing.append(cap.name)

        if common_version is None or missing_required:
            return {
                "outcome": NegotiationOutcome.REJECTED.value,
                "version": None,
                "missing_required": missing_required,
                "optional_missing": optional_missing,
            }

        # Determine if we had to downgrade (client's best != common)
        client_best = self.highest_common(client_versions, client_versions)
        server_best = self.highest_common(self.supported_versions, self.supported_versions)

        if client_best is not None and server_best is not None:
            ideal = min(client_best, server_best)
            outcome = (
                NegotiationOutcome.DOWNGRADED
                if common_version < ideal
                else NegotiationOutcome.ACCEPTED
            )
        else:
            outcome = NegotiationOutcome.ACCEPTED

        return {
            "outcome": outcome.value,
            "version": str(common_version),
            "missing_required": missing_required,
            "optional_missing": optional_missing,
        }

    def highest_common(
        self,
        a: list[ProtocolVersion],
        b: list[ProtocolVersion],
    ) -> ProtocolVersion | None:
        """Return the highest :class:`ProtocolVersion` present in both *a* and *b*.

        Comparison is performed on ``(major, minor, patch)`` tuples.
        Returns ``None`` if there is no overlap.
        """
        set_b = {v._as_tuple() for v in b}
        common = [v for v in a if v._as_tuple() in set_b]
        if not common:
            return None
        return max(common, key=lambda v: v._as_tuple())


# ---------------------------------------------------------------------------
# Module-level registry
# ---------------------------------------------------------------------------

PROTOCOL_NEGOTIATOR_REGISTRY: dict = {
    "default": ProtocolNegotiator,
}
