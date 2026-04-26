"""Trust boundary validator — reject cross-boundary calls.

Trail of Bits: enforce trust boundaries at the protocol layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrustBoundary:
    name: str
    allowed_caller_prefixes: list[str] | None = None
    allowed_methods: list[str] | None = None


@dataclass
class TrustBoundaryValidator:
    """Validate that cross-boundary calls are authorized."""

    boundaries: dict[str, TrustBoundary] = field(default_factory=dict, repr=False)

    def register(self, boundary: TrustBoundary) -> None:
        self.boundaries[boundary.name] = boundary

    def check(self, boundary_name: str, caller: str, method: str) -> tuple[bool, str]:
        boundary = self.boundaries.get(boundary_name)
        if boundary is None:
            return False, f"unknown boundary: {boundary_name}"

        if boundary.allowed_caller_prefixes:
            if not any(caller.startswith(p) for p in boundary.allowed_caller_prefixes):
                return False, f"caller '{caller}' not allowed in boundary '{boundary_name}'"

        if boundary.allowed_methods:
            if method not in boundary.allowed_methods:
                return False, f"method '{method}' not allowed in boundary '{boundary_name}'"

        return True, "ok"

    def unregister(self, name: str) -> None:
        self.boundaries.pop(name, None)


TRUST_BOUNDARY_VALIDATOR = TrustBoundaryValidator()
