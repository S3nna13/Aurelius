"""Model router for Aurelius serving infrastructure.

Routes inference requests across registered ModelEndpoints using one of four
policies: ROUND_ROBIN, LEAST_LOADED, HASH_CONSISTENT, or LATENCY_AWARE.

Thread-safe via threading.Lock.  Pure stdlib + project-local deps only.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from enum import StrEnum


class RoutingPolicy(StrEnum):
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    HASH_CONSISTENT = "hash_consistent"
    LATENCY_AWARE = "latency_aware"


@dataclass
class ModelEndpoint:
    endpoint_id: str
    model_name: str
    host: str
    port: int
    weight: float = 1.0
    latency_ms: float = 0.0
    active_requests: int = 0


class ModelRouter:
    """Routes requests to registered ModelEndpoints based on a RoutingPolicy."""

    def __init__(self) -> None:
        self._endpoints: dict[str, ModelEndpoint] = {}
        self._rr_index: int = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, endpoint: ModelEndpoint) -> None:
        """Add or replace an endpoint."""
        with self._lock:
            self._endpoints[endpoint.endpoint_id] = endpoint

    def deregister(self, endpoint_id: str) -> None:
        """Remove an endpoint by ID (no-op if not found)."""
        with self._lock:
            self._endpoints.pop(endpoint_id, None)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select(
        self,
        request_key: str,
        policy: RoutingPolicy = RoutingPolicy.LEAST_LOADED,
    ) -> ModelEndpoint | None:
        """Select an endpoint according to *policy*.

        Returns ``None`` when no endpoints are registered.
        """
        with self._lock:
            active = list(self._endpoints.values())
            if not active:
                return None

            if policy == RoutingPolicy.ROUND_ROBIN:
                ep = active[self._rr_index % len(active)]
                self._rr_index = (self._rr_index + 1) % len(active)
                return ep

            if policy == RoutingPolicy.LEAST_LOADED:
                return min(active, key=lambda e: e.active_requests)

            if policy == RoutingPolicy.HASH_CONSISTENT:
                idx = int(
                    hashlib.sha256(request_key.encode(), usedforsecurity=False).hexdigest(),
                    16,
                ) % len(active)
                return active[idx]

            if policy == RoutingPolicy.LATENCY_AWARE:
                eligible = [e for e in active if e.latency_ms > 0]
                if not eligible:
                    # Fall back to round-robin when no latency data available
                    ep = active[self._rr_index % len(active)]
                    self._rr_index = (self._rr_index + 1) % len(active)
                    return ep
                # Select endpoint with highest weight / latency score
                return max(eligible, key=lambda e: e.weight / e.latency_ms)

            # Unknown policy — default to least loaded
            return min(active, key=lambda e: e.active_requests)

    # ------------------------------------------------------------------
    # Load tracking
    # ------------------------------------------------------------------

    def increment_load(self, endpoint_id: str) -> None:
        """Increment active request counter for *endpoint_id*."""
        with self._lock:
            ep = self._endpoints.get(endpoint_id)
            if ep is not None:
                ep.active_requests += 1

    def decrement_load(self, endpoint_id: str) -> None:
        """Decrement active request counter for *endpoint_id* (floor 0)."""
        with self._lock:
            ep = self._endpoints.get(endpoint_id)
            if ep is not None:
                ep.active_requests = max(0, ep.active_requests - 1)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def endpoints(self) -> list[ModelEndpoint]:
        """Return a snapshot of registered endpoints."""
        with self._lock:
            return list(self._endpoints.values())


# ---------------------------------------------------------------------------
# Module-level singleton (additive registry entry)
# ---------------------------------------------------------------------------

SERVING_REGISTRY: dict = {
    "model_router": ModelRouter(),
}
