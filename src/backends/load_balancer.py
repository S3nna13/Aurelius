"""Load balancer across multiple backend instances.

Provides round-robin, weighted round-robin, least-connections, and random
selection strategies. All logic is stdlib-only; no foreign dependencies.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "BackendInstance",
    "LBAlgorithm",
    "LoadBalancer",
    "LOAD_BALANCER_REGISTRY",
]


@dataclass
class BackendInstance:
    """A single backend server instance tracked by the load balancer."""

    instance_id: str
    host: str
    port: int
    weight: float = 1.0
    active_requests: int = 0
    total_requests: int = 0


class LBAlgorithm(Enum):
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    RANDOM = "random"


class LoadBalancer:
    """Distributes requests across a pool of :class:`BackendInstance` objects.

    Parameters
    ----------
    instances:
        Initial list of backend instances.
    algorithm:
        Load-balancing strategy; defaults to ``ROUND_ROBIN``.
    """

    def __init__(
        self,
        instances: list[BackendInstance],
        algorithm: LBAlgorithm = LBAlgorithm.ROUND_ROBIN,
    ) -> None:
        self._instances: list[BackendInstance] = list(instances)
        self._algorithm = algorithm
        self._rr_index: int = 0

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select(self) -> BackendInstance | None:
        """Pick an instance according to the configured algorithm.

        Returns ``None`` when the pool is empty.
        """
        if not self._instances:
            return None

        if self._algorithm is LBAlgorithm.ROUND_ROBIN:
            return self._select_round_robin()
        if self._algorithm is LBAlgorithm.WEIGHTED_ROUND_ROBIN:
            return self._select_weighted_round_robin()
        if self._algorithm is LBAlgorithm.LEAST_CONNECTIONS:
            return self._select_least_connections()
        if self._algorithm is LBAlgorithm.RANDOM:
            return random.choice(self._instances)

        # Fallback (should not be reachable with a valid enum value)
        return self._select_round_robin()

    def _select_round_robin(self) -> BackendInstance:
        idx = self._rr_index % len(self._instances)
        self._rr_index = (idx + 1) % len(self._instances)
        return self._instances[idx]

    def _select_weighted_round_robin(self) -> BackendInstance:
        # Score = weight * 1 / (active_requests + 1); highest score wins.
        best = max(
            self._instances,
            key=lambda inst: inst.weight * (1.0 / (inst.active_requests + 1)),
        )
        return best

    def _select_least_connections(self) -> BackendInstance:
        # Lowest active_requests; ties broken lexicographically by instance_id.
        return min(
            self._instances,
            key=lambda inst: (inst.active_requests, inst.instance_id),
        )

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def acquire(self, instance: BackendInstance) -> None:
        """Mark a request as in-flight on *instance*."""
        instance.active_requests += 1
        instance.total_requests += 1

    def release(self, instance: BackendInstance) -> None:
        """Mark a request as completed on *instance* (floor at 0)."""
        instance.active_requests = max(0, instance.active_requests - 1)

    # ------------------------------------------------------------------
    # Pool management
    # ------------------------------------------------------------------

    def add_instance(self, instance: BackendInstance) -> None:
        """Add a new instance to the pool."""
        self._instances.append(instance)

    def remove_instance(self, instance_id: str) -> bool:
        """Remove the instance with the given *instance_id*.

        Returns ``True`` if an instance was removed, ``False`` otherwise.
        """
        before = len(self._instances)
        self._instances = [i for i in self._instances if i.instance_id != instance_id]
        # Reset round-robin cursor if the pool shrank to avoid out-of-range.
        if self._instances:
            self._rr_index = self._rr_index % len(self._instances)
        else:
            self._rr_index = 0
        return len(self._instances) < before

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def stats(self) -> list[dict]:
        """Return per-instance statistics."""
        return [
            {
                "id": inst.instance_id,
                "active": inst.active_requests,
                "total": inst.total_requests,
                "weight": inst.weight,
            }
            for inst in self._instances
        ]


LOAD_BALANCER_REGISTRY: dict[str, type[LoadBalancer]] = {"default": LoadBalancer}
