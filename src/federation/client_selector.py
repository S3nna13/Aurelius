"""Client selector: strategy-based selection of federated learning clients."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum


class SelectionStrategy(str, Enum):
    RANDOM = "RANDOM"
    POWER_OF_CHOICE = "POWER_OF_CHOICE"
    RESOURCE_AWARE = "RESOURCE_AWARE"
    ROUND_ROBIN = "ROUND_ROBIN"


@dataclass
class ClientProfile:
    client_id: str
    compute_score: float = 1.0
    data_size: int = 0
    last_selected_round: int = -1


class ClientSelector:
    """Selects clients for federated rounds based on a configured strategy."""

    def __init__(self, strategy: SelectionStrategy = SelectionStrategy.RANDOM) -> None:
        self.strategy = strategy
        self._profiles: dict[str, ClientProfile] = {}

    def register(self, profile: ClientProfile) -> None:
        """Register a client profile."""
        self._profiles[profile.client_id] = profile

    def select(
        self,
        num_clients: int,
        round_idx: int = 0,
        seed: int | None = None,
    ) -> list[ClientProfile]:
        """Select up to num_clients profiles using the configured strategy."""
        profiles = list(self._profiles.values())
        k = min(num_clients, len(profiles))
        if k == 0:
            return []

        rng = random.Random(seed)

        if self.strategy == SelectionStrategy.RANDOM:
            selected = rng.sample(profiles, k)

        elif self.strategy == SelectionStrategy.POWER_OF_CHOICE:
            candidate_k = min(2 * k, len(profiles))
            candidates = rng.sample(profiles, candidate_k)
            candidates.sort(key=lambda p: p.compute_score, reverse=True)
            selected = candidates[:k]

        elif self.strategy == SelectionStrategy.RESOURCE_AWARE:
            sorted_profiles = sorted(
                profiles, key=lambda p: p.compute_score, reverse=True
            )
            selected = sorted_profiles[:k]

        elif self.strategy == SelectionStrategy.ROUND_ROBIN:
            n = len(profiles)
            # Stable ordering for reproducibility: sort by client_id then cycle
            ordered = sorted(profiles, key=lambda p: p.client_id)
            start = (round_idx * k) % n
            indices = [(start + i) % n for i in range(k)]
            selected = [ordered[i] for i in indices]

        else:
            selected = profiles[:k]

        for profile in selected:
            profile.last_selected_round = round_idx

        return selected

    def deregister(self, client_id: str) -> bool:
        """Remove a client. Returns True if found and removed, else False."""
        if client_id in self._profiles:
            del self._profiles[client_id]
            return True
        return False

    def client_count(self) -> int:
        """Return total number of registered clients."""
        return len(self._profiles)


CLIENT_SELECTOR_REGISTRY: dict[str, type] = {"default": ClientSelector}
