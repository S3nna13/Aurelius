"""Federated learning: client updates, server aggregation, round management."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ClientUpdate:
    """Represents a single client's model update for a federated round."""

    client_id: str
    round_id: int
    weight_delta: list[float]
    n_samples: int
    loss: float = 0.0


class FederatedClient:
    """Simulates a federated learning client."""

    def __init__(self, client_id: str, n_samples: int = 100) -> None:
        self.client_id = client_id
        self.n_samples = n_samples

    def compute_update(
        self,
        global_weights: list[float],
        local_weights: list[float],
        round_id: int,
    ) -> ClientUpdate:
        """Compute the delta between local and global weights."""
        delta = [l - g for l, g in zip(local_weights, global_weights)]
        return ClientUpdate(
            client_id=self.client_id,
            round_id=round_id,
            weight_delta=delta,
            n_samples=self.n_samples,
        )

    def apply_update(
        self,
        weights: list[float],
        delta: list[float],
        lr: float = 1.0,
    ) -> list[float]:
        """Apply a delta to weights with a learning rate."""
        return [w + lr * d for w, d in zip(weights, delta)]


class FederatedServer:
    """Manages global model aggregation and round tracking."""

    def __init__(self) -> None:
        self._global_round: int = 0

    @property
    def global_round(self) -> int:
        return self._global_round

    def aggregate(self, updates: list[ClientUpdate]) -> list[float]:
        """Weighted FedAvg: aggregate client updates weighted by n_samples."""
        if not updates:
            return []

        total_samples = sum(u.n_samples for u in updates)
        n_params = len(updates[0].weight_delta)

        aggregated = []
        for i in range(n_params):
            weighted_sum = sum(u.weight_delta[i] * u.n_samples for u in updates)
            aggregated.append(weighted_sum / total_samples)

        return aggregated

    def advance_round(self) -> int:
        """Increment the global round counter and return the new round number."""
        self._global_round += 1
        return self._global_round

    def client_participation_rate(
        self, n_total: int, updates: list[ClientUpdate]
    ) -> float:
        """Return fraction of total clients that submitted updates."""
        return len(updates) / n_total


FEDERATION_REGISTRY: dict[str, object] = {
    "server": FederatedServer(),
    "client_factory": FederatedClient,
}
