"""Federated aggregation with multiple robust strategies.

Supports FedAvg, coordinate-wise median, and trimmed mean.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


class FederatedAggregator:
    """Aggregate client model updates using a chosen strategy."""

    _strategies: dict[
        str, Callable[[list[dict[str, torch.Tensor]], list[float]], dict[str, torch.Tensor]]
    ]

    def __init__(self, strategy: str = "fedavg") -> None:
        self.strategy = strategy
        self._strategies = {
            "fedavg": self._fedavg,
            "median": self._median,
            "trimmed_mean": self._trimmed_mean,
        }
        if strategy not in self._strategies:
            raise ValueError(
                f"Unknown strategy {strategy!r}. Choose from {list(self._strategies.keys())}."
            )

    def aggregate(
        self,
        client_updates: list[dict[str, torch.Tensor]],
        client_weights: list[float] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Aggregate model updates from multiple clients.

        Args:
            client_updates: List of per-client parameter dictionaries.
            client_weights: Optional weight for each client. If None, equal weights are used.

        Returns:
            Aggregated parameter dictionary.

        Raises:
            ValueError: If updates have mismatched keys or shapes, or if weights length is wrong.
        """
        if not client_updates:
            return {}

        if client_weights is not None and len(client_weights) != len(client_updates):
            raise ValueError(
                f"client_weights length ({len(client_weights)}) must match "
                f"client_updates length ({len(client_updates)})"
            )

        self._validate_updates(client_updates)

        weights = (
            client_weights
            if client_weights is not None
            else [1.0 / len(client_updates)] * len(client_updates)
        )
        return self._strategies[self.strategy](client_updates, weights)

    def _validate_updates(self, client_updates: list[dict[str, torch.Tensor]]) -> None:
        expected_keys = set(client_updates[0].keys())
        for i, update in enumerate(client_updates):
            if set(update.keys()) != expected_keys:
                raise ValueError(
                    f"Client update {i} has mismatched keys. "
                    f"Expected {sorted(expected_keys)}, got {sorted(update.keys())}"
                )
            for key in expected_keys:
                if update[key].shape != client_updates[0][key].shape:
                    raise ValueError(
                        f"Client update {i}, key {key!r}: shape "
                        f"{tuple(update[key].shape)} does not match expected shape "
                        f"{tuple(client_updates[0][key].shape)}"
                    )

    def _fedavg(
        self,
        client_updates: list[dict[str, torch.Tensor]],
        weights: list[float],
    ) -> dict[str, torch.Tensor]:
        total_weight = sum(weights)
        if total_weight == 0:
            raise ValueError("Sum of client weights must be > 0")

        result: dict[str, torch.Tensor] = {}
        for key in client_updates[0]:
            weighted = torch.zeros_like(client_updates[0][key], dtype=torch.float64)
            for update, w in zip(client_updates, weights):
                weighted += update[key].to(torch.float64) * (w / total_weight)
            result[key] = weighted.to(client_updates[0][key].dtype)
        return result

    def _median(
        self,
        client_updates: list[dict[str, torch.Tensor]],
        _weights: list[float],
    ) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        for key in client_updates[0]:
            stacked = torch.stack([u[key] for u in client_updates])
            median_values = torch.median(stacked, dim=0).values
            result[key] = median_values.to(client_updates[0][key].dtype)
        return result

    def _trimmed_mean(
        self,
        client_updates: list[dict[str, torch.Tensor]],
        _weights: list[float],
    ) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        for key in client_updates[0]:
            stacked = torch.stack([u[key] for u in client_updates])
            n = stacked.size(0)
            if n <= 2:
                # Cannot remove both min and max without emptying; fall back to mean.
                result[key] = stacked.mean(dim=0).to(client_updates[0][key].dtype)
            else:
                sorted_values, _ = torch.sort(stacked, dim=0)
                trimmed = sorted_values[1:-1]
                result[key] = trimmed.mean(dim=0).to(client_updates[0][key].dtype)
        return result


FEDERATED_AGGREGATOR_REGISTRY: dict[str, type[FederatedAggregator]] = {
    "default": FederatedAggregator,
}
