"""Federated learning aggregator with simulated secure aggregation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn


@dataclass
class ClientUpdate:
    """Holds a single client's model update."""

    client_id: int
    state_dict: Dict[str, torch.Tensor]
    n_samples: int


class FederatedAggregator:
    """Aggregates model updates from multiple clients using FedAvg.

    Optionally applies additive Gaussian noise masking to simulate secure
    aggregation before loading the result into the global model.
    """

    def __init__(
        self,
        model: nn.Module,
        n_clients: int,
        noise_sigma: float = 0.0,
    ) -> None:
        self.model = model
        self.n_clients = n_clients
        self.noise_sigma = noise_sigma

    def _weighted_average(
        self, updates: List[ClientUpdate]
    ) -> Dict[str, torch.Tensor]:
        """Compute FedAvg weighted average of client state dicts."""
        total_samples = sum(u.n_samples for u in updates)
        keys = list(updates[0].state_dict.keys())
        aggregated: Dict[str, torch.Tensor] = {}
        for key in keys:
            weighted_sum = torch.zeros_like(updates[0].state_dict[key], dtype=torch.float32)
            for update in updates:
                weight = update.n_samples / total_samples
                weighted_sum = weighted_sum + weight * update.state_dict[key].float()
            # Cast back to the original dtype
            aggregated[key] = weighted_sum.to(updates[0].state_dict[key].dtype)
        return aggregated

    def _add_aggregation_noise(
        self, state_dict: Dict[str, torch.Tensor], sigma: float
    ) -> Dict[str, torch.Tensor]:
        """Add iid N(0, sigma^2) noise to every tensor in state_dict."""
        noisy: Dict[str, torch.Tensor] = {}
        for key, tensor in state_dict.items():
            noise = torch.randn_like(tensor.float()) * sigma
            noisy[key] = (tensor.float() + noise).to(tensor.dtype)
        return noisy

    def aggregate(self, updates: List[ClientUpdate]) -> Dict[str, torch.Tensor]:
        """Compute weighted average and optionally apply noise."""
        aggregated = self._weighted_average(updates)
        if self.noise_sigma > 0.0:
            aggregated = self._add_aggregation_noise(aggregated, self.noise_sigma)
        return aggregated

    def apply(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load state dict into the global model."""
        self.model.load_state_dict(state_dict)

    def compute_update_norm(
        self, update: ClientUpdate, global_state: Dict[str, torch.Tensor]
    ) -> float:
        """L2 norm of the difference between client update and global model parameters."""
        diff_tensors = [
            (update.state_dict[k].float() - global_state[k].float()).reshape(-1)
            for k in global_state.keys()
        ]
        flat_diff = torch.cat(diff_tensors)
        return float(torch.norm(flat_diff, p=2).item())

    def clip_update(self, update: ClientUpdate, max_norm: float) -> ClientUpdate:
        """Scale client update so its difference from global model has L2 norm <= max_norm."""
        global_state = {k: v for k, v in self.model.state_dict().items()}
        current_norm = self.compute_update_norm(update, global_state)
        if current_norm <= max_norm or current_norm == 0.0:
            return update
        scale = max_norm / current_norm
        clipped_state: Dict[str, torch.Tensor] = {}
        for key in global_state:
            diff = (update.state_dict[key].float() - global_state[key].float()) * scale
            clipped_state[key] = (global_state[key].float() + diff).to(update.state_dict[key].dtype)
        return ClientUpdate(
            client_id=update.client_id,
            state_dict=clipped_state,
            n_samples=update.n_samples,
        )
