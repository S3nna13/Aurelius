"""Federated Learning: FedAvg and FedProx for distributed training simulation."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FedConfig:
    """Configuration for federated learning simulation."""

    n_clients: int = 4
    local_epochs: int = 2
    local_lr: float = 1e-3
    mu: float = 0.01                  # FedProx proximal term coefficient
    aggregation: str = "fedavg"       # "fedavg" | "fedprox" | "weighted_avg"
    clip_norm: float | None = None    # optional gradient clipping


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def fedavg_aggregate(
    client_state_dicts: list[dict],
    weights: list[float] | None = None,
) -> dict:
    """Weighted average of model state dicts.

    Args:
        client_state_dicts: List of state dicts, one per client.
        weights: Relative data sizes per client (normalized internally).
                 None means uniform weighting.

    Returns:
        Averaged state dict with all tensors.
    """
    n = len(client_state_dicts)
    if n == 0:
        raise ValueError("client_state_dicts must not be empty")

    # Normalize weights
    if weights is None:
        w = [1.0 / n] * n
    else:
        if len(weights) != n:
            raise ValueError(
                f"len(weights)={len(weights)} must equal len(client_state_dicts)={n}"
            )
        total = sum(weights)
        w = [wi / total for wi in weights]

    # Compute weighted average
    avg_state: dict = {}
    keys = list(client_state_dicts[0].keys())
    for key in keys:
        stacked = torch.stack(
            [client_state_dicts[i][key].float() * w[i] for i in range(n)]
        )
        avg_state[key] = stacked.sum(dim=0).to(client_state_dicts[0][key].dtype)

    return avg_state


def fedprox_loss(
    local_model: nn.Module,
    global_model: nn.Module,
    mu: float,
) -> Tensor:
    """FedProx proximal regularization term.

    Computes: (mu/2) * sum(||w_local - w_global||^2)

    Args:
        local_model: Client's local model.
        global_model: Server's global model (frozen reference).
        mu: Proximal coefficient.

    Returns:
        Scalar tensor representing the proximal loss.
    """
    local_params = dict(local_model.named_parameters())
    global_params = dict(global_model.named_parameters())

    prox = torch.tensor(0.0, device=next(local_model.parameters()).device)
    for name, local_p in local_params.items():
        if name in global_params:
            global_p = global_params[name].detach()
            prox = prox + (local_p - global_p).pow(2).sum()

    return (mu / 2.0) * prox


# ---------------------------------------------------------------------------
# FederatedClient
# ---------------------------------------------------------------------------

class FederatedClient:
    """Simulates a single federated learning client."""

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        config: FedConfig,
    ) -> None:
        self.client_id = client_id
        self.model = copy.deepcopy(model)
        self.config = config
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=config.local_lr
        )

    def local_train(
        self,
        data_loader: list[tuple[Tensor, Tensor]],
        global_model: nn.Module | None = None,
    ) -> dict[str, float]:
        """Run local_epochs of training.

        Args:
            data_loader: List of (input_ids, labels) tuples.
            global_model: Reference global model for FedProx regularization.

        Returns:
            dict with keys "train_loss" (float) and "n_samples" (int).
        """
        self.model.train()
        total_loss = 0.0
        n_samples = 0

        use_fedprox = (
            global_model is not None
            and self.config.aggregation == "fedprox"
        )

        for _epoch in range(self.config.local_epochs):
            for input_ids, labels in data_loader:
                self.optimizer.zero_grad()

                loss, _logits, _ = self.model(input_ids, labels=labels)

                if use_fedprox:
                    loss = loss + fedprox_loss(self.model, global_model, self.config.mu)

                loss.backward()

                if self.config.clip_norm is not None:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.clip_norm
                    )

                self.optimizer.step()

                total_loss += loss.item() * input_ids.size(0)
                n_samples += input_ids.size(0)

        mean_loss = total_loss / n_samples if n_samples > 0 else 0.0
        return {"train_loss": mean_loss, "n_samples": n_samples}

    def get_state_dict(self) -> dict:
        """Return the client model's state dict."""
        return copy.deepcopy(self.model.state_dict())

    def set_state_dict(self, state_dict: dict) -> None:
        """Load a state dict into the client model (copy from global)."""
        self.model.load_state_dict(copy.deepcopy(state_dict))


# ---------------------------------------------------------------------------
# FederatedServer
# ---------------------------------------------------------------------------

class FederatedServer:
    """Coordinates federated learning across multiple clients."""

    def __init__(self, global_model: nn.Module, config: FedConfig) -> None:
        self.global_model = global_model
        self.config = config
        self.clients: list[FederatedClient] = []
        self.round: int = 0

    def register_client(self, client: FederatedClient) -> None:
        """Add a client to the federation."""
        self.clients.append(client)

    def broadcast(self) -> None:
        """Push current global model weights to all clients."""
        global_sd = self.global_model.state_dict()
        for client in self.clients:
            client.set_state_dict(global_sd)

    def aggregate(self, client_weights: list[float] | None = None) -> None:
        """Collect client state dicts and update global model via FedAvg."""
        state_dicts = [c.get_state_dict() for c in self.clients]
        avg_sd = fedavg_aggregate(state_dicts, weights=client_weights)
        self.global_model.load_state_dict(avg_sd)

    def federated_round(
        self,
        client_data: list[list[tuple[Tensor, Tensor]]],
        client_weights: list[float] | None = None,
    ) -> dict[str, float]:
        """Execute one full federated round: broadcast → local train → aggregate.

        Args:
            client_data: Per-client list of (input_ids, labels) batches.
            client_weights: Optional per-client data weights for aggregation.

        Returns:
            dict with keys "round" (int), "mean_client_loss" (float),
            "n_clients" (int).
        """
        self.broadcast()

        # Decide whether to pass global model for FedProx
        global_ref = (
            self.global_model if self.config.aggregation == "fedprox" else None
        )

        client_losses = []
        for client, data in zip(self.clients, client_data):
            result = client.local_train(data, global_model=global_ref)
            client_losses.append(result["train_loss"])

        self.aggregate(client_weights=client_weights)
        self.round += 1

        mean_loss = sum(client_losses) / len(client_losses) if client_losses else 0.0
        return {
            "round": self.round,
            "mean_client_loss": mean_loss,
            "n_clients": len(self.clients),
        }


# ---------------------------------------------------------------------------
# Data heterogeneity simulation
# ---------------------------------------------------------------------------

def simulate_data_heterogeneity(
    n_clients: int,
    vocab_size: int,
    seq_len: int,
    n_batches: int,
    seed: int = 42,
) -> list[list[tuple[Tensor, Tensor]]]:
    """Generate non-IID synthetic data for each client.

    Each client receives data biased toward a different sub-range of the
    vocabulary, simulating realistic data heterogeneity in federated settings.

    Args:
        n_clients: Number of clients.
        vocab_size: Vocabulary size used to bias token ranges.
        seq_len: Sequence length for each batch.
        n_batches: Number of (input_ids, labels) batches per client.
        seed: Random seed for reproducibility.

    Returns:
        List of n_clients data loaders, each a list of
        (input_ids, labels) tuples with shapes (1, seq_len).
    """
    rng = torch.Generator()
    rng.manual_seed(seed)

    # Partition vocab into n_clients ranges (may overlap at edges)
    range_size = max(1, vocab_size // n_clients)

    all_data: list[list[tuple[Tensor, Tensor]]] = []
    for client_idx in range(n_clients):
        low = client_idx * range_size
        high = min(low + range_size, vocab_size)

        client_batches: list[tuple[Tensor, Tensor]] = []
        for _ in range(n_batches):
            # Sample tokens biased to this client's range
            token_range = high - low
            tokens = torch.randint(
                0, token_range, (1, seq_len), generator=rng
            ) + low
            tokens = tokens.clamp(0, vocab_size - 1)

            input_ids = tokens
            labels = tokens.clone()
            client_batches.append((input_ids, labels))

        all_data.append(client_batches)

    return all_data
