"""Federated learning: FedAvg, FedProx, and SCAFFOLD simulation for Aurelius LLM."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FedConfig:
    """Configuration for federated learning simulation."""

    n_clients: int = 10
    fraction_fit: float = 0.3       # fraction of clients selected per round
    local_epochs: int = 2           # local training epochs per client
    local_lr: float = 1e-3          # local SGD learning rate
    aggregation: str = "fedavg"     # "fedavg" | "fedprox" | "scaffold"
    mu: float = 0.01                # FedProx proximal coefficient


# ---------------------------------------------------------------------------
# ClientUpdate
# ---------------------------------------------------------------------------

@dataclass
class ClientUpdate:
    """Result of a single client's local training."""

    client_id: int
    delta_weights: dict[str, Tensor]
    n_samples: int
    loss: float


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def fedavg_aggregate(updates: list[ClientUpdate]) -> dict[str, Tensor]:
    """Weighted average of weight deltas (FedAvg).

    Weight for each client = n_samples_i / sum(n_samples).

    Args:
        updates: List of ClientUpdate objects from participating clients.

    Returns:
        Dict mapping parameter name to aggregated delta tensor.
    """
    if not updates:
        return {}

    total_samples = sum(u.n_samples for u in updates)
    weights = [u.n_samples / total_samples for u in updates]

    keys = list(updates[0].delta_weights.keys())
    aggregated: dict[str, Tensor] = {}
    for key in keys:
        stacked = torch.stack(
            [updates[i].delta_weights[key].float() * weights[i] for i in range(len(updates))]
        )
        ref = updates[0].delta_weights[key]
        aggregated[key] = stacked.sum(dim=0).to(ref.dtype)

    return aggregated


# ---------------------------------------------------------------------------
# FedProx proximal term
# ---------------------------------------------------------------------------

def fedprox_loss(
    local_params: dict[str, Tensor],
    global_params: dict[str, Tensor],
    mu: float,
) -> Tensor:
    """Compute the FedProx proximal regularization term.

    Term: (mu/2) * sum(||w - w_global||^2)

    Args:
        local_params: Local model parameter dict.
        global_params: Global (server) parameter dict.
        mu: Proximal coefficient.

    Returns:
        Scalar tensor.
    """
    device = next(iter(local_params.values())).device
    prox = torch.tensor(0.0, device=device)
    for key, local_p in local_params.items():
        if key in global_params:
            global_p = global_params[key].detach().to(device)
            prox = prox + (local_p.float() - global_p.float()).pow(2).sum()
    return (mu / 2.0) * prox


# ---------------------------------------------------------------------------
# Client simulation
# ---------------------------------------------------------------------------

def simulate_client_update(
    model: nn.Module,
    client_data: list[Tensor],
    config: FedConfig,
    global_params: Optional[dict[str, Tensor]] = None,
    client_id: int = 0,
) -> ClientUpdate:
    """Simulate local training on a single client.

    Copies the model, runs local_epochs of gradient steps on client_data,
    optionally adds FedProx proximal term, then computes the weight delta.

    Args:
        model: Global model (will not be modified).
        client_data: List of input_ids tensors for this client.
        config: Federated training configuration.
        global_params: Global parameter dict needed for FedProx; optional otherwise.
        client_id: Integer identifier for this client.

    Returns:
        ClientUpdate with delta_weights, n_samples, and mean loss.
    """
    local_model = copy.deepcopy(model)
    local_model.train()

    # Capture initial (global) parameters
    initial_params = {k: v.detach().clone() for k, v in local_model.named_parameters()}

    if global_params is None:
        global_params = initial_params

    optimizer = torch.optim.SGD(local_model.parameters(), lr=config.local_lr)

    total_loss = 0.0
    n_steps = 0
    n_samples = sum(x.shape[0] for x in client_data) if client_data else 1

    for _epoch in range(config.local_epochs):
        for batch in client_data:
            optimizer.zero_grad()

            # Model API: loss, logits, pkv = model(input_ids)
            # Pass labels=batch to get a real loss from the model
            try:
                loss, _logits, _pkv = local_model(batch, labels=batch)
            except TypeError:
                loss, _logits, _pkv = local_model(batch)

            if loss is None:
                loss = _logits.mean()

            # FedProx proximal regularization
            if config.aggregation == "fedprox":
                local_named = dict(local_model.named_parameters())
                loss = loss + fedprox_loss(local_named, global_params, config.mu)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_steps += 1

    mean_loss = total_loss / max(n_steps, 1)

    # Compute delta = local - initial
    local_params_now = {k: v.detach().clone() for k, v in local_model.named_parameters()}
    delta = {k: local_params_now[k] - initial_params[k] for k in initial_params}

    return ClientUpdate(
        client_id=client_id,
        delta_weights=delta,
        n_samples=n_samples,
        loss=mean_loss,
    )


# ---------------------------------------------------------------------------
# FederatedServer
# ---------------------------------------------------------------------------

class FederatedServer:
    """Coordinates federated training rounds."""

    def __init__(self, model: nn.Module, config: FedConfig) -> None:
        """Initialize the server with a global model and config.

        Args:
            model: The server-side global model.
            config: Federated training configuration.
        """
        self.model = model
        self.config = config
        self._round: int = 0

    def get_global_params(self) -> dict[str, Tensor]:
        """Return a copy of the current global model state dict.

        Returns:
            Dict mapping parameter name to detached clone tensor.
        """
        return {k: v.detach().clone() for k, v in self.model.named_parameters()}

    def select_clients(self, n_clients: int) -> list[int]:
        """Randomly select fraction_fit * n_clients client IDs.

        Args:
            n_clients: Total number of available clients.

        Returns:
            Sorted list of selected client integer IDs.
        """
        n_select = max(1, int(self.config.fraction_fit * n_clients))
        n_select = min(n_select, n_clients)
        selected = random.sample(range(n_clients), k=n_select)
        return sorted(selected)

    def aggregate_and_update(self, updates: list[ClientUpdate]) -> dict:
        """Apply FedAvg aggregation and update the global model.

        Applies the aggregated delta to model parameters:
            new_param = old_param + aggregated_delta

        Args:
            updates: List of ClientUpdate from selected clients.

        Returns:
            Dict with keys: "round", "n_clients", "mean_loss", "weight_norm".
        """
        if not updates:
            return {
                "round": self._round,
                "n_clients": 0,
                "mean_loss": 0.0,
                "weight_norm": 0.0,
            }

        aggregated_delta = fedavg_aggregate(updates)

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in aggregated_delta:
                    param.add_(aggregated_delta[name].to(param.device))

        mean_loss = sum(u.loss for u in updates) / len(updates)
        weight_norm = sum(
            p.norm().item() ** 2 for p in self.model.parameters()
        ) ** 0.5

        self._round += 1

        return {
            "round": self._round,
            "n_clients": len(updates),
            "mean_loss": mean_loss,
            "weight_norm": weight_norm,
        }

    def federated_round(self, client_datasets: list[list[Tensor]]) -> dict:
        """Execute one full federated round.

        Selects clients, simulates local updates, aggregates, and updates model.

        Args:
            client_datasets: List of per-client datasets (each a list of tensors).

        Returns:
            Round statistics dict with at least "mean_loss".
        """
        n_clients = len(client_datasets)
        selected_ids = self.select_clients(n_clients)
        global_params = self.get_global_params()

        updates: list[ClientUpdate] = []
        for cid in selected_ids:
            data = client_datasets[cid]
            update = simulate_client_update(
                model=self.model,
                client_data=data,
                config=self.config,
                global_params=global_params,
                client_id=cid,
            )
            updates.append(update)

        return self.aggregate_and_update(updates)
