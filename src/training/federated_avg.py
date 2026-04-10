"""Federated Learning: FedAvg, FedProx, and FedMedian aggregation strategies."""

from __future__ import annotations

import copy
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class FederatedConfig:
    """Configuration for federated learning (FedAvg / FedProx / FedMedian)."""

    n_clients: int = 10
    fraction: float = 0.1         # fraction of clients selected per round
    local_epochs: int = 5         # local training epochs per client per round
    local_lr: float = 0.01        # local SGD learning rate
    mu: float = 0.0               # FedProx proximal regularisation coefficient
    aggregation: str = "fedavg"   # "fedavg" | "fedprox" | "fedmedian"


# ---------------------------------------------------------------------------
# Weight utilities
# ---------------------------------------------------------------------------


def get_model_weights(model: nn.Module) -> OrderedDict:
    """Return a detached copy of model parameters as an OrderedDict.

    Args:
        model: PyTorch module.

    Returns:
        OrderedDict mapping parameter name -> detached CPU tensor copy.
    """
    return OrderedDict(
        (name, param.detach().clone())
        for name, param in model.named_parameters()
    )


def set_model_weights(model: nn.Module, weights: OrderedDict) -> None:
    """Load a weight dict into *model* in-place (no_grad).

    Args:
        model:   PyTorch module whose parameters will be overwritten.
        weights: OrderedDict mapping parameter name -> tensor.
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in weights:
                param.copy_(weights[name])


# ---------------------------------------------------------------------------
# Aggregation functions
# ---------------------------------------------------------------------------


def fedavg_aggregate(
    client_weights: List[OrderedDict],
    client_sizes: List[int],
) -> OrderedDict:
    """Weighted-average aggregation (FedAvg).

    Computes  w_global = sum_i( n_i * w_i ) / sum_i( n_i ).

    Args:
        client_weights: List of weight dicts, one per client.
        client_sizes:   Number of samples each client trained on.

    Returns:
        Aggregated weight OrderedDict.
    """
    if not client_weights:
        raise ValueError("client_weights must not be empty")
    if len(client_weights) != len(client_sizes):
        raise ValueError("client_weights and client_sizes must have the same length")

    total = sum(client_sizes)
    if total == 0:
        raise ValueError("Total client sizes must be > 0")

    aggregated: OrderedDict = OrderedDict()
    keys = list(client_weights[0].keys())
    for key in keys:
        stacked = torch.stack(
            [client_weights[i][key].float() * client_sizes[i] for i in range(len(client_weights))]
        )
        aggregated[key] = stacked.sum(dim=0) / total

    # Cast back to original dtype
    for key in keys:
        ref_dtype = client_weights[0][key].dtype
        aggregated[key] = aggregated[key].to(ref_dtype)

    return aggregated


def fedmedian_aggregate(client_weights: List[OrderedDict]) -> OrderedDict:
    """Coordinate-wise median aggregation (Byzantine-robust alternative to FedAvg).

    Args:
        client_weights: List of weight dicts, one per client (>= 1 required).

    Returns:
        Aggregated weight OrderedDict where each element is the median across clients.
    """
    if not client_weights:
        raise ValueError("client_weights must not be empty")

    aggregated: OrderedDict = OrderedDict()
    keys = list(client_weights[0].keys())
    for key in keys:
        stacked = torch.stack([cw[key].float() for cw in client_weights])
        median_vals, _ = torch.median(stacked, dim=0)
        ref_dtype = client_weights[0][key].dtype
        aggregated[key] = median_vals.to(ref_dtype)

    return aggregated


# ---------------------------------------------------------------------------
# FedProx penalty
# ---------------------------------------------------------------------------


def fedprox_penalty(
    model: nn.Module,
    global_weights: OrderedDict,
    mu: float,
) -> torch.Tensor:
    """Compute the FedProx proximal regularisation term.

    penalty = (mu / 2) * || w_local - w_global ||^2

    Args:
        model:          Local model being trained.
        global_weights: Global model weights (from the server).
        mu:             Proximal coefficient.

    Returns:
        Scalar tensor representing the penalty (0 when mu == 0).
    """
    if mu == 0.0:
        return torch.tensor(0.0)

    penalty = torch.tensor(0.0)
    for name, param in model.named_parameters():
        if name in global_weights:
            global_param = global_weights[name].to(param.device)
            penalty = penalty + (param - global_param).pow(2).sum()

    return (mu / 2.0) * penalty


# ---------------------------------------------------------------------------
# FederatedClient
# ---------------------------------------------------------------------------


class FederatedClient:
    """Simulates a single federated learning client.

    Performs local_epochs of SGD on its data and returns updated weights.
    """

    def __init__(self, model: nn.Module, config: FederatedConfig) -> None:
        self.model = model
        self.config = config

    def local_train(
        self,
        input_ids: torch.Tensor,
        global_weights: OrderedDict | None = None,
    ) -> Tuple[OrderedDict, int]:
        """Run local training for *local_epochs* steps.

        Args:
            input_ids:      (batch, seq_len) integer token tensor.
            global_weights: Server weights, required for FedProx penalty.

        Returns:
            Tuple of (updated_weights, n_samples) where n_samples = batch size.
        """
        # Load global weights if provided (start from server's checkpoint)
        if global_weights is not None:
            set_model_weights(self.model, global_weights)

        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.local_lr)

        n_samples = input_ids.shape[0]

        for _ in range(self.config.local_epochs):
            optimizer.zero_grad()
            loss, _logits, _pkv = self.model(input_ids)

            if loss is None:
                # No labels provided — use next-token prediction targets
                logits = _logits  # (B, S, V)
                targets = input_ids[:, 1:].contiguous()
                shift_logits = logits[:, :-1, :].contiguous()
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    targets.view(-1),
                )

            # FedProx proximal term
            if (
                self.config.aggregation in ("fedprox",)
                and global_weights is not None
                and self.config.mu > 0.0
            ):
                loss = loss + fedprox_penalty(self.model, global_weights, self.config.mu)

            loss.backward()
            optimizer.step()

        updated_weights = get_model_weights(self.model)
        return updated_weights, n_samples


# ---------------------------------------------------------------------------
# FederatedServer
# ---------------------------------------------------------------------------


class FederatedServer:
    """Simulates the federated learning server.

    Holds the global model and aggregates client updates each round.
    """

    def __init__(self, model: nn.Module, config: FederatedConfig) -> None:
        self.model = model
        self.config = config

    def aggregate_round(
        self,
        client_results: List[Tuple[OrderedDict, int]],
    ) -> OrderedDict:
        """Aggregate client updates and update the server model.

        Args:
            client_results: List of (weights, n_samples) tuples from clients.

        Returns:
            Aggregated global weight dict (also loaded into self.model).
        """
        if not client_results:
            raise ValueError("client_results must not be empty")

        client_weights = [r[0] for r in client_results]
        client_sizes = [r[1] for r in client_results]

        if self.config.aggregation == "fedmedian":
            aggregated = fedmedian_aggregate(client_weights)
        else:
            # "fedavg" or "fedprox" both use weighted averaging at the server
            aggregated = fedavg_aggregate(client_weights, client_sizes)

        set_model_weights(self.model, aggregated)
        return aggregated
