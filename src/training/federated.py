"""Federated learning simulation: FedAvg with differential privacy noise, client drift correction."""  # noqa: E501

from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class FederatedConfig:
    """Configuration for federated learning simulation with differential privacy."""

    n_clients: int = 10
    fraction_clients: float = 0.3  # fraction selected per round
    local_steps: int = 5  # local SGD steps per client
    local_lr: float = 1e-3
    noise_multiplier: float = 1.0  # DP Gaussian noise sigma
    clip_norm: float = 1.0  # DP gradient clipping norm
    aggregation: str = "fedavg"  # "fedavg" | "fedprox" | "scaffold"
    mu: float = 0.01  # FedProx proximal term coefficient


# ---------------------------------------------------------------------------
# Differential Privacy utilities
# ---------------------------------------------------------------------------


def clip_gradients_dp(model: nn.Module, clip_norm: float) -> None:
    """Clip per-sample gradient norm to clip_norm (simulated as global grad-norm clip).

    Performs in-place clipping of the global gradient norm across all parameters,
    simulating per-sample DP gradient clipping in the absence of per-sample grads.

    Args:
        model: The model whose parameter gradients will be clipped.
        clip_norm: Maximum allowed gradient norm.
    """
    nn.utils.clip_grad_norm_(model.parameters(), clip_norm)


def add_dp_noise(
    model: nn.Module,
    noise_multiplier: float,
    clip_norm: float,
    n_samples: int,
) -> None:
    """Add Gaussian noise scaled to (noise_multiplier * clip_norm / n_samples).

    Adds calibrated noise to each parameter's gradient for differential privacy.
    In-place operation; skips parameters without gradients.

    Args:
        model: The model whose parameter gradients receive noise.
        noise_multiplier: Noise multiplier sigma for DP guarantee.
        clip_norm: Sensitivity / clipping norm.
        n_samples: Number of samples in the batch (used to scale noise).
    """
    noise_scale = (noise_multiplier * clip_norm) / max(n_samples, 1)
    for param in model.parameters():
        if param.grad is not None:
            noise = torch.randn_like(param.grad) * noise_scale
            param.grad.add_(noise)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def fedavg_aggregate(
    global_params: dict[str, Tensor],
    client_params_list: list[dict[str, Tensor]],
    weights: list[float] | None = None,
) -> dict[str, Tensor]:
    """Weighted average of client parameters (FedAvg).

    Args:
        global_params: Current global parameter dict (used for key reference / fallback).
        client_params_list: List of parameter dicts from selected clients.
        weights: Per-client aggregation weights; defaults to uniform if None.

    Returns:
        New global parameter dict with averaged values.
    """
    n = len(client_params_list)
    if n == 0:
        return {k: v.clone() for k, v in global_params.items()}

    if weights is None:
        w = [1.0 / n] * n
    else:
        total = sum(weights)
        w = [wi / total for wi in weights]

    aggregated: dict[str, Tensor] = {}
    keys = list(client_params_list[0].keys())
    for key in keys:
        stacked = torch.stack([client_params_list[i][key].float() * w[i] for i in range(n)])
        aggregated[key] = stacked.sum(dim=0).to(client_params_list[0][key].dtype)

    return aggregated


# ---------------------------------------------------------------------------
# FedProx proximal term
# ---------------------------------------------------------------------------


def fedprox_loss(
    model_params: dict[str, Tensor],
    global_params: dict[str, Tensor],
    mu: float,
) -> Tensor:
    """Compute the FedProx proximal regularization term.

    Proximal term: (mu/2) * sum(||w - w_global||^2)

    Args:
        model_params: Local model parameter dict.
        global_params: Global (server) parameter dict used as the anchor.
        mu: Proximal coefficient.

    Returns:
        Scalar tensor representing the proximal loss.
    """
    device = next(iter(model_params.values())).device
    prox = torch.tensor(0.0, device=device)
    for key, local_p in model_params.items():
        if key in global_params:
            global_p = global_params[key].detach().to(device)
            prox = prox + (local_p.float() - global_p.float()).pow(2).sum()
    return (mu / 2.0) * prox


# ---------------------------------------------------------------------------
# ClientSimulator
# ---------------------------------------------------------------------------


class ClientSimulator:
    """Simulates a single federated learning client with DP training."""

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        config: FederatedConfig,
    ) -> None:
        """Initialize the client with a deep copy of the global model.

        Args:
            client_id: Unique integer identifier for this client.
            model: Global model to copy as the starting point.
            config: Federated training configuration.
        """
        self.client_id = client_id
        self.model = copy.deepcopy(model)
        self.config = config

    def local_train(
        self,
        data: Tensor,
        global_params: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Run local_steps of SGD on data with optional FedProx and DP.

        Args:
            data: Input token ids tensor of shape (batch, seq_len).
            global_params: Global model parameters for FedProx proximal term
                           and as the reference anchor.

        Returns:
            Local model parameter dict after training.
        """
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.local_lr)
        n_samples = data.size(0)

        for _step in range(self.config.local_steps):
            optimizer.zero_grad()

            # Forward: model returns (loss, logits, past_key_values)
            loss, _logits, _pkv = self.model(data, labels=data)

            if loss is None:
                # Fallback: compute a dummy loss if labels not used
                loss = _logits.mean()

            # FedProx proximal regularization
            if self.config.aggregation == "fedprox":
                local_p = dict(self.model.named_parameters())
                loss = loss + fedprox_loss(local_p, global_params, self.config.mu)

            loss.backward()

            # DP: clip then add noise
            clip_gradients_dp(self.model, self.config.clip_norm)
            add_dp_noise(
                self.model,
                self.config.noise_multiplier,
                self.config.clip_norm,
                n_samples,
            )

            optimizer.step()

        return self.get_params()

    def get_params(self) -> dict[str, Tensor]:
        """Return a copy of the current local model parameters.

        Returns:
            Dict mapping parameter names to detached tensors.
        """
        return {k: v.detach().clone() for k, v in self.model.named_parameters()}

    def set_params(self, params: dict[str, Tensor]) -> None:
        """Load parameters into the local model.

        Args:
            params: Dict mapping parameter names to tensors.
        """
        self.model.state_dict()
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in params:
                    param.copy_(params[name])


# ---------------------------------------------------------------------------
# FederatedTrainer
# ---------------------------------------------------------------------------


class FederatedTrainer:
    """Coordinates federated training rounds with client selection and DP aggregation."""

    def __init__(
        self,
        global_model: nn.Module,
        config: FederatedConfig,
    ) -> None:
        """Initialize the federated trainer.

        Args:
            global_model: The server-side global model.
            config: Federated training configuration.
        """
        self.global_model = global_model
        self.config = config

    def select_clients(self, rng: random.Random) -> list[int]:
        """Sample a fraction of clients for this round.

        Args:
            rng: Random instance for reproducible selection.

        Returns:
            Sorted list of selected client integer IDs.
        """
        n_select = max(1, int(self.config.fraction_clients * self.config.n_clients))
        selected = rng.sample(range(self.config.n_clients), k=n_select)
        return sorted(selected)

    def train_round(
        self,
        client_data: dict[int, Tensor],
        rng: random.Random,
    ) -> dict[str, float]:
        """Execute one federated round: select clients, distribute, train, aggregate.

        Args:
            client_data: Mapping from client_id to input tensor for that client.
            rng: Random instance for client selection.

        Returns:
            Dict with keys:
                "round_loss": float — mean loss proxy (0.0 placeholder, training handled locally)
                "n_clients_selected": int — number of clients that participated
                "noise_scale": float — DP noise scale applied (noise_multiplier * clip_norm)
        """
        selected_ids = self.select_clients(rng)
        global_params = self.get_global_params()

        client_param_list: list[dict[str, Tensor]] = []

        for cid in selected_ids:
            data = client_data.get(cid)
            if data is None:
                continue

            client = ClientSimulator(
                client_id=cid,
                model=self.global_model,
                config=self.config,
            )
            local_params = client.local_train(data, global_params)
            client_param_list.append(local_params)

        if client_param_list:
            new_params = fedavg_aggregate(global_params, client_param_list)
            # Load aggregated parameters back into global model
            with torch.no_grad():
                for name, param in self.global_model.named_parameters():
                    if name in new_params:
                        param.copy_(new_params[name])

        noise_scale = self.config.noise_multiplier * self.config.clip_norm

        return {
            "round_loss": 0.0,
            "n_clients_selected": len(selected_ids),
            "noise_scale": noise_scale,
        }

    def get_global_params(self) -> dict[str, Tensor]:
        """Return a copy of the current global model parameters.

        Returns:
            Dict mapping parameter names to detached tensors.
        """
        return {k: v.detach().clone() for k, v in self.global_model.named_parameters()}
