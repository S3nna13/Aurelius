"""Federated Fine-Tuning for LLMs: FedProx and Federated LoRA.

FedProx: adds proximal penalty to local updates to prevent client drift.
FedLoRA: clients train and communicate only LoRA adapters (rank-r updates),
reducing communication cost by factor d_model/r.

References:
    Li et al. 2020 (FedProx) — https://arxiv.org/abs/1812.06127
    Che et al. 2023 (FedPEFT/FedLoRA) — various works
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# ClientDataShard
# ---------------------------------------------------------------------------


class ClientDataShard:
    """Represents a single client's local dataset.

    Args:
        client_id: Unique integer identifier for this client.
        data: Local training data of shape (N_local, d_input).
        labels: Corresponding labels of shape (N_local,) or (N_local, d_output).
    """

    def __init__(self, client_id: int, data: Tensor, labels: Tensor) -> None:
        self.client_id = client_id
        self.data = data
        self.labels = labels
        self._n = data.shape[0]

    def sample_batch(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        """Return a random batch of (data, labels), wrapping around if needed.

        Args:
            batch_size: Number of samples to return.

        Returns:
            Tuple of (data_batch, labels_batch).
        """
        indices = torch.randint(0, self._n, (batch_size,))
        return self.data[indices], self.labels[indices]


# ---------------------------------------------------------------------------
# FedProxClient
# ---------------------------------------------------------------------------


class FedProxClient:
    """Local client with FedProx proximal term to prevent drift.

    Args:
        client_id: Unique integer identifier.
        model: Local PyTorch model.
        mu: FedProx proximal coefficient (default 0.01).
    """

    def __init__(self, client_id: int, model: nn.Module, mu: float = 0.01) -> None:
        self.client_id = client_id
        self.model = model
        self.mu = mu
        self.global_params: Dict[str, Tensor] = {}

    def set_global_params(self, global_state: Dict[str, Tensor]) -> None:
        """Load global parameters into model and save a reference copy.

        Args:
            global_state: State dict from the global model.
        """
        self.model.load_state_dict(global_state)
        self.global_params = {k: v.clone() for k, v in global_state.items()}

    def proximal_term(self) -> Tensor:
        """Compute the FedProx proximal penalty.

        Returns:
            Scalar tensor: (mu/2) * sum_k ||param_k - global_k||^2
        """
        prox = torch.tensor(0.0)
        for name, param in self.model.named_parameters():
            if name in self.global_params:
                diff = param - self.global_params[name]
                prox = prox + (diff * diff).sum()
        return (self.mu / 2.0) * prox

    def local_train(
        self,
        shard: ClientDataShard,
        loss_fn: Callable,
        n_steps: int = 5,
        lr: float = 0.01,
    ) -> Dict[str, float]:
        """Run n_steps of gradient descent with Adam + proximal term.

        Args:
            shard: Client's local data shard.
            loss_fn: Callable(predictions, labels) -> scalar loss tensor.
            n_steps: Number of local gradient steps.
            lr: Adam learning rate.

        Returns:
            Dict with keys 'avg_task_loss', 'avg_proximal', 'n_steps'.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        total_task_loss = 0.0
        total_proximal = 0.0

        self.model.train()
        for _ in range(n_steps):
            data, labels = shard.sample_batch(min(32, shard._n))
            optimizer.zero_grad()
            preds = self.model(data)
            task_loss = loss_fn(preds, labels)
            prox = self.proximal_term()
            total_loss = task_loss + prox
            total_loss.backward()
            optimizer.step()
            total_task_loss += task_loss.item()
            total_proximal += prox.item()

        return {
            "avg_task_loss": total_task_loss / n_steps,
            "avg_proximal": total_proximal / n_steps,
            "n_steps": n_steps,
        }

    def get_update(self) -> Dict[str, Tensor]:
        """Return the parameter delta (current - global) for each param.

        Returns:
            Dict mapping param name to delta tensor.
        """
        update = {}
        current_state = self.model.state_dict()
        for name, param in current_state.items():
            if name in self.global_params:
                update[name] = param.data - self.global_params[name]
            else:
                update[name] = param.data.clone()
        return update


# ---------------------------------------------------------------------------
# FedLoRAAdapter
# ---------------------------------------------------------------------------


class FedLoRAAdapter(nn.Module):
    """LoRA adapter for federated communication efficiency.

    Only low-rank matrices A and B are trained and communicated, reducing
    communication cost by a factor of d_model / rank.

    Args:
        d_in: Input feature dimension.
        d_out: Output feature dimension.
        rank: LoRA rank (default 4).
        alpha: Scaling factor; effective weight is (alpha/rank) * B @ A.
    """

    def __init__(self, d_in: int, d_out: int, rank: int = 4, alpha: float = 1.0) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.rank = rank
        self.alpha = alpha

        # A: (rank, d_in) — random normal init
        self.A = nn.Parameter(torch.randn(rank, d_in) * 0.02)
        # B: (d_out, rank) — zero init so adapter starts at identity contribution
        self.B = nn.Parameter(torch.zeros(d_out, rank))

    def forward(self, x: Tensor) -> Tensor:
        """Compute LoRA contribution.

        Args:
            x: Input tensor of shape (B, d_in).

        Returns:
            Output tensor of shape (B, d_out).
        """
        # effective_weight: (d_out, d_in) = (alpha/rank) * B @ A
        scale = self.alpha / self.rank
        effective_weight = scale * (self.B @ self.A)  # (d_out, d_in)
        return x @ effective_weight.T  # (B, d_out)

    def get_adapter_params(self) -> Dict[str, Tensor]:
        """Return cloned copies of A and B.

        Returns:
            Dict with keys 'A' and 'B'.
        """
        return {"A": self.A.data.clone(), "B": self.B.data.clone()}

    def set_adapter_params(self, params: Dict[str, Tensor]) -> None:
        """Load A and B from a params dict.

        Args:
            params: Dict with keys 'A' and 'B'.
        """
        with torch.no_grad():
            self.A.copy_(params["A"])
            self.B.copy_(params["B"])

    def communication_bytes(self) -> int:
        """Return the number of bytes needed to communicate this adapter.

        Returns:
            (rank * d_in + d_out * rank) * 4  (float32 = 4 bytes)
        """
        return (self.rank * self.d_in + self.d_out * self.rank) * 4


# ---------------------------------------------------------------------------
# FedAggregator
# ---------------------------------------------------------------------------


class FedAggregator:
    """Server-side aggregation for federated learning.

    Args:
        n_clients: Total number of clients in the federation.
    """

    def __init__(self, n_clients: int) -> None:
        self.n_clients = n_clients

    def fedavg(
        self,
        client_states: List[Dict[str, Tensor]],
        weights: Optional[List[float]] = None,
    ) -> Dict[str, Tensor]:
        """Weighted average of client state dicts.

        Args:
            client_states: List of state dicts from each participating client.
            weights: Per-client weights. Defaults to uniform 1/n.

        Returns:
            Averaged state dict.
        """
        n = len(client_states)
        if weights is None:
            weights = [1.0 / n] * n
        else:
            total = sum(weights)
            weights = [w / total for w in weights]

        averaged: Dict[str, Tensor] = {}
        for key in client_states[0]:
            averaged[key] = sum(
                w * state[key].float() for w, state in zip(weights, client_states)
            )
        return averaged

    def fedprox_aggregate(
        self,
        global_state: Dict[str, Tensor],
        client_deltas: List[Dict[str, Tensor]],
        weights: Optional[List[float]] = None,
    ) -> Dict[str, Tensor]:
        """Aggregate client deltas onto the global model.

        new_global = global + weighted_mean(deltas)

        Args:
            global_state: Current global model state dict.
            client_deltas: List of delta dicts (current - global) per client.
            weights: Per-client weights. Defaults to uniform 1/n.

        Returns:
            New global state dict.
        """
        n = len(client_deltas)
        if weights is None:
            weights = [1.0 / n] * n
        else:
            total = sum(weights)
            weights = [w / total for w in weights]

        new_state: Dict[str, Tensor] = {}
        for key in global_state:
            mean_delta = sum(
                w * delta[key].float() for w, delta in zip(weights, client_deltas)
            )
            new_state[key] = global_state[key].float() + mean_delta
        return new_state

    def aggregate_lora(
        self,
        adapter_params_list: List[Dict[str, Tensor]],
    ) -> Dict[str, Tensor]:
        """Average LoRA adapter parameters across clients.

        Args:
            adapter_params_list: List of {'A': ..., 'B': ...} dicts from clients.

        Returns:
            Averaged {'A': ..., 'B': ...} dict.
        """
        n = len(adapter_params_list)
        avg_A = sum(p["A"].float() for p in adapter_params_list) / n
        avg_B = sum(p["B"].float() for p in adapter_params_list) / n
        return {"A": avg_A, "B": avg_B}
