"""AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning.

Implements SVD-based LoRA with dynamic rank allocation across weight matrices
based on singular value importance scoring (Zhang et al. 2023).
"""

from __future__ import annotations

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# SVDLoRALayer
# ---------------------------------------------------------------------------


class SVDLoRALayer(nn.Module):
    """LoRA layer parameterized via SVD decomposition.

    The weight delta is factored as:
        delta_W = P @ diag(Lambda) @ Q
    scaled by alpha / rank.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha

        # SVD factors: P (out, rank), Lambda (rank,), Q (rank, in)
        self.P = nn.Parameter(torch.empty(out_features, rank))
        self.Lambda = nn.Parameter(torch.ones(rank))
        self.Q = nn.Parameter(torch.empty(rank, in_features))

        # Truncated normal init for P and Q
        nn.init.trunc_normal_(self.P, std=0.02)
        nn.init.trunc_normal_(self.Q, std=0.02)

    def _delta_W(self) -> torch.Tensor:
        """Compute delta_W = P @ diag(Lambda) @ Q, shape (out, in)."""
        # P: (out, rank), Lambda: (rank,), Q: (rank, in)
        return self.P * self.Lambda.unsqueeze(0) @ self.Q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SVD-LoRA delta to input x.

        Args:
            x: (..., in_features)

        Returns:
            Tensor of shape (..., out_features)
        """
        scale = self.alpha / self.rank
        delta_W = self._delta_W()  # (out, in)
        return scale * (x @ delta_W.T)

    def effective_rank(self, threshold: float = 1e-3) -> int:
        """Count singular values (Lambda) above threshold."""
        with torch.no_grad():
            return int((self.Lambda.abs() > threshold).sum().item())

    def importance_score(self) -> torch.Tensor:
        """Return importance scores as |Lambda| values, shape (rank,)."""
        return self.Lambda.detach().abs()


# ---------------------------------------------------------------------------
# RankBudgetAllocator
# ---------------------------------------------------------------------------


class RankBudgetAllocator:
    """Allocates rank budget across AdaLoRA layers based on importance scores."""

    def __init__(
        self,
        total_rank_budget: int,
        n_layers: int,
        initial_rank: int,
    ) -> None:
        self.total_rank_budget = total_rank_budget
        self.n_layers = n_layers
        self.initial_rank = initial_rank

    def compute_allocation(
        self,
        importance_scores: dict[str, torch.Tensor],
    ) -> dict[str, int]:
        """Compute new rank per layer proportional to importance scores.

        Args:
            importance_scores: {layer_name: (rank,) tensor of |Lambda| values}

        Returns:
            Dict mapping layer_name to new rank (int), total ≤ total_rank_budget,
            minimum 1 per layer, maximum initial_rank per layer.
        """
        layer_names = list(importance_scores.keys())
        n = len(layer_names)
        if n == 0:
            return {}

        # Compute per-layer importance as sum of scores
        layer_importance = {
            name: float(scores.sum().item()) for name, scores in importance_scores.items()
        }

        total_importance = sum(layer_importance.values())

        # Allocate budget proportionally
        if total_importance <= 0.0:
            # Uniform split
            base = max(1, self.total_rank_budget // n)
            allocation = {name: min(base, self.initial_rank) for name in layer_names}
        else:
            raw = {
                name: (layer_importance[name] / total_importance) * self.total_rank_budget
                for name in layer_names
            }
            # Floor, clamp to [1, initial_rank]
            allocation = {
                name: max(1, min(self.initial_rank, int(raw[name]))) for name in layer_names
            }

        # Enforce global budget: greedily reduce layers with most rank
        current_total = sum(allocation.values())
        while current_total > self.total_rank_budget:
            # Reduce the layer with the highest allocation (that's above 1)
            candidates = [(name, r) for name, r in allocation.items() if r > 1]
            if not candidates:
                break
            max_name = max(candidates, key=lambda kv: kv[1])[0]
            allocation[max_name] -= 1
            current_total -= 1

        # If we are under budget, that is fine — we never exceed it.
        return allocation

    def prune_decision(
        self,
        layer_name: str,
        new_rank: int,
        current_rank: int,
    ) -> list[int]:
        """Return indices of singular values to prune (zero out).

        Pruned positions are those with the lowest importance that push the
        active count above new_rank. We return the indices that should be
        zeroed so that only new_rank singular values remain active.

        Args:
            layer_name: name of the layer (unused here, for API symmetry)
            new_rank: target number of active singular values
            current_rank: current total rank of the layer

        Returns:
            List of indices to zero out (sorted ascending).
        """
        n_prune = max(0, current_rank - new_rank)
        if n_prune == 0:
            return []
        # Prune the last n_prune indices (assumed sorted by importance descending
        # after masking; caller is responsible for ordering).
        # In practice AdaLoRAMask uses importance ordering; here we return the
        # tail indices as the least-important ones.
        return list(range(current_rank - n_prune, current_rank))


# ---------------------------------------------------------------------------
# AdaLoRAMask
# ---------------------------------------------------------------------------


class AdaLoRAMask:
    """Manages which singular values are active for an SVDLoRALayer."""

    def __init__(self, rank: int) -> None:
        self.rank = rank
        self._mask: torch.Tensor = torch.ones(rank, dtype=torch.bool)

    def set_mask(self, active_indices: list[int]) -> None:
        """Set which ranks are active; all others will be zeroed.

        Args:
            active_indices: list of integer indices that remain active.
        """
        self._mask = torch.zeros(self.rank, dtype=torch.bool)
        for idx in active_indices:
            if 0 <= idx < self.rank:
                self._mask[idx] = True

    def apply(self, layer: SVDLoRALayer) -> None:
        """Zero out Lambda at masked (inactive) positions in-place, no grad."""
        with torch.no_grad():
            layer.Lambda.data[~self._mask] = 0.0

    def active_count(self) -> int:
        """Return the number of active singular values."""
        return int(self._mask.sum().item())

    @property
    def mask(self) -> torch.Tensor:
        """Bool tensor of shape (rank,): True = active."""
        return self._mask


# ---------------------------------------------------------------------------
# AdaLoRARegularizer
# ---------------------------------------------------------------------------


class AdaLoRARegularizer:
    """Orthogonality regularization for P and Q matrices."""

    def __init__(self, beta: float = 0.001) -> None:
        self.beta = beta

    def loss(self, layer: SVDLoRALayer) -> torch.Tensor:
        """Compute orthogonality penalty for a single layer.

        Penalty = beta * (||P.T @ P - I||_F^2 + ||Q @ Q.T - I||_F^2)

        Args:
            layer: SVDLoRALayer

        Returns:
            Scalar tensor.
        """
        rank = layer.rank

        # P: (out, rank)  →  P.T @ P should be (rank, rank) identity
        PtP = layer.P.T @ layer.P
        I_r = torch.eye(rank, device=layer.P.device, dtype=layer.P.dtype)
        p_penalty = (PtP - I_r).pow(2).sum()

        # Q: (rank, in)  →  Q @ Q.T should be (rank, rank) identity
        QQt = layer.Q @ layer.Q.T
        q_penalty = (QQt - I_r).pow(2).sum()

        return self.beta * (p_penalty + q_penalty)

    def total_loss(self, lora_layers: dict[str, SVDLoRALayer]) -> torch.Tensor:
        """Sum orthogonality penalty over all layers.

        Args:
            lora_layers: {name: SVDLoRALayer}

        Returns:
            Scalar tensor.
        """
        total = None
        for layer in lora_layers.values():
            lo = self.loss(layer)
            total = lo if total is None else total + lo
        if total is None:
            return torch.tensor(0.0)
        return total


# ---------------------------------------------------------------------------
# AdaLoRATrainer
# ---------------------------------------------------------------------------


class AdaLoRATrainer:
    """Training wrapper with adaptive rank updates for SVDLoRA layers."""

    def __init__(
        self,
        model: nn.Module,
        lora_layers: dict[str, SVDLoRALayer],
        optimizer: torch.optim.Optimizer,
        rank_update_interval: int = 10,
        total_rank_budget: int = 32,
    ) -> None:
        self.model = model
        self.lora_layers = lora_layers
        self.optimizer = optimizer
        self.rank_update_interval = rank_update_interval
        self.total_rank_budget = total_rank_budget

        self._step = 0
        self._masks: dict[str, AdaLoRAMask] = {
            name: AdaLoRAMask(layer.rank) for name, layer in lora_layers.items()
        }
        # Initial mask: all active
        for name, mask in self._masks.items():
            mask.set_mask(list(range(lora_layers[name].rank)))

        # Initial rank per layer = layer.rank
        initial_rank = next(iter(lora_layers.values())).rank if lora_layers else 1
        self._allocator = RankBudgetAllocator(
            total_rank_budget=total_rank_budget,
            n_layers=len(lora_layers),
            initial_rank=initial_rank,
        )

    def train_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict:
        """Run one training step.

        Args:
            input_ids: (B, T) long tensor
            labels: (B, T) long tensor

        Returns:
            dict with keys: loss, total_active_rank, per_layer_ranks
        """
        self.model.train()
        self.optimizer.zero_grad()

        # model: input_ids -> logits (B, T, V)
        logits = self.model(input_ids)
        B, T, V = logits.shape

        # CE loss over all positions
        loss = nn.functional.cross_entropy(
            logits.reshape(B * T, V),
            labels.reshape(B * T),
        )
        loss.backward()
        self.optimizer.step()

        self._step += 1
        if self._step % self.rank_update_interval == 0:
            self.update_ranks()

        per_layer_ranks = {name: self._masks[name].active_count() for name in self.lora_layers}
        total_active_rank = sum(per_layer_ranks.values())

        return {
            "loss": loss.item(),
            "total_active_rank": total_active_rank,
            "per_layer_ranks": per_layer_ranks,
        }

    def update_ranks(self) -> None:
        """Reallocate rank budget based on current importance scores."""
        importance_scores = {
            name: layer.importance_score() for name, layer in self.lora_layers.items()
        }
        new_allocation = self._allocator.compute_allocation(importance_scores)

        for name, layer in self.lora_layers.items():
            new_rank = new_allocation.get(name, 1)

            # Determine active indices: keep top-new_rank by importance
            scores = importance_scores[name]  # (rank,)
            sorted_indices = torch.argsort(scores, descending=True).tolist()
            active_indices = sorted_indices[:new_rank]

            self._masks[name].set_mask(active_indices)
            self._masks[name].apply(layer)
