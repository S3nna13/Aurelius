"""
Task Arithmetic (Ilharco et al. 2023)
Arithmetic operations on task vectors (delta weights) to combine,
negate, and compose fine-tuned model capabilities.

Pure PyTorch only — no third-party ML libraries.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# TaskVector
# ---------------------------------------------------------------------------


class TaskVector:
    """Represents the delta between a fine-tuned and a base model.

    Can be constructed either from two models (base + fine-tuned) or
    from a pre-computed parameter dictionary.
    """

    def __init__(
        self,
        base_model: nn.Module | None = None,
        finetuned_model: nn.Module | None = None,
        vector: dict[str, Tensor] | None = None,
    ) -> None:
        if vector is not None:
            # Store clones so mutations don't bleed through
            self.vector: dict[str, Tensor] = {k: v.clone() for k, v in vector.items()}
        elif base_model is not None and finetuned_model is not None:
            base_sd = dict(base_model.named_parameters())
            ft_sd = dict(finetuned_model.named_parameters())
            self.vector = {
                name: ft_sd[name].detach().clone() - base_sd[name].detach().clone()
                for name in base_sd
                if name in ft_sd
            }
        else:
            raise ValueError("Provide either (base_model, finetuned_model) or vector.")

    # ------------------------------------------------------------------
    # Arithmetic operators
    # ------------------------------------------------------------------

    def __add__(self, other: TaskVector) -> TaskVector:
        result = {k: self.vector[k] + other.vector[k] for k in self.vector if k in other.vector}
        return TaskVector(vector=result)

    def __sub__(self, other: TaskVector) -> TaskVector:
        result = {k: self.vector[k] - other.vector[k] for k in self.vector if k in other.vector}
        return TaskVector(vector=result)

    def __mul__(self, scalar: float) -> TaskVector:
        result = {k: v * scalar for k, v in self.vector.items()}
        return TaskVector(vector=result)

    def __rmul__(self, scalar: float) -> TaskVector:
        return self.__mul__(scalar)

    def __neg__(self) -> TaskVector:
        return self.__mul__(-1.0)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def norm(self) -> float:
        """Total L2 norm across all components."""
        sq_sum = sum(v.pow(2).sum().item() for v in self.vector.values())
        return math.sqrt(sq_sum)

    def apply(self, base_model: nn.Module, scale: float = 1.0) -> None:
        """Add scale * vector to base_model parameters in-place."""
        with torch.no_grad():
            for name, param in base_model.named_parameters():
                if name in self.vector:
                    param.add_(self.vector[name] * scale)


# ---------------------------------------------------------------------------
# TaskComposer
# ---------------------------------------------------------------------------


class TaskComposer:
    """Combine multiple TaskVectors via different composition strategies."""

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Weighted sum
    # ------------------------------------------------------------------

    def sum(
        self,
        task_vectors: list[TaskVector],
        weights: list[float] | None = None,
    ) -> TaskVector:
        """Weighted sum of task vectors (default: equal weights)."""
        if not task_vectors:
            raise ValueError("task_vectors must be non-empty.")

        n = len(task_vectors)
        if weights is None:
            weights = [1.0 / n] * n

        if len(weights) != n:
            raise ValueError("len(weights) must equal len(task_vectors).")

        keys = list(task_vectors[0].vector.keys())
        result: dict[str, Tensor] = {}
        for k in keys:
            acc = torch.zeros_like(task_vectors[0].vector[k])
            for tv, w in zip(task_vectors, weights):
                if k in tv.vector:
                    acc = acc + tv.vector[k] * w
            result[k] = acc
        return TaskVector(vector=result)

    # ------------------------------------------------------------------
    # Orthogonalization via QR
    # ------------------------------------------------------------------

    def orthogonalize(self, task_vectors: list[TaskVector]) -> list[TaskVector]:
        """Remove mutual interference via QR decomposition.

        Flatten all vectors into rows of a matrix, QR-decompose, then
        separate back into individual TaskVectors with the same keys.
        """
        if not task_vectors:
            return []

        keys = list(task_vectors[0].vector.keys())
        # Build (N, D) matrix where each row is a flattened task vector
        rows = []
        for tv in task_vectors:
            flat = torch.cat([tv.vector[k].reshape(-1) for k in keys])
            rows.append(flat)

        mat = torch.stack(rows, dim=0)  # (N, D)

        # QR gives an orthonormal basis in the row space
        # torch.linalg.qr operates on columns, so transpose
        q, _ = torch.linalg.qr(mat.T)  # q: (D, min(N,D))
        ortho = q.T  # (min(N,D), D)

        # Re-scale each orthogonalized row to have the same norm as the original
        result_tvs: list[TaskVector] = []
        for i, tv in enumerate(task_vectors):
            if i >= ortho.shape[0]:
                # More vectors than dimensions — append zero vector
                result_tvs.append(
                    TaskVector(vector={k: torch.zeros_like(v) for k, v in tv.vector.items()})
                )
                continue

            row = ortho[i]
            orig_norm = rows[i].norm()
            row_norm = row.norm()
            if row_norm > 1e-12:
                row = row * (orig_norm / row_norm)

            # Split back into per-parameter tensors
            new_vector: dict[str, Tensor] = {}
            offset = 0
            for k in keys:
                numel = tv.vector[k].numel()
                new_vector[k] = row[offset : offset + numel].reshape(tv.vector[k].shape).clone()
                offset += numel
            result_tvs.append(TaskVector(vector=new_vector))

        return result_tvs

    # ------------------------------------------------------------------
    # Consensus merge
    # ------------------------------------------------------------------

    def consensus(self, task_vectors: list[TaskVector]) -> TaskVector:
        """Keep only parameters where all vectors agree on sign.

        Parameters with sign conflicts across vectors are zeroed out.
        """
        if not task_vectors:
            raise ValueError("task_vectors must be non-empty.")

        keys = list(task_vectors[0].vector.keys())
        result: dict[str, Tensor] = {}
        for k in keys:
            # Stack all values for this parameter: (N, *shape)
            stacked = torch.stack([tv.vector[k] for tv in task_vectors if k in tv.vector])
            # Sign: +1, -1, or 0
            signs = torch.sign(stacked)
            # Agreement: all signs equal the first non-zero sign
            # Practical check: product of signs along dim=0 is +1 iff all same sign,
            # 0 if any element is exactly zero, -1 if odd number of negatives.
            # We use: keep element if sign product == +1 (all agree positively or negatively).
            # For two vectors: product of signs is +1 iff both same sign.
            sign_product = signs.prod(dim=0)
            mask = (sign_product > 0).float()
            # Use mean of values where agreed
            mean_val = stacked.mean(dim=0)
            result[k] = mean_val * mask
        return TaskVector(vector=result)


# ---------------------------------------------------------------------------
# NegationOperator
# ---------------------------------------------------------------------------


class NegationOperator:
    """Negate / suppress a task vector to remove a fine-tuned capability."""

    def __init__(self, negation_scale: float = 1.0) -> None:
        self.negation_scale = negation_scale

    def negate(self, positive_vector: TaskVector) -> TaskVector:
        """Return -positive_vector * negation_scale."""
        return positive_vector * (-self.negation_scale)

    def apply_negation(
        self,
        base_model: nn.Module,
        positive_vector: TaskVector,
        scale: float = 0.5,
    ) -> None:
        """Subtract scale * positive_vector from base_model parameters."""
        with torch.no_grad():
            for name, param in base_model.named_parameters():
                if name in positive_vector.vector:
                    param.sub_(positive_vector.vector[name] * scale)


# ---------------------------------------------------------------------------
# TaskInterferenceAnalyzer
# ---------------------------------------------------------------------------


class TaskInterferenceAnalyzer:
    """Measure interference and structure between task vectors."""

    def __init__(self) -> None:
        pass

    def _flatten(self, tv: TaskVector) -> Tensor:
        """Flatten all components of a TaskVector into a 1-D tensor."""
        parts = [v.reshape(-1) for v in tv.vector.values()]
        return torch.cat(parts)

    def cosine_similarity(self, tv1: TaskVector, tv2: TaskVector) -> float:
        """Cosine similarity between two task vectors (flattened)."""
        v1 = self._flatten(tv1).float()
        v2 = self._flatten(tv2).float()
        dot = (v1 * v2).sum()
        denom = v1.norm() * v2.norm()
        if denom < 1e-12:
            return 0.0
        return (dot / denom).item()

    def interference_matrix(self, task_vectors: list[TaskVector]) -> Tensor:
        """Pairwise cosine similarity matrix, shape (N, N)."""
        n = len(task_vectors)
        mat = torch.zeros(n, n)
        for i in range(n):
            for j in range(n):
                mat[i, j] = self.cosine_similarity(task_vectors[i], task_vectors[j])
        return mat

    def dominant_parameters(self, tv: TaskVector, k: int = 10) -> list[tuple[str, float]]:
        """Top-k parameters by L2 magnitude (descending)."""
        magnitudes: list[tuple[str, float]] = []
        for name, val in tv.vector.items():
            magnitudes.append((name, val.norm().item()))
        magnitudes.sort(key=lambda x: x[1], reverse=True)
        return magnitudes[:k]


# ---------------------------------------------------------------------------
# TaskArithmeticEvaluator
# ---------------------------------------------------------------------------


class TaskArithmeticEvaluator:
    """Evaluate task arithmetic compositions against a scoring function."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate_task(
        self,
        task_vector: TaskVector,
        base_model: nn.Module,
        eval_fn: Callable[[nn.Module], float],
        scale: float = 0.8,
    ) -> float:
        """Apply task_vector to a copy of base_model, score, then restore."""
        # Work on a deep copy so the original is never mutated
        model_copy = copy.deepcopy(base_model)
        task_vector.apply(model_copy, scale=scale)
        score = eval_fn(model_copy)
        return score

    def find_optimal_scale(
        self,
        task_vector: TaskVector,
        base_model: nn.Module,
        eval_fn: Callable[[nn.Module], float],
        scales: list[float],
    ) -> float:
        """Try each scale; return the one that maximises eval_fn."""
        if not scales:
            raise ValueError("scales must be non-empty.")

        best_scale = scales[0]
        best_score = float("-inf")
        for s in scales:
            score = self.evaluate_task(task_vector, base_model, eval_fn, scale=s)
            if score > best_score:
                best_score = score
                best_scale = s
        return best_scale
