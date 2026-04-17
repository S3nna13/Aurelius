"""
Model merging techniques: SLERP, TIES, and DARE.

Provides methods to combine multiple fine-tuned models without additional training.
Only stdlib + torch — no third-party ML libraries.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class ParameterVector:
    """Represent a model as a flat parameter vector for arithmetic operations."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def to_vector(self) -> Tensor:
        """Return all parameters concatenated into a single 1-D tensor."""
        parts = [p.data.detach().reshape(-1) for p in self.model.parameters()]
        return torch.cat(parts)

    def from_vector(self, vector: Tensor) -> None:
        """Load parameters back from a flat vector into the model in-place."""
        offset = 0
        for p in self.model.parameters():
            numel = p.numel()
            p.data.copy_(vector[offset : offset + numel].reshape(p.shape))
            offset += numel

    def task_vector(self, base_vector: Tensor) -> Tensor:
        """Return the delta (task vector) relative to base_vector."""
        return self.to_vector() - base_vector

    def n_params(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.model.parameters())


class SLERPMerger:
    """Spherical linear interpolation between two parameter vectors."""

    def __init__(self, t: float = 0.5) -> None:
        if not (0.0 <= t <= 1.0):
            raise ValueError(f"t must be in [0, 1], got {t}")
        self.t = t

    def merge(self, v1: Tensor, v2: Tensor) -> Tensor:
        """SLERP between v1 and v2 at interpolation parameter t.

        Falls back to linear interpolation when vectors are nearly parallel.
        """
        v1_norm = v1 / (v1.norm() + 1e-12)
        v2_norm = v2 / (v2.norm() + 1e-12)

        cos_sim = torch.dot(v1_norm, v2_norm).clamp(-1.0, 1.0)

        if cos_sim.item() > 0.9999:
            # Nearly parallel — linear interpolation fallback
            return (1.0 - self.t) * v1 + self.t * v2

        theta = torch.acos(cos_sim)
        sin_theta = torch.sin(theta)

        coeff1 = torch.sin((1.0 - self.t) * theta) / sin_theta
        coeff2 = torch.sin(self.t * theta) / sin_theta
        return coeff1 * v1 + coeff2 * v2

    def merge_models(
        self,
        base_model: nn.Module,
        model_a: nn.Module,
        model_b: nn.Module,
    ) -> Tensor:
        """Merge model_a and model_b relative to base_model via SLERP on task vectors.

        Returns the merged flat parameter vector.
        """
        base_pv = ParameterVector(base_model)
        base_vec = base_pv.to_vector()

        tv_a = ParameterVector(model_a).task_vector(base_vec)
        tv_b = ParameterVector(model_b).task_vector(base_vec)

        merged_delta = self.merge(tv_a, tv_b)
        return base_vec + merged_delta


class TIESMerger:
    """Trim, Elect Sign & Merge (TIES) for combining multiple task vectors."""

    def __init__(self, top_k_fraction: float = 0.2, density: float = 0.5) -> None:
        if not (0.0 < top_k_fraction <= 1.0):
            raise ValueError("top_k_fraction must be in (0, 1]")
        self.top_k_fraction = top_k_fraction
        self.density = density

    def trim(self, task_vector: Tensor, fraction: float) -> Tensor:
        """Keep only the top-fraction parameters by absolute magnitude; zero the rest."""
        if fraction >= 1.0:
            return task_vector.clone()
        k = max(1, int(fraction * task_vector.numel()))
        threshold_idx = task_vector.abs().topk(k, sorted=False).indices
        trimmed = torch.zeros_like(task_vector)
        trimmed[threshold_idx] = task_vector[threshold_idx]
        return trimmed

    def elect_sign(self, task_vectors: list) -> Tensor:
        """Elect a unified sign per parameter via majority vote across task vectors.

        Returns a tensor with values in {-1, 0, +1}.
        """
        if not task_vectors:
            raise ValueError("task_vectors must be non-empty")
        stacked = torch.stack(task_vectors, dim=0)  # (n_models, n_params)
        sign_sum = stacked.sign().sum(dim=0)
        # +1 if majority positive, -1 if majority negative, 0 if tie
        elected = sign_sum.sign()
        return elected

    def merge(self, task_vectors: list, base_vector: Tensor) -> Tensor:
        """TIES merge: trim → elect sign → mask → average → add to base."""
        if not task_vectors:
            raise ValueError("task_vectors must be non-empty")

        # Step 1: trim each task vector
        trimmed = [self.trim(tv, self.top_k_fraction) for tv in task_vectors]

        # Step 2: elect unified sign
        sign_vec = self.elect_sign(trimmed)  # (n_params,)

        # Step 3: mask — keep only parameters that match the elected sign
        masked = []
        for tv in trimmed:
            mask = (tv.sign() == sign_vec) & (sign_vec != 0)
            m = tv.clone()
            m[~mask] = 0.0
            masked.append(m)

        # Step 4: average non-zero contributions per parameter
        stacked = torch.stack(masked, dim=0)  # (n_models, n_params)
        non_zero_count = (stacked != 0).float().sum(dim=0).clamp(min=1.0)
        averaged = stacked.sum(dim=0) / non_zero_count

        return base_vector + averaged


class DAREPruner:
    """Drop And REscale (DARE) — random pruning before merging."""

    def __init__(self, drop_rate: float = 0.9, seed: int = 42) -> None:
        if not (0.0 <= drop_rate < 1.0):
            raise ValueError("drop_rate must be in [0, 1)")
        self.drop_rate = drop_rate
        self.seed = seed

    def prune(self, task_vector: Tensor) -> Tensor:
        """Randomly zero out drop_rate fraction of parameters and rescale survivors."""
        gen = torch.Generator()
        gen.manual_seed(self.seed)

        mask = torch.bernoulli(
            torch.full(task_vector.shape, 1.0 - self.drop_rate),
            generator=gen,
        ).to(task_vector.device)

        scale = 1.0 / (1.0 - self.drop_rate)
        return task_vector * mask * scale

    def dare_merge(
        self,
        task_vectors: list,
        base_vector: Tensor,
        weights: Optional[list] = None,
    ) -> Tensor:
        """DARE prune each task vector, then weighted-average, then add to base."""
        if not task_vectors:
            raise ValueError("task_vectors must be non-empty")

        n = len(task_vectors)
        if weights is None:
            weights = [1.0 / n] * n
        else:
            if len(weights) != n:
                raise ValueError("weights length must match task_vectors length")
            total = sum(weights)
            weights = [w / total for w in weights]

        pruned = [self.prune(tv) for tv in task_vectors]
        merged_delta = sum(w * tv for w, tv in zip(weights, pruned))
        return base_vector + merged_delta


class ModelMergeEvaluator:
    """Evaluate the quality of merged models."""

    def __init__(self) -> None:
        pass

    def _cosine_sim(self, a: Tensor, b: Tensor) -> float:
        norm_a = a.norm()
        norm_b = b.norm()
        if norm_a < 1e-12 or norm_b < 1e-12:
            return 0.0
        return (torch.dot(a, b) / (norm_a * norm_b)).item()

    def parameter_interference(self, task_vectors: list) -> float:
        """Mean pairwise cosine similarity between task vectors.

        Higher value indicates less interference between fine-tuned models.
        Returns value in [-1, 1].
        """
        n = len(task_vectors)
        if n < 2:
            return 1.0

        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += self._cosine_sim(task_vectors[i], task_vectors[j])
                count += 1
        return total / count

    def sign_agreement(self, task_vectors: list) -> float:
        """Fraction of parameters where all task vectors agree on sign.

        Returns value in [0, 1].
        """
        if not task_vectors:
            return 0.0
        stacked = torch.stack(task_vectors, dim=0)  # (n, d)
        signs = stacked.sign()
        # A parameter has agreement if all signs are the same (and non-zero)
        first = signs[0]
        all_agree = (signs == first).all(dim=0) & (first != 0)
        return all_agree.float().mean().item()

    def magnitude_preservation(self, original: Tensor, merged: Tensor) -> float:
        """Cosine similarity between merged task vector and mean of originals.

        Returns value in [-1, 1]; 1.0 means perfect preservation.
        """
        return self._cosine_sim(original, merged)
