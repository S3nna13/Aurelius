"""
model_merging_v2.py — Model Merging: SLERP, TIES, DARE, and Task Arithmetic.

Supported merge methods:
  - linear        : weighted average of parameters
  - slerp         : spherical linear interpolation (two models)
  - ties          : TIES merging via trim + majority-sign election
  - dare          : DARE random-masking of task vectors
  - task_arithmetic: task vector arithmetic (add scaled deltas to base)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MergeConfig:
    """Configuration for model merging operations."""

    merge_method: str = "linear"
    """One of: 'linear', 'slerp', 'ties', 'dare', 'task_arithmetic'."""

    alpha: float = 0.5
    """Interpolation weight / task-vector scale factor."""

    dare_density: float = 0.9
    """Fraction of delta elements to keep in DARE pruning."""

    ties_k: float = 0.2
    """Top-k fraction to keep during TIES trimming (by magnitude)."""


# ---------------------------------------------------------------------------
# Linear merging
# ---------------------------------------------------------------------------

class LinearMerge:
    """Weighted (or uniform) linear average of model state dicts."""

    def merge(
        self,
        state_dicts: List[Dict[str, Tensor]],
        weights: Optional[List[float]] = None,
    ) -> Dict[str, Tensor]:
        """Weighted average of *state_dicts*.

        Args:
            state_dicts: List of state dicts with identical keys.
            weights: Optional per-model weights; must sum to 1 (or will be
                     normalised).  Defaults to uniform weights.

        Returns:
            A new state dict whose tensors are the weighted average.
        """
        if not state_dicts:
            raise ValueError("state_dicts must be non-empty")

        n = len(state_dicts)
        if weights is None:
            weights = [1.0 / n] * n
        else:
            if len(weights) != n:
                raise ValueError("len(weights) must equal len(state_dicts)")
            total = sum(weights)
            weights = [w / total for w in weights]

        keys = list(state_dicts[0].keys())
        result: Dict[str, Tensor] = {}
        for key in keys:
            stacked = torch.stack(
                [sd[key].float() * w for sd, w in zip(state_dicts, weights)]
            )
            result[key] = stacked.sum(dim=0).to(state_dicts[0][key].dtype)
        return result

    def merge_two(
        self,
        sd_a: Dict[str, Tensor],
        sd_b: Dict[str, Tensor],
        alpha: float = 0.5,
    ) -> Dict[str, Tensor]:
        """Interpolate two state dicts: ``(1 - alpha) * A + alpha * B``.

        Args:
            sd_a: First state dict (weight 1 - alpha).
            sd_b: Second state dict (weight alpha).
            alpha: Blending factor in [0, 1].

        Returns:
            Merged state dict.
        """
        result: Dict[str, Tensor] = {}
        for key in sd_a:
            a = sd_a[key].float()
            b = sd_b[key].float()
            result[key] = ((1.0 - alpha) * a + alpha * b).to(sd_a[key].dtype)
        return result


# ---------------------------------------------------------------------------
# SLERP merging
# ---------------------------------------------------------------------------

class SLERPMerge:
    """Spherical linear interpolation between two model state dicts."""

    def slerp(self, v0: Tensor, v1: Tensor, t: float) -> Tensor:
        """Spherical linear interpolation between two flat vectors.

        Formula: ``v0 * sin((1-t)*Ω) / sin(Ω) + v1 * sin(t*Ω) / sin(Ω)``
        where ``Ω = arccos(dot(v0_hat, v1_hat))``.

        Falls back to linear interpolation when ``sin(Ω) < 1e-6`` (vectors
        nearly parallel or anti-parallel).

        Args:
            v0: Starting vector (arbitrary shape, flattened internally).
            v1: Ending vector (same shape as v0).
            t: Interpolation parameter in [0, 1].

        Returns:
            Interpolated vector with the same shape as v0.
        """
        orig_shape = v0.shape
        v0_f = v0.float().flatten()
        v1_f = v1.float().flatten()

        # Normalise for angle calculation
        norm0 = v0_f.norm()
        norm1 = v1_f.norm()

        # Handle zero vectors — fall back to linear
        if norm0 < 1e-10 or norm1 < 1e-10:
            return ((1.0 - t) * v0_f + t * v1_f).reshape(orig_shape).to(v0.dtype)

        v0_hat = v0_f / norm0
        v1_hat = v1_f / norm1

        dot = torch.clamp(torch.dot(v0_hat, v1_hat), -1.0, 1.0)
        omega = torch.acos(dot)
        sin_omega = torch.sin(omega)

        if sin_omega.abs().item() < 1e-6:
            # Nearly parallel — linear fallback
            result = (1.0 - t) * v0_f + t * v1_f
        else:
            coeff0 = torch.sin((1.0 - t) * omega) / sin_omega
            coeff1 = torch.sin(t * omega) / sin_omega
            result = coeff0 * v0_f + coeff1 * v1_f

        return result.reshape(orig_shape).to(v0.dtype)

    def merge(
        self,
        sd_a: Dict[str, Tensor],
        sd_b: Dict[str, Tensor],
        t: float = 0.5,
    ) -> Dict[str, Tensor]:
        """SLERP-merge two state dicts at interpolation factor *t*.

        Each parameter tensor is flattened, SLERPed, then reshaped.

        Args:
            sd_a: State dict at t=0.
            sd_b: State dict at t=1.
            t: Interpolation factor in [0, 1].

        Returns:
            Merged state dict.
        """
        result: Dict[str, Tensor] = {}
        for key in sd_a:
            result[key] = self.slerp(sd_a[key], sd_b[key], t)
        return result


# ---------------------------------------------------------------------------
# TIES merging
# ---------------------------------------------------------------------------

class TIESMerge:
    """TIES (Trim, Elect Sign & Merge) model merging.

    Reference: Yadav et al., "TIES-Merging: Resolving Interference When
    Merging Models", NeurIPS 2023.
    """

    def trim_delta(self, delta: Tensor, k: float) -> Tensor:
        """Keep only the top-k fraction of *delta* by absolute magnitude.

        Elements below the k-th percentile threshold are zeroed out.

        Args:
            delta: Task vector tensor (arbitrary shape).
            k: Fraction to keep, e.g. 0.2 keeps the top-20 % by magnitude.

        Returns:
            Trimmed delta tensor with the same shape.
        """
        flat = delta.float().flatten()
        n_keep = max(1, int(math.ceil(k * flat.numel())))
        if n_keep >= flat.numel():
            return delta.clone()

        threshold = torch.topk(flat.abs(), n_keep, largest=True).values.min()
        mask = flat.abs() >= threshold
        trimmed = flat * mask.float()
        return trimmed.reshape(delta.shape).to(delta.dtype)

    def resolve_signs(self, deltas: List[Tensor]) -> Tensor:
        """Compute the majority sign per element across *deltas*.

        Sign is determined by ``sign(sum_of_signs)``.  Ties (sum == 0) yield
        sign +1 (torch.sign returns 0 for exact ties; we map 0 → +1).

        Args:
            deltas: List of task-vector tensors with identical shape.

        Returns:
            Tensor of {-1, +1} with the same shape.
        """
        stacked = torch.stack([d.float() for d in deltas], dim=0)  # (M, ...)
        sign_sum = stacked.sign().sum(dim=0)
        majority = sign_sum.sign()
        # Replace 0 (tie) with +1
        majority[majority == 0] = 1
        return majority.to(deltas[0].dtype)

    def merge(
        self,
        base: Dict[str, Tensor],
        fine_tuned_list: List[Dict[str, Tensor]],
        k: float = 0.2,
    ) -> Dict[str, Tensor]:
        """TIES-merge a list of fine-tuned models into *base*.

        Steps per parameter:
        1. Compute deltas (FT - base).
        2. Trim each delta (keep top-k by magnitude).
        3. Resolve signs via majority vote.
        4. Mask trimmed deltas to the majority sign, then average.
        5. Add averaged delta back to base.

        Args:
            base: Base model state dict.
            fine_tuned_list: List of fine-tuned state dicts.
            k: Top-k fraction for trimming.

        Returns:
            Merged state dict.
        """
        result: Dict[str, Tensor] = {}
        for key in base:
            base_param = base[key].float()
            deltas = [sd[key].float() - base_param for sd in fine_tuned_list]

            trimmed = [self.trim_delta(d, k) for d in deltas]
            majority_sign = self.resolve_signs(trimmed)

            # Keep only elements that agree with majority sign
            agreed = [
                t * (t.sign() == majority_sign).float() for t in trimmed
            ]
            # Average (ignoring zeros) — sum / count_of_agreements per element
            stacked = torch.stack(agreed, dim=0)
            counts = (stacked != 0).float().sum(dim=0).clamp(min=1)
            mean_delta = stacked.sum(dim=0) / counts

            merged = base_param + mean_delta
            result[key] = merged.to(base[key].dtype)
        return result


# ---------------------------------------------------------------------------
# DARE merging
# ---------------------------------------------------------------------------

class DAREMerge:
    """DARE (Drop And REscale) model merging.

    Reference: Yu et al., "Language Models are Super Mario: Absorbing Abilities
    from Homologous Models as a Free Lunch", 2023.
    """

    def dare_prune(self, delta: Tensor, density: float) -> Tensor:
        """Randomly prune *delta*, keeping each element with probability *density*.

        Surviving elements are rescaled by ``1 / density`` to preserve the
        expected value.

        Args:
            delta: Task vector tensor.
            density: Probability of keeping each element (0 < density ≤ 1).

        Returns:
            Pruned and rescaled delta with the same shape.
        """
        if density >= 1.0:
            return delta.clone()
        mask = torch.bernoulli(
            torch.full(delta.shape, density, dtype=torch.float32)
        )
        return (delta.float() * mask / density).to(delta.dtype)

    def merge(
        self,
        base: Dict[str, Tensor],
        fine_tuned: Dict[str, Tensor],
        density: float = 0.9,
        alpha: float = 0.5,
    ) -> Dict[str, Tensor]:
        """DARE-merge a single fine-tuned model into *base*.

        Formula: ``new_params = base + alpha * dare_prune(FT - base, density)``

        Args:
            base: Base model state dict.
            fine_tuned: Fine-tuned model state dict.
            density: Fraction of delta elements to keep.
            alpha: Scale factor applied to the pruned delta.

        Returns:
            Merged state dict.
        """
        result: Dict[str, Tensor] = {}
        for key in base:
            base_param = base[key].float()
            delta = fine_tuned[key].float() - base_param
            pruned = self.dare_prune(delta, density)
            merged = base_param + alpha * pruned.float()
            result[key] = merged.to(base[key].dtype)
        return result


# ---------------------------------------------------------------------------
# Task Arithmetic merging
# ---------------------------------------------------------------------------

class TaskArithmeticMerge:
    """Task Arithmetic model merging.

    Reference: Ilharco et al., "Editing Models with Task Arithmetic", ICLR 2023.
    """

    def merge(
        self,
        base: Dict[str, Tensor],
        fine_tuned_list: List[Dict[str, Tensor]],
        alpha: float = 0.5,
    ) -> Dict[str, Tensor]:
        """Add scaled task vectors to *base*.

        Formula: ``new_params = base + alpha * sum(FT_i - base)``

        Args:
            base: Base model state dict.
            fine_tuned_list: List of fine-tuned state dicts.
            alpha: Scale factor for the summed task vector.

        Returns:
            Merged state dict.
        """
        result: Dict[str, Tensor] = {}
        for key in base:
            base_param = base[key].float()
            task_vector_sum = sum(
                sd[key].float() - base_param for sd in fine_tuned_list
            )
            merged = base_param + alpha * task_vector_sum
            result[key] = merged.to(base[key].dtype)
        return result


# ---------------------------------------------------------------------------
# Facade: ModelMerger
# ---------------------------------------------------------------------------

class ModelMerger:
    """High-level facade that dispatches to the appropriate merge strategy.

    Args:
        config: A :class:`MergeConfig` instance controlling which method to use
                and its hyperparameters.
    """

    def __init__(self, config: MergeConfig) -> None:
        self.config = config
        self._linear = LinearMerge()
        self._slerp = SLERPMerge()
        self._ties = TIESMerge()
        self._dare = DAREMerge()
        self._task_arith = TaskArithmeticMerge()

    def merge(
        self,
        base_sd: Dict[str, Tensor],
        *fine_tuned_sds: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """Merge *fine_tuned_sds* into *base_sd* using the configured method.

        Args:
            base_sd: Base model state dict (used as reference for TIES/DARE/
                     task_arithmetic; treated as the first model for
                     linear/slerp).
            *fine_tuned_sds: One or more fine-tuned state dicts.

        Returns:
            Merged state dict.

        Raises:
            ValueError: If the merge method is unknown or no fine-tuned models
                        are provided.
        """
        if not fine_tuned_sds:
            raise ValueError("At least one fine-tuned state dict is required.")

        method = self.config.merge_method
        cfg = self.config

        if method == "linear":
            all_sds = [base_sd] + list(fine_tuned_sds)
            return self._linear.merge(all_sds)

        elif method == "slerp":
            # SLERP is defined for two models; use first fine-tuned
            ft = fine_tuned_sds[0]
            return self._slerp.merge(base_sd, ft, t=cfg.alpha)

        elif method == "ties":
            return self._ties.merge(base_sd, list(fine_tuned_sds), k=cfg.ties_k)

        elif method == "dare":
            ft = fine_tuned_sds[0]
            return self._dare.merge(
                base_sd, ft, density=cfg.dare_density, alpha=cfg.alpha
            )

        elif method == "task_arithmetic":
            return self._task_arith.merge(
                base_sd, list(fine_tuned_sds), alpha=cfg.alpha
            )

        else:
            raise ValueError(
                f"Unknown merge_method '{method}'. "
                "Choose from: linear, slerp, ties, dare, task_arithmetic."
            )
