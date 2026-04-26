"""Neural network pruning analysis tools.

Implements magnitude pruning, gradient sensitivity analysis, and
post-pruning quality measurement. Distinct from pruning.py (basic iterative)
and structured_pruning.py (structured sparsity).

References:
    Han et al. 2015 (Deep Compression) — https://arxiv.org/abs/1510.00149
    Frankle & Carlin 2019 (Lottery Ticket) — https://arxiv.org/abs/1803.03635
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# SparsityStats
# ---------------------------------------------------------------------------


class SparsityStats:
    """Measures the fraction of zero-valued weights in a model or tensor."""

    def __init__(self) -> None:
        pass

    def layer_sparsity(self, param: Tensor) -> float:
        """Return fraction of zero elements in *param*."""
        total = param.numel()
        if total == 0:
            return 0.0
        zeros = (param == 0).sum().item()
        return zeros / total

    def model_sparsity(self, model: nn.Module) -> dict[str, float]:
        """Return ``{name: sparsity_fraction}`` for every named parameter."""
        return {name: self.layer_sparsity(param.data) for name, param in model.named_parameters()}

    def total_sparsity(self, model: nn.Module) -> float:
        """Overall fraction of zero parameters across all weights."""
        total_zeros = 0
        total_params = 0
        for param in model.parameters():
            total_zeros += (param.data == 0).sum().item()
            total_params += param.data.numel()
        if total_params == 0:
            return 0.0
        return total_zeros / total_params


# ---------------------------------------------------------------------------
# MagnitudePruner
# ---------------------------------------------------------------------------


class MagnitudePruner:
    """Prunes the smallest-magnitude weights to achieve a target sparsity."""

    def __init__(self, sparsity: float = 0.5) -> None:
        if not 0.0 <= sparsity <= 1.0:
            raise ValueError(f"sparsity must be in [0, 1], got {sparsity}")
        self.sparsity = sparsity

    def prune_tensor(self, param: Tensor) -> Tensor:
        """Return a new tensor with the bottom-*sparsity* fraction zeroed out."""
        result = param.clone()
        n = result.numel()
        if n == 0 or self.sparsity == 0.0:
            return result

        k = int(math.floor(self.sparsity * n))
        if k == 0:
            return result

        flat = result.view(-1).abs()
        threshold = flat.kthvalue(k).values
        mask = param.abs() > threshold
        result = result * mask
        return result

    def prune_layer(self, param: nn.Parameter, mask_only: bool = False) -> Tensor:
        """Prune a single parameter layer.

        Args:
            param: The parameter to prune.
            mask_only: If True, return the boolean keep-mask without modifying
                *param.data*.  Otherwise zero *param.data* in-place where the
                mask is False and return the mask.

        Returns:
            Boolean tensor of shape ``param.shape`` where True = keep.
        """
        n = param.data.numel()
        if n == 0 or self.sparsity == 0.0:
            mask = torch.ones_like(param.data, dtype=torch.bool)
            return mask

        k = int(math.floor(self.sparsity * n))
        flat_abs = param.data.view(-1).abs()

        if k == 0:
            mask = torch.ones_like(param.data, dtype=torch.bool)
        else:
            threshold = flat_abs.kthvalue(k).values
            mask = param.data.abs() > threshold

        if not mask_only:
            param.data[~mask] = 0.0

        return mask

    def prune_model(self, model: nn.Module) -> dict[str, Tensor]:
        """Prune all parameters of *model*.

        Returns:
            Dict mapping parameter name to its boolean keep-mask.
        """
        masks: dict[str, Tensor] = {}
        for name, param in model.named_parameters():
            masks[name] = self.prune_layer(param, mask_only=False)
        return masks


# ---------------------------------------------------------------------------
# GradientSensitivityPruner
# ---------------------------------------------------------------------------


class GradientSensitivityPruner:
    """Prunes weights by gradient sensitivity: |weight * gradient|."""

    def __init__(self, sparsity: float = 0.5) -> None:
        if not 0.0 <= sparsity <= 1.0:
            raise ValueError(f"sparsity must be in [0, 1], got {sparsity}")
        self.sparsity = sparsity

    def compute_sensitivity(self, param: nn.Parameter) -> Tensor:
        """Return element-wise sensitivity ``|w * grad|``.

        Falls back to ``|w|`` when *param.grad* is None.
        """
        if param.grad is not None:
            return (param.data * param.grad).abs()
        return param.data.abs()

    def prune_by_sensitivity(self, model: nn.Module) -> dict[str, Tensor]:
        """Prune each parameter by sensitivity, return keep-masks.

        Parameters whose grad is None fall back to magnitude-based pruning.
        """
        masks: dict[str, Tensor] = {}
        for name, param in model.named_parameters():
            sensitivity = self.compute_sensitivity(param)
            n = sensitivity.numel()
            if n == 0 or self.sparsity == 0.0:
                masks[name] = torch.ones_like(param.data, dtype=torch.bool)
                continue

            k = int(math.floor(self.sparsity * n))
            if k == 0:
                mask = torch.ones_like(param.data, dtype=torch.bool)
            else:
                flat_sens = sensitivity.view(-1)
                threshold = flat_sens.kthvalue(k).values
                mask = sensitivity > threshold

            param.data[~mask] = 0.0
            masks[name] = mask
        return masks


# ---------------------------------------------------------------------------
# LotteryTicketAnalyzer
# ---------------------------------------------------------------------------


class LotteryTicketAnalyzer:
    """Implements the Lottery Ticket Hypothesis (Frankle & Carlin 2019).

    Usage::
        analyzer = LotteryTicketAnalyzer()
        ticket   = analyzer.save_initial_weights(model)
        # ... train + prune ...
        masks    = pruner.prune_model(model)
        analyzer.reset_to_ticket(model, ticket, masks)
    """

    def __init__(self) -> None:
        pass

    def save_initial_weights(self, model: nn.Module) -> dict[str, Tensor]:
        """Return ``{name: param.data.clone()}`` for all parameters."""
        return {name: param.data.clone() for name, param in model.named_parameters()}

    def reset_to_ticket(
        self,
        model: nn.Module,
        initial_weights: dict[str, Tensor],
        masks: dict[str, Tensor],
    ) -> None:
        """Reset *model* to ``initial_weights * masks`` (rewind with sparsity).

        Parameters not present in *masks* are reset to their initial weights
        without masking.
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in initial_weights:
                    w0 = initial_weights[name]
                    if name in masks:
                        param.data.copy_(w0 * masks[name].to(dtype=w0.dtype))
                    else:
                        param.data.copy_(w0)

    def ticket_similarity(
        self,
        weights1: dict[str, Tensor],
        weights2: dict[str, Tensor],
    ) -> float:
        """Cosine similarity between the flattened weight vectors.

        Only keys present in both dicts are used.

        Returns:
            Float in [-1, 1].
        """
        common_keys = sorted(set(weights1) & set(weights2))
        if not common_keys:
            return 0.0

        v1 = torch.cat([weights1[k].view(-1).float() for k in common_keys])
        v2 = torch.cat([weights2[k].view(-1).float() for k in common_keys])

        norm1 = v1.norm()
        norm2 = v2.norm()
        if norm1 == 0 or norm2 == 0:
            return 0.0

        return (v1 @ v2 / (norm1 * norm2)).item()


# ---------------------------------------------------------------------------
# PruningScheduler
# ---------------------------------------------------------------------------


class PruningScheduler:
    """Gradually increases sparsity from *initial_sparsity* to *final_sparsity*.

    Uses cubic interpolation (polynomial schedule from Zhu & Gupta 2017).

    Formula::
        s(t) = s_f + (s_i - s_f) * (1 - (t - t_0) / (t_n - t_0))^3
    """

    def __init__(
        self,
        initial_sparsity: float = 0.0,
        final_sparsity: float = 0.9,
        start_step: int = 0,
        end_step: int = 1000,
        frequency: int = 100,
    ) -> None:
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.start_step = start_step
        self.end_step = end_step
        self.frequency = frequency

    def sparsity_at_step(self, step: int) -> float:
        """Return target sparsity at *step* via cubic interpolation."""
        s_i = self.initial_sparsity
        s_f = self.final_sparsity
        t0 = self.start_step
        tn = self.end_step

        if step <= t0:
            return s_i
        if step >= tn:
            return s_f

        progress = (step - t0) / (tn - t0)
        sparsity = s_f + (s_i - s_f) * (1.0 - progress) ** 3

        # Clamp to [initial_sparsity, final_sparsity]
        lo, hi = min(s_i, s_f), max(s_i, s_f)
        return float(max(lo, min(hi, sparsity)))

    def should_prune(self, step: int) -> bool:
        """Return True if pruning should occur at *step*."""
        return step >= self.start_step and step <= self.end_step and step % self.frequency == 0
