"""
Spectral Norm Regularizer
=========================
Penalizes the largest singular value (spectral norm) of weight matrices to
improve training stability and generalization.  Applied to Linear (and
optionally Embedding) layers in a module.

Algorithm
---------
Power-iteration estimate of the top singular value (avoids a full SVD):

    u  ← W^T v / ‖W^T v‖
    v  ← W u  / ‖W u‖
    σ ≈ v^T W u

Running k iterations trades accuracy against compute.

Registered under TRAINING_REGISTRY["spectral_norm_reg"].
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SpectralNormConfig:
    """Hyper-parameters for SpectralNormRegularizer."""

    penalty_weight: float = 0.001   # λ — scales the spectral penalty in loss
    power_iterations: int = 1       # k — number of power-iteration steps
    target_sigma: float = 1.0       # desired maximum singular value
    apply_to_linear: bool = True    # apply to nn.Linear weights
    apply_to_embedding: bool = False  # apply to nn.Embedding weights


# ---------------------------------------------------------------------------
# SpectralNormEstimator
# ---------------------------------------------------------------------------

class SpectralNormEstimator:
    """
    Estimates the spectral norm (largest singular value) of a weight matrix
    via power iteration.

    Parameters
    ----------
    config : SpectralNormConfig
        Hyper-parameter bundle (only ``power_iterations`` is used here).
    """

    def __init__(self, config: SpectralNormConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _power_iteration(self, W: Tensor, k: int) -> Tensor:
        """
        Estimate top singular value of W ∈ R^{out × in} via k steps of
        power iteration.

        Returns
        -------
        Tensor
            Scalar estimate of σ_max(W).
        """
        out_dim, in_dim = W.shape
        # Initialise unit vectors on the correct device/dtype.
        v = F.normalize(torch.randn(out_dim, device=W.device, dtype=W.dtype), dim=0)

        for _ in range(k):
            # u = W^T v / ‖W^T v‖   [in]
            u = F.normalize(W.t().mv(v), dim=0)
            # v = W u / ‖W u‖       [out]
            v = F.normalize(W.mv(u), dim=0)

        # σ ≈ v^T W u  (scalar)
        sigma = v.dot(W.mv(u))
        return sigma

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(self, W: Tensor) -> Tensor:
        """
        Estimate σ_max for a weight tensor of any rank.

        The tensor is reshaped to 2-D before the power iteration:
          * 1-D  → [1, n]
          * 2-D  → unchanged
          * 3-D+ → [prod(dims[:-1]), dims[-1]]

        Parameters
        ----------
        W : Tensor
            Weight tensor (any shape).

        Returns
        -------
        Tensor
            Scalar spectral-norm estimate.
        """
        if W.ndim == 1:
            W2 = W.unsqueeze(0)          # [1, n]
        elif W.ndim == 2:
            W2 = W                       # [out, in]
        else:
            # Flatten all leading dims: [*, in] → [prod(*), in]
            W2 = W.reshape(-1, W.shape[-1])

        return self._power_iteration(W2, self.config.power_iterations)

    def exact_sigma(self, W: Tensor) -> Tensor:
        """
        Compute the exact largest singular value using ``torch.linalg.svdvals``.

        Parameters
        ----------
        W : Tensor
            Weight tensor (any shape).

        Returns
        -------
        Tensor
            Scalar — exact σ_max(W).
        """
        if W.ndim == 1:
            W2 = W.unsqueeze(0)
        else:
            W2 = W.reshape(-1, W.shape[-1])

        return torch.linalg.svdvals(W2)[0]


# ---------------------------------------------------------------------------
# SpectralNormRegularizer
# ---------------------------------------------------------------------------

class SpectralNormRegularizer:
    """
    Computes a one-sided L2 penalty on the spectral norm of weight matrices.

    penalty(σ) = λ · relu(σ − σ_target)²

    This is zero when σ ≤ σ_target and grows quadratically above it.

    Parameters
    ----------
    config : SpectralNormConfig
        Hyper-parameter bundle.
    """

    def __init__(self, config: SpectralNormConfig) -> None:
        self.config = config
        self.estimator = SpectralNormEstimator(config)

    # ------------------------------------------------------------------
    # Core penalty
    # ------------------------------------------------------------------

    def penalty(self, sigma: Tensor) -> Tensor:
        """
        One-sided L2 penalty: λ · relu(σ − σ_target)².

        Parameters
        ----------
        sigma : Tensor
            Scalar spectral-norm estimate.

        Returns
        -------
        Tensor
            Scalar penalty value (0 when σ ≤ target).
        """
        excess = F.relu(sigma - self.config.target_sigma)
        return self.config.penalty_weight * excess ** 2

    # ------------------------------------------------------------------
    # Module-level penalty
    # ------------------------------------------------------------------

    def module_penalty(self, module: nn.Module) -> dict[str, Tensor]:
        """
        Walk *module* and compute spectral-norm penalties for every
        qualifying weight tensor.

        Qualifying layers:
          * ``nn.Linear``     when ``config.apply_to_linear`` is True
          * ``nn.Embedding``  when ``config.apply_to_embedding`` is True

        Returns
        -------
        dict with keys:
            ``"total_penalty"``      — sum of per-layer penalties (scalar)
            ``"max_sigma"``          — largest σ across all layers (scalar)
            ``"mean_sigma"``         — mean σ across all layers (scalar)
            ``"n_layers_over_target"`` — count of layers with σ > target (int tensor)
        """
        sigmas: list[Tensor] = []
        penalties: list[Tensor] = []

        for mod in module.modules():
            weight: Tensor | None = None
            if self.config.apply_to_linear and isinstance(mod, nn.Linear):
                weight = mod.weight
            elif self.config.apply_to_embedding and isinstance(mod, nn.Embedding):
                weight = mod.weight

            if weight is not None:
                sigma = self.estimator.estimate(weight.detach() if not weight.requires_grad else weight)
                sigmas.append(sigma)
                penalties.append(self.penalty(sigma))

        if not sigmas:
            # No qualifying layers found — return zero tensors.
            zero = torch.tensor(0.0)
            return {
                "total_penalty": zero,
                "max_sigma": zero,
                "mean_sigma": zero,
                "n_layers_over_target": torch.tensor(0, dtype=torch.long),
            }

        sigma_stack = torch.stack(sigmas)          # [n_layers]
        total_penalty = torch.stack(penalties).sum()
        max_sigma = sigma_stack.max()
        mean_sigma = sigma_stack.mean()
        n_over = (sigma_stack > self.config.target_sigma).sum()

        return {
            "total_penalty": total_penalty,
            "max_sigma": max_sigma,
            "mean_sigma": mean_sigma,
            "n_layers_over_target": n_over,
        }

    # ------------------------------------------------------------------
    # Combined loss
    # ------------------------------------------------------------------

    def loss_with_penalty(
        self,
        base_loss: Tensor,
        module: nn.Module,
    ) -> dict[str, Tensor]:
        """
        Add the spectral-norm penalty to *base_loss*.

        Parameters
        ----------
        base_loss : Tensor
            Scalar task loss (e.g. cross-entropy).
        module : nn.Module
            Model whose weights are regularized.

        Returns
        -------
        dict with keys:
            ``"loss"``             — base_loss + total_penalty  (differentiable)
            ``"base_loss"``        — original base_loss (detached copy)
            ``"spectral_penalty"`` — total spectral penalty scalar
            ``"max_sigma"``        — largest σ across all layers
            ``"mean_sigma"``       — mean σ across all layers
        """
        pen_dict = self.module_penalty(module)
        total_penalty = pen_dict["total_penalty"]

        return {
            "loss": base_loss + total_penalty,
            "base_loss": base_loss.detach(),
            "spectral_penalty": total_penalty.detach(),
            "max_sigma": pen_dict["max_sigma"],
            "mean_sigma": pen_dict["mean_sigma"],
        }


# ---------------------------------------------------------------------------
# Registry registration
# ---------------------------------------------------------------------------

from src.training import TRAINING_REGISTRY  # noqa: E402

TRAINING_REGISTRY["spectral_norm_reg"] = SpectralNormRegularizer

__all__ = ["SpectralNormConfig", "SpectralNormEstimator", "SpectralNormRegularizer"]
