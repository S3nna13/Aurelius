"""
src/interpretability/distributed_alignment_search.py

Distributed Alignment Search (DAS) — arXiv:2305.08809
"Finding Alignments Between Interpretable Causal Variables and
 Distributed Representations"

DAS finds which *directions* (a subspace) in a model's hidden states
correspond to interpretable causal variables.  It does this by learning
an orthogonal rotation R of the residual stream such that a fixed
low-dimensional prefix of the rotated space carries a known causal
variable.  Given matched (base, source) pairs and a causal label, we
perform an "interchange intervention":

  1. Rotate base hidden states h_b and source hidden states h_s.
  2. Copy the first n_directions dimensions of rotated h_s into h_b.
  3. Rotate back.
  4. Measure whether the model produces the source-label outcome.
  5. Gradient-descend on R while maintaining orthogonality.

Orthogonality is enforced after each gradient step via QR decomposition
(Gram-Schmidt projection onto the Stiefel manifold).

Pure PyTorch — no HuggingFace, einops, scipy, or sklearn.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# DASConfig
# ---------------------------------------------------------------------------


@dataclass
class DASConfig:
    """Hyper-parameters for a DAS training run."""

    n_variables: int = 1
    """Number of distinct causal variables to find."""

    n_directions: int = 1
    """Dimensionality of the subspace assigned to each variable."""

    lr: float = 1e-3
    """Learning rate for the rotation optimiser."""

    n_steps: int = 100
    """Number of gradient steps."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_orthogonal(matrix: Tensor) -> Tensor:
    """Project *matrix* onto the Stiefel manifold via QR decomposition.

    Given a square or tall matrix M, returns Q from M = QR where Q has
    orthonormal columns (Q^T Q = I).

    Args:
        matrix: (..., m, n) or (n, n) real tensor.

    Returns:
        Q with the same shape as *matrix* and orthonormal columns.
    """
    Q, _ = torch.linalg.qr(matrix)
    return Q


# ---------------------------------------------------------------------------
# DistributedAlignmentSearch
# ---------------------------------------------------------------------------


class DistributedAlignmentSearch(nn.Module):
    """Learn an orthogonal rotation that aligns causal variables to subspaces.

    Parameters
    ----------
    d_model:
        Dimensionality of the hidden states.
    config:
        DAS hyper-parameters.  Defaults to ``DASConfig()``.
    """

    def __init__(self, d_model: int, config: DASConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = DASConfig()
        self.config = config
        self.d_model = d_model

        # R is stored as a raw (unconstrained) parameter; orthogonality is
        # imposed explicitly after each gradient step.
        R_init = torch.eye(d_model)
        self.R = nn.Parameter(R_init.clone())  # (d_model, d_model)

    # ------------------------------------------------------------------
    # Rotation helpers
    # ------------------------------------------------------------------

    def _get_R(self) -> Tensor:
        """Return current orthogonal rotation (re-orthogonalised on the fly)."""
        return make_orthogonal(self.R)

    def rotate(self, h: Tensor) -> Tensor:
        """Apply the learned rotation to hidden states.

        Args:
            h: (..., d_model) tensor of hidden states.

        Returns:
            Rotated hidden states with the same shape as *h*.
        """
        R = self._get_R()  # (d, d)
        # h @ R^T  — works for any leading batch dimensions.
        return h @ R.T

    def rotate_back(self, h_rot: Tensor) -> Tensor:
        """Apply the inverse (transpose) of R to rotated hidden states.

        Args:
            h_rot: (..., d_model) rotated hidden states.

        Returns:
            Un-rotated hidden states.
        """
        R = self._get_R()  # (d, d)
        return h_rot @ R

    # ------------------------------------------------------------------
    # Interchange intervention
    # ------------------------------------------------------------------

    def interchange_intervention(
        self,
        base_h: Tensor,
        source_h: Tensor,
        variable_idx: int = 0,
    ) -> Tensor:
        """Swap the subspace for *variable_idx* from source into base.

        The subspace for variable ``k`` occupies columns
        ``[k * n_directions : (k+1) * n_directions]`` of the rotated
        representation.

        Args:
            base_h:       (B, d_model) hidden states to modify.
            source_h:     (B, d_model) source of the causal variable.
            variable_idx: which causal variable's subspace to swap.

        Returns:
            Intervened hidden states with shape (B, d_model).
        """
        nd = self.config.n_directions
        start = variable_idx * nd
        end = start + nd

        # Rotate both into the aligned basis.
        base_rot = self.rotate(base_h)  # (B, d_model)
        source_rot = self.rotate(source_h)  # (B, d_model)

        # Copy the causal-variable slice from source into base.
        intervened_rot = base_rot.clone()
        intervened_rot[:, start:end] = source_rot[:, start:end]

        # Rotate back to the original basis.
        return self.rotate_back(intervened_rot)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        base_inputs: Tensor,
        source_inputs: Tensor,
        labels: Tensor,
        metric_fn: Callable[[Tensor, Tensor], Tensor],
    ) -> list[float]:
        """Train the rotation to maximise causal alignment.

        At each step:
          1. Perform the interchange intervention.
          2. Compute the loss via *metric_fn*.
          3. Gradient-descend on R.
          4. Re-orthogonalise R via QR.

        Args:
            base_inputs:   (N, d_model) base hidden states.
            source_inputs: (N, d_model) source hidden states.
            labels:        (N,) causal labels (float or long).
            metric_fn:     ``(intervened_h, labels) -> scalar loss``.
                           The loss should be *minimised* (e.g. cross-entropy).

        Returns:
            List of per-step loss values (length == config.n_steps).
        """
        optimizer = torch.optim.Adam([self.R], lr=self.config.lr)
        loss_history: list[float] = []

        for _ in range(self.config.n_steps):
            optimizer.zero_grad()

            intervened = self.interchange_intervention(base_inputs, source_inputs)
            loss = metric_fn(intervened, labels)
            loss.backward()
            optimizer.step()

            # Re-project R onto SO(d_model) after the gradient step.
            with torch.no_grad():
                Q = make_orthogonal(self.R.data)
                self.R.data.copy_(Q)

            loss_history.append(float(loss.item()))

        return loss_history
