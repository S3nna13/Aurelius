"""
LEACE (Least-squares Concept Erasure) for the Aurelius LLM project.

Implements the optimal linear concept eraser from:
  "Concept Erasure as Targeted Model Editing" — arXiv:2306.03819

Variable notation follows the paper:
  x   — representation vector in R^d
  y   — binary label in {0, 1}
  μ_0, μ_1 — class-conditional means
  Σ_W — pooled within-class covariance
  w   — concept direction (Fisher discriminant in whitened space)
  P   — orthogonal projection matrix (eraser)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sym_matrix_power(A: Tensor, p: float, eps: float = 1e-6) -> Tensor:
    """Compute A^p for a symmetric PSD matrix via eigendecomposition.

    Args:
        A:   Symmetric PSD matrix of shape (d, d).
        p:   Exponent (e.g. -0.5 for inverse square root).
        eps: Floor for eigenvalues before exponentiation (numerical stability).

    Returns:
        A^p of shape (d, d).
    """
    # Symmetrise to reduce floating-point asymmetry
    A = (A + A.T) / 2.0
    eigenvalues, eigenvectors = torch.linalg.eigh(A)
    # Clamp small/negative eigenvalues
    eigenvalues = eigenvalues.clamp(min=eps)
    # A^p = V diag(λ^p) V^T
    return eigenvectors @ torch.diag(eigenvalues ** p) @ eigenvectors.T


# ---------------------------------------------------------------------------
# LeaceEraser  (batch, stateless)
# ---------------------------------------------------------------------------

@dataclass
class LeaceEraser:
    """Batch LEACE eraser: fit once on a dataset, transform new representations.

    Attributes:
        P:   Orthogonal projection matrix P = I - w wᵀ / (wᵀ w), shape (d, d).
        mu:  Overall mean μ of the training representations, shape (d,).
        w:   Concept direction in the whitened space, shape (d,).
        d:   Representation dimensionality.
    """

    P: Tensor
    mu: Tensor
    w: Tensor
    d: int

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def fit(
        cls,
        X: Tensor,
        y: Tensor,
        eps: float = 1e-6,
    ) -> "LeaceEraser":
        """Fit the LEACE eraser.

        Args:
            X:   Representations of shape (N, d).
            y:   Binary labels of shape (N,) with values in {0, 1}.
            eps: Regularisation added to Σ_W before inversion.

        Returns:
            Fitted LeaceEraser instance.
        """
        X = X.float()
        y = y.float()
        N, d = X.shape

        mask_0 = y == 0
        mask_1 = y == 1
        X_0 = X[mask_0]
        X_1 = X[mask_1]

        # Class-conditional means
        mu_0 = X_0.mean(dim=0)
        mu_1 = X_1.mean(dim=0)

        # Overall mean
        mu = X.mean(dim=0)

        # Pooled within-class covariance  Σ_W = (Σ_0 + Σ_1) / N
        def _class_scatter(X_c: Tensor, mu_c: Tensor) -> Tensor:
            diff = X_c - mu_c.unsqueeze(0)        # (N_c, d)
            return diff.T @ diff                   # (d, d)

        Sigma_W = (_class_scatter(X_0, mu_0) + _class_scatter(X_1, mu_1)) / N
        # Regularise for numerical stability
        Sigma_W = Sigma_W + eps * torch.eye(d, dtype=X.dtype, device=X.device)

        # Σ_W^{-1/2}
        Sigma_W_inv_sqrt = _sym_matrix_power(Sigma_W, -0.5, eps=eps)

        # Concept direction: w = Σ_W^{-1/2} (μ_1 − μ_0), normalised
        delta = mu_1 - mu_0
        w_raw = Sigma_W_inv_sqrt @ delta
        w = w_raw / (w_raw.norm() + 1e-12)

        # Projection P = I − w wᵀ / (wᵀ w)
        P = torch.eye(d, dtype=X.dtype, device=X.device) - torch.outer(w, w)

        return cls(P=P, mu=mu, w=w, d=d)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def transform(self, X: Tensor) -> Tensor:
        """Project out the concept direction.

        x̃ = P x + (I − P) μ   for each row x in X.

        Args:
            X: Representations of shape (N, d) or (d,).

        Returns:
            Erased representations of the same shape as X.
        """
        original_dtype = X.dtype
        X = X.float()
        squeeze = X.dim() == 1
        if squeeze:
            X = X.unsqueeze(0)

        # x̃ = P x + (I − P) μ
        # (I − P) μ is the residual mean component — constant shift
        I_minus_P = torch.eye(self.d, dtype=X.dtype, device=X.device) - self.P
        mu = self.mu.to(X.device)
        shift = (I_minus_P @ mu).unsqueeze(0)           # (1, d)
        X_erased = X @ self.P.T.to(X.device) + shift

        if squeeze:
            X_erased = X_erased.squeeze(0)
        return X_erased.to(original_dtype)


# ---------------------------------------------------------------------------
# ConceptEraser  (online, incremental — Welford-style)
# ---------------------------------------------------------------------------

class ConceptEraser(nn.Module):
    """Online LEACE eraser with Welford-style incremental updates.

    Accumulates sufficient statistics (count, mean, within-class scatter)
    and recomputes the eraser projection on demand.

    Args:
        d_model: Dimensionality of the representations.
        eps:     Covariance regularisation.
        dtype:   Storage dtype for accumulators.
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-6,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.dtype = dtype

        # Sufficient statistics — not nn.Parameters (no gradient tracking)
        self.register_buffer("n_total", torch.tensor(0, dtype=torch.long))
        self.register_buffer("n_0",     torch.tensor(0, dtype=torch.long))
        self.register_buffer("n_1",     torch.tensor(0, dtype=torch.long))
        # Running sums for means
        self.register_buffer("sum_all", torch.zeros(d_model, dtype=dtype))
        self.register_buffer("sum_0",   torch.zeros(d_model, dtype=dtype))
        self.register_buffer("sum_1",   torch.zeros(d_model, dtype=dtype))
        # Running within-class scatter matrices  M2_c = Σ (x - μ_c)(x - μ_c)^T
        # Stored as flattened (d*d,) for buffer compatibility
        self.register_buffer("M2_0", torch.zeros(d_model, d_model, dtype=dtype))
        self.register_buffer("M2_1", torch.zeros(d_model, d_model, dtype=dtype))

        # Cached eraser — invalidated on each update
        self._eraser: Optional[LeaceEraser] = None

    # ------------------------------------------------------------------
    # Online update
    # ------------------------------------------------------------------

    def update(self, X: Tensor, y: Tensor) -> None:
        """Incorporate a new batch of representations.

        Uses Chan et al. parallel Welford update for scatter matrices.

        Args:
            X: Representations (N, d_model).
            y: Binary labels (N,) in {0, 1}.
        """
        X = X.float()
        y = y.float()

        mask_0 = y == 0
        mask_1 = y == 1
        X_0 = X[mask_0]
        X_1 = X[mask_1]

        def _update_class(
            X_c: Tensor,
            n_c: Tensor,
            sum_c: Tensor,
            M2_c: Tensor,
        ) -> None:
            """In-place Welford update for one class."""
            n_new = X_c.shape[0]
            if n_new == 0:
                return
            new_sum = X_c.sum(dim=0)
            new_mean = new_sum / n_new
            old_mean = sum_c / n_c.clamp(min=1).float()

            # Scatter of new batch around its own mean
            diff_new = X_c - new_mean.unsqueeze(0)
            scatter_new = diff_new.T @ diff_new

            # Chan parallel update
            combined_n = n_c.float() + n_new
            delta_mean = new_mean - old_mean
            correction = (
                n_c.float() * n_new / combined_n
            ) * torch.outer(delta_mean, delta_mean)

            M2_c.add_(scatter_new + correction)
            sum_c.add_(new_sum)
            n_c.add_(n_new)

        _update_class(X_0, self.n_0, self.sum_0, self.M2_0)
        _update_class(X_1, self.n_1, self.sum_1, self.M2_1)
        self.sum_all.add_(X.sum(dim=0))
        self.n_total.add_(X.shape[0])

        # Invalidate cache
        self._eraser = None

    # ------------------------------------------------------------------
    # Eraser property
    # ------------------------------------------------------------------

    @property
    def eraser(self) -> LeaceEraser:
        """Return (cached) LeaceEraser built from current sufficient statistics."""
        if self._eraser is not None:
            return self._eraser

        n_total = self.n_total.item()
        if n_total == 0:
            raise RuntimeError("ConceptEraser has no data — call update() first.")

        d = self.d_model
        device = self.sum_all.device

        mu_0 = self.sum_0 / self.n_0.float().clamp(min=1)
        mu_1 = self.sum_1 / self.n_1.float().clamp(min=1)
        mu   = self.sum_all / float(n_total)

        # Pooled within-class covariance
        Sigma_W = (self.M2_0 + self.M2_1) / float(n_total)
        Sigma_W = Sigma_W + self.eps * torch.eye(d, dtype=torch.float32, device=device)

        Sigma_W_inv_sqrt = _sym_matrix_power(Sigma_W, -0.5, eps=self.eps)

        delta = mu_1 - mu_0
        w_raw = Sigma_W_inv_sqrt @ delta
        w = w_raw / (w_raw.norm() + 1e-12)

        P = torch.eye(d, dtype=torch.float32, device=device) - torch.outer(w, w)

        self._eraser = LeaceEraser(P=P, mu=mu, w=w, d=d)
        return self._eraser

    # ------------------------------------------------------------------
    # nn.Module forward
    # ------------------------------------------------------------------

    def forward(self, X: Tensor) -> Tensor:
        """Erase concept from X using current accumulated statistics.

        Args:
            X: Representations (N, d_model) or (d_model,).

        Returns:
            Erased representations of the same shape.
        """
        return self.eraser.transform(X)
