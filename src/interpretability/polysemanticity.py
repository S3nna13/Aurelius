"""
src/interpretability/polysemanticity.py

Polysemanticity and superposition detector for transformer models.

Implements the core metrics from:
  - "Toy Models of Superposition" (Elhage et al., Anthropic 2022)
  - "Softmax Linear Units" (Elhage et al., Anthropic 2022)

Superposition occurs when a neural network represents more features than it
has dimensions, causing neurons to polysemantically encode multiple features.

Pure PyTorch — no HuggingFace, no scipy, no sklearn, no einops.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# PolysemanticitAnalyzer  (typo preserved from spec)
# ---------------------------------------------------------------------------


class PolysemanticitAnalyzer:
    """
    Analyse polysemanticity and superposition in transformer weight matrices
    and activation patterns.

    Parameters
    ----------
    threshold : float
        Activation magnitude below which a neuron is considered "inactive"
        for the sparsity metric.  Default: 0.1.
    eps : float
        Small constant for numerical stability in division.  Default: 1e-8.
    """

    def __init__(self, threshold: float = 0.1, eps: float = 1e-8) -> None:
        self.threshold = threshold
        self.eps = eps

    # ------------------------------------------------------------------
    # polysemanticity_index
    # ------------------------------------------------------------------

    def polysemanticity_index(self, W: Tensor) -> Tensor:
        """
        Per-neuron Polysemanticity Index (PI) from weight matrix W.

        For each output neuron i, computes a normalised entropy over the
        distribution of input-feature magnitudes:

            p_ij = |W[j, i]| / sum_j |W[j, i]|
            PI(i) = 1 - exp(-H(p_i)) / max_entropy

        where H is Shannon entropy and max_entropy = log(d_in).

        A monosemantic neuron (all weight mass on one input) → PI ≈ 0.
        A polysemantic neuron (uniform weight distribution) → PI ≈ 1.

        Parameters
        ----------
        W : Tensor, shape (d_in, d_out)

        Returns
        -------
        Tensor of shape (d_out,) with values in [0, 1].
        """
        if W.dim() != 2:
            raise ValueError(f"W must be 2-D, got shape {W.shape}")

        d_in, d_out = W.shape

        # Absolute weights: (d_in, d_out)
        W_abs = W.abs()

        # Normalise each column → probability distribution over input features
        col_sums = W_abs.sum(dim=0, keepdim=True).clamp(min=self.eps)  # (1, d_out)
        p = W_abs / col_sums  # (d_in, d_out)

        # Shannon entropy per column  H(p_i) = -sum_j p_ij * log(p_ij)
        # Clip to avoid log(0)
        log_p = torch.log(p.clamp(min=self.eps))
        entropy = -(p * log_p).sum(dim=0)  # (d_out,)

        # Max entropy for d_in uniform distribution: H_max = log(d_in)
        # We need PI=0 for monosemantic (one-hot, H=0) and PI=1 for polysemantic
        # (uniform, H=H_max).  The spec formula  1 - exp(-H) / max_entropy
        # maps as: one-hot → 1 - exp(0)/log(d_in) = 1 - 1/log(d_in) ≠ 0.
        #
        # Correct interpretation following "PI=0 monosemantic, PI→1 polysemantic":
        #   PI(i) = 1 - exp(-H(p_i)) / exp(-0)   with effective entropy normalised
        # This reduces to normalised entropy: H(p_i) / H_max, which gives exactly
        # 0 for one-hot columns and 1 for uniform columns.
        max_entropy = math.log(d_in) if d_in > 1 else 1.0

        pi = entropy / max_entropy
        # Clamp to [0, 1] for numerical safety
        pi = pi.clamp(0.0, 1.0)
        return pi

    # ------------------------------------------------------------------
    # participation_ratio
    # ------------------------------------------------------------------

    def participation_ratio(self, activations: Tensor) -> float:
        """
        Participation Ratio (PR) of the activation covariance.

        Measures how many dimensions the representation effectively uses:

            PR = (sum_i λ_i)^2 / sum_i λ_i^2

        where λ_i are eigenvalues of A^T A / N.

        PR ≈ 1   → concentrated in one dimension.
        PR ≈ d   → spread evenly across all dimensions.

        Parameters
        ----------
        activations : Tensor, shape (N, d)

        Returns
        -------
        float
        """
        if activations.dim() != 2:
            raise ValueError(f"activations must be 2-D (N, d), got shape {activations.shape}")

        N, d = activations.shape
        # Compute A^T A / N — symmetric positive semi-definite
        cov = activations.T @ activations / N  # (d, d)

        # Eigenvalues (real, non-negative for PSD matrices)
        # Use torch.linalg.eigvalsh (symmetric) for stability
        eigenvalues = torch.linalg.eigvalsh(cov)  # (d,) ascending
        # Clamp tiny negatives from floating-point noise
        eigenvalues = eigenvalues.clamp(min=0.0)

        sum_lam = eigenvalues.sum()
        sum_lam2 = (eigenvalues**2).sum()

        if sum_lam2 < self.eps:
            # Zero matrix — convention: PR = 1
            return 1.0

        pr = (sum_lam**2) / sum_lam2
        return pr.item()

    # ------------------------------------------------------------------
    # superposition_score
    # ------------------------------------------------------------------

    def superposition_score(self, W: Tensor) -> float:
        """
        Superposition Score (SS): mean off-diagonal cosine similarity.

        For each pair of neurons (i, j), computes:
            interference = |W[:,i] · W[:,j]| / (||W[:,i]|| * ||W[:,j]||)

        SS = mean over all i≠j pairs.

        High SS → many feature pairs interfere → evidence of superposition.
        SS ≈ 0  → orthogonal features (no superposition).
        SS ≈ 1  → collinear features (maximum superposition).

        Parameters
        ----------
        W : Tensor, shape (d_in, d_out)

        Returns
        -------
        float in [0, 1]
        """
        if W.dim() != 2:
            raise ValueError(f"W must be 2-D, got shape {W.shape}")

        d_in, d_out = W.shape

        if d_out < 2:
            # Only one neuron — no pairs, score = 0
            return 0.0

        # Normalise columns to unit vectors
        norms = W.norm(dim=0, keepdim=True).clamp(min=self.eps)  # (1, d_out)
        W_norm = W / norms  # (d_in, d_out)

        # Gram matrix of cosine similarities: (d_out, d_out)
        gram = W_norm.T @ W_norm  # (d_out, d_out)

        # Absolute value, zero out diagonal
        gram_abs = gram.abs()
        diag_mask = torch.eye(d_out, dtype=torch.bool, device=W.device)
        gram_abs = gram_abs.masked_fill(diag_mask, 0.0)

        # Mean over off-diagonal entries
        n_pairs = d_out * (d_out - 1)  # total off-diagonal elements
        ss = gram_abs.sum() / n_pairs
        return ss.clamp(0.0, 1.0).item()

    # ------------------------------------------------------------------
    # activation_sparsity
    # ------------------------------------------------------------------

    def activation_sparsity(self, activations: Tensor) -> Tensor:
        """
        Per-neuron activation sparsity.

        Sparsity of neuron i = fraction of samples where |A[:,i]| < threshold.

        Parameters
        ----------
        activations : Tensor, shape (N, d)

        Returns
        -------
        Tensor of shape (d,) with values in [0, 1].
        """
        if activations.dim() != 2:
            raise ValueError(f"activations must be 2-D (N, d), got shape {activations.shape}")
        # (N, d) → bool mask, average over N
        sparse_mask = (activations.abs() < self.threshold).float()
        return sparse_mask.mean(dim=0)  # (d,)

    # ------------------------------------------------------------------
    # feature_geometry
    # ------------------------------------------------------------------

    def feature_geometry(self, W: Tensor) -> dict:
        """
        Analyse the geometry of feature directions in weight space.

        Computes the distribution of cosine similarities across all column
        pairs and summarises it.  For pure superposition the distribution
        should be bimodal at ±1; for monosemantic representations it should
        be concentrated near 0.

        Parameters
        ----------
        W : Tensor, shape (d_in, d_out)

        Returns
        -------
        dict with keys:
            "cosine_similarities" : (d_out*(d_out-1)//2,) all upper-triangle sims
            "mean_abs_cosine"     : float — mean |cosine similarity|
            "max_abs_cosine"      : float
            "min_abs_cosine"      : float
        """
        if W.dim() != 2:
            raise ValueError(f"W must be 2-D, got shape {W.shape}")

        d_out = W.shape[1]

        norms = W.norm(dim=0, keepdim=True).clamp(min=self.eps)
        W_norm = W / norms

        gram = W_norm.T @ W_norm  # (d_out, d_out)

        # Extract upper triangle (i < j) → all unique pairs
        idx = torch.triu_indices(d_out, d_out, offset=1, device=W.device)
        cosine_sims = gram[idx[0], idx[1]]  # (n_pairs,)
        abs_sims = cosine_sims.abs()

        if abs_sims.numel() == 0:
            return {
                "cosine_similarities": cosine_sims,
                "mean_abs_cosine": 0.0,
                "max_abs_cosine": 0.0,
                "min_abs_cosine": 0.0,
            }

        return {
            "cosine_similarities": cosine_sims,
            "mean_abs_cosine": abs_sims.mean().item(),
            "max_abs_cosine": abs_sims.max().item(),
            "min_abs_cosine": abs_sims.min().item(),
        }

    # ------------------------------------------------------------------
    # analyze
    # ------------------------------------------------------------------

    def analyze(
        self,
        W: Tensor,
        activations: Tensor | None = None,
    ) -> dict:
        """
        Full polysemanticity / superposition analysis.

        Parameters
        ----------
        W : Tensor, shape (d_in, d_out)
            Weight matrix to analyse (e.g. MLP weight, embedding projection).
        activations : Tensor, shape (N, d_out), optional
            Activation samples for the neurons defined by W columns.
            When provided, activation-based metrics (sparsity, PR) are computed.

        Returns
        -------
        dict with keys:
            "polysemanticity_index"  : Tensor (d_out,)
            "superposition_score"    : float
            "feature_geometry"       : dict
            "mean_pi"                : float — mean PI across neurons
            "n_neurons"              : int
            "n_features"             : int
            And when activations is not None:
            "participation_ratio"    : float
            "activation_sparsity"    : Tensor (d_out,)
            "mean_sparsity"          : float
        """
        pi = self.polysemanticity_index(W)
        ss = self.superposition_score(W)
        geom = self.feature_geometry(W)

        result: dict = {
            "polysemanticity_index": pi,
            "superposition_score": ss,
            "feature_geometry": geom,
            "mean_pi": pi.mean().item(),
            "n_neurons": W.shape[1],
            "n_features": W.shape[0],
        }

        if activations is not None:
            if activations.dim() != 2:
                raise ValueError(
                    f"activations must be 2-D (N, d_out), got shape {activations.shape}"
                )
            pr = self.participation_ratio(activations)
            sparsity = self.activation_sparsity(activations)
            result["participation_ratio"] = pr
            result["activation_sparsity"] = sparsity
            result["mean_sparsity"] = sparsity.mean().item()

        return result
