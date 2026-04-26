"""Representation Similarity Analysis for LLMs.

Compares internal representations between layers or models.

CKA: measures similarity robust to orthogonal transformations and scaling.
RSA: correlates representational geometry (pairwise distances).
Procrustes: finds best alignment between representation spaces.

References:
    Kornblith et al. 2019 (CKA) — https://arxiv.org/abs/1905.00414
    Kriegeskorte et al. 2008 (RSA)
"""

from __future__ import annotations

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# CKA (Centered Kernel Alignment)
# ---------------------------------------------------------------------------


class CKAAnalyzer:
    """Centered Kernel Alignment similarity between representation matrices.

    Measures similarity robust to orthogonal transformations and isotropic
    scaling (Kornblith et al. 2019).
    """

    def __init__(self, unbiased: bool = False) -> None:
        """
        Args:
            unbiased: if True, use the unbiased HSIC estimator (better for
                      small n). Currently the biased estimator is implemented;
                      this flag is reserved for future extension.
        """
        self.unbiased = unbiased

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hsic(A: Tensor, B: Tensor) -> Tensor:
        """Biased HSIC estimator using the linear kernel.

        HSIC(A, B) = ||A_c.T @ B_c||_F^2 / (N-1)^2
        where A_c and B_c are column-centered matrices.
        """
        N = A.shape[0]
        A_c = A - A.mean(dim=0, keepdim=True)
        B_c = B - B.mean(dim=0, keepdim=True)
        dot = torch.linalg.matrix_norm(A_c.T @ B_c, ord="fro") ** 2
        return dot / (N - 1) ** 2

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def linear_cka(self, X: Tensor, Y: Tensor) -> float:
        """Linear CKA similarity between two representation matrices.

        Args:
            X: ``(N, d1)`` representation matrix.
            Y: ``(N, d2)`` representation matrix.

        Returns:
            CKA similarity in [0, 1].
        """
        hsic_xx = self._hsic(X, X)
        hsic_yy = self._hsic(Y, Y)
        hsic_xy = self._hsic(X, Y)

        denom = torch.sqrt(hsic_xx * hsic_yy)
        if denom.item() < 1e-10:
            return 0.0

        cka = hsic_xy / denom
        # Clamp to [0, 1] to handle floating point noise.
        return float(cka.clamp(0.0, 1.0).item())

    def layer_cka_matrix(self, layer_reprs: list[Tensor]) -> Tensor:
        """Compute pairwise CKA matrix across layers.

        Args:
            layer_reprs: list of ``(N, d_i)`` tensors, one per layer.

        Returns:
            ``(n_layers, n_layers)`` symmetric CKA similarity matrix.
        """
        n = len(layer_reprs)
        mat = torch.zeros(n, n)
        for i in range(n):
            mat[i, i] = 1.0
            for j in range(i + 1, n):
                val = self.linear_cka(layer_reprs[i], layer_reprs[j])
                mat[i, j] = val
                mat[j, i] = val
        return mat


# ---------------------------------------------------------------------------
# RSA (Representational Similarity Analysis)
# ---------------------------------------------------------------------------


class RSAAnalyzer:
    """Representational Similarity Analysis via RDMs.

    Follows Kriegeskorte et al. 2008: compute representational dissimilarity
    matrices (RDMs) and compare their geometry.
    """

    def __init__(self, distance_fn: str = "euclidean") -> None:
        """
        Args:
            distance_fn: ``'euclidean'`` for L2 distances or ``'cosine'``
                         for ``1 - cosine_similarity``.
        """
        if distance_fn not in ("euclidean", "cosine"):
            raise ValueError(f"distance_fn must be 'euclidean' or 'cosine', got {distance_fn!r}")
        self.distance_fn = distance_fn

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _euclidean_rdm(self, X: Tensor) -> Tensor:
        """Pairwise L2 distance matrix."""
        # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 x_i . x_j
        sq_norms = (X * X).sum(dim=1, keepdim=True)  # (N, 1)
        sq_dist = sq_norms + sq_norms.T - 2.0 * (X @ X.T)
        sq_dist = sq_dist.clamp(min=0.0)  # guard against numerical noise
        dist = sq_dist.sqrt()
        # Explicitly zero the diagonal — distance from a point to itself is 0
        # by definition; the clamp + sqrt can leave tiny residuals otherwise.
        dist.fill_diagonal_(0.0)
        return dist

    def _cosine_rdm(self, X: Tensor) -> Tensor:
        """Pairwise 1 - cosine_similarity matrix."""
        norms = X.norm(dim=1, keepdim=True).clamp(min=1e-8)
        X_norm = X / norms
        cos_sim = X_norm @ X_norm.T  # (N, N)
        return 1.0 - cos_sim

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rdm(self, X: Tensor) -> Tensor:
        """Compute the Representational Dissimilarity Matrix.

        Args:
            X: ``(N, d)`` representation matrix.

        Returns:
            ``(N, N)`` RDM.
        """
        if self.distance_fn == "euclidean":
            return self._euclidean_rdm(X)
        return self._cosine_rdm(X)

    def _upper_tri(self, M: Tensor) -> Tensor:
        """Extract upper triangle (excluding diagonal) as a 1-D vector."""
        N = M.shape[0]
        idx = torch.triu_indices(N, N, offset=1)
        return M[idx[0], idx[1]]

    def rsa_correlation(self, X: Tensor, Y: Tensor) -> float:
        """Pearson correlation between upper triangles of RDMs.

        Args:
            X: ``(N, d_x)`` representation matrix.
            Y: ``(N, d_y)`` representation matrix.

        Returns:
            Pearson correlation in [-1, 1].
        """
        rdm_x = self._upper_tri(self.rdm(X))
        rdm_y = self._upper_tri(self.rdm(Y))
        return float(_pearson(rdm_x, rdm_y).item())

    def spearman_rsa(self, X: Tensor, Y: Tensor) -> float:
        """Spearman rank correlation between upper triangles of RDMs.

        Ranks are computed via argsort-of-argsort, then Pearson is applied.

        Args:
            X: ``(N, d_x)`` representation matrix.
            Y: ``(N, d_y)`` representation matrix.

        Returns:
            Spearman correlation in [-1, 1].
        """
        rdm_x = self._upper_tri(self.rdm(X))
        rdm_y = self._upper_tri(self.rdm(Y))
        rank_x = _rank(rdm_x).float()
        rank_y = _rank(rdm_y).float()
        return float(_pearson(rank_x, rank_y).item())


# ---------------------------------------------------------------------------
# Procrustes Analysis
# ---------------------------------------------------------------------------


class ProcrustesAnalyzer:
    """Orthogonal Procrustes alignment between two representation spaces."""

    def __init__(self) -> None:
        pass  # stateless

    def procrustes(self, X: Tensor, Y: Tensor) -> dict[str, float]:
        """Find the optimal orthogonal rotation Q minimising ``||X Q - Y||_F``.

        Algorithm:
            SVD of ``Y.T @ X`` → ``U S Vᴴ``
            ``Q = (U @ Vᴴ).T``  (optimal rotation)

        Args:
            X: ``(N, d)`` source representation matrix.
            Y: ``(N, d)`` target representation matrix.

        Returns:
            dict with keys:
              - ``'residual'``: Frobenius norm of ``X @ Q - Y``.
              - ``'similarity'``: ``1 - residual / (||Y||_F + 1e-8)``.
        """
        U, S, Vh = torch.linalg.svd(Y.T @ X, full_matrices=False)
        Q = (U @ Vh).T  # (d, d) orthogonal rotation

        residual_mat = X @ Q - Y
        residual = float(torch.linalg.matrix_norm(residual_mat, ord="fro").item())
        norm_y = float(torch.linalg.matrix_norm(Y, ord="fro").item())
        similarity = 1.0 - residual / (norm_y + 1e-8)

        return {"residual": residual, "similarity": similarity}


# ---------------------------------------------------------------------------
# Layer Similarity Report
# ---------------------------------------------------------------------------


class LayerSimilarityReport:
    """Aggregate representation similarity statistics across model layers."""

    def __init__(self, n_layers: int) -> None:
        self.n_layers = n_layers
        self._cka = CKAAnalyzer()

    def compute_cka_profile(self, layer_reprs: list[Tensor]) -> Tensor:
        """CKA between each pair of consecutive layers.

        Args:
            layer_reprs: list of ``(N, d_i)`` tensors, one per layer.

        Returns:
            ``(n_layers - 1,)`` tensor of consecutive CKA values.
        """
        values = [
            self._cka.linear_cka(layer_reprs[i], layer_reprs[i + 1])
            for i in range(len(layer_reprs) - 1)
        ]
        return torch.tensor(values)

    def find_similar_layers(
        self,
        layer_reprs: list[Tensor],
        threshold: float = 0.9,
    ) -> list[tuple[int, int]]:
        """Return pairs (i, j) with i < j where CKA(layer_i, layer_j) > threshold.

        Args:
            layer_reprs: list of ``(N, d_i)`` tensors.
            threshold: CKA threshold above which layers are considered similar.

        Returns:
            List of ``(i, j)`` index pairs.
        """
        mat = self._cka.layer_cka_matrix(layer_reprs)
        n = mat.shape[0]
        pairs: list[tuple[int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                if mat[i, j].item() > threshold:
                    pairs.append((i, j))
        return pairs

    def layer_change_rate(self, layer_reprs: list[Tensor]) -> list[float]:
        """``1 - CKA`` for each pair of consecutive layers.

        Higher values indicate more representational change at that transition.

        Args:
            layer_reprs: list of ``(N, d_i)`` tensors.

        Returns:
            List of ``n_layers - 1`` change-rate values.
        """
        profile = self.compute_cka_profile(layer_reprs)
        return (1.0 - profile).tolist()


# ---------------------------------------------------------------------------
# Private utilities
# ---------------------------------------------------------------------------


def _pearson(a: Tensor, b: Tensor) -> Tensor:
    """Pearson correlation coefficient between two 1-D tensors."""
    a_c = a - a.mean()
    b_c = b - b.mean()
    num = (a_c * b_c).sum()
    denom = torch.sqrt((a_c**2).sum() * (b_c**2).sum()).clamp(min=1e-10)
    return num / denom


def _rank(x: Tensor) -> Tensor:
    """Convert a 1-D tensor to its rank vector (argsort of argsort)."""
    return x.argsort().argsort()
