"""Dictionary learning over activations via K-SVD + OMP sparse coding.

Solves: min_{D, alpha} ||X - alpha @ D^T||_F^2  s.t. ||alpha_i||_0 <= k_sparse.

Distinct from sae_trainer.py: this module learns an overcomplete dictionary
by alternating minimization between sparse code assignment (Orthogonal
Matching Pursuit) and a K-SVD-style atom-by-atom dictionary update.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class DictionaryResult:
    """Result of dictionary learning fit."""

    D: Tensor  # [D_dim, n_atoms], columns are L2-normalized atoms
    alpha: Tensor  # [N, n_atoms], sparse codes
    reconstruction_error: float
    n_iters: int


class DictionaryLearner:
    """Overcomplete dictionary learning with K-SVD update and OMP coding."""

    def __init__(
        self,
        n_atoms: int,
        sparsity_target: int = 8,
        l1_lambda: float = 0.01,
        max_iters: int = 50,
        tol: float = 1e-4,
    ):
        if n_atoms <= 0:
            raise ValueError(f"n_atoms must be > 0, got {n_atoms}")
        if sparsity_target <= 0:
            raise ValueError(
                f"sparsity_target must be > 0, got {sparsity_target}"
            )
        if max_iters <= 0:
            raise ValueError(f"max_iters must be > 0, got {max_iters}")
        self.n_atoms = n_atoms
        self.sparsity_target = sparsity_target
        self.l1_lambda = float(l1_lambda)
        self.max_iters = max_iters
        self.tol = float(tol)

    # -------- helpers --------
    @staticmethod
    def _normalize_cols(D: Tensor, eps: float = 1e-12) -> Tensor:
        norms = D.norm(dim=0, keepdim=True).clamp_min(eps)
        return D / norms

    def _init_dictionary(self, X: Tensor) -> Tensor:
        """Initialize atoms from normalized random data samples (with fallback)."""
        N, D_dim = X.shape
        device, dtype = X.device, X.dtype
        k = self.n_atoms
        if N >= k:
            idx = torch.randperm(N, device=device)[:k]
            D = X[idx].T.clone()  # [D_dim, k]
        else:
            # sample with replacement, then mix in noise
            idx = torch.randint(0, N, (k,), device=device)
            D = X[idx].T.clone()
            D = D + 0.01 * torch.randn(D_dim, k, device=device, dtype=dtype)
        # replace any zero columns
        col_norms = D.norm(dim=0)
        zero_mask = col_norms < 1e-10
        if zero_mask.any():
            n_zero = int(zero_mask.sum().item())
            D[:, zero_mask] = torch.randn(
                D_dim, n_zero, device=device, dtype=dtype
            )
        return self._normalize_cols(D)

    # -------- public API --------
    def encode(self, X: Tensor, D: Tensor) -> Tensor:
        """Orthogonal Matching Pursuit sparse coding.

        Args:
            X: [N, D_dim] data
            D: [D_dim, n_atoms] dictionary with L2-normalized columns

        Returns:
            alpha: [N, n_atoms] sparse codes; each row has <= sparsity_target nonzeros.
        """
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {tuple(X.shape)}")
        if D.ndim != 2:
            raise ValueError(f"D must be 2D, got shape {tuple(D.shape)}")
        N, D_dim = X.shape
        D_dim2, K = D.shape
        if D_dim != D_dim2:
            raise ValueError(
                f"X/D dim mismatch: X has {D_dim}, D has {D_dim2}"
            )
        device, dtype = X.device, X.dtype
        alpha = torch.zeros(N, K, device=device, dtype=dtype)
        k_sparse = min(self.sparsity_target, K, D_dim)
        if k_sparse == 0:
            return alpha

        # per-sample OMP (N is typically small-to-moderate for activation slices)
        for i in range(N):
            x = X[i]
            if x.norm() < 1e-12:
                continue
            residual = x.clone()
            selected: list[int] = []
            for _ in range(k_sparse):
                # pick atom with maximum |<residual, d_j>| among unselected
                corr = D.T @ residual  # [K]
                if selected:
                    mask = torch.zeros(K, device=device, dtype=torch.bool)
                    mask[torch.tensor(selected, device=device)] = True
                    corr = corr.masked_fill(mask, 0.0)
                j = int(torch.argmax(corr.abs()).item())
                if abs(corr[j].item()) < 1e-10:
                    break
                selected.append(j)
                D_sel = D[:, selected]  # [D_dim, s]
                # least-squares solve: min ||D_sel c - x||
                sol = torch.linalg.lstsq(D_sel, x.unsqueeze(1)).solution
                c = sol.squeeze(1)
                residual = x - D_sel @ c
                if residual.norm().item() < 1e-10:
                    break
            if selected:
                alpha[i, torch.tensor(selected, device=device)] = c.to(dtype)
        return alpha

    def decode(self, alpha: Tensor, D: Tensor) -> Tensor:
        """Reconstruct X_hat = alpha @ D^T  -> [N, D_dim]."""
        if alpha.ndim != 2 or D.ndim != 2:
            raise ValueError("alpha and D must both be 2D")
        return alpha @ D.T

    def _ksvd_update(self, X: Tensor, D: Tensor, alpha: Tensor) -> tuple[Tensor, Tensor]:
        """K-SVD-style per-atom update via rank-1 SVD of the masked residual.

        For each atom k used by at least one sample, restrict to those samples,
        compute E_k = X_used^T - sum_{j!=k} d_j alpha_{used, j}^T, take its
        top singular vector for the new atom (d_k <- u_1) and update the
        active codes (alpha_{used, k} <- sigma_1 * v_1).
        """
        N, D_dim = X.shape
        K = D.shape[1]
        X_T = X.T  # [D_dim, N]
        for k in range(K):
            used = alpha[:, k].abs() > 0
            n_used = int(used.sum().item())
            if n_used == 0:
                # atom unused: reseed from the worst-reconstructed sample
                recon = alpha @ D.T  # [N, D_dim]
                errs = (X - recon).norm(dim=1)
                worst = int(torch.argmax(errs).item())
                new_atom = X[worst].clone()
                n = new_atom.norm()
                if n < 1e-10:
                    new_atom = torch.randn_like(new_atom)
                    n = new_atom.norm().clamp_min(1e-10)
                D[:, k] = new_atom / n
                continue

            used_idx = used.nonzero(as_tuple=False).squeeze(1)
            X_used = X_T[:, used_idx]  # [D_dim, n_used]
            alpha_used = alpha[used_idx]  # [n_used, K]
            # residual excluding atom k
            alpha_excl = alpha_used.clone()
            alpha_excl[:, k] = 0.0
            E_k = X_used - D @ alpha_excl.T  # [D_dim, n_used]
            # rank-1 approximation via SVD
            try:
                U, S, Vh = torch.linalg.svd(E_k, full_matrices=False)
            except Exception:
                continue
            if S.numel() == 0 or S[0].item() < 1e-12:
                continue
            D[:, k] = U[:, 0]
            alpha[used_idx, k] = S[0] * Vh[0, :]
        # renormalize defensively
        D = self._normalize_cols(D)
        return D, alpha

    def fit(self, X: Tensor) -> DictionaryResult:
        if X.ndim != 2:
            raise ValueError(f"X must be 2D [N, D_dim], got {tuple(X.shape)}")
        N, D_dim = X.shape
        if N == 0:
            raise ValueError("X must have at least one sample")

        # zero-variance fast path
        if X.abs().max().item() < 1e-12:
            D = torch.zeros(D_dim, self.n_atoms, device=X.device, dtype=X.dtype)
            # give atoms a valid unit direction so shape/norm tests still pass
            k = min(self.n_atoms, D_dim)
            eye = torch.eye(D_dim, device=X.device, dtype=X.dtype)
            D[:, :k] = eye[:, :k]
            if self.n_atoms > D_dim:
                extra = torch.randn(
                    D_dim, self.n_atoms - D_dim, device=X.device, dtype=X.dtype
                )
                D[:, D_dim:] = self._normalize_cols(extra)
            alpha = torch.zeros(N, self.n_atoms, device=X.device, dtype=X.dtype)
            return DictionaryResult(D=D, alpha=alpha, reconstruction_error=0.0, n_iters=0)

        D = self._init_dictionary(X)
        prev_err = float("inf")
        alpha = torch.zeros(N, self.n_atoms, device=X.device, dtype=X.dtype)
        err = float("inf")

        for it in range(1, self.max_iters + 1):
            alpha = self.encode(X, D)
            # soft-threshold with l1_lambda for gentle shrinkage (keeps
            # sparsity but dampens tiny coefficients).
            if self.l1_lambda > 0:
                alpha = torch.sign(alpha) * torch.clamp(
                    alpha.abs() - self.l1_lambda, min=0.0
                )
            D, alpha = self._ksvd_update(X, D, alpha)
            recon = alpha @ D.T
            err = float(((X - recon) ** 2).mean().item())
            if abs(prev_err - err) < self.tol:
                return DictionaryResult(
                    D=D, alpha=alpha, reconstruction_error=err, n_iters=it
                )
            prev_err = err

        return DictionaryResult(
            D=D, alpha=alpha, reconstruction_error=err, n_iters=self.max_iters
        )
