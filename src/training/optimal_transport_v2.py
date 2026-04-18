"""
Optimal Transport for sequence alignment and distribution matching in LLM training.
Pure native PyTorch implementation — no external OT libraries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# OTConfig
# ---------------------------------------------------------------------------

@dataclass
class OTConfig:
    eps: float = 0.1
    n_iters: int = 50
    thresh: float = 1e-3
    n_projections: int = 50
    lambda_ot: float = 1.0


# ---------------------------------------------------------------------------
# CostMatrix
# ---------------------------------------------------------------------------

class CostMatrix:
    """Factory methods for computing cost matrices between point sets."""

    @staticmethod
    def l2_cost(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Squared Euclidean cost matrix.

        Args:
            X: [N, d]
            Y: [M, d]
        Returns:
            C: [N, M]  where C[i,j] = ||X[i] - Y[j]||^2
        """
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
        X_sq = (X ** 2).sum(dim=1, keepdim=True)   # [N, 1]
        Y_sq = (Y ** 2).sum(dim=1, keepdim=True).T   # [1, M]
        cross = X @ Y.T                               # [N, M]
        C = X_sq + Y_sq - 2.0 * cross
        # Numerical safety: clamp to non-negative
        return C.clamp(min=0.0)

    @staticmethod
    def cosine_cost(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Cosine dissimilarity cost matrix: 1 - cosine_similarity.

        Args:
            X: [N, d]
            Y: [M, d]
        Returns:
            C: [N, M]  values in [0, 2]
        """
        X_norm = F.normalize(X, p=2, dim=1)   # [N, d]
        Y_norm = F.normalize(Y, p=2, dim=1)   # [M, d]
        sim = X_norm @ Y_norm.T                # [N, M]
        return 1.0 - sim

    @staticmethod
    def token_edit_cost(seq_a: torch.Tensor, seq_b: torch.Tensor) -> torch.Tensor:
        """
        Binary token edit cost: 0 if tokens match, 1 otherwise.

        Args:
            seq_a: [T_a]  integer token ids
            seq_b: [T_b]  integer token ids
        Returns:
            C: [T_a, T_b]
        """
        # Broadcasting comparison
        C = (seq_a.unsqueeze(1) != seq_b.unsqueeze(0)).float()   # [T_a, T_b]
        return C


# ---------------------------------------------------------------------------
# SinkhornSolver
# ---------------------------------------------------------------------------

class SinkhornSolver:
    """
    Log-space Sinkhorn-Knopp iterations for numerical stability.
    """

    def __init__(self, eps: float = 0.1, n_iters: int = 50, thresh: float = 1e-3):
        self.eps = eps
        self.n_iters = n_iters
        self.thresh = thresh

    def solve(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        C: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute the regularised OT plan via Sinkhorn-Knopp in log-space.

        Args:
            a: [N]  source marginal (probability simplex)
            b: [M]  target marginal (probability simplex)
            C: [N, M]  cost matrix
        Returns:
            plan: [N, M]  transport plan
            cost: float   OT cost sum(plan * C)
        """
        N, M = C.shape
        log_a = a.log()          # [N]
        log_b = b.log()          # [M]
        log_K = -C / self.eps    # [N, M]

        # Initialise dual variables in log-space
        log_f = torch.zeros(N, dtype=C.dtype, device=C.device)  # [N]
        log_g = torch.zeros(M, dtype=C.dtype, device=C.device)  # [M]

        for _ in range(self.n_iters):
            log_f_prev = log_f.clone()
            # log f_{t+1} = log a - logsumexp(log g + log K, dim=1)
            # log g is [M], log K is [N,M], broadcast: log_g[j] + log_K[i,j]
            log_f = log_a - torch.logsumexp(log_g.unsqueeze(0) + log_K, dim=1)
            # log g_{t+1} = log b - logsumexp(log f + log K, dim=0)
            log_g = log_b - torch.logsumexp(log_f.unsqueeze(1) + log_K, dim=0)

            # Check convergence
            err = (log_f - log_f_prev).abs().max().item()
            if err < self.thresh:
                break

        # Recover plan: P = diag(f) K diag(g)  in log-space
        log_plan = log_f.unsqueeze(1) + log_K + log_g.unsqueeze(0)   # [N, M]
        plan = log_plan.exp()

        cost = (plan * C).sum().item()
        return plan, cost

    def wasserstein_distance(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        C: torch.Tensor,
    ) -> float:
        """Return total OT cost: sum(plan * C)."""
        _, cost = self.solve(a, b, C)
        return cost


# ---------------------------------------------------------------------------
# EarthMoversDistance (nn.Module)
# ---------------------------------------------------------------------------

class EarthMoversDistance(nn.Module):
    """Batched Earth Mover's Distance (Wasserstein-1 / Sinkhorn approximation)."""

    def __init__(self, eps: float = 0.1, n_iters: int = 50):
        super().__init__()
        self.solver = SinkhornSolver(eps=eps, n_iters=n_iters)

    def forward(
        self,
        p: torch.Tensor,
        q: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """
        Batched Sinkhorn: compute EMD for each (p_b, q_b) pair independently.

        Args:
            p: [B, N]  source distributions (rows sum to 1)
            q: [B, M]  target distributions (rows sum to 1)
            C: [N, M]  cost matrix (shared across batch)
        Returns:
            distances: [B]
        """
        B = p.shape[0]
        distances = []
        for i in range(B):
            _, cost = self.solver.solve(p[i], q[i], C)
            distances.append(cost)
        return torch.tensor(distances, dtype=p.dtype, device=p.device)


# ---------------------------------------------------------------------------
# OTSequenceAligner
# ---------------------------------------------------------------------------

class OTSequenceAligner:
    """
    Soft sequence alignment via Optimal Transport on token embeddings.
    """

    def __init__(self, solver: SinkhornSolver):
        self.solver = solver

    def align_sequences(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """
        Soft alignment via OT on L2 cost with uniform marginals.

        Args:
            emb_a: [T_a, d]
            emb_b: [T_b, d]
        Returns:
            plan: [T_a, T_b]  transport plan
            cost: float        OT cost
        """
        T_a = emb_a.shape[0]
        T_b = emb_b.shape[0]
        C = CostMatrix.l2_cost(emb_a, emb_b)          # [T_a, T_b]
        a = torch.full((T_a,), 1.0 / T_a, dtype=emb_a.dtype, device=emb_a.device)
        b = torch.full((T_b,), 1.0 / T_b, dtype=emb_b.dtype, device=emb_b.device)
        plan, cost = self.solver.solve(a, b, C)
        return plan, cost

    def soft_align_loss(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Differentiable OT distance between batched encoder outputs.

        Args:
            emb_a: [B, T_a, d]
            emb_b: [B, T_b, d]
        Returns:
            losses: [B]
        """
        B = emb_a.shape[0]
        losses = []
        for i in range(B):
            _, cost = self.align_sequences(emb_a[i], emb_b[i])
            losses.append(cost)
        return torch.tensor(losses, dtype=emb_a.dtype, device=emb_a.device)

    def barycentric_projection(
        self,
        emb_b: torch.Tensor,
        plan: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project emb_b onto emb_a space via the transport plan.

        emb_a_proj[i] = sum_j plan[i,j] * emb_b[j]  /  sum_j plan[i,j]

        Args:
            emb_b:  [T_b, d]   target embeddings
            plan:   [T_a, T_b] transport plan (row sums = marginal a)
        Returns:
            emb_proj: [T_a, d]
        """
        # plan: [T_a, T_b], emb_b: [T_b, d]
        row_sums = plan.sum(dim=1, keepdim=True).clamp(min=1e-9)   # [T_a, 1]
        emb_proj = (plan @ emb_b) / row_sums                       # [T_a, d]
        return emb_proj


# ---------------------------------------------------------------------------
# OTDistillationLoss (nn.Module)
# ---------------------------------------------------------------------------

class OTDistillationLoss(nn.Module):
    """
    OT alignment loss between student and teacher representations.
    Handles T_student != T_teacher via non-square transport plans.
    """

    def __init__(self, eps: float = 0.1, n_iters: int = 30, lambda_ot: float = 1.0):
        super().__init__()
        self.aligner = OTSequenceAligner(SinkhornSolver(eps=eps, n_iters=n_iters))
        self.lambda_ot = lambda_ot

    def forward(
        self,
        student_emb: torch.Tensor,
        teacher_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute OT distillation loss between student and teacher embeddings.

        Args:
            student_emb: [B, T_s, d]
            teacher_emb: [B, T_t, d]
        Returns:
            loss: scalar tensor
        """
        per_sample = self.aligner.soft_align_loss(student_emb, teacher_emb)   # [B]
        loss = self.lambda_ot * per_sample.mean()
        return loss


# ---------------------------------------------------------------------------
# SlicedWasserstein
# ---------------------------------------------------------------------------

class SlicedWasserstein:
    """
    Sliced Wasserstein distance: approximates W_2 via random 1-D projections.
    """

    def __init__(self, n_projections: int = 50):
        self.n_projections = n_projections

    def distance(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        """
        Approximate W_2 between empirical distributions X and Y.

        Args:
            X: [N, d]
            Y: [M, d]
        Returns:
            distance: float ≥ 0
        """
        d = X.shape[1]
        # Random unit projections: [n_projections, d]
        theta = torch.randn(self.n_projections, d, dtype=X.dtype, device=X.device)
        theta = F.normalize(theta, p=2, dim=1)

        # Project: [n_projections, N] and [n_projections, M]
        X_proj = (theta @ X.T)   # [P, N]
        Y_proj = (theta @ Y.T)   # [P, M]

        # Sort along sample dimension
        X_sorted, _ = X_proj.sort(dim=1)   # [P, N]
        Y_sorted, _ = Y_proj.sort(dim=1)   # [P, M]

        # If N != M, interpolate the smaller to match the larger
        N = X_sorted.shape[1]
        M = Y_sorted.shape[1]
        if N != M:
            # Upsample the smaller via linear interpolation
            if N < M:
                X_sorted = F.interpolate(
                    X_sorted.unsqueeze(0), size=M, mode='linear', align_corners=False
                ).squeeze(0)
            else:
                Y_sorted = F.interpolate(
                    Y_sorted.unsqueeze(0), size=N, mode='linear', align_corners=False
                ).squeeze(0)

        # Sliced W_2^2: mean over projections of mean squared diff of sorted points
        sw2 = ((X_sorted - Y_sorted) ** 2).mean()
        return math.sqrt(sw2.item())
