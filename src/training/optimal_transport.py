"""
Optimal transport distances (Sinkhorn, Wasserstein) for sequence alignment
and training signals.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class OTConfig:
    """Configuration for optimal transport computations."""
    epsilon: float = 0.05       # Sinkhorn regularisation strength
    n_iters: int = 100          # number of Sinkhorn iterations
    p: int = 2                  # Lp distance exponent for cost matrix
    normalize: bool = True      # normalise cost matrix before Sinkhorn


# ---------------------------------------------------------------------------
# Core primitives
# ---------------------------------------------------------------------------

def cost_matrix(x: Tensor, y: Tensor, p: int = 2) -> Tensor:
    """
    Compute pairwise Lp distance matrix between rows of x and y.

    Args:
        x: (M, D) source point cloud
        y: (N, D) target point cloud
        p: distance exponent (p=2 → squared Euclidean by convention)

    Returns:
        C: (M, N) cost matrix where C[i, j] = ||x[i] - y[j]||_p^p
    """
    # x: (M, D), y: (N, D)
    # Broadcast: (M, 1, D) - (1, N, D) → (M, N, D)
    diff = x.unsqueeze(1) - y.unsqueeze(0)          # (M, N, D)
    C = diff.abs().pow(p).sum(dim=-1)                # (M, N)
    return C


def sinkhorn(
    cost: Tensor,
    a: Tensor,
    b: Tensor,
    epsilon: float,
    n_iters: int,
) -> Tensor:
    """
    Sinkhorn-Knopp algorithm (log-domain for numerical stability).

    Args:
        cost:    (M, N) cost matrix
        a:       (M,) source marginal distribution (sums to 1)
        b:       (N,) target marginal distribution (sums to 1)
        epsilon: regularisation strength (> 0)
        n_iters: number of Sinkhorn iterations

    Returns:
        transport plan T: (M, N), rows sum ≈ a, cols sum ≈ b
    """
    log_a = a.log()          # (M,)
    log_b = b.log()          # (N,)
    M_cost = cost / epsilon  # (M, N)

    # Log-domain Sinkhorn
    log_u = torch.zeros_like(log_a)  # (M,)
    log_v = torch.zeros_like(log_b)  # (N,)

    for _ in range(n_iters):
        # u update: log_u = log_a - logsumexp(log_v - M_cost, dim=1)
        log_u = log_a - torch.logsumexp(log_v.unsqueeze(0) - M_cost, dim=1)
        # v update: log_v = log_b - logsumexp(log_u - M_cost, dim=0)
        log_v = log_b - torch.logsumexp(log_u.unsqueeze(1) - M_cost, dim=0)

    # Transport plan: T[i,j] = exp(log_u[i] + log_v[j] - M_cost[i,j])
    log_T = log_u.unsqueeze(1) + log_v.unsqueeze(0) - M_cost
    return log_T.exp()


# ---------------------------------------------------------------------------
# Wasserstein distance
# ---------------------------------------------------------------------------

def wasserstein_distance(x: Tensor, y: Tensor, config: OTConfig) -> Tensor:
    """
    Sinkhorn-approximate Wasserstein distance between point clouds x and y.

    Args:
        x:      (N, D) source point cloud
        y:      (N, D) target point cloud
        config: OTConfig

    Returns:
        scalar Wasserstein distance
    """
    M_src, _ = x.shape
    N_tgt, _ = y.shape

    C = cost_matrix(x, y, p=config.p)  # (M, N)

    if config.normalize:
        max_c = C.max()
        if max_c > 0:
            C = C / max_c

    # Uniform marginals
    a = torch.ones(M_src, dtype=x.dtype, device=x.device) / M_src
    b = torch.ones(N_tgt, dtype=y.dtype, device=y.device) / N_tgt

    T = sinkhorn(C, a, b, config.epsilon, config.n_iters)  # (M, N)
    return (T * C).sum()


# ---------------------------------------------------------------------------
# Sequence OT loss
# ---------------------------------------------------------------------------

def sequence_ot_loss(
    logits_p: Tensor,
    logits_q: Tensor,
    config: OTConfig,
) -> Tensor:
    """
    Compute OT distance between token probability distributions.

    Both logits_p and logits_q are treated as 1-D discrete distributions
    over the vocabulary (after softmax).  The cost is the squared difference
    of vocabulary indices, normalised to [0, 1].

    Args:
        logits_p: (V,) or (1, V) logits for distribution p
        logits_q: (V,) or (1, V) logits for distribution q
        config:   OTConfig

    Returns:
        scalar OT loss
    """
    logits_p = logits_p.reshape(-1)
    logits_q = logits_q.reshape(-1)

    p_dist = F.softmax(logits_p, dim=0)  # (V,)
    q_dist = F.softmax(logits_q, dim=0)  # (V,)

    V = p_dist.shape[0]

    # Cost: squared index distance normalised to [0, 1]
    indices = torch.arange(V, dtype=logits_p.dtype, device=logits_p.device)
    C = cost_matrix(indices.unsqueeze(1), indices.unsqueeze(1), p=config.p)  # (V, V)

    if config.normalize:
        max_c = C.max()
        if max_c > 0:
            C = C / max_c

    T = sinkhorn(C, p_dist, q_dist, config.epsilon, config.n_iters)  # (V, V)
    return (T * C).sum()


# ---------------------------------------------------------------------------
# OTAligner
# ---------------------------------------------------------------------------

class OTAligner:
    """Aligns two sequences using optimal transport."""

    def __init__(self, config: OTConfig):
        self.config = config

    def align(self, seq_a: Tensor, seq_b: Tensor) -> Tensor:
        """
        Compute soft alignment between two embedding sequences.

        Args:
            seq_a: (M, D) source embeddings
            seq_b: (N, D) target embeddings

        Returns:
            transport plan T: (M, N)
        """
        C = cost_matrix(seq_a, seq_b, p=self.config.p)  # (M, N)

        if self.config.normalize:
            max_c = C.max()
            if max_c > 0:
                C = C / max_c

        M = seq_a.shape[0]
        N = seq_b.shape[0]
        a = torch.ones(M, dtype=seq_a.dtype, device=seq_a.device) / M
        b = torch.ones(N, dtype=seq_b.dtype, device=seq_b.device) / N

        return sinkhorn(C, a, b, self.config.epsilon, self.config.n_iters)

    def soft_alignment_loss(self, emb_a: Tensor, emb_b: Tensor) -> Tensor:
        """
        OT distance as a training signal.

        Args:
            emb_a: (M, D)
            emb_b: (N, D)

        Returns:
            scalar OT distance
        """
        C = cost_matrix(emb_a, emb_b, p=self.config.p)

        if self.config.normalize:
            max_c = C.max()
            if max_c > 0:
                C = C / max_c

        M = emb_a.shape[0]
        N = emb_b.shape[0]
        a = torch.ones(M, dtype=emb_a.dtype, device=emb_a.device) / M
        b = torch.ones(N, dtype=emb_b.dtype, device=emb_b.device) / N

        T = sinkhorn(C, a, b, self.config.epsilon, self.config.n_iters)
        return (T * C).sum()
