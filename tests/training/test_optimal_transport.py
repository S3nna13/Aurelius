"""
Tests for src/training/optimal_transport.py
At least 14 tests covering OTConfig, cost_matrix, sinkhorn,
wasserstein_distance, sequence_ot_loss, and OTAligner.
"""

import torch

from src.training.optimal_transport import (
    OTAligner,
    OTConfig,
    cost_matrix,
    sequence_ot_loss,
    sinkhorn,
    wasserstein_distance,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
M, N, D = 8, 8, 16
VOCAB_SIZE = 32
TOL = 1e-4


# ---------------------------------------------------------------------------
# OTConfig
# ---------------------------------------------------------------------------


def test_otconfig_defaults():
    cfg = OTConfig()
    assert cfg.epsilon == 0.05
    assert cfg.n_iters == 100
    assert cfg.p == 2
    assert cfg.normalize is True


# ---------------------------------------------------------------------------
# cost_matrix
# ---------------------------------------------------------------------------


def test_cost_matrix_shape():
    x = torch.randn(M, D)
    y = torch.randn(N, D)
    C = cost_matrix(x, y)
    assert C.shape == (M, N)


def test_cost_matrix_same_point_zero():
    """Distance from a point to itself should be 0."""
    x = torch.randn(M, D)
    C = cost_matrix(x, x)
    diag = torch.diagonal(C)
    assert diag.abs().max().item() < TOL


def test_cost_matrix_symmetric_p2():
    """cost_matrix(x, y) and cost_matrix(y, x) should be transposes."""
    x = torch.randn(M, D)
    y = torch.randn(N, D)
    C_xy = cost_matrix(x, y, p=2)
    C_yx = cost_matrix(y, x, p=2)
    assert torch.allclose(C_xy, C_yx.T, atol=TOL)


# ---------------------------------------------------------------------------
# sinkhorn
# ---------------------------------------------------------------------------


def _uniform(n: int) -> torch.Tensor:
    return torch.ones(n) / n


def test_sinkhorn_output_shape():
    C = torch.rand(M, N)
    a = _uniform(M)
    b = _uniform(N)
    T = sinkhorn(C, a, b, epsilon=0.05, n_iters=100)
    assert T.shape == (M, N)


def test_sinkhorn_row_marginals():
    """Row sums of transport plan should be close to a."""
    C = torch.rand(M, N)
    a = _uniform(M)
    b = _uniform(N)
    T = sinkhorn(C, a, b, epsilon=0.05, n_iters=200)
    assert torch.allclose(T.sum(dim=1), a, atol=1e-3)


def test_sinkhorn_col_marginals():
    """Column sums of transport plan should be close to b."""
    C = torch.rand(M, N)
    a = _uniform(M)
    b = _uniform(N)
    T = sinkhorn(C, a, b, epsilon=0.05, n_iters=200)
    assert torch.allclose(T.sum(dim=0), b, atol=1e-3)


# ---------------------------------------------------------------------------
# wasserstein_distance
# ---------------------------------------------------------------------------


def test_wasserstein_same_distribution_near_zero():
    """OT distance between identical point clouds should be near 0."""
    x = torch.randn(M, D)
    cfg = OTConfig(epsilon=0.05, n_iters=200)
    d = wasserstein_distance(x, x.clone(), cfg)
    # Sinkhorn regularisation leaves a small residual; allow up to 0.01
    assert d.item() < 0.01


def test_wasserstein_different_distributions_positive():
    """OT distance between well-separated clouds should be > 0."""
    torch.manual_seed(42)
    x = torch.randn(M, D)
    y = torch.randn(N, D) + 10.0  # shift far away
    cfg = OTConfig()
    d = wasserstein_distance(x, y, cfg)
    assert d.item() > 0.0


def test_wasserstein_returns_scalar():
    x = torch.randn(M, D)
    y = torch.randn(N, D)
    cfg = OTConfig()
    d = wasserstein_distance(x, y, cfg)
    assert d.shape == torch.Size([])


# ---------------------------------------------------------------------------
# sequence_ot_loss
# ---------------------------------------------------------------------------


def test_sequence_ot_loss_returns_scalar():
    logits_p = torch.randn(VOCAB_SIZE)
    logits_q = torch.randn(VOCAB_SIZE)
    cfg = OTConfig()
    loss = sequence_ot_loss(logits_p, logits_q, cfg)
    assert loss.shape == torch.Size([])


def test_sequence_ot_loss_same_logits_near_zero():
    """OT loss between identical distributions should be near 0."""
    logits = torch.randn(VOCAB_SIZE)
    cfg = OTConfig(epsilon=0.05, n_iters=200)
    loss = sequence_ot_loss(logits, logits.clone(), cfg)
    # Sinkhorn regularisation leaves a small residual; allow up to 0.05
    assert loss.item() < 0.05


# ---------------------------------------------------------------------------
# OTAligner
# ---------------------------------------------------------------------------


def test_otaligner_align_shape():
    cfg = OTConfig()
    aligner = OTAligner(cfg)
    seq_a = torch.randn(M, D)
    seq_b = torch.randn(N, D)
    T = aligner.align(seq_a, seq_b)
    assert T.shape == (M, N)


def test_otaligner_soft_alignment_loss_scalar():
    cfg = OTConfig()
    aligner = OTAligner(cfg)
    emb_a = torch.randn(M, D)
    emb_b = torch.randn(N, D)
    loss = aligner.soft_alignment_loss(emb_a, emb_b)
    assert loss.shape == torch.Size([])
