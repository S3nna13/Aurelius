"""
Tests for optimal_transport_v2.py
Covers CostMatrix, SinkhornSolver, EarthMoversDistance, OTSequenceAligner,
OTDistillationLoss, and SlicedWasserstein.

Test parameters: d=16, N=8, M=6, B=2, T_a=6, T_b=4
"""

import math
import torch
import pytest

from src.training.optimal_transport_v2 import (
    CostMatrix,
    EarthMoversDistance,
    OTConfig,
    OTDistillationLoss,
    OTSequenceAligner,
    SinkhornSolver,
    SlicedWasserstein,
)

# ---------------------------------------------------------------------------
# Shared fixtures / constants
# ---------------------------------------------------------------------------

D = 16
N = 8
M = 6
B = 2
T_A = 6
T_B = 4

torch.manual_seed(0)


def make_uniform(n: int) -> torch.Tensor:
    return torch.full((n,), 1.0 / n)


def make_cost(n: int, m: int) -> torch.Tensor:
    X = torch.randn(n, D)
    Y = torch.randn(m, D)
    return CostMatrix.l2_cost(X, Y)


# ---------------------------------------------------------------------------
# CostMatrix tests
# ---------------------------------------------------------------------------

def test_l2_cost_shape():
    X = torch.randn(N, D)
    Y = torch.randn(M, D)
    C = CostMatrix.l2_cost(X, Y)
    assert C.shape == (N, M), f"Expected ({N}, {M}), got {C.shape}"


def test_l2_cost_non_negative():
    X = torch.randn(N, D)
    Y = torch.randn(M, D)
    C = CostMatrix.l2_cost(X, Y)
    assert (C >= 0).all(), "l2_cost contains negative values"


def test_cosine_cost_shape():
    X = torch.randn(N, D)
    Y = torch.randn(M, D)
    C = CostMatrix.cosine_cost(X, Y)
    assert C.shape == (N, M)


def test_cosine_cost_range():
    X = torch.randn(N, D)
    Y = torch.randn(M, D)
    C = CostMatrix.cosine_cost(X, Y)
    # 1 - cos(theta) is in [0, 2]
    assert (C >= -1e-6).all(), "cosine_cost below 0"
    assert (C <= 2.0 + 1e-6).all(), "cosine_cost above 2"


def test_token_edit_cost_diagonal_zeros():
    """Same tokens should have zero cost on the diagonal."""
    vocab_size = 100
    T = 10
    seq = torch.randint(0, vocab_size, (T,))
    C = CostMatrix.token_edit_cost(seq, seq)
    assert C.shape == (T, T)
    diag = torch.diag(C)
    assert (diag == 0).all(), "Diagonal of token_edit_cost for identical sequences should be 0"


def test_token_edit_cost_off_diagonal_one():
    """Different tokens must have cost 1."""
    # Two non-overlapping token sequences
    seq_a = torch.zeros(4, dtype=torch.long)     # all token 0
    seq_b = torch.ones(4, dtype=torch.long)      # all token 1
    C = CostMatrix.token_edit_cost(seq_a, seq_b)
    assert (C == 1).all(), "token_edit_cost between different tokens should be 1"


# ---------------------------------------------------------------------------
# SinkhornSolver tests
# ---------------------------------------------------------------------------

def test_sinkhorn_plan_shape():
    solver = SinkhornSolver(eps=0.1, n_iters=50)
    a = make_uniform(N)
    b = make_uniform(M)
    C = make_cost(N, M)
    plan, _ = solver.solve(a, b, C)
    assert plan.shape == (N, M), f"Expected plan shape ({N}, {M}), got {plan.shape}"


def test_sinkhorn_plan_row_marginals():
    """Row sums of plan should approximate source marginal a."""
    solver = SinkhornSolver(eps=0.05, n_iters=200, thresh=1e-7)
    a = make_uniform(N)
    b = make_uniform(M)
    C = make_cost(N, M)
    plan, _ = solver.solve(a, b, C)
    row_sums = plan.sum(dim=1)
    assert torch.allclose(row_sums, a, atol=0.06), (
        f"Row marginals off: max error {(row_sums - a).abs().max().item():.4f}"
    )


def test_sinkhorn_plan_col_marginals():
    """Column sums of plan should approximate target marginal b."""
    solver = SinkhornSolver(eps=0.05, n_iters=200, thresh=1e-7)
    a = make_uniform(N)
    b = make_uniform(M)
    C = make_cost(N, M)
    plan, _ = solver.solve(a, b, C)
    col_sums = plan.sum(dim=0)
    assert torch.allclose(col_sums, b, atol=0.06), (
        f"Col marginals off: max error {(col_sums - b).abs().max().item():.4f}"
    )


def test_sinkhorn_wasserstein_non_negative():
    solver = SinkhornSolver()
    a = make_uniform(N)
    b = make_uniform(M)
    C = make_cost(N, M)
    dist = solver.wasserstein_distance(a, b, C)
    assert dist >= 0.0, f"Wasserstein distance must be non-negative, got {dist}"


def test_sinkhorn_identical_distributions_near_zero():
    """Identical source and target on the same cost should yield low OT cost."""
    solver = SinkhornSolver(eps=0.05, n_iters=100)
    X = torch.randn(N, D)
    C = CostMatrix.l2_cost(X, X)
    a = make_uniform(N)
    dist = solver.wasserstein_distance(a, a, C)
    # Optimal plan is identity; total cost should be near 0
    assert dist < 1e-3, f"Distance between identical distributions: {dist:.6f}"


def test_sinkhorn_plan_non_negative():
    solver = SinkhornSolver()
    a = make_uniform(N)
    b = make_uniform(M)
    C = make_cost(N, M)
    plan, _ = solver.solve(a, b, C)
    assert (plan >= 0).all(), "Transport plan has negative entries"


# ---------------------------------------------------------------------------
# EarthMoversDistance tests
# ---------------------------------------------------------------------------

def test_emd_output_shape():
    emd = EarthMoversDistance(eps=0.1, n_iters=30)
    p = torch.ones(B, N) / N
    q = torch.ones(B, M) / M
    C = make_cost(N, M)
    out = emd(p, q, C)
    assert out.shape == (B,), f"Expected shape ({B},), got {out.shape}"


def test_emd_non_negative():
    emd = EarthMoversDistance(eps=0.1, n_iters=30)
    p = torch.ones(B, N) / N
    q = torch.ones(B, M) / M
    C = make_cost(N, M)
    out = emd(p, q, C)
    assert (out >= 0).all(), "EMD values must be non-negative"


def test_emd_uniform_returns_tensor():
    emd = EarthMoversDistance()
    p = torch.ones(B, N) / N
    q = torch.ones(B, M) / M
    C = make_cost(N, M)
    out = emd(p, q, C)
    assert isinstance(out, torch.Tensor)
    assert out.numel() == B


# ---------------------------------------------------------------------------
# OTSequenceAligner tests
# ---------------------------------------------------------------------------

def test_align_sequences_plan_shape():
    solver = SinkhornSolver(eps=0.1, n_iters=50)
    aligner = OTSequenceAligner(solver)
    emb_a = torch.randn(T_A, D)
    emb_b = torch.randn(T_B, D)
    plan, _ = aligner.align_sequences(emb_a, emb_b)
    assert plan.shape == (T_A, T_B), f"Expected ({T_A}, {T_B}), got {plan.shape}"


def test_soft_align_loss_output_shape():
    solver = SinkhornSolver(eps=0.1, n_iters=30)
    aligner = OTSequenceAligner(solver)
    emb_a = torch.randn(B, T_A, D)
    emb_b = torch.randn(B, T_B, D)
    losses = aligner.soft_align_loss(emb_a, emb_b)
    assert losses.shape == (B,), f"Expected ({B},), got {losses.shape}"


def test_soft_align_loss_non_negative():
    solver = SinkhornSolver(eps=0.1, n_iters=30)
    aligner = OTSequenceAligner(solver)
    emb_a = torch.randn(B, T_A, D)
    emb_b = torch.randn(B, T_B, D)
    losses = aligner.soft_align_loss(emb_a, emb_b)
    assert (losses >= 0).all(), "Alignment losses must be non-negative"


def test_barycentric_projection_shape():
    solver = SinkhornSolver(eps=0.1, n_iters=50)
    aligner = OTSequenceAligner(solver)
    emb_a = torch.randn(T_A, D)
    emb_b = torch.randn(T_B, D)
    plan, _ = aligner.align_sequences(emb_a, emb_b)
    proj = aligner.barycentric_projection(emb_b, plan)
    assert proj.shape == (T_A, D), f"Expected ({T_A}, {D}), got {proj.shape}"


def test_barycentric_projection_finite():
    solver = SinkhornSolver(eps=0.1, n_iters=50)
    aligner = OTSequenceAligner(solver)
    emb_a = torch.randn(T_A, D)
    emb_b = torch.randn(T_B, D)
    plan, _ = aligner.align_sequences(emb_a, emb_b)
    proj = aligner.barycentric_projection(emb_b, plan)
    assert torch.isfinite(proj).all(), "Barycentric projection contains non-finite values"


# ---------------------------------------------------------------------------
# OTDistillationLoss tests
# ---------------------------------------------------------------------------

def test_ot_distillation_loss_scalar():
    loss_fn = OTDistillationLoss(eps=0.1, n_iters=30, lambda_ot=1.0)
    student = torch.randn(B, T_A, D)
    teacher = torch.randn(B, T_B, D)
    loss = loss_fn(student, teacher)
    assert loss.ndim == 0, "Loss should be a scalar (0-d tensor)"


def test_ot_distillation_loss_positive():
    loss_fn = OTDistillationLoss(eps=0.1, n_iters=30)
    student = torch.randn(B, T_A, D)
    teacher = torch.randn(B, T_B, D)
    loss = loss_fn(student, teacher)
    assert loss.item() >= 0.0, f"Distillation loss must be non-negative, got {loss.item()}"


def test_ot_distillation_loss_finite():
    loss_fn = OTDistillationLoss(eps=0.1, n_iters=30)
    student = torch.randn(B, T_A, D)
    teacher = torch.randn(B, T_B, D)
    loss = loss_fn(student, teacher)
    assert math.isfinite(loss.item()), "Distillation loss is not finite"


def test_ot_distillation_loss_same_length():
    """Should work with equal sequence lengths too (square plan)."""
    loss_fn = OTDistillationLoss(eps=0.1, n_iters=30)
    student = torch.randn(B, T_A, D)
    teacher = torch.randn(B, T_A, D)
    loss = loss_fn(student, teacher)
    assert loss.item() >= 0.0


# ---------------------------------------------------------------------------
# SlicedWasserstein tests
# ---------------------------------------------------------------------------

def test_sliced_wasserstein_non_negative():
    sw = SlicedWasserstein(n_projections=50)
    X = torch.randn(N, D)
    Y = torch.randn(M, D)
    dist = sw.distance(X, Y)
    assert dist >= 0.0, f"SlicedWasserstein must be non-negative, got {dist}"


def test_sliced_wasserstein_identical_distributions_near_zero():
    """Same cloud of points should yield ~0 distance."""
    sw = SlicedWasserstein(n_projections=100)
    X = torch.randn(N, D)
    dist = sw.distance(X, X)
    assert dist < 1e-5, f"SW distance between identical sets should be ~0, got {dist:.8f}"


def test_sliced_wasserstein_symmetric():
    """SW distance should be symmetric: d(X, Y) == d(Y, X) (same N for both)."""
    sw = SlicedWasserstein(n_projections=50)
    torch.manual_seed(1)
    X = torch.randn(N, D)
    Y = torch.randn(N, D)
    dist_xy = sw.distance(X, Y)
    dist_yx = sw.distance(Y, X)
    # With the same seed the projections differ, so check approximate equality
    assert abs(dist_xy - dist_yx) < 0.5, (
        f"SW distance should be approximately symmetric: {dist_xy:.4f} vs {dist_yx:.4f}"
    )


# ---------------------------------------------------------------------------
# OTConfig dataclass test
# ---------------------------------------------------------------------------

def test_ot_config_defaults():
    cfg = OTConfig()
    assert cfg.eps == 0.1
    assert cfg.n_iters == 50
    assert cfg.thresh == 1e-3
    assert cfg.n_projections == 50
    assert cfg.lambda_ot == 1.0
