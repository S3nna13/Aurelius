"""Tests for hyperbolic_embeddings.py — Poincaré ball module."""

import math

import torch

from src.model.hyperbolic_embeddings import (
    HyperbolicConfig,
    HyperbolicLinear,
    PoincareEmbedding,
    expmap0,
    logmap0,
    poincare_distance,
    project_to_ball,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
DIM = 16
N_EMBS = 32
B = 4
C = 1.0


# ---------------------------------------------------------------------------
# HyperbolicConfig
# ---------------------------------------------------------------------------


def test_hyperbolic_config_defaults():
    cfg = HyperbolicConfig()
    assert cfg.dim == 64
    assert cfg.curvature == 1.0
    assert cfg.clip_r == 2.3
    assert cfg.eps == 1e-5


# ---------------------------------------------------------------------------
# poincare_distance
# ---------------------------------------------------------------------------


def test_poincare_distance_non_negative():
    u = torch.randn(B, DIM) * 0.3
    v = torch.randn(B, DIM) * 0.3
    d = poincare_distance(u, v, c=C)
    assert (d >= 0).all(), "Distances must be non-negative"


def test_poincare_distance_same_point_is_zero():
    u = torch.randn(B, DIM) * 0.3
    d = poincare_distance(u, u, c=C)
    # same point → distance should be ~0
    assert (d < 1e-3).all(), f"Same-point distance should be ~0, got {d}"


def test_poincare_distance_symmetric():
    u = torch.randn(B, DIM) * 0.3
    v = torch.randn(B, DIM) * 0.3
    d_uv = poincare_distance(u, v, c=C)
    d_vu = poincare_distance(v, u, c=C)
    assert torch.allclose(d_uv, d_vu, atol=1e-5), "Distance must be symmetric"


# ---------------------------------------------------------------------------
# expmap0
# ---------------------------------------------------------------------------


def test_expmap0_output_inside_unit_ball():
    """For c=1 the Poincaré ball has radius 1; output norms should be < 1."""
    v = torch.randn(B, DIM)
    y = expmap0(v, c=C)
    norms = y.norm(dim=-1)
    assert (norms < 1.0).all(), (
        f"expmap0 output should be inside unit ball, got max norm {norms.max()}"
    )


# ---------------------------------------------------------------------------
# logmap0
# ---------------------------------------------------------------------------


def test_logmap0_inverse_of_expmap0():
    # Use small-norm vectors so tanh doesn't saturate and roundtrip is exact.
    v = torch.randn(B, DIM) * 0.2
    y = expmap0(v, c=C)
    v_recovered = logmap0(y, c=C)
    assert torch.allclose(v, v_recovered, atol=1e-5), "logmap0(expmap0(v)) should recover v"


# ---------------------------------------------------------------------------
# project_to_ball
# ---------------------------------------------------------------------------


def test_project_to_ball_norm_clipped():
    cfg = HyperbolicConfig(dim=DIM, curvature=C, clip_r=2.3)
    x = torch.randn(B, DIM) * 10.0  # large vectors well outside ball
    projected = project_to_ball(x, c=cfg.curvature, clip_r=cfg.clip_r)
    max_norm = cfg.clip_r / math.sqrt(cfg.curvature)
    norms = projected.norm(dim=-1)
    assert (norms <= max_norm + 1e-6).all(), (
        f"Projected norms should be <= {max_norm}, got {norms.max()}"
    )


def test_project_to_ball_inball_points_unchanged():
    """Points already inside the ball must not be moved."""
    cfg = HyperbolicConfig(dim=DIM, curvature=C, clip_r=2.3)
    max_norm = cfg.clip_r / math.sqrt(cfg.curvature)
    # Create vectors with norm = 0.5 * max_norm
    x = torch.randn(B, DIM)
    x = x / x.norm(dim=-1, keepdim=True) * (0.5 * max_norm)
    projected = project_to_ball(x, c=cfg.curvature, clip_r=cfg.clip_r)
    assert torch.allclose(x, projected, atol=1e-6), "In-ball points should not be moved"


# ---------------------------------------------------------------------------
# PoincareEmbedding
# ---------------------------------------------------------------------------


def test_poincare_embedding_output_shape():
    cfg = HyperbolicConfig(dim=DIM)
    emb = PoincareEmbedding(N_EMBS, DIM, cfg)
    idx = torch.randint(0, N_EMBS, (B,))
    out = emb(idx)
    assert out.shape == (B, DIM), f"Expected shape {(B, DIM)}, got {out.shape}"


def test_poincare_embedding_output_inside_ball():
    """c=1 → Poincaré ball has radius 1; norms < 1."""
    cfg = HyperbolicConfig(dim=DIM, curvature=1.0)
    emb = PoincareEmbedding(N_EMBS, DIM, cfg)
    idx = torch.randint(0, N_EMBS, (B,))
    out = emb(idx)
    norms = out.norm(dim=-1)
    assert (norms < 1.0).all(), f"Embeddings must lie inside unit ball, got max norm {norms.max()}"


def test_poincare_embedding_differentiable():
    cfg = HyperbolicConfig(dim=DIM)
    emb = PoincareEmbedding(N_EMBS, DIM, cfg)
    idx = torch.randint(0, N_EMBS, (B,))
    out = emb(idx)
    loss = out.sum()
    loss.backward()
    assert emb.weight.grad is not None, "Gradients must flow back to embedding weights"


# ---------------------------------------------------------------------------
# HyperbolicLinear
# ---------------------------------------------------------------------------


def test_hyperbolic_linear_output_inside_ball():
    cfg = HyperbolicConfig(dim=DIM, curvature=1.0)
    layer = HyperbolicLinear(DIM, DIM, cfg)
    # Create inputs already inside unit ball via expmap0
    v = torch.randn(B, DIM) * 0.3
    x = expmap0(v, c=cfg.curvature)
    out = layer(x)
    norms = out.norm(dim=-1)
    assert (norms < 1.0).all(), (
        f"HyperbolicLinear output must be inside unit ball, got max norm {norms.max()}"
    )


def test_hyperbolic_linear_output_shape():
    in_dim, out_dim = DIM, DIM * 2
    cfg = HyperbolicConfig(dim=DIM)
    layer = HyperbolicLinear(in_dim, out_dim, cfg)
    v = torch.randn(B, in_dim) * 0.3
    x = expmap0(v, c=cfg.curvature)
    out = layer(x)
    assert out.shape == (B, out_dim), f"Expected shape {(B, out_dim)}, got {out.shape}"


# ---------------------------------------------------------------------------
# Different indices → different embeddings
# ---------------------------------------------------------------------------


def test_different_indices_give_different_embeddings():
    cfg = HyperbolicConfig(dim=DIM)
    emb = PoincareEmbedding(N_EMBS, DIM, cfg)
    # Pick two distinct indices
    idx_a = torch.tensor([0])
    idx_b = torch.tensor([1])
    out_a = emb(idx_a)
    out_b = emb(idx_b)
    assert not torch.allclose(out_a, out_b), "Different indices must produce different embeddings"
