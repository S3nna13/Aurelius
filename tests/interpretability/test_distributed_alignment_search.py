"""
tests/interpretability/test_distributed_alignment_search.py

Tests for src/interpretability/distributed_alignment_search.py

Covers all 13 rigor-floor items:
  1.  Shape: rotate(h) preserves shape (B, d_model)
  2.  Orthogonality: R^T R ≈ I after construction (atol=1e-5)
  3.  Rotation preserves norms
  4.  Interchange intervention shape
  5.  Interchange changes output for different source
  6.  Self-intervention is identity
  7.  fit() returns list of floats
  8.  fit() reduces loss over iterations
  9.  Determinism under torch.manual_seed
  10. make_orthogonal: output is orthogonal
  11. Gradient flow: rotation is differentiable
  12. Edge case: d_model == n_directions (full rotation)
  13. No NaN/Inf after fitting on random data

Pure PyTorch — no scipy, sklearn, HuggingFace, einops.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.interpretability.distributed_alignment_search import (
    DASConfig,
    DistributedAlignmentSearch,
    make_orthogonal,
)

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

D_MODEL = 32
BATCH = 8
SEED = 42


def _make_das(d_model: int = D_MODEL, **cfg_kwargs) -> DistributedAlignmentSearch:
    config = DASConfig(**cfg_kwargs)
    return DistributedAlignmentSearch(d_model, config)


def _rand(shape, seed=SEED):
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randn(*shape, generator=g)


def _simple_metric(intervened_h: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy between a linear projection of intervened_h and labels."""
    # Project to 2 logits with a fixed (non-trained) linear layer for simplicity.
    torch.manual_seed(0)
    W = torch.randn(intervened_h.shape[-1], 2)
    logits = intervened_h @ W
    return F.cross_entropy(logits, labels.long())


# ---------------------------------------------------------------------------
# 1. Shape: rotate preserves shape
# ---------------------------------------------------------------------------

def test_rotate_preserves_shape():
    das = _make_das()
    h = _rand((BATCH, D_MODEL))
    out = das.rotate(h)
    assert out.shape == h.shape, f"Expected {h.shape}, got {out.shape}"


# ---------------------------------------------------------------------------
# 2. Orthogonality: R^T R ≈ I after construction
# ---------------------------------------------------------------------------

def test_rotation_matrix_is_orthogonal():
    das = _make_das()
    R = das._get_R()
    I_approx = R.T @ R
    I_exact = torch.eye(D_MODEL)
    assert torch.allclose(I_approx, I_exact, atol=1e-5), \
        f"R^T R not close to I; max deviation = {(I_approx - I_exact).abs().max():.2e}"


# ---------------------------------------------------------------------------
# 3. Rotation preserves L2 norms
# ---------------------------------------------------------------------------

def test_rotate_preserves_norms():
    das = _make_das()
    h = _rand((BATCH, D_MODEL))
    out = das.rotate(h)
    norms_in = h.norm(dim=-1)
    norms_out = out.norm(dim=-1)
    assert torch.allclose(norms_in, norms_out, atol=1e-5), \
        "rotate() should be an isometry (norm-preserving)"


# ---------------------------------------------------------------------------
# 4. Interchange intervention shape
# ---------------------------------------------------------------------------

def test_interchange_intervention_shape():
    das = _make_das()
    base = _rand((BATCH, D_MODEL), seed=1)
    source = _rand((BATCH, D_MODEL), seed=2)
    out = das.interchange_intervention(base, source)
    assert out.shape == (BATCH, D_MODEL), \
        f"Expected ({BATCH}, {D_MODEL}), got {out.shape}"


# ---------------------------------------------------------------------------
# 5. Interchange changes output when source != base
# ---------------------------------------------------------------------------

def test_interchange_changes_output_for_different_source():
    das = _make_das(n_directions=4)
    base = _rand((BATCH, D_MODEL), seed=10)
    source = _rand((BATCH, D_MODEL), seed=20)   # deliberately different
    out = das.interchange_intervention(base, source)
    # At least some elements should differ from base.
    assert not torch.allclose(out, base, atol=1e-6), \
        "interchange_intervention should change base when source != base"


# ---------------------------------------------------------------------------
# 6. Self-intervention is identity
# ---------------------------------------------------------------------------

def test_interchange_self_is_identity():
    das = _make_das(n_directions=D_MODEL)
    h = _rand((BATCH, D_MODEL))
    out = das.interchange_intervention(h, h)
    assert torch.allclose(out, h, atol=1e-5), \
        "interchange(base, base) should equal base (self-intervention = identity)"


# ---------------------------------------------------------------------------
# 7. fit() returns list of floats
# ---------------------------------------------------------------------------

def test_fit_returns_list_of_floats():
    torch.manual_seed(SEED)
    das = _make_das(n_steps=5)
    base = _rand((BATCH, D_MODEL), seed=1)
    source = _rand((BATCH, D_MODEL), seed=2)
    labels = torch.randint(0, 2, (BATCH,))
    history = das.fit(base, source, labels, _simple_metric)
    assert isinstance(history, list), "fit() should return a list"
    assert len(history) == 5, f"Expected 5 loss values, got {len(history)}"
    assert all(isinstance(v, float) for v in history), \
        "All loss history entries should be float"


# ---------------------------------------------------------------------------
# 8. fit() reduces loss over iterations
# ---------------------------------------------------------------------------

def test_fit_reduces_loss():
    torch.manual_seed(SEED)
    das = _make_das(n_steps=50, lr=5e-2)
    base = _rand((BATCH, D_MODEL), seed=3)
    source = _rand((BATCH, D_MODEL), seed=4)
    labels = torch.randint(0, 2, (BATCH,))
    history = das.fit(base, source, labels, _simple_metric)
    # Compare first quarter mean vs last quarter mean.
    q = max(1, len(history) // 4)
    early = sum(history[:q]) / q
    late = sum(history[-q:]) / q
    assert late <= early + 0.05, \
        f"Loss did not decrease: early={early:.4f}, late={late:.4f}"


# ---------------------------------------------------------------------------
# 9. Determinism under torch.manual_seed
# ---------------------------------------------------------------------------

def test_determinism_under_seed():
    def _run(seed):
        torch.manual_seed(seed)
        das = _make_das(n_steps=3, lr=1e-2)
        base = _rand((BATCH, D_MODEL), seed=seed)
        source = _rand((BATCH, D_MODEL), seed=seed + 1)
        labels = torch.randint(0, 2, (BATCH,))
        return das.fit(base, source, labels, _simple_metric)

    h1 = _run(SEED)
    h2 = _run(SEED)
    assert h1 == h2, "Results should be identical under the same seed"


# ---------------------------------------------------------------------------
# 10. make_orthogonal: output is orthogonal
# ---------------------------------------------------------------------------

def test_make_orthogonal_output_is_orthogonal():
    torch.manual_seed(SEED)
    M = torch.randn(D_MODEL, D_MODEL)
    Q = make_orthogonal(M)
    I_approx = Q.T @ Q
    I_exact = torch.eye(D_MODEL)
    assert torch.allclose(I_approx, I_exact, atol=1e-5), \
        f"Q^T Q not close to I; max deviation = {(I_approx - I_exact).abs().max():.2e}"


# ---------------------------------------------------------------------------
# 11. Gradient flow: rotation is differentiable
# ---------------------------------------------------------------------------

def test_gradient_flows_through_rotation():
    das = _make_das()
    h = _rand((BATCH, D_MODEL))
    out = das.rotate(h)
    loss = out.sum()
    loss.backward()
    assert das.R.grad is not None, "Gradient should flow back to R"
    assert not torch.all(das.R.grad == 0), "R.grad should not be all zeros"


# ---------------------------------------------------------------------------
# 12. Edge case: d_model == n_directions (full rotation)
# ---------------------------------------------------------------------------

def test_full_rotation_case():
    """When n_directions == d_model, the entire hidden state is the subspace."""
    d = 16
    das = _make_das(d_model=d, n_directions=d)
    base = _rand((BATCH, d), seed=5)
    source = _rand((BATCH, d), seed=6)

    # Shape check.
    out = das.interchange_intervention(base, source)
    assert out.shape == (BATCH, d)

    # When d_model == n_directions, the full source is copied, so result
    # should equal source (because we swap *all* directions).
    assert torch.allclose(out, source, atol=1e-5), \
        "Full-rotation interchange should reproduce source entirely"


# ---------------------------------------------------------------------------
# 13. No NaN/Inf after fitting on random data
# ---------------------------------------------------------------------------

def test_no_nan_or_inf_after_fitting():
    torch.manual_seed(SEED)
    das = _make_das(n_steps=20, lr=1e-2)
    base = _rand((BATCH, D_MODEL), seed=7)
    source = _rand((BATCH, D_MODEL), seed=8)
    labels = torch.randint(0, 2, (BATCH,))
    history = das.fit(base, source, labels, _simple_metric)

    assert all(not (v != v) for v in history), "NaN detected in loss history"   # NaN != NaN
    assert all(v < float("inf") for v in history), "Inf detected in loss history"

    R = das._get_R()
    assert not torch.isnan(R).any(), "NaN detected in rotation matrix after fitting"
    assert not torch.isinf(R).any(), "Inf detected in rotation matrix after fitting"
