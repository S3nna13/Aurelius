"""Tests for GaLore (Gradient Low-Rank Projection) optimizer.

Reference: Zhao et al., arXiv:2403.03507
Module: src/training/galore.py
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from src.training.galore import GaLoreAdamW, GaLoreProjector, _compute_P

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

M, N, RANK = 64, 128, 8
T_PROJ = 5


def _make_grad(m: int = M, n: int = N, seed: int = 0) -> torch.Tensor:
    g = torch.manual_seed(seed) and torch.randn(m, n)
    return torch.randn(m, n)


def _projector(rank: int = RANK, gap: int = T_PROJ) -> GaLoreProjector:
    return GaLoreProjector(rank=rank, update_proj_gap=gap, scale=1.0)


def _tiny_model() -> nn.Linear:
    torch.manual_seed(42)
    return nn.Linear(N, M, bias=True)  # weight: (M, N), bias: (M,)


def _tiny_loss(model: nn.Linear) -> torch.Tensor:
    x = torch.randn(4, N)
    return model(x).sum()


# ---------------------------------------------------------------------------
# 1. project() output shape is (rank, n)
# ---------------------------------------------------------------------------

def test_project_output_shape():
    proj = _projector()
    G = torch.randn(M, N)
    G_tilde = proj.project(G)
    assert G_tilde.shape == (RANK, N), (
        f"Expected ({RANK}, {N}), got {G_tilde.shape}"
    )


# ---------------------------------------------------------------------------
# 2. unproject(project(G)) has same shape as G
# ---------------------------------------------------------------------------

def test_unproject_restores_shape():
    proj = _projector()
    G = torch.randn(M, N)
    G_tilde = proj.project(G)
    delta_W = proj.unproject(G_tilde)
    assert delta_W.shape == G.shape, (
        f"Expected {G.shape}, got {delta_W.shape}"
    )


# ---------------------------------------------------------------------------
# 3. P is updated every T_proj steps, not every step
# ---------------------------------------------------------------------------

def test_projection_updates_only_at_gap():
    proj = _projector(gap=T_PROJ)
    G = torch.randn(M, N)

    # First project — step 0 → update happens, step becomes 1
    proj.project(G)
    P_after_first = proj.P_t.clone()

    # Steps 1..T_PROJ-1: same P
    for _ in range(T_PROJ - 1):
        proj.project(G + torch.randn(M, N) * 10)  # different gradients
        assert torch.allclose(proj.P_t, P_after_first), (
            "P_t changed before T_proj steps elapsed"
        )

    # Step T_PROJ: P should update
    proj.project(G * -5)  # very different gradient
    assert not torch.allclose(proj.P_t, P_after_first, atol=1e-6), (
        "P_t did NOT change at T_proj step"
    )


# ---------------------------------------------------------------------------
# 4. P columns are orthonormal (P^T P ≈ I, atol=1e-5)
# ---------------------------------------------------------------------------

def test_P_columns_orthonormal():
    proj = _projector()
    G = torch.randn(M, N)
    proj.project(G)
    P = proj.P_t  # (M, RANK)
    PtP = P.T @ P  # (RANK, RANK)
    eye = torch.eye(RANK, dtype=P.dtype)
    assert torch.allclose(PtP, eye, atol=1e-5), (
        f"P^T P is not identity. Max deviation: {(PtP - eye).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# 5. GaLoreAdamW step does not crash on a tiny linear model
# ---------------------------------------------------------------------------

def test_optimizer_step_no_crash():
    model = _tiny_model()
    optimizer = GaLoreAdamW(model.parameters(), lr=1e-3, rank=RANK,
                             update_proj_gap=T_PROJ)
    loss = _tiny_loss(model)
    loss.backward()
    optimizer.step()  # must not raise


# ---------------------------------------------------------------------------
# 6. Weight changes after step (params actually move)
# ---------------------------------------------------------------------------

def test_weights_change_after_step():
    model = _tiny_model()
    w_before = model.weight.data.clone()
    optimizer = GaLoreAdamW(model.parameters(), lr=1e-3, rank=RANK,
                             update_proj_gap=T_PROJ)
    loss = _tiny_loss(model)
    loss.backward()
    optimizer.step()
    assert not torch.allclose(model.weight.data, w_before), (
        "Weight did not change after optimizer step"
    )


# ---------------------------------------------------------------------------
# 7. Optimizer state (m1, m2) stored in rank×n space, not m×n
# ---------------------------------------------------------------------------

def test_optimizer_state_is_low_rank():
    model = _tiny_model()
    optimizer = GaLoreAdamW(model.parameters(), lr=1e-3, rank=RANK,
                             update_proj_gap=T_PROJ)
    loss = _tiny_loss(model)
    loss.backward()
    optimizer.step()

    # Find the weight parameter state
    w_param = model.weight
    state = optimizer.state[w_param]
    m1 = state["m1"]
    m2 = state["m2"]

    expected_r = min(RANK, M, N)  # clamped rank
    assert m1.shape == (expected_r, N), (
        f"m1 shape {m1.shape} != ({expected_r}, {N})"
    )
    assert m2.shape == (expected_r, N), (
        f"m2 shape {m2.shape} != ({expected_r}, {N})"
    )
    # Verify they're NOT full-rank (m1 rows < M)
    assert m1.shape[0] < M, "m1 should have fewer rows than M (memory saving)"


# ---------------------------------------------------------------------------
# 8. 1D params (bias) use standard update, not projected
# ---------------------------------------------------------------------------

def test_bias_uses_standard_adamw():
    model = _tiny_model()
    optimizer = GaLoreAdamW(model.parameters(), lr=1e-3, rank=RANK,
                             update_proj_gap=T_PROJ)
    bias_before = model.bias.data.clone()
    loss = _tiny_loss(model)
    loss.backward()
    optimizer.step()

    state = optimizer.state[model.bias]
    # Standard AdamW state: m1/m2 same shape as bias (1D)
    assert "m1" in state, "1D bias should have m1 moment in state"
    assert state["m1"].shape == model.bias.shape, (
        f"Bias m1 shape {state['m1'].shape} != bias shape {model.bias.shape}"
    )
    # No projector for 1D
    assert "projector" not in state, "Bias should NOT have a GaLoreProjector"
    # Bias should have changed
    assert not torch.allclose(model.bias.data, bias_before), (
        "Bias did not change after optimizer step"
    )


# ---------------------------------------------------------------------------
# 9. Gradient flow: loss.backward() + step() yields finite params
# ---------------------------------------------------------------------------

def test_gradient_flow_finite_params():
    model = _tiny_model()
    optimizer = GaLoreAdamW(model.parameters(), lr=1e-3, rank=RANK,
                             update_proj_gap=T_PROJ)
    for _ in range(3):
        optimizer.zero_grad()
        loss = _tiny_loss(model)
        loss.backward()
        optimizer.step()

    for name, p in model.named_parameters():
        assert torch.isfinite(p).all(), f"Non-finite values in {name}"


# ---------------------------------------------------------------------------
# 10. Determinism: same result with same seed
# ---------------------------------------------------------------------------

def test_determinism():
    def _run(seed: int) -> torch.Tensor:
        torch.manual_seed(seed)
        model = nn.Linear(N, M, bias=True)
        optimizer = GaLoreAdamW(model.parameters(), lr=1e-3, rank=RANK,
                                 update_proj_gap=T_PROJ)
        torch.manual_seed(seed + 100)
        x = torch.randn(4, N)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        return model.weight.data.clone()

    w1 = _run(7)
    w2 = _run(7)
    assert torch.allclose(w1, w2), "Results differ for same seed — not deterministic"


# ---------------------------------------------------------------------------
# 11. Edge: rank >= min(m,n) → clamp, no crash
# ---------------------------------------------------------------------------

def test_rank_clamp_no_crash():
    m, n = 8, 4
    huge_rank = 100  # >> min(m, n) = 4
    proj = GaLoreProjector(rank=huge_rank, update_proj_gap=1)
    G = torch.randn(m, n)
    G_tilde = proj.project(G)
    # rank is clamped to min(m, n) = 4
    assert G_tilde.shape[0] <= min(m, n), (
        f"rank not clamped: G_tilde.shape = {G_tilde.shape}"
    )
    delta_W = proj.unproject(G_tilde)
    assert delta_W.shape == (m, n)


# ---------------------------------------------------------------------------
# 12. Numerical stability: no NaN/Inf with large gradients (scale 1000x)
# ---------------------------------------------------------------------------

def test_numerical_stability_large_gradients():
    model = _tiny_model()
    optimizer = GaLoreAdamW(model.parameters(), lr=1e-3, rank=RANK,
                             update_proj_gap=T_PROJ)
    optimizer.zero_grad()
    loss = _tiny_loss(model)
    loss.backward()

    # Scale up gradients 1000x
    for p in model.parameters():
        if p.grad is not None:
            p.grad.mul_(1000.0)

    optimizer.step()

    for name, p in model.named_parameters():
        assert torch.isfinite(p).all(), (
            f"NaN/Inf in {name} after large-gradient step"
        )


# ---------------------------------------------------------------------------
# 13. SVD recomputation gap: after T_proj steps, projection direction changes
# ---------------------------------------------------------------------------

def test_svd_recomputation_changes_projection():
    proj = _projector(gap=T_PROJ)

    # First projection
    torch.manual_seed(0)
    G0 = torch.randn(M, N)
    proj.project(G0)
    P_initial = proj.P_t.clone()

    # Fill remaining steps in the gap with similar gradients
    for _ in range(T_PROJ - 1):
        proj.project(G0.clone())  # identical grad → same step counter advance

    # At step T_PROJ the projection must re-run with a very different gradient
    G_new = torch.randn(M, N) * 50  # completely different
    proj.project(G_new)
    P_new = proj.P_t.clone()

    # Subspaces should differ (won't be identical for random unrelated gradients)
    cos_sim = (P_initial * P_new).sum().abs() / (
        P_initial.norm() * P_new.norm() + 1e-12
    )
    # They should not be identical (columns in a different order would still differ
    # in the Frobenius sense for unrelated random matrices)
    assert not torch.allclose(P_initial, P_new, atol=1e-4), (
        "Projection matrix did not change at T_proj step with different gradient"
    )


# ---------------------------------------------------------------------------
# 14. Scale parameter: scale=0.5 halves the effective update magnitude
# ---------------------------------------------------------------------------

def test_scale_parameter_halves_update():
    """scale=0.5 should halve the unprojected update compared to scale=1.0."""
    torch.manual_seed(99)
    G = torch.randn(M, N)

    # Projector with scale=1.0
    proj1 = GaLoreProjector(rank=RANK, update_proj_gap=1, scale=1.0)
    G_tilde1 = proj1.project(G.clone())
    delta1 = proj1.unproject(G_tilde1)

    # Projector with scale=0.5 — use same gradient so P_t is identical
    proj2 = GaLoreProjector(rank=RANK, update_proj_gap=1, scale=0.5)
    # Force same P_t
    proj2.P_t = proj1.P_t.clone()
    proj2._step = 1  # skip SVD on next project call since step % gap != 0
    G_tilde2 = proj2.project(G.clone())
    delta2 = proj2.unproject(G_tilde2)

    ratio = delta1.norm() / (delta2.norm() + 1e-12)
    assert abs(ratio.item() - 2.0) < 1e-4, (
        f"Expected scale=0.5 to give half update; ratio was {ratio.item():.6f}"
    )
