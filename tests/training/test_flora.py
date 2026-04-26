"""Tests for src/training/flora.py — FLoRA: Full-rank LoRA via Random Projection.

Coverage targets (14 tests):
 1.  project() output shape is (rank, rank)
 2.  unproject(project(G)) has same shape as G
 3.  project() produces different outputs at different steps (fresh randomness)
 4.  FLoRAOptimizer: params move after one step
 5.  Param movement is finite (no NaN/Inf)
 6.  1-D params use standard Adam (no FLoRA projection path)
 7.  Convergence: loss decreases over 20 steps on simple quadratic
 8.  Determinism: same seed_offset + step → same projection
 9.  rank=1 works without crash
10.  Large gradient stability: no NaN/Inf with 1000× scale
11.  R and C drawn fresh each step (different matrices at step 0 vs step 1)
12.  Adam moments stored in (rank, rank) shape (memory efficiency)
13.  weight_decay modifies update correctly when non-zero
14.  Gradient accumulation: multiple backward passes before step works
"""

import torch

from src.training.flora import FLoRAOptimizer, FLoRAProjector

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_param(m: int, n: int, requires_grad: bool = True) -> torch.Tensor:
    """Create a 2-D parameter with grad."""
    p = torch.nn.Parameter(torch.randn(m, n))
    return p


def _quadratic_loss(param: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Simple quadratic ‖param - target‖² / 2."""
    return 0.5 * ((param - target) ** 2).sum()


# ---------------------------------------------------------------------------
# Test 1: project() output shape is (rank, rank)
# ---------------------------------------------------------------------------


def test_project_output_shape():
    m, n, r = 16, 12, 4
    proj = FLoRAProjector(m, n, r)
    G = torch.randn(m, n)
    G_tilde = proj.project(G, step=0)
    assert G_tilde.shape == (r, r), f"Expected ({r}, {r}), got {G_tilde.shape}"


# ---------------------------------------------------------------------------
# Test 2: unproject(project(G)) has same shape as G
# ---------------------------------------------------------------------------


def test_unproject_restores_shape():
    m, n, r = 8, 6, 3
    proj = FLoRAProjector(m, n, r)
    G = torch.randn(m, n)
    G_tilde = proj.project(G, step=0)
    delta_W = proj.unproject(G_tilde, step=0)
    assert delta_W.shape == (m, n), f"Expected ({m}, {n}), got {delta_W.shape}"


# ---------------------------------------------------------------------------
# Test 3: project() produces different outputs at different steps
# ---------------------------------------------------------------------------


def test_project_fresh_randomness_per_step():
    m, n, r = 10, 8, 4
    proj = FLoRAProjector(m, n, r)
    G = torch.randn(m, n)
    G_tilde_0 = proj.project(G, step=0)
    G_tilde_1 = proj.project(G, step=1)
    # Different seeds → different projections → different compressed gradients
    assert not torch.allclose(G_tilde_0, G_tilde_1), (
        "project() should yield different results at different steps"
    )


# ---------------------------------------------------------------------------
# Test 4: FLoRAOptimizer — params move after one step
# ---------------------------------------------------------------------------


def test_optimizer_params_move():
    p = _make_param(8, 6)
    p_initial = p.data.clone()
    opt = FLoRAOptimizer([p], lr=0.01, rank=4)
    loss = (p**2).sum()
    loss.backward()
    opt.step()
    assert not torch.allclose(p.data, p_initial), "Parameter should have moved after step"


# ---------------------------------------------------------------------------
# Test 5: Param movement is finite (no NaN/Inf)
# ---------------------------------------------------------------------------


def test_no_nan_inf_after_step():
    p = _make_param(12, 10)
    opt = FLoRAOptimizer([p], lr=0.01, rank=4)
    loss = (p**2).sum()
    loss.backward()
    opt.step()
    assert torch.isfinite(p.data).all(), "Parameter contains NaN or Inf after step"


# ---------------------------------------------------------------------------
# Test 6: 1-D params use standard Adam (no FLoRA path)
# ---------------------------------------------------------------------------


def test_1d_param_uses_standard_adam():
    """1-D parameter should update via Adam without crashing (no projector created)."""
    p_1d = torch.nn.Parameter(torch.randn(16))
    opt = FLoRAOptimizer([p_1d], lr=0.01, rank=4)
    p_initial = p_1d.data.clone()
    loss = (p_1d**2).sum()
    loss.backward()
    opt.step()
    # Should have moved and have no NaN
    assert not torch.allclose(p_1d.data, p_initial), "1-D param should have moved"
    assert torch.isfinite(p_1d.data).all(), "1-D param has NaN/Inf"
    # Confirm no projector stored in state
    state = opt.state[p_1d]
    assert "projector" not in state, "1-D param should not have a FLoRAProjector in state"


# ---------------------------------------------------------------------------
# Test 7: Convergence on simple quadratic over 20 steps
# ---------------------------------------------------------------------------


def test_convergence_quadratic():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.ones(8, 8))
    target = torch.zeros(8, 8)
    opt = FLoRAOptimizer([p], lr=0.05, rank=4, betas=(0.9, 0.999))

    losses = []
    for _ in range(20):
        opt.zero_grad()
        loss = _quadratic_loss(p, target)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], (
        f"Loss should decrease over 20 steps; first={losses[0]:.4f}, last={losses[-1]:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 8: Determinism — same seed_offset + step → same projection
# ---------------------------------------------------------------------------


def test_determinism_same_seed():
    m, n, r = 8, 6, 3
    seed = 42
    proj_a = FLoRAProjector(m, n, r, seed_offset=seed)
    proj_b = FLoRAProjector(m, n, r, seed_offset=seed)
    G = torch.randn(m, n)

    for step in [0, 5, 99]:
        a = proj_a.project(G, step=step)
        b = proj_b.project(G, step=step)
        assert torch.allclose(a, b), f"Projections with same seed should match at step={step}"


# ---------------------------------------------------------------------------
# Test 9: rank=1 works without crash
# ---------------------------------------------------------------------------


def test_rank_1_no_crash():
    m, n = 6, 5
    proj = FLoRAProjector(m, n, rank=1)
    G = torch.randn(m, n)
    G_tilde = proj.project(G, step=0)
    assert G_tilde.shape == (1, 1)
    delta_W = proj.unproject(G_tilde, step=0)
    assert delta_W.shape == (m, n)

    # Also test through optimizer
    p = _make_param(m, n)
    opt = FLoRAOptimizer([p], lr=0.01, rank=1)
    loss = (p**2).sum()
    loss.backward()
    opt.step()  # must not crash


# ---------------------------------------------------------------------------
# Test 10: Large gradient stability (1000× scale → no NaN/Inf)
# ---------------------------------------------------------------------------


def test_large_gradient_stability():
    p = _make_param(8, 6)
    opt = FLoRAOptimizer([p], lr=0.01, rank=4)
    loss = (p * 1000.0) ** 2
    loss.sum().backward()
    opt.step()
    assert torch.isfinite(p.data).all(), "NaN/Inf after large-gradient step"


# ---------------------------------------------------------------------------
# Test 11: R and C drawn fresh each step (different matrices step 0 vs step 1)
# ---------------------------------------------------------------------------


def test_fresh_R_C_each_step():
    """Verify that the projector internally uses different R_t, C_t per step."""
    m, n, r = 8, 6, 3
    proj = FLoRAProjector(m, n, r, seed_offset=0)

    # Use a fixed unit gradient so any difference is purely from R/C changing
    G = torch.ones(m, n)
    G_tilde_0 = proj.project(G, step=0)
    G_tilde_1 = proj.project(G, step=1)

    assert not torch.allclose(G_tilde_0, G_tilde_1), (
        "Compressed gradient should differ at step 0 vs step 1 due to fresh R_t, C_t"
    )


# ---------------------------------------------------------------------------
# Test 12: Adam moments stored in (rank, rank) shape
# ---------------------------------------------------------------------------


def test_adam_moments_in_compressed_space():
    m, n, r = 12, 10, 4
    p = _make_param(m, n)
    opt = FLoRAOptimizer([p], lr=0.01, rank=r)
    loss = (p**2).sum()
    loss.backward()
    opt.step()

    state = opt.state[p]
    assert "m1" in state and "m2" in state, "State should contain m1, m2"
    assert state["m1"].shape == (r, r), f"m1 should be ({r}, {r}), got {state['m1'].shape}"
    assert state["m2"].shape == (r, r), f"m2 should be ({r}, {r}), got {state['m2'].shape}"


# ---------------------------------------------------------------------------
# Test 13: weight_decay modifies the update when non-zero
# ---------------------------------------------------------------------------


def test_weight_decay_effect():
    """Parameters with weight_decay>0 should shrink more than without."""
    torch.manual_seed(7)
    base = torch.randn(6, 5)

    p_no_wd = torch.nn.Parameter(base.clone())
    p_wd = torch.nn.Parameter(base.clone())

    opt_no_wd = FLoRAOptimizer([p_no_wd], lr=0.01, rank=3, weight_decay=0.0, seed_offset=0)
    opt_wd = FLoRAOptimizer([p_wd], lr=0.01, rank=3, weight_decay=1.0, seed_offset=0)

    for opt, param in [(opt_no_wd, p_no_wd), (opt_wd, p_wd)]:
        loss = (param**2).sum()
        loss.backward()
        opt.step()

    # Weight decay should produce a smaller norm
    assert p_wd.data.norm() < p_no_wd.data.norm(), (
        "weight_decay>0 should shrink parameter norm compared to weight_decay=0"
    )


# ---------------------------------------------------------------------------
# Test 14: Gradient accumulation — multiple backward passes before step
# ---------------------------------------------------------------------------


def test_gradient_accumulation():
    """Accumulating gradients over several backward passes before calling step()
    should work without NaN/Inf, and params should move."""
    p = _make_param(8, 6)
    opt = FLoRAOptimizer([p], lr=0.01, rank=4)
    p_initial = p.data.clone()

    opt.zero_grad()
    # Simulate gradient accumulation: three micro-batches
    for _ in range(3):
        loss = (p**2).sum() / 3.0
        loss.backward()  # gradients accumulate in p.grad

    opt.step()

    assert torch.isfinite(p.data).all(), "NaN/Inf after gradient accumulation step"
    assert not torch.allclose(p.data, p_initial), "Params should move after accumulation step"
