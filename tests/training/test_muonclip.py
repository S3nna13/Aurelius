"""Tests for MuonClip optimizer."""

import torch
import torch.nn as nn
import pytest

from src.training.muonclip import MuonClip


# ---------------------------------------------------------------------------
# Test 1: basic convergence
# ---------------------------------------------------------------------------

def test_muonclip_step_reduces_loss():
    """Optimizing a simple quadratic over 10 steps should reduce loss."""
    torch.manual_seed(0)
    W = nn.Parameter(torch.randn(8, 8))
    target = torch.zeros(8, 8)

    optimizer = MuonClip([W], lr=0.02)

    initial_loss = None
    for _ in range(10):
        optimizer.zero_grad()
        loss = (W - target).pow(2).sum()
        if initial_loss is None:
            initial_loss = loss.item()
        loss.backward()
        optimizer.step()

    final_loss = (W - target).pow(2).sum().item()
    assert final_loss < initial_loss, (
        f"Loss did not decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 2: Newton-Schulz orthogonality
# ---------------------------------------------------------------------------

def test_newton_schulz_orthogonalizes():
    """After orthogonalization, G @ G.T should be approximately proportional to I."""
    torch.manual_seed(1)
    opt = MuonClip([nn.Parameter(torch.randn(1))], lr=0.02)  # dummy param to instantiate

    G = torch.randn(16, 16)
    G_orth = opt.newton_schulz_orthogonalize(G, steps=5)

    # Normalize columns so we can check near-orthogonality up to scale
    # Use the Frobenius-normalized version: divide by its own norm first
    norm = G_orth.norm()
    G_norm = G_orth / norm * (16 ** 0.5)   # scale so expected singular values ≈ 1

    product = G_norm @ G_norm.T
    identity = torch.eye(16)

    # Off-diagonal elements should be small, diagonal ≈ 1
    error = (product - identity).abs().mean()
    assert error < 0.5, f"G @ G.T is not close enough to I (mean abs error={error:.4f})"


# ---------------------------------------------------------------------------
# Test 3: gradient clipping reduces spike
# ---------------------------------------------------------------------------

def test_gradient_clipping():
    """Injecting a spike gradient, clipping should reduce the max absolute value."""
    opt = MuonClip([nn.Parameter(torch.randn(1))], lr=0.02)

    grad = torch.ones(100)
    spike_idx = 50
    grad[spike_idx] = 1e6

    clipped = opt._clip_gradient(grad, clip_percentile=99.0)

    # The spike should be dramatically reduced
    assert clipped[spike_idx].item() < 1e5, (
        f"Spike was not clipped: clipped value = {clipped[spike_idx].item()}"
    )
    # Normal values should be preserved (they are below the 99th percentile)
    assert clipped[0].item() == pytest.approx(1.0, abs=1e-5), (
        "Normal values were incorrectly clipped"
    )


# ---------------------------------------------------------------------------
# Test 4: momentum buffer populated after first step (matrix params)
# ---------------------------------------------------------------------------

def test_matrix_param_uses_momentum():
    """After the first step, a 2D parameter should have a momentum_buffer in state."""
    torch.manual_seed(2)
    W = nn.Parameter(torch.randn(8, 8))
    optimizer = MuonClip([W], lr=0.02)

    loss = W.pow(2).sum()
    loss.backward()
    optimizer.step()

    assert "momentum_buffer" in optimizer.state[W], (
        "momentum_buffer not found in optimizer state after first step"
    )
    assert optimizer.state[W]["momentum_buffer"].shape == (8, 8)


# ---------------------------------------------------------------------------
# Test 5: 1D parameter falls back to Adam state
# ---------------------------------------------------------------------------

def test_vector_param_fallback():
    """A 1D parameter should accumulate Adam-style state (exp_avg, exp_avg_sq)."""
    torch.manual_seed(3)
    bias = nn.Parameter(torch.randn(16))
    optimizer = MuonClip([bias], lr=0.02)

    loss = bias.pow(2).sum()
    loss.backward()
    optimizer.step()

    state = optimizer.state[bias]
    assert "exp_avg" in state, "exp_avg not found in state for 1D parameter"
    assert "exp_avg_sq" in state, "exp_avg_sq not found in state for 1D parameter"
    assert state["exp_avg"].shape == (16,)
    assert state["exp_avg_sq"].shape == (16,)


# ---------------------------------------------------------------------------
# Test 6: tiny MoE simulation — no NaN/Inf in outputs
# ---------------------------------------------------------------------------

def test_muonclip_with_moe_params():
    """MuonClip should handle a tiny MoE (router + expert FFNs) without NaN/Inf."""
    torch.manual_seed(4)

    num_experts = 4
    d_model = 64
    d_ff = 128
    batch = 8

    # Router: Linear 64 → 8 (top-k selection)
    router = nn.Linear(d_model, num_experts * 2, bias=False)

    # Expert FFNs (simple 2-layer MLPs)
    experts = nn.ModuleList([
        nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False),
        )
        for _ in range(num_experts)
    ])

    # Collect 2D weight params (Muon) and 1D params (Adam fallback)
    matrix_params = []
    vector_params = []
    for module in [router] + list(experts):
        for p in module.parameters():
            if p.ndim >= 2:
                matrix_params.append(p)
            else:
                vector_params.append(p)

    # MuonClip handles both in one group (or two groups with different lr)
    all_params = matrix_params + vector_params
    optimizer = MuonClip(all_params, lr=0.02, clip_percentile=99.0)

    x = torch.randn(batch, d_model)

    # Forward: simple router + expert sum (no top-k for simplicity)
    router_logits = router(x)                          # (batch, num_experts*2)
    router_weights = router_logits.softmax(dim=-1)     # (batch, num_experts*2)

    # Use only first num_experts weights for the actual experts
    expert_out = torch.stack(
        [expert(x) for expert in experts], dim=1
    )  # (batch, num_experts, d_model)

    w = router_weights[:, :num_experts].unsqueeze(-1)  # (batch, num_experts, 1)
    output = (w * expert_out).sum(dim=1)               # (batch, d_model)

    loss = output.pow(2).mean()
    loss.backward()
    optimizer.step()

    # Check no NaN/Inf in any parameter after the update
    for module in [router] + list(experts):
        for p in module.parameters():
            assert not torch.isnan(p).any(), f"NaN found in {p.shape} parameter after step"
            assert not torch.isinf(p).any(), f"Inf found in {p.shape} parameter after step"
