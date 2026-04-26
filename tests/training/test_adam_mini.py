"""Tests for Adam-mini optimizer (src/training/adam_mini.py).

Covers the 14 test cases specified in the implementation spec:
1.  Params move after one step
2.  Param movement is finite (no NaN/Inf)
3.  Determinism under seed
4.  Adam-mini memory: v state has fewer elements than param (n_heads > 1 case)
5.  v_block is scalar per block — check state storage shape
6.  Single-block mode: v has shape () or (1,)
7.  Converges on quadratic loss over 20 steps
8.  weight_decay non-zero modifies update
9.  beta1=0: m = g_t (no momentum, pure gradient)
10. Large gradient stability: no NaN/Inf at scale 1000
11. 1D params (biases): single-block v (scalar)
12. 2D param without head config: single-block v (one scalar per param)
13. 2D param with head config (d_model=64, n_heads=4, head_dim=16): 4 v scalars
14. Head-blocked update: different blocks can have different v values
"""

import torch

from src.training.adam_mini import AdamMini

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_param(shape, requires_grad=True, seed=42):
    torch.manual_seed(seed)
    return torch.nn.Parameter(torch.randn(*shape))


def _one_step(param, opt, loss_fn=None):
    opt.zero_grad()
    if loss_fn is None:
        loss = param.sum()
    else:
        loss = loss_fn(param)
    loss.backward()
    opt.step()
    return loss.item()


# ---------------------------------------------------------------------------
# Test 1: Params move after one step
# ---------------------------------------------------------------------------


def test_params_move_after_one_step():
    p = _make_param((8, 8))
    p_init = p.data.clone()
    opt = AdamMini([p], lr=1e-3)
    _one_step(p, opt)
    assert not torch.allclose(p.data, p_init), "Parameter did not change after one step."


# ---------------------------------------------------------------------------
# Test 2: Param movement is finite (no NaN/Inf)
# ---------------------------------------------------------------------------


def test_param_finite_after_step():
    p = _make_param((16, 16))
    opt = AdamMini([p], lr=1e-3)
    _one_step(p, opt)
    assert torch.isfinite(p.data).all(), "Parameter contains NaN or Inf after step."


# ---------------------------------------------------------------------------
# Test 3: Determinism under seed
# ---------------------------------------------------------------------------


def test_determinism_under_seed():
    def run():
        torch.manual_seed(0)
        p = torch.nn.Parameter(torch.randn(8, 8))
        opt = AdamMini([p], lr=1e-3)
        _one_step(p, opt)
        return p.data.clone()

    r1 = run()
    r2 = run()
    assert torch.allclose(r1, r2), "Results differ across two runs with same seed."


# ---------------------------------------------------------------------------
# Test 4: v state has fewer elements than param (n_heads > 1)
# ---------------------------------------------------------------------------


def test_v_has_fewer_elements_than_param_head_blocked():
    n_heads, head_dim = 4, 16
    d_model = n_heads * head_dim  # 64
    p = _make_param((d_model, d_model))
    opt = AdamMini([p], lr=1e-3, n_heads=n_heads, head_dim=head_dim)
    _one_step(p, opt)
    state = opt.state[p]
    v_numel = state["v"].numel()
    param_numel = p.numel()
    assert v_numel < param_numel, (
        f"v should have fewer elements than param: v={v_numel}, param={param_numel}"
    )


# ---------------------------------------------------------------------------
# Test 5: v_block is scalar per block — check state storage shape
# ---------------------------------------------------------------------------


def test_v_shape_is_per_block_not_per_element():
    n_heads, head_dim = 4, 16
    d_model = n_heads * head_dim
    p = _make_param((d_model, d_model))
    opt = AdamMini([p], lr=1e-3, n_heads=n_heads, head_dim=head_dim)
    _one_step(p, opt)
    state = opt.state[p]
    # v should be (n_heads,) — one scalar per head, not per element
    assert state["v"].shape == (n_heads,), f"Expected v shape ({n_heads},), got {state['v'].shape}"


# ---------------------------------------------------------------------------
# Test 6: Single-block mode — v is a scalar tensor ()
# ---------------------------------------------------------------------------


def test_single_block_v_is_scalar():
    p = _make_param((8, 8))
    opt = AdamMini([p], lr=1e-3)  # no n_heads / head_dim
    _one_step(p, opt)
    state = opt.state[p]
    assert state["v"].shape == torch.Size([]), (
        f"Expected scalar v (shape ()), got {state['v'].shape}"
    )


# ---------------------------------------------------------------------------
# Test 7: Converges on quadratic loss over 20 steps
# ---------------------------------------------------------------------------


def test_converges_on_quadratic():
    torch.manual_seed(7)
    target = torch.zeros(16)
    p = torch.nn.Parameter(torch.ones(16) * 2.0)
    opt = AdamMini([p], lr=0.1)
    losses = []
    for _ in range(20):
        opt.zero_grad()
        loss = ((p - target) ** 2).sum()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], "Loss did not decrease over 20 steps."
    # Adam-mini uses a single block-averaged v, so convergence is more
    # conservative than standard Adam — verify >95 % loss reduction in 20 steps.
    reduction = (losses[0] - losses[-1]) / losses[0]
    assert reduction > 0.95, (
        f"Expected >95% loss reduction in 20 steps, got {reduction:.2%} "
        f"(initial={losses[0]:.4f}, final={losses[-1]:.4f})"
    )


# ---------------------------------------------------------------------------
# Test 8: weight_decay non-zero modifies update
# ---------------------------------------------------------------------------


def test_weight_decay_modifies_update():
    torch.manual_seed(8)
    shape = (8,)

    p_no_wd = _make_param(shape, seed=8)
    p_wd = _make_param(shape, seed=8)

    opt_no = AdamMini([p_no_wd], lr=1e-3, weight_decay=0.0)
    opt_wd = AdamMini([p_wd], lr=1e-3, weight_decay=0.1)

    _one_step(p_no_wd, opt_no)
    _one_step(p_wd, opt_wd)

    assert not torch.allclose(p_no_wd.data, p_wd.data), (
        "weight_decay=0.1 produced identical result to weight_decay=0.0"
    )


# ---------------------------------------------------------------------------
# Test 9: beta1=0 → m equals g_t (no momentum)
# ---------------------------------------------------------------------------


def test_beta1_zero_m_equals_gradient():
    p = _make_param((4, 4), seed=9)
    opt = AdamMini([p], lr=1e-3, betas=(0.0, 0.999))

    # Manually compute expected gradient
    p.data.clone()
    loss = p.sum()
    loss.backward()
    grad = p.grad.clone()

    opt.step()

    state = opt.state[p]
    # With β₁=0: m_t = 0*m_{t-1} + 1*g_t = g_t
    assert torch.allclose(state["m"], grad), "With beta1=0, m should equal the raw gradient g_t."


# ---------------------------------------------------------------------------
# Test 10: Large gradient stability — no NaN/Inf
# ---------------------------------------------------------------------------


def test_large_gradient_stability():
    torch.manual_seed(10)
    p = torch.nn.Parameter(torch.randn(16, 16) * 1000.0)
    opt = AdamMini([p], lr=1e-3)
    opt.zero_grad()
    loss = (p * 1000.0).sum()
    loss.backward()
    opt.step()
    assert torch.isfinite(p.data).all(), "NaN/Inf in params after large-gradient step."


# ---------------------------------------------------------------------------
# Test 11: 1D params (biases) — single-block scalar v
# ---------------------------------------------------------------------------


def test_1d_param_single_block_scalar():
    p = _make_param((64,))  # 1D bias
    opt = AdamMini([p], lr=1e-3, n_heads=4, head_dim=16)
    _one_step(p, opt)
    state = opt.state[p]
    # 1D param → single block regardless of n_heads/head_dim
    assert state["v"].shape == torch.Size([]), (
        f"1D param should have scalar v, got {state['v'].shape}"
    )
    assert state["n_blocks"] == 1


# ---------------------------------------------------------------------------
# Test 12: 2D param without head config — single-block scalar v
# ---------------------------------------------------------------------------


def test_2d_param_no_head_config_single_block():
    p = _make_param((64, 64))
    opt = AdamMini([p], lr=1e-3)  # no head config
    _one_step(p, opt)
    state = opt.state[p]
    assert state["v"].shape == torch.Size([]), (
        f"2D param without head config should have scalar v, got {state['v'].shape}"
    )
    assert state["n_blocks"] == 1


# ---------------------------------------------------------------------------
# Test 13: 2D param with head config — 4 v scalars
# ---------------------------------------------------------------------------


def test_2d_param_with_head_config_four_v_scalars():
    n_heads, head_dim = 4, 16
    d_model = n_heads * head_dim  # 64
    p = _make_param((d_model, d_model))
    opt = AdamMini([p], lr=1e-3, n_heads=n_heads, head_dim=head_dim)
    _one_step(p, opt)
    state = opt.state[p]
    assert state["n_blocks"] == n_heads, f"Expected n_blocks={n_heads}, got {state['n_blocks']}"
    assert state["v"].shape == (n_heads,), f"Expected v shape ({n_heads},), got {state['v'].shape}"
    assert state["v"].numel() == n_heads  # 4 scalars


# ---------------------------------------------------------------------------
# Test 14: Head-blocked update — different blocks can have different v values
# ---------------------------------------------------------------------------


def test_head_blocked_different_v_per_block():
    """Construct a gradient with unequal magnitude per head so v diverges."""
    n_heads, head_dim = 4, 16
    d_model = n_heads * head_dim  # 64
    d_in = 32

    p = torch.nn.Parameter(torch.zeros(d_model, d_in))
    opt = AdamMini([p], lr=1e-3, n_heads=n_heads, head_dim=head_dim)

    # Craft a gradient where each head block has a different magnitude
    g = torch.zeros(d_model, d_in)
    for h in range(n_heads):
        g[h * head_dim : (h + 1) * head_dim, :] = float(h + 1)  # magnitudes 1,2,3,4
    p.grad = g.clone()
    opt.step()

    state = opt.state[p]
    v = state["v"]  # (n_heads,)
    # All four head blocks should have different v values
    assert v.numel() == n_heads
    # Because head gradients have different magnitudes, v values must differ
    assert not torch.allclose(v[0], v[-1]), (
        f"Expected different v values per head, got v={v.tolist()}"
    )
    # Sanity: values should be increasing with head index
    for i in range(n_heads - 1):
        assert v[i].item() < v[i + 1].item(), (
            f"v[{i}]={v[i].item():.6f} should be < v[{i + 1}]={v[i + 1].item():.6f}"
        )
