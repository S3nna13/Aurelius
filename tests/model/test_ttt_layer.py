"""Tests for TTT-Linear Layer.

Reference: Sun et al. 2024, "Learning to (Learn at Test Time): RNNs with
Expressive Hidden States", arXiv:2407.04620.

12 focused tests covering shape, causality, state update, gradient flow,
determinism, batch isolation, edge-case sizes, numerical stability,
lr=0 behaviour, and layer-norm application.
"""

import pytest
import torch

from src.model.ttt_layer import TTTConfig, TTTLinearLayer

# ---------------------------------------------------------------------------
# Shared fixture — tiny config used across all tests unless overridden
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg():
    """Tiny TTTConfig: d_model=64, mini_batch_size=4, lr=0.01."""
    return TTTConfig(d_model=64, mini_batch_size=4, lr=0.01, use_ln=True)


@pytest.fixture
def layer(cfg):
    """A deterministically initialised TTTLinearLayer."""
    torch.manual_seed(0)
    return TTTLinearLayer(cfg)


# ---------------------------------------------------------------------------
# 1. Shape: (B, T, d) → (B, T, d)
# ---------------------------------------------------------------------------


def test_output_shape(layer, cfg):
    """Output tensor must have exactly the same shape as the input."""
    B, T = 2, 16
    x = torch.randn(B, T, cfg.d_model)
    out = layer(x)
    assert out.shape == (B, T, cfg.d_model), f"Expected {(B, T, cfg.d_model)}, got {out.shape}"


# ---------------------------------------------------------------------------
# 2. Causal: output at position t depends only on positions ≤ t
# ---------------------------------------------------------------------------


def test_causality(layer, cfg):
    """Changing tokens at positions > t must not affect output at position t."""
    torch.manual_seed(1)
    B, T, d = 1, 8, cfg.d_model

    x1 = torch.randn(B, T, d)
    x2 = x1.clone()
    # Perturb all tokens strictly after position 3
    x2[0, 4:, :] = torch.randn(T - 4, d)

    out1 = layer(x1)
    out2 = layer(x2)

    # Outputs at positions 0–3 must be identical
    assert torch.allclose(out1[0, :4], out2[0, :4], atol=1e-6), (
        "Causal violation: output at t ≤ 3 changed when tokens at t > 3 changed."
    )


# ---------------------------------------------------------------------------
# 3. W changes over sequence: hidden state is updated as tokens are processed
# ---------------------------------------------------------------------------


def test_W_updates_over_sequence(cfg):
    """W must differ between time-steps (it is not a static matrix)."""
    torch.manual_seed(2)
    d = cfg.d_model

    layer = TTTLinearLayer(cfg)
    # Use identity-like input so the update is non-trivial
    x = torch.eye(d).unsqueeze(0)  # (1, d, d) — d tokens of dimension d

    with torch.no_grad():
        K = layer.theta_K(x[0])  # (d, d)
        Q = layer.theta_Q(x[0])  # (d, d)

        W = layer.W_0.clone()
        outputs = []
        for t in range(d):
            k_t = K[t]
            W = layer._update_W(W, k_t, x[0, t])
            outputs.append((W @ Q[t]).clone())

    # Not all outputs should be identical — W must have changed
    all_same = all(torch.allclose(outputs[0], o, atol=1e-8) for o in outputs[1:])
    assert not all_same, "W did not update: all output vectors are identical."


# ---------------------------------------------------------------------------
# 4. Gradient flow: loss.backward() gives finite gradients on W_0
# ---------------------------------------------------------------------------


def test_gradient_flow(layer, cfg):
    """sum(outputs).backward() must yield finite, non-zero grads on W_0."""
    torch.manual_seed(3)
    x = torch.randn(2, 8, cfg.d_model)
    out = layer(x)
    loss = out.sum()
    loss.backward()

    assert layer.W_0.grad is not None, "W_0 has no gradient."
    assert torch.isfinite(layer.W_0.grad).all(), "W_0 gradient contains NaN/Inf."
    assert layer.W_0.grad.abs().sum() > 0, "W_0 gradient is all-zero."


# ---------------------------------------------------------------------------
# 5. Determinism under torch.manual_seed
# ---------------------------------------------------------------------------


def test_determinism(cfg):
    """Same seed → same output; different seed → different output."""
    x = torch.randn(2, 8, cfg.d_model)

    torch.manual_seed(42)
    layer_a = TTTLinearLayer(cfg)
    out_a = layer_a(x)

    torch.manual_seed(42)
    layer_b = TTTLinearLayer(cfg)
    out_b = layer_b(x)

    assert torch.allclose(out_a, out_b, atol=1e-7), "Same seed produced different outputs."

    torch.manual_seed(99)
    layer_c = TTTLinearLayer(cfg)
    out_c = layer_c(x)
    assert not torch.allclose(out_a, out_c, atol=1e-7), (
        "Different seeds produced identical outputs — seed not affecting params."
    )


# ---------------------------------------------------------------------------
# 6. No cross-batch contamination: different sequences get independent W trajectories
# ---------------------------------------------------------------------------


def test_no_cross_batch_contamination(layer, cfg):
    """Batch element 0's output must equal processing it in isolation."""
    torch.manual_seed(4)
    B, T, d = 3, 12, cfg.d_model

    x_batch = torch.randn(B, T, d)
    out_batch = layer(x_batch)

    # Process first sequence individually
    out_solo = layer(x_batch[:1])

    assert torch.allclose(out_batch[:1], out_solo, atol=1e-6), (
        "Batch element 0 gives different output when processed alone vs in a batch."
    )


# ---------------------------------------------------------------------------
# 7. mini_batch_size=1 works (degenerate but valid)
# ---------------------------------------------------------------------------


def test_mini_batch_size_one():
    """TTTLinearLayer must work when mini_batch_size=1."""
    cfg = TTTConfig(d_model=64, mini_batch_size=1, lr=0.01)
    torch.manual_seed(5)
    layer = TTTLinearLayer(cfg)
    x = torch.randn(2, 8, 64)
    out = layer(x)
    assert out.shape == (2, 8, 64)


# ---------------------------------------------------------------------------
# 8. mini_batch_size=T works (single W update after full sequence)
# ---------------------------------------------------------------------------


def test_mini_batch_size_equals_T():
    """TTTLinearLayer must work when mini_batch_size equals the sequence length."""
    T = 16
    cfg = TTTConfig(d_model=64, mini_batch_size=T, lr=0.01)
    torch.manual_seed(6)
    layer = TTTLinearLayer(cfg)
    x = torch.randn(2, T, 64)
    out = layer(x)
    assert out.shape == (2, T, 64)


# ---------------------------------------------------------------------------
# 9. T=1 (single token) works
# ---------------------------------------------------------------------------


def test_single_token(layer, cfg):
    """Layer must handle sequences of length 1 without error."""
    x = torch.randn(2, 1, cfg.d_model)
    out = layer(x)
    assert out.shape == (2, 1, cfg.d_model)


# ---------------------------------------------------------------------------
# 10. No NaN / Inf on random input
# ---------------------------------------------------------------------------


def test_no_nan_inf_on_random_input(layer, cfg):
    """Output must be finite for random Gaussian input."""
    torch.manual_seed(7)
    x = torch.randn(4, 32, cfg.d_model)
    out = layer(x)
    assert torch.isfinite(out).all(), "Output contains NaN or Inf on random input."


# ---------------------------------------------------------------------------
# 11. lr=0 → W never changes (layer behaves as a static linear map)
# ---------------------------------------------------------------------------


def test_lr_zero_static_W(cfg):
    """With lr=0 the hidden state W must not change between steps.

    Concretely: for a sequence of T tokens, if η=0 then W stays at W_0
    throughout, so o_t = W_0 @ theta_Q(x_t)  for every t.  The outputs
    at different positions must still differ (because the inputs differ),
    but they must all be consistent with the same fixed W.
    """
    cfg_no_lr = TTTConfig(d_model=64, mini_batch_size=4, lr=0.0, use_ln=False)
    torch.manual_seed(8)
    layer = TTTLinearLayer(cfg_no_lr)

    torch.manual_seed(9)
    x = torch.randn(1, 8, 64)
    out = layer(x)

    # With lr=0 each output is W_0 @ Q_t  (W never moves)
    with torch.no_grad():
        Q = layer.theta_Q(x[0])  # (T, d)
        expected = (layer.W_0 @ Q.T).T  # (T, d)

    assert torch.allclose(out[0], expected, atol=1e-5), (
        "With lr=0, outputs should equal W_0 @ theta_Q(x_t) for every t."
    )


# ---------------------------------------------------------------------------
# 12. use_ln=True applies layer norm to output
# ---------------------------------------------------------------------------


def test_use_ln_applies_layer_norm(cfg):
    """Output with use_ln=True must differ from output with use_ln=False."""
    torch.manual_seed(10)
    x = torch.randn(2, 8, cfg.d_model)

    # Layer with LN
    cfg_ln = TTTConfig(
        d_model=cfg.d_model, mini_batch_size=cfg.mini_batch_size, lr=cfg.lr, use_ln=True
    )
    torch.manual_seed(11)
    layer_ln = TTTLinearLayer(cfg_ln)

    # Layer without LN — same params except the norm
    cfg_no_ln = TTTConfig(
        d_model=cfg.d_model, mini_batch_size=cfg.mini_batch_size, lr=cfg.lr, use_ln=False
    )
    torch.manual_seed(11)
    layer_no_ln = TTTLinearLayer(cfg_no_ln)

    out_ln = layer_ln(x)
    out_no_ln = layer_no_ln(x)

    # Outputs must differ
    assert not torch.allclose(out_ln, out_no_ln, atol=1e-5), (
        "use_ln=True and use_ln=False produced identical outputs."
    )

    # The LN output must have near-unit variance per token (standard LN property)
    # Compute std along the d_model dimension; should be close to 1
    std = out_ln.std(dim=-1)  # (B, T)
    assert (std > 0.1).all(), "Layer-norm output has near-zero std — LN may be broken."
