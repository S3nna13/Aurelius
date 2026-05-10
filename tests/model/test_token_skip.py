"""Tests for src/model/token_skip.py"""

import pytest
import torch
from aurelius.model.token_skip import (
    ConfidenceGate,
    SkippableLayer,
    SkipRateLoss,
    TokenSkipModel,
)

# ---------------------------------------------------------------------------
# Shared fixtures / constants
# ---------------------------------------------------------------------------

D_MODEL = 16
N_LAYERS = 3
B = 2
T = 6


@pytest.fixture()
def x() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(B, T, D_MODEL)


@pytest.fixture()
def gate() -> ConfidenceGate:
    return ConfidenceGate(D_MODEL, threshold=0.5)


@pytest.fixture()
def model() -> TokenSkipModel:
    return TokenSkipModel(D_MODEL, N_LAYERS, threshold=0.5)


# ---------------------------------------------------------------------------
# ConfidenceGate tests  (1–4)
# ---------------------------------------------------------------------------


def test_confidence_gate_output_shape(x, gate):
    """1. forward() returns (B, T) tensor."""
    scores = gate(x)
    assert scores.shape == (B, T), f"Expected ({B}, {T}), got {scores.shape}"


def test_confidence_gate_values_in_0_1(x, gate):
    """2. Confidence scores are in [0, 1]."""
    scores = gate(x)
    assert scores.min().item() >= 0.0
    assert scores.max().item() <= 1.0


def test_exit_mask_dtype_bool(x, gate):
    """3. exit_mask() returns a bool tensor."""
    mask = gate.exit_mask(x)
    assert mask.dtype == torch.bool


def test_exit_mask_depends_on_threshold(x):
    """4. A higher threshold produces fewer exits than a lower one."""
    gate_low = ConfidenceGate(D_MODEL, threshold=0.01)
    gate_high = ConfidenceGate(D_MODEL, threshold=0.99)
    # Force identical weights so only threshold differs.
    gate_high.gate.weight.data = gate_low.gate.weight.data.clone()
    gate_high.gate.bias.data = gate_low.gate.bias.data.clone()

    mask_low = gate_low.exit_mask(x)
    mask_high = gate_high.exit_mask(x)

    # Low threshold → more exits; high threshold → fewer exits.
    assert mask_low.sum() >= mask_high.sum(), (
        "Lower threshold should produce at least as many exits as higher threshold"
    )


# ---------------------------------------------------------------------------
# SkippableLayer tests  (5–8)
# ---------------------------------------------------------------------------


def _make_skippable_layer() -> SkippableLayer:
    import torch.nn as nn

    layer = nn.Sequential(nn.Linear(D_MODEL, D_MODEL), nn.GELU())
    gate = ConfidenceGate(D_MODEL, threshold=0.5)
    return SkippableLayer(layer, gate)


def test_skippable_layer_output_shape(x):
    """5. output tensor has shape (B, T, d_model)."""
    sl = _make_skippable_layer()
    out, _ = sl(x)
    assert out.shape == (B, T, D_MODEL)


def test_skippable_layer_no_skip_mask(x):
    """6. With no skip_mask all tokens are processed (output differs from input)."""
    torch.manual_seed(42)
    sl = _make_skippable_layer()
    out, _ = sl(x, skip_mask=None)
    # The layer transforms the input, so output should generally differ.
    assert out.shape == (B, T, D_MODEL)
    # We can't guarantee every value changes, but the layer should not be an identity.
    assert not torch.allclose(out, x), "Layer with no skip should transform input"


def test_skippable_layer_all_true_skip_mask_returns_input(x):
    """7. With all-True skip_mask the layer output equals the input."""
    sl = _make_skippable_layer()
    all_skip = torch.ones(B, T, dtype=torch.bool)
    out, _ = sl(x, skip_mask=all_skip)
    assert torch.allclose(out, x), "All-skip mask should preserve input unchanged"


def test_skippable_layer_output_finite(x):
    """8. Output values are finite (no NaN / Inf)."""
    sl = _make_skippable_layer()
    out, _ = sl(x)
    assert torch.isfinite(out).all(), "SkippableLayer output contains non-finite values"


# ---------------------------------------------------------------------------
# TokenSkipModel tests  (9–13)
# ---------------------------------------------------------------------------


def test_token_skip_model_output_shape(x, model):
    """9. Model output tensor has shape (B, T, d_model)."""
    out, _ = model(x)
    assert out.shape == (B, T, D_MODEL)


def test_token_skip_model_stats_keys(x, model):
    """10. stats dict contains 'exit_fractions' and 'mean_layers_computed'."""
    _, stats = model(x)
    assert "exit_fractions" in stats
    assert "mean_layers_computed" in stats


def test_exit_fractions_length(x, model):
    """11. exit_fractions has exactly n_layers elements."""
    _, stats = model(x)
    assert len(stats["exit_fractions"]) == N_LAYERS


def test_mean_layers_computed_in_range(x, model):
    """12. mean_layers_computed is in [0, n_layers]."""
    _, stats = model(x)
    mlc = stats["mean_layers_computed"]
    assert 0.0 <= mlc <= N_LAYERS, f"mean_layers_computed={mlc} out of [0, {N_LAYERS}]"


def test_gradients_flow_through_model(x, model):
    """13. Gradients flow back through the model to the input."""
    x_req = x.requires_grad_(True)
    out, _ = model(x_req)
    loss = out.sum()
    loss.backward()
    assert x_req.grad is not None, "No gradient reached the input tensor"
    assert torch.isfinite(x_req.grad).all(), "Input gradient contains non-finite values"


# ---------------------------------------------------------------------------
# SkipRateLoss tests  (14–15)
# ---------------------------------------------------------------------------


def test_skip_rate_loss_returns_scalar():
    """14. SkipRateLoss.forward() returns a scalar tensor."""
    loss_fn = SkipRateLoss(target_skip_rate=0.5)
    fractions = [0.2, 0.5, 0.8]
    loss = loss_fn(fractions)
    assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"


def test_skip_rate_loss_zero_when_mean_equals_target():
    """15. Loss is 0 when mean(exit_fractions) equals target_skip_rate."""
    target = 0.4
    loss_fn = SkipRateLoss(target_skip_rate=target)
    # Fractions that average to exactly target.
    fractions = [0.2, 0.4, 0.6]  # mean = 0.4
    loss = loss_fn(fractions)
    assert loss.item() == pytest.approx(0.0, abs=1e-6), f"Expected loss ≈ 0, got {loss.item()}"
