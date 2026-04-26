"""Tests for src/eval/integrated_gradients.py."""

import pytest
import torch

from src.eval.integrated_gradients import (
    IGConfig,
    IntegratedGradientsExplainer,
    compute_integrated_gradients,
    interpolate_inputs,
    smooth_grad,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )
    model = AureliusTransformer(cfg)
    model.eval()
    return model


@pytest.fixture(scope="module")
def input_ids():
    torch.manual_seed(0)
    return torch.randint(0, 256, (1, 8))


# ---------------------------------------------------------------------------
# interpolate_inputs tests
# ---------------------------------------------------------------------------


def test_interpolate_inputs_zero_alpha():
    """alpha=0 should return the baseline exactly."""
    torch.manual_seed(0)
    baseline = torch.zeros(1, 8, 64)
    inputs = torch.randn(1, 8, 64)
    result = interpolate_inputs(baseline, inputs, alpha=0.0)
    assert torch.allclose(result, baseline), "alpha=0 should return baseline"


def test_interpolate_inputs_one_alpha():
    """alpha=1 should return the inputs exactly."""
    torch.manual_seed(0)
    baseline = torch.zeros(1, 8, 64)
    inputs = torch.randn(1, 8, 64)
    result = interpolate_inputs(baseline, inputs, alpha=1.0)
    assert torch.allclose(result, inputs), "alpha=1 should return inputs"


def test_interpolate_inputs_shape():
    """Output shape must match input shape."""
    torch.manual_seed(0)
    baseline = torch.zeros(2, 5, 32)
    inputs = torch.randn(2, 5, 32)
    result = interpolate_inputs(baseline, inputs, alpha=0.5)
    assert result.shape == inputs.shape, f"Expected shape {inputs.shape}, got {result.shape}"


# ---------------------------------------------------------------------------
# compute_integrated_gradients tests
# ---------------------------------------------------------------------------


def test_integrated_gradients_shape(tiny_model, input_ids):
    """compute_integrated_gradients should return a (T,) tensor."""
    T = input_ids.shape[1]
    attrs = compute_integrated_gradients(
        model=tiny_model,
        input_ids=input_ids,
        target_token_idx=1,
        target_pos=0,
        n_steps=5,
    )
    assert attrs.shape == (T,), f"Expected shape ({T},), got {attrs.shape}"


def test_integrated_gradients_nonnegative(tiny_model, input_ids):
    """All attribution values must be >= 0 (absolute values are returned)."""
    attrs = compute_integrated_gradients(
        model=tiny_model,
        input_ids=input_ids,
        target_token_idx=1,
        target_pos=0,
        n_steps=5,
    )
    assert (attrs >= 0).all(), "Attribution scores must be non-negative"


# ---------------------------------------------------------------------------
# IGConfig tests
# ---------------------------------------------------------------------------


def test_ig_config_defaults():
    """IGConfig() should have the specified default values."""
    cfg = IGConfig()
    assert cfg.n_steps == 50
    assert cfg.baseline_token == 0
    assert cfg.normalize is True


# ---------------------------------------------------------------------------
# IntegratedGradientsExplainer tests
# ---------------------------------------------------------------------------


def test_explainer_explain_shape(tiny_model, input_ids):
    """explain() should return a (T,) tensor."""
    T = input_ids.shape[1]
    cfg = IGConfig(n_steps=5)
    explainer = IntegratedGradientsExplainer(tiny_model, cfg)
    attrs = explainer.explain(input_ids, target_pos=0, target_token_idx=1)
    assert attrs.shape == (T,), f"Expected shape ({T},), got {attrs.shape}"


def test_explainer_explain_normalized(tiny_model, input_ids):
    """With normalize=True, attributions should sum to ~1.0."""
    cfg = IGConfig(n_steps=5, normalize=True)
    explainer = IntegratedGradientsExplainer(tiny_model, cfg)
    attrs = explainer.explain(input_ids, target_pos=0, target_token_idx=1)
    total = attrs.sum().item()
    assert abs(total - 1.0) < 1e-5, f"Normalized attributions should sum to 1.0, got {total}"


def test_explainer_top_k_tokens(tiny_model, input_ids):
    """top_k_tokens should return k indices sorted by attribution descending."""
    T = input_ids.shape[1]
    cfg = IGConfig(n_steps=5, normalize=True)
    explainer = IntegratedGradientsExplainer(tiny_model, cfg)
    attrs = explainer.explain(input_ids, target_pos=0, target_token_idx=1)

    k = 3
    top_k = explainer.top_k_tokens(attrs, k=k)

    assert len(top_k) == k, f"Expected {k} indices, got {len(top_k)}"
    # Verify descending order: each element should have attribution >= next
    for i in range(len(top_k) - 1):
        assert attrs[top_k[i]] >= attrs[top_k[i + 1]], (
            f"Indices not sorted by attribution: {attrs[top_k[i]]} < {attrs[top_k[i + 1]]}"
        )
    # All indices must be in valid range
    for idx in top_k:
        assert 0 <= idx < T, f"Index {idx} out of range [0, {T})"


# ---------------------------------------------------------------------------
# smooth_grad tests
# ---------------------------------------------------------------------------


def test_smooth_grad_shape(tiny_model, input_ids):
    """smooth_grad should return a (T,) tensor."""
    T = input_ids.shape[1]
    attrs = smooth_grad(
        model=tiny_model,
        input_ids=input_ids,
        target_token_idx=1,
        target_pos=0,
        n_samples=2,
        noise_std=0.01,
    )
    assert attrs.shape == (T,), f"Expected shape ({T},), got {attrs.shape}"
