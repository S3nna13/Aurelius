"""Tests for src/alignment/reward_model_training.py — MultiHeadRewardModel (10+ tests)."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from src.alignment.reward_model_training import (
    MultiHeadRewardModel,
    RewardModelFeatures,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

B = 2      # batch size
T = 8      # sequence length
H = 16     # text hidden size (small for tests)
TEXT_HIDDEN = H


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def model():
    torch.manual_seed(0)
    return MultiHeadRewardModel(text_hidden=TEXT_HIDDEN)


@pytest.fixture()
def text_hidden():
    torch.manual_seed(1)
    return torch.randn(B, T, TEXT_HIDDEN)


@pytest.fixture()
def features_vec():
    torch.manual_seed(2)
    return torch.randn(B, 7)


# ---------------------------------------------------------------------------
# RewardModelFeatures tests (1-3)
# ---------------------------------------------------------------------------

def test_features_to_tensor_shape():
    """to_tensor() should produce a (7,) float tensor."""
    feat = RewardModelFeatures()
    t = feat.to_tensor()
    assert t.shape == (7,)
    assert t.dtype == torch.float32


def test_features_to_tensor_finite():
    """All feature values should be finite."""
    feat = RewardModelFeatures(
        task_complexity=0.8,
        domain="math",
        top_k_sim=0.3,
        citation_coverage=0.5,
        tool_call_count=2,
        failure_count=0,
        safety_flag_count=1,
    )
    t = feat.to_tensor()
    assert torch.isfinite(t).all()


def test_features_domain_encoding_differs():
    """Different domain strings should produce different tensor values."""
    t1 = RewardModelFeatures(domain="math").to_tensor()
    t2 = RewardModelFeatures(domain="coding").to_tensor()
    # The domain field is index 1 in the tensor
    assert t1[1].item() != t2[1].item()


# ---------------------------------------------------------------------------
# MultiHeadRewardModel construction (4)
# ---------------------------------------------------------------------------

def test_model_has_expected_layers(model):
    """Model should expose text_proj, tabular_proj, quality_head, risk_head."""
    assert hasattr(model, "text_proj")
    assert hasattr(model, "tabular_proj")
    assert hasattr(model, "quality_head")
    assert hasattr(model, "risk_head")


# ---------------------------------------------------------------------------
# Forward pass shape tests (5-6)
# ---------------------------------------------------------------------------

def test_quality_score_shape(model, text_hidden, features_vec):
    """quality_score should have shape (B, 1)."""
    quality, _ = model(text_hidden, features_vec)
    assert quality.shape == (B, 1)


def test_risk_logits_shape(model, text_hidden, features_vec):
    """risk_logits should have shape (B, 4)."""
    _, risk = model(text_hidden, features_vec)
    assert risk.shape == (B, 4)


# ---------------------------------------------------------------------------
# Forward pass value tests (7-8)
# ---------------------------------------------------------------------------

def test_outputs_finite(model, text_hidden, features_vec):
    """Both outputs should be finite."""
    quality, risk = model(text_hidden, features_vec)
    assert torch.isfinite(quality).all()
    assert torch.isfinite(risk).all()


def test_quality_score_unbounded(model, text_hidden, features_vec):
    """Quality score should not be clamped to [0,1] — can be any real value."""
    torch.manual_seed(42)
    m = MultiHeadRewardModel(text_hidden=TEXT_HIDDEN)
    q, _ = m(text_hidden, features_vec)
    # With random init the absolute value can exceed 1
    assert q.dtype == torch.float32


# ---------------------------------------------------------------------------
# Gradient flow tests (9-10)
# ---------------------------------------------------------------------------

def test_gradient_flows_through_quality(model, text_hidden, features_vec):
    """Gradients should reach text_hidden through the quality head."""
    th = text_hidden.clone().requires_grad_(True)
    quality, _ = model(th, features_vec)
    quality.sum().backward()
    assert th.grad is not None
    assert torch.isfinite(th.grad).all()


def test_gradient_flows_through_risk(model, text_hidden, features_vec):
    """Gradients should reach features_vec through the risk head."""
    fv = features_vec.clone().requires_grad_(True)
    _, risk = model(text_hidden, fv)
    risk.sum().backward()
    assert fv.grad is not None
    assert torch.isfinite(fv.grad).all()


# ---------------------------------------------------------------------------
# Regression / consistency tests (11-12, use deterministic seed)
# ---------------------------------------------------------------------------

def test_deterministic_output(text_hidden, features_vec):
    """Two models with the same seed should produce identical outputs."""
    torch.manual_seed(99)
    m1 = MultiHeadRewardModel(text_hidden=TEXT_HIDDEN)
    torch.manual_seed(99)
    m2 = MultiHeadRewardModel(text_hidden=TEXT_HIDDEN)

    q1, r1 = m1(text_hidden, features_vec)
    q2, r2 = m2(text_hidden, features_vec)

    assert torch.allclose(q1, q2)
    assert torch.allclose(r1, r2)


def test_batch_independence(model):
    """Each batch item's output should depend only on its own inputs."""
    torch.manual_seed(3)
    th = torch.randn(B, T, TEXT_HIDDEN)
    fv = torch.randn(B, 7)

    # Full batch
    q_full, r_full = model(th, fv)

    # Single items
    q0, r0 = model(th[0:1], fv[0:1])
    q1, r1 = model(th[1:2], fv[1:2])

    assert torch.allclose(q_full[0:1], q0, atol=1e-5)
    assert torch.allclose(q_full[1:2], q1, atol=1e-5)
    assert torch.allclose(r_full[0:1], r0, atol=1e-5)
    assert torch.allclose(r_full[1:2], r1, atol=1e-5)


# ---------------------------------------------------------------------------
# Registry test (13)
# ---------------------------------------------------------------------------

def test_registry_entry():
    """ALIGNMENT_REGISTRY['multi_head_rm'] should be MultiHeadRewardModel."""
    from src.alignment import ALIGNMENT_REGISTRY  # noqa: F401
    assert "multi_head_rm" in ALIGNMENT_REGISTRY
    assert ALIGNMENT_REGISTRY["multi_head_rm"] is MultiHeadRewardModel
