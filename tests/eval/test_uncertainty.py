"""Tests for src/eval/uncertainty.py -- pure PyTorch, no HuggingFace."""

from __future__ import annotations

import math
import sys
import os

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Make sure the project root is on sys.path so the import works whether pytest
# is run from the repo root or from a sub-directory.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.eval.uncertainty import (
    UncertaintyConfig,
    MCDropoutEstimator,
    compute_epistemic_aleatoric_split,
    compute_mutual_information,
    compute_predictive_entropy,
    compute_sequence_confidence,
    compute_token_uncertainty,
    detect_uncertainty_threshold,
)

# ---------------------------------------------------------------------------
# Tiny dims used throughout the tests
# ---------------------------------------------------------------------------
B = 3        # batch size
T = 4        # sequence length
V = 16       # vocabulary size  (small but > 1)
S = 5        # MC samples


# ===========================================================================
# 1. UncertaintyConfig defaults
# ===========================================================================

def test_uncertainty_config_defaults():
    cfg = UncertaintyConfig()
    assert cfg.n_mc_samples == 10
    assert cfg.dropout_rate == 0.1
    assert cfg.temperature == 1.0
    assert cfg.top_k == 0
    assert cfg.use_ensemble is False


# ===========================================================================
# 2. compute_predictive_entropy -- shape (B,) from (B, vocab)
# ===========================================================================

def test_predictive_entropy_shape_2d():
    probs = torch.softmax(torch.randn(B, V), dim=-1)
    out = compute_predictive_entropy(probs)
    assert out.shape == (B,), f"Expected ({B},), got {out.shape}"


# ===========================================================================
# 3. compute_predictive_entropy -- shape (B, T) from (B, T, vocab)
# ===========================================================================

def test_predictive_entropy_shape_3d():
    probs = torch.softmax(torch.randn(B, T, V), dim=-1)
    out = compute_predictive_entropy(probs)
    assert out.shape == (B, T), f"Expected ({B},{T}), got {out.shape}"


# ===========================================================================
# 4. compute_predictive_entropy -- non-negative
# ===========================================================================

def test_predictive_entropy_non_negative():
    probs = torch.softmax(torch.randn(B, V), dim=-1)
    out = compute_predictive_entropy(probs)
    assert (out >= -1e-6).all(), "Entropy should be non-negative"


# ===========================================================================
# 5. compute_predictive_entropy -- uniform probs -> max entropy = log(vocab)
# ===========================================================================

def test_predictive_entropy_uniform_is_max():
    uniform = torch.full((B, V), 1.0 / V)
    out = compute_predictive_entropy(uniform)
    expected = math.log(V)
    assert torch.allclose(out, torch.full((B,), expected), atol=1e-5), (
        f"Uniform entropy should be log({V})={expected:.4f}, got {out}"
    )


# ===========================================================================
# 6. compute_predictive_entropy -- one-hot probs -> zero entropy
# ===========================================================================

def test_predictive_entropy_one_hot_is_zero():
    one_hot = torch.zeros(B, V)
    one_hot[:, 0] = 1.0
    out = compute_predictive_entropy(one_hot)
    assert torch.allclose(out, torch.zeros(B), atol=1e-5), (
        f"One-hot entropy should be 0, got {out}"
    )


# ===========================================================================
# 7. compute_mutual_information -- shape (B,)
# ===========================================================================

def test_mutual_information_shape():
    probs_samples = torch.softmax(torch.randn(S, B, V), dim=-1)
    out = compute_mutual_information(probs_samples)
    assert out.shape == (B,), f"Expected ({B},), got {out.shape}"


# ===========================================================================
# 8. compute_mutual_information -- non-negative
# ===========================================================================

def test_mutual_information_non_negative():
    probs_samples = torch.softmax(torch.randn(S, B, V), dim=-1)
    out = compute_mutual_information(probs_samples)
    assert (out >= -1e-6).all(), "MI should be non-negative"


# ===========================================================================
# 9. compute_mutual_information -- 0 when all samples identical
# ===========================================================================

def test_mutual_information_zero_when_identical():
    # All S samples are the same distribution -> MI = 0
    single = torch.softmax(torch.randn(1, B, V), dim=-1)
    probs_samples = single.expand(S, B, V)
    out = compute_mutual_information(probs_samples)
    assert torch.allclose(out, torch.zeros(B), atol=1e-5), (
        f"MI should be 0 for identical samples, got {out}"
    )


# ===========================================================================
# 10. compute_token_uncertainty -- shape (B, T)
# ===========================================================================

def test_token_uncertainty_shape():
    logits = torch.randn(B, T, V)
    out = compute_token_uncertainty(logits, temperature=1.0)
    assert out.shape == (B, T), f"Expected ({B},{T}), got {out.shape}"


# ===========================================================================
# 11. compute_token_uncertainty -- non-negative
# ===========================================================================

def test_token_uncertainty_non_negative():
    logits = torch.randn(B, T, V)
    out = compute_token_uncertainty(logits)
    assert (out >= -1e-6).all(), "Token uncertainty should be non-negative"


# ===========================================================================
# 12. MCDropoutEstimator.sample -- output shape (n_mc_samples, B, T, vocab)
# ===========================================================================

class _TinyLM(nn.Module):
    """Minimal causal LM for testing: embedding + dropout + linear head."""

    def __init__(self, vocab: int, d_model: int = 8) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.drop = nn.Dropout(p=0.5)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:  # (B, T) -> (B, T, V)
        x = self.embed(input_ids)
        x = self.drop(x)
        return self.head(x)


def test_mc_dropout_sample_shape():
    cfg = UncertaintyConfig(n_mc_samples=S)
    model = _TinyLM(vocab=V)
    estimator = MCDropoutEstimator(model, cfg)
    input_ids = torch.randint(0, V, (B, T))
    out = estimator.sample(input_ids)
    assert out.shape == (S, B, T, V), f"Expected ({S},{B},{T},{V}), got {out.shape}"


# ===========================================================================
# 13. compute_sequence_confidence -- in (0, 1] for valid log_probs
# ===========================================================================

def test_sequence_confidence_range():
    # log-probs should be <= 0; use log of uniform
    log_probs = torch.full((T,), math.log(1.0 / V))
    conf = compute_sequence_confidence(log_probs)
    assert 0.0 < conf.item() <= 1.0 + 1e-7, (
        f"Confidence should be in (0,1], got {conf.item()}"
    )


# ===========================================================================
# 14. detect_uncertainty_threshold -- returns bool tensor (B,)
# ===========================================================================

def test_detect_uncertainty_threshold_shape_and_dtype():
    uncertainties = torch.rand(B, T)
    mask = detect_uncertainty_threshold(uncertainties, threshold=0.5)
    assert mask.shape == (B,), f"Expected ({B},), got {mask.shape}"
    assert mask.dtype == torch.bool, f"Expected bool dtype, got {mask.dtype}"


# ===========================================================================
# 15. compute_epistemic_aleatoric_split -- shapes (B,) each
# ===========================================================================

def test_epistemic_aleatoric_shapes():
    probs_samples = torch.softmax(torch.randn(S, B, V), dim=-1)
    epistemic, aleatoric = compute_epistemic_aleatoric_split(probs_samples)
    assert epistemic.shape == (B,), f"epistemic: expected ({B},), got {epistemic.shape}"
    assert aleatoric.shape == (B,), f"aleatoric: expected ({B},), got {aleatoric.shape}"


# ===========================================================================
# 16. epistemic + aleatoric ~= total entropy
# ===========================================================================

def test_epistemic_plus_aleatoric_equals_total():
    probs_samples = torch.softmax(torch.randn(S, B, V), dim=-1)
    epistemic, aleatoric = compute_epistemic_aleatoric_split(probs_samples)
    total = compute_predictive_entropy(probs_samples.mean(dim=0))
    reconstructed = epistemic + aleatoric
    assert torch.allclose(reconstructed, total, atol=1e-5), (
        f"epistemic+aleatoric should equal total entropy.\n"
        f"  total={total}\n  reconstructed={reconstructed}"
    )
