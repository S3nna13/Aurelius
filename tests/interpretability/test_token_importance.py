"""Tests for src/interpretability/token_importance.py

Tiny config: D=8, T=6, B=2, VOCAB=16
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.interpretability.token_importance import (
    ImportanceConfig,
    gradient_norm_importance,
    attention_rollout,
    integrated_gradients,
    smooth_importance,
    normalize_importance,
    TokenImportanceScorer,
)

D = 8
T = 6
B = 2
VOCAB = 16


# ---------------------------------------------------------------------------
# Tiny model: Embedding → Linear (differentiable)
# ---------------------------------------------------------------------------

class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(D, VOCAB, bias=False)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.proj(emb)  # (B, T, VOCAB)


def _loss_fn(logits, labels):
    return F.cross_entropy(logits.reshape(-1, VOCAB), labels.reshape(-1))


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = ImportanceConfig()
    assert cfg.method == "gradient_norm"
    assert cfg.normalize is True
    assert cfg.smooth_window == 1


# ---------------------------------------------------------------------------
# gradient_norm_importance
# ---------------------------------------------------------------------------

def test_gradient_norm_importance_shape():
    model = _TinyModel()
    emb = torch.randn(B, T, D, requires_grad=True)
    labels = torch.randint(0, VOCAB, (B, T))
    imp = gradient_norm_importance(model, emb, labels, _loss_fn)
    assert imp.shape == (B, T)


def test_gradient_norm_importance_non_negative():
    model = _TinyModel()
    emb = torch.randn(B, T, D, requires_grad=True)
    labels = torch.randint(0, VOCAB, (B, T))
    imp = gradient_norm_importance(model, emb, labels, _loss_fn)
    assert (imp >= 0).all()


# ---------------------------------------------------------------------------
# attention_rollout
# ---------------------------------------------------------------------------

def test_attention_rollout_shape():
    attn_weights = [torch.softmax(torch.randn(B, 2, T, T), dim=-1) for _ in range(3)]
    rollout = attention_rollout(attn_weights, add_residual=True)
    assert rollout.shape == (B, T, T)


def test_attention_rollout_sums_to_one():
    """Each row of the rollout should approximately sum to 1."""
    attn_weights = [torch.softmax(torch.randn(B, 2, T, T), dim=-1) for _ in range(3)]
    rollout = attention_rollout(attn_weights, add_residual=True)
    row_sums = rollout.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


# ---------------------------------------------------------------------------
# integrated_gradients
# ---------------------------------------------------------------------------

def test_integrated_gradients_shape():
    model = _TinyModel()
    emb = torch.randn(B, T, D)
    baseline = torch.zeros(B, T, D)
    labels = torch.randint(0, VOCAB, (B, T))
    imp = integrated_gradients(model, emb, baseline, labels, _loss_fn, n_steps=5)
    assert imp.shape == (B, T)


# ---------------------------------------------------------------------------
# smooth_importance
# ---------------------------------------------------------------------------

def test_smooth_importance_shape_preserved():
    imp = torch.rand(B, T)
    out = smooth_importance(imp, window=3)
    assert out.shape == (B, T)


def test_smooth_importance_window_1_unchanged():
    imp = torch.rand(B, T)
    out = smooth_importance(imp, window=1)
    assert torch.allclose(out, imp)


# ---------------------------------------------------------------------------
# normalize_importance
# ---------------------------------------------------------------------------

def test_normalize_importance_sums_to_one():
    imp = torch.rand(B, T)
    out = normalize_importance(imp)
    row_sums = out.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(B), atol=1e-6)


def test_normalize_importance_non_negative():
    imp = torch.rand(B, T)
    out = normalize_importance(imp)
    assert (out >= 0).all()


# ---------------------------------------------------------------------------
# TokenImportanceScorer
# ---------------------------------------------------------------------------

def test_scorer_score_shape():
    cfg = ImportanceConfig(method="gradient_norm", normalize=True)
    scorer = TokenImportanceScorer(cfg)
    model = _TinyModel()
    emb = torch.randn(B, T, D, requires_grad=True)
    labels = torch.randint(0, VOCAB, (B, T))
    imp = scorer.score(model, emb, labels, _loss_fn)
    assert imp.shape == (B, T)


def test_scorer_get_top_tokens_shape():
    cfg = ImportanceConfig()
    scorer = TokenImportanceScorer(cfg)
    imp = torch.rand(B, T)
    k = 3
    top = scorer.get_top_tokens(imp, k)
    assert top.shape == (B, k)


def test_scorer_get_top_tokens_valid_indices():
    cfg = ImportanceConfig()
    scorer = TokenImportanceScorer(cfg)
    imp = torch.rand(B, T)
    k = 3
    top = scorer.get_top_tokens(imp, k)
    assert (top >= 0).all()
    assert (top < T).all()
